#!/usr/bin/env python3
"""Validate reCamera Media1 discovery and its RTSP ONVIF metadata track."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET


SOAP_OPEN = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
 xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
<s:Body>"""
SOAP_CLOSE = "</s:Body></s:Envelope>"
ROOT_START = re.compile(
    rb"<(?P<prefix>[A-Za-z_][\w.-]*:)?(?P<name>MetadataStream|MetaDataStream)\b"
)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def soap(endpoint: str, operation: str) -> ET.Element:
    body = (SOAP_OPEN + operation + SOAP_CLOSE).encode()
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/soap+xml; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return ET.fromstring(response.read())


def first_text(root: ET.Element, name: str) -> str | None:
    for element in root.iter():
        if local_name(element.tag) == name and element.text:
            return element.text.strip()
    return None


def extract_document(data: bytes) -> bytes | None:
    start = ROOT_START.search(data)
    if start is None:
        return None
    prefix = start.group("prefix") or b""
    name = start.group("name")
    closing = re.compile(rb"</" + re.escape(prefix + name) + rb"\s*>")
    end = closing.search(data, start.end())
    return None if end is None else data[start.start() : end.end()]


def capture_metadata(ffmpeg: str, uri: str, seconds: float) -> tuple[bytes, str]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-allowed_media_types",
        "data",
        "-i",
        uri,
        "-map",
        "0:d:0",
        "-c",
        "copy",
        "-f",
        "data",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
    return stdout, stderr.decode("utf-8", "replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="192.168.42.1")
    parser.add_argument("--service-port", type=int, default=8000)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--capture-seconds", type=float, default=4.0)
    args = parser.parse_args()

    endpoint = f"http://{args.host}:{args.service_port}/onvif/device_service"
    services = soap(
        endpoint,
        '<tds:GetServices xmlns:tds="http://www.onvif.org/ver10/device/wsdl">'
        "<tds:IncludeCapability>false</tds:IncludeCapability></tds:GetServices>",
    )
    namespaces = [
        (element.text or "").strip()
        for element in services.iter()
        if local_name(element.tag) == "Namespace"
    ]
    media1_advertised = "http://www.onvif.org/ver10/media/wsdl" in namespaces

    profiles = soap(endpoint, "<trt:GetProfiles/>")
    metadata_profiles = []
    for profile in profiles.iter():
        if local_name(profile.tag) != "Profiles":
            continue
        if any(local_name(child.tag) == "MetadataConfiguration" for child in profile):
            metadata_profiles.append(profile.attrib.get("token", ""))

    stream = soap(
        endpoint,
        "<trt:GetStreamUri><trt:StreamSetup>"
        "<tt:Stream xmlns:tt=\"http://www.onvif.org/ver10/schema\">RTP-Unicast</tt:Stream>"
        "<tt:Transport xmlns:tt=\"http://www.onvif.org/ver10/schema\">"
        "<tt:Protocol>RTSP</tt:Protocol></tt:Transport></trt:StreamSetup>"
        f"<trt:ProfileToken>{metadata_profiles[0] if metadata_profiles else 'live0'}</trt:ProfileToken>"
        "</trt:GetStreamUri>",
    )
    uri = first_text(stream, "Uri") or f"rtsp://{args.host}:8554/live0"

    probe = subprocess.run(
        [
            args.ffprobe,
            "-v",
            "error",
            "-rtsp_transport",
            "tcp",
            "-show_entries",
            "stream=index,codec_type,codec_name",
            "-of",
            "json",
            uri,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    streams = json.loads(probe.stdout or "{}").get("streams", [])
    video_seen = any(item.get("codec_type") == "video" for item in streams)
    data_seen = any(item.get("codec_type") == "data" for item in streams)

    # cvi_rtsp releases a just-probed TCP session asynchronously.
    time.sleep(0.5)
    raw, ffmpeg_error = capture_metadata(args.ffmpeg, uri, args.capture_seconds)
    document = extract_document(raw)
    frame_count = object_count = human_count = 0
    xml_valid = False
    if document is not None:
        try:
            root = ET.fromstring(document)
            xml_valid = True
            for element in root.iter():
                name = local_name(element.tag)
                frame_count += name == "Frame"
                object_count += name == "Object"
                if name == "Type" and (element.text or "").strip().lower() == "human":
                    human_count += 1
        except ET.ParseError:
            pass

    result = {
        "endpoint": endpoint,
        "media1_advertised": media1_advertised,
        "metadata_profiles": metadata_profiles,
        "rtsp_uri": uri,
        "streams": streams,
        "video_seen": video_seen,
        "data_seen": data_seen,
        "metadata_bytes": len(raw),
        "xml_valid": xml_valid,
        "frame_count": frame_count,
        "object_count": object_count,
        "human_count": human_count,
        "ffmpeg_error": ffmpeg_error.strip(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if media1_advertised and metadata_profiles and video_seen and data_seen and xml_valid else 1


if __name__ == "__main__":
    sys.exit(main())
