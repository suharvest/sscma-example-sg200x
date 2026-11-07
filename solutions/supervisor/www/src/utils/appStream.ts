import { baseIP } from "@/utils/supervisorRequest";
import { IAppManifest, IDebugWsInfo } from "@/api/app/app";

/** Device hostname the browser should use to reach camera services. */
export function getDeviceHost(): string {
  try {
    return new URL(baseIP).hostname;
  } catch (e) {
    return window.location.hostname;
  }
}

/** Fill the `{host}` placeholder in manifest rtsp_url. */
export function resolveRtspUrl(app?: IAppManifest | null): string {
  if (!app?.rtsp_url) return "";
  return app.rtsp_url.replace(/\{host\}/g, getDeviceHost());
}

export type StreamMode = "ws" | "rtsp" | "both" | "none";

/**
 * Stream mode the app is expected to run with: the manifest's declared
 * default, else inferred from the debug_ws / rtsp_url endpoints.
 *
 * `onvifService`: the console ONVIF device service is on. A VMS needs RTSP,
 * so yolo-detector adds it to its WebSocket-only default at startup; a
 * declared "ws" with an rtsp_url is then reported as "both". Apps that do not
 * (face-recognition) are reported the same way: their RTSP address is shown
 * without the "off by default" hint, as before stream_mode existed.
 */
export function resolveStreamMode(
  app?: IAppManifest | null,
  onvifService = false
): StreamMode {
  if (!app) return "none";
  const ws = !!app.debug_ws?.port;
  const rtsp = !!app.rtsp_url;
  const mode: StreamMode =
    app.stream_mode ??
    (ws && rtsp ? "both" : ws ? "ws" : rtsp ? "rtsp" : "none");
  return mode === "ws" && rtsp && onvifService ? "both" : mode;
}

export const streamModeHasRtsp = (mode: StreamMode): boolean =>
  mode === "rtsp" || mode === "both";

function normalizePath(path: string | undefined, fallback: string): string {
  const p = path || fallback;
  return p.startsWith("/") ? p : `/${p}`;
}

/** ws:// URL for the debug H.264 video stream, null when unsupported. */
export function resolveDebugVideoUrl(app?: IAppManifest | null): string | null {
  const dbg: IDebugWsInfo | undefined = app?.debug_ws;
  if (!dbg?.port) return null;
  return `ws://${getDeviceHost()}:${dbg.port}${normalizePath(
    dbg.video_path,
    "/"
  )}`;
}

/** ws:// URL for the inference results channel, null when unsupported. */
export function resolveDebugResultsUrl(
  app?: IAppManifest | null
): string | null {
  const dbg: IDebugWsInfo | undefined = app?.debug_ws;
  if (!dbg?.port) return null;
  return `ws://${getDeviceHost()}:${dbg.port}${normalizePath(
    dbg.results_path,
    "/results"
  )}`;
}
