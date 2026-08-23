/*
 * ONVIF Device plus Media1/Media2 SOAP services over HTTP.
 *
 * Transport is libwebsockets, on its own context and its own thread. The
 * reasoning is in docs/onvif-implementation-spec.md 14.9.1: lws is statically
 * linked into every binary already, so reusing it costs no dependency, whereas
 * the hand-rolled minimal HTTP server this file replaces in the plan would have
 * been ~200 lines of new, unreviewed, network-facing code -- and section 14.9
 * itself conceded that TLS or static assets would later force a move onto lws
 * anyway.
 *
 * Own context rather than sharing debug_stream's, because onvif_service has to
 * link into applications that have no debug stream at all, and because tying
 * ONVIF availability to whether the Live preview happens to be enabled is the
 * wrong coupling.
 *
 * On the XML: assembled from string templates, and requests are parsed only far
 * enough to recognise the operation name. That is the same bargain
 * onvif_discovery.cpp makes when it reads nothing from a Probe but the
 * MessageID, and it is what keeps "no XML library" a safe choice rather than a
 * reckless one -- there is no document model to confuse, no entity expansion,
 * no namespace resolution. Everything the device says is built from values it
 * already owns; everything the client says is a verb and nothing more.
 *
 * NOT implemented, deliberately, and tracked rather than silently missing:
 * Events (PullPoint). Analytics results travel over MQTT today (onvif_meta), and
 * discovery-plus-streaming -- the phase 1 goal -- does not pass through Events.
 * Profile T conformance would require it.
 */

#include "onvif_service.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include <libwebsockets.h>

#include "rtsp_server.h"

#define OSOAP_TAG "onvif_soap"

namespace {

/* ------------------------------------------------------------------------ */
/* State                                                                     */
/* ------------------------------------------------------------------------ */

struct SoapState {
    struct lws_context* ctx = nullptr;
    std::thread worker;
    std::atomic<bool> running { false };
    std::atomic<unsigned long> served { 0 };
    std::atomic<unsigned long> unknown { 0 };
    onvif_service_config cfg;
    std::string uuid;
};

SoapState g_ss;

/* Per-connection: the request body accumulates across HTTP_BODY callbacks and
 * the reply is written from HTTP_WRITEABLE, so both need somewhere to live for
 * the length of the transaction. */
struct PerSession {
    std::string* body = nullptr;
    std::string* reply = nullptr;
    size_t sent = 0;
};

/* ------------------------------------------------------------------------ */
/* Small helpers                                                             */
/* ------------------------------------------------------------------------ */

std::string xml_escape(const std::string& in)
{
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        switch (c) {
        case '&':  out += "&amp;";  break;
        case '<':  out += "&lt;";   break;
        case '>':  out += "&gt;";   break;
        case '"':  out += "&quot;"; break;
        case '\'': out += "&apos;"; break;
        default:   out += c;
        }
    }
    return out;
}

/*
 * Recognise the operation from the SOAP body.
 *
 * Namespace-prefix agnostic on purpose: clients write tds:GetCapabilities,
 * ns0:GetCapabilities or bare GetCapabilities interchangeably, and rejecting an
 * unfamiliar prefix presents as "works with ODM, invisible to Milestone" --
 * exactly the class of bug the same decision in onvif_discovery.cpp avoids.
 *
 * Matching requires the name to appear as an element tag, i.e. after '<' with
 * only prefix characters in between, and to be followed by a tag terminator.
 * Without that last check GetProfiles would also match GetProfilesResponse, and
 * a device that echoes a client's own response back at it is a puzzle nobody
 * enjoys.
 */
bool has_op(const std::string& body, const char* op)
{
    const size_t oplen = strlen(op);
    size_t pos = 0;
    while ((pos = body.find(op, pos)) != std::string::npos) {
        const size_t after = pos + oplen;
        if (after >= body.size()) return false;
        const char t = body[after];
        if (t != '>' && t != ' ' && t != '/' && t != '\t' && t != '\r' && t != '\n') {
            pos = after;
            continue;
        }
        const size_t lt = body.rfind('<', pos);
        if (lt == std::string::npos) { pos = after; continue; }
        bool tag = true;
        for (size_t k = lt + 1; k < pos; ++k) {
            const char c = body[k];
            if (!(isalnum(static_cast<unsigned char>(c)) || c == ':' || c == '_' || c == '-')) {
                tag = false;
                break;
            }
        }
        if (tag) return true;
        pos = after;
    }
    return false;
}

/* Contents of the first <...:Tag>value</...:Tag>. Used for the handful of
 * request parameters that matter (a profile token, a clock value); anything
 * absent yields "" and the caller falls back to a default. */
std::string tag_value(const std::string& body, const char* tag)
{
    const size_t oplen = strlen(tag);
    size_t pos = 0;
    while ((pos = body.find(tag, pos)) != std::string::npos) {
        const size_t lt = body.rfind('<', pos);
        if (lt == std::string::npos || (lt + 1 < body.size() && body[lt + 1] == '/')) {
            pos += oplen;
            continue;
        }
        const size_t gt = body.find('>', pos);
        if (gt == std::string::npos) return "";
        const size_t end = body.find('<', gt + 1);
        if (end == std::string::npos) return "";
        std::string v = body.substr(gt + 1, end - gt - 1);
        const size_t b = v.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) return "";
        const size_t e = v.find_last_not_of(" \t\r\n");
        return v.substr(b, e - b + 1);
    }
    return "";
}

/* The address this request arrived on, so a device reachable over usb0 and eth0
 * hands each client URLs it can actually route to. Mirrors what discovery does
 * for XAddrs; a wrong answer degrades to unreachable URLs, not a crash. */
std::string local_ip_of(struct lws* wsi)
{
    char name[64] = { 0 };
    char rip[64] = { 0 };
    const int fd = lws_get_socket_fd(wsi);
    if (fd >= 0) {
        struct sockaddr_in me {};
        socklen_t len = sizeof(me);
        if (getsockname(fd, (struct sockaddr*)&me, &len) == 0) {
            char buf[INET_ADDRSTRLEN];
            if (inet_ntop(AF_INET, &me.sin_addr, buf, sizeof(buf))) return buf;
        }
    }
    (void)name; (void)rip;
    return "127.0.0.1";
}

std::string device_service_url(const std::string& ip)
{
    char b[192];
    snprintf(b, sizeof(b), "http://%s:%d%s", ip.c_str(), g_ss.cfg.service_port,
        g_ss.cfg.service_path.c_str());
    return b;
}

std::string utc_now()
{
    struct timespec ts {};
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tmv {};
    time_t t = ts.tv_sec;
    gmtime_r(&t, &tmv);
    char b[40];
    snprintf(b, sizeof(b), "%04d-%02d-%02dT%02d:%02d:%02dZ",
        tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
        tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
    return b;
}

/* ------------------------------------------------------------------------ */
/* Envelope                                                                  */
/* ------------------------------------------------------------------------ */

/*
 * Every namespace a response might use is declared once on the Envelope.
 * Declaring them all unconditionally costs a few hundred bytes per reply and
 * removes an entire failure mode: a response that uses a prefix it forgot to
 * declare is not well-formed XML, and strict clients drop it without a word.
 */
const char kEnvOpen[] =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
    "<s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\""
    " xmlns:tt=\"http://www.onvif.org/ver10/schema\""
    " xmlns:tds=\"http://www.onvif.org/ver10/device/wsdl\""
    " xmlns:trt=\"http://www.onvif.org/ver10/media/wsdl\""
    " xmlns:tr2=\"http://www.onvif.org/ver20/media/wsdl\""
    ">"
    "<s:Body>";

const char kEnvClose[] = "</s:Body></s:Envelope>";

std::string envelope(const std::string& body)
{
    return std::string(kEnvOpen) + body + kEnvClose;
}

std::string soap_fault(const char* code, const char* subcode, const char* reason)
{
    std::string x;
    x += "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";
    x += "<s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\""
         " xmlns:ter=\"http://www.onvif.org/ver10/error\">";
    x += "<s:Body><s:Fault>";
    x += "<s:Code><s:Value>s:";
    x += code;
    x += "</s:Value><s:Subcode><s:Value>ter:";
    x += subcode;
    x += "</s:Value></s:Subcode></s:Code>";
    x += "<s:Reason><s:Text xml:lang=\"en\">";
    x += xml_escape(reason);
    x += "</s:Text></s:Reason>";
    x += "</s:Fault></s:Body></s:Envelope>";
    return x;
}

/* ------------------------------------------------------------------------ */
/* Media facts, asked of the components that own them                        */
/* ------------------------------------------------------------------------ */

/*
 * Profiles come from the RTSP server rather than a table here. That is the
 * whole reason rtsp_server was given self-description: GetStreamUri has to
 * answer with the port and session name actually in use, and the alternative --
 * a second copy of those values in this file -- is precisely how the ":554"
 * string stayed wrong in eight applications for as long as it did.
 */
int profile_count()
{
    const int n = rtsp_server_session_count();
    return n > 0 ? n : 0;
}

std::string profile_token(int idx)
{
    const char* s = rtsp_server_session_name(idx);
    return s != nullptr ? std::string(s) : ("profile" + std::to_string(idx));
}

std::string stream_uri(const std::string& ip, int idx)
{
    char b[256];
    if (rtsp_server_url(b, sizeof(b), ip.c_str(), idx) < 0) return "";
    return b;
}

std::string snapshot_uri(const std::string& ip)
{
    if (g_ss.cfg.snapshot_port <= 0) return "";
    char b[192];
    snprintf(b, sizeof(b), "http://%s:%d%s", ip.c_str(), g_ss.cfg.snapshot_port,
        g_ss.cfg.snapshot_path.c_str());
    return b;
}

/* Video source and encoder configuration, as one profile's worth of XML.
 * Values come from rtsp_server's running VENC session, so SOAP and SDP describe
 * the same stream. Fallbacks only cover a request during partial startup. */
std::string video_config_xml(const std::string& token, int idx)
{
    const int width = rtsp_server_width(idx) > 0 ? rtsp_server_width(idx) : 1920;
    const int height = rtsp_server_height(idx) > 0 ? rtsp_server_height(idx) : 1080;
    const int frame_rate = rtsp_server_frame_rate(idx) > 0
        ? rtsp_server_frame_rate(idx) : 30;
    const int bitrate = rtsp_server_encoder_bitrate(idx) > 0
        ? rtsp_server_encoder_bitrate(idx) : 4096;
    std::string x;
    x += "<tt:VideoSourceConfiguration token=\"vsc0\">";
    x += "<tt:Name>VideoSource</tt:Name><tt:UseCount>1</tt:UseCount>";
    x += "<tt:SourceToken>vs0</tt:SourceToken>";
    x += "<tt:Bounds x=\"0\" y=\"0\" width=\"" + std::to_string(width) +
         "\" height=\"" + std::to_string(height) + "\"/>";
    x += "</tt:VideoSourceConfiguration>";
    x += "<tt:VideoEncoderConfiguration token=\"vec_" + xml_escape(token) + "\">";
    x += "<tt:Name>" + xml_escape(token) + "</tt:Name><tt:UseCount>1</tt:UseCount>";
    x += "<tt:Encoding>H264</tt:Encoding>";
    x += "<tt:Resolution><tt:Width>" + std::to_string(width) +
         "</tt:Width><tt:Height>" + std::to_string(height) +
         "</tt:Height></tt:Resolution>";
    x += "<tt:Quality>5</tt:Quality>";
    x += "<tt:RateControl><tt:FrameRateLimit>" + std::to_string(frame_rate) +
         "</tt:FrameRateLimit>"
         "<tt:EncodingInterval>1</tt:EncodingInterval>"
         "<tt:BitrateLimit>" + std::to_string(bitrate) +
         "</tt:BitrateLimit></tt:RateControl>";
    x += "<tt:H264><tt:GovLength>30</tt:GovLength><tt:H264Profile>High</tt:H264Profile></tt:H264>";
    x += "<tt:SessionTimeout>PT60S</tt:SessionTimeout>";
    x += "</tt:VideoEncoderConfiguration>";
    return x;
}

std::string metadata_config_xml(const std::string& token)
{
    if (!rtsp_server_metadata_enabled()) return "";
    std::string x;
    x += "<tt:MetadataConfiguration token=\"meta_" + xml_escape(token) + "\">";
    x += "<tt:Name>ONVIF Metadata</tt:Name><tt:UseCount>1</tt:UseCount>";
    x += "<tt:Analytics>true</tt:Analytics>";
    x += "<tt:Multicast><tt:Address><tt:Type>IPv4</tt:Type>"
         "<tt:IPv4Address>0.0.0.0</tt:IPv4Address></tt:Address>"
         "<tt:Port>0</tt:Port><tt:TTL>0</tt:TTL>"
         "<tt:AutoStart>false</tt:AutoStart></tt:Multicast>";
    x += "<tt:SessionTimeout>PT60S</tt:SessionTimeout>";
    x += "</tt:MetadataConfiguration>";
    return x;
}

/* ------------------------------------------------------------------------ */
/* Device service                                                            */
/* ------------------------------------------------------------------------ */

std::string op_get_system_date_and_time()
{
    struct timespec ts {};
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm g {};
    time_t t = ts.tv_sec;
    gmtime_r(&t, &g);

    char b[640];
    snprintf(b, sizeof(b),
        "<tds:GetSystemDateAndTimeResponse><tds:SystemDateAndTime>"
        "<tt:DateTimeType>Manual</tt:DateTimeType>"
        "<tt:DaylightSavings>false</tt:DaylightSavings>"
        "<tt:TimeZone><tt:TZ>UTC0</tt:TZ></tt:TimeZone>"
        "<tt:UTCDateTime>"
        "<tt:Time><tt:Hour>%d</tt:Hour><tt:Minute>%d</tt:Minute><tt:Second>%d</tt:Second></tt:Time>"
        "<tt:Date><tt:Year>%d</tt:Year><tt:Month>%d</tt:Month><tt:Day>%d</tt:Day></tt:Date>"
        "</tt:UTCDateTime>"
        "</tds:SystemDateAndTime></tds:GetSystemDateAndTimeResponse>",
        g.tm_hour, g.tm_min, g.tm_sec,
        g.tm_year + 1900, g.tm_mon + 1, g.tm_mday);
    return b;
}

/*
 * SetSystemDateAndTime.
 *
 * Worth supporting rather than faulting: reCamera boots at 1970 because the RTC
 * driver is not loaded, and a VMS that can set the clock turns that from a
 * permanent wrong-timestamp problem into a self-correcting one. Best effort --
 * settimeofday needs privilege the application may not have, and failing to set
 * the clock is not a reason to fail the request in a way that makes the client
 * mark the device unusable.
 */
std::string op_set_system_date_and_time(const std::string& body)
{
    const std::string y = tag_value(body, "Year");
    const std::string mo = tag_value(body, "Month");
    const std::string d = tag_value(body, "Day");
    const std::string h = tag_value(body, "Hour");
    const std::string mi = tag_value(body, "Minute");
    const std::string se = tag_value(body, "Second");

    if (!y.empty() && !mo.empty() && !d.empty()) {
        struct tm tmv {};
        tmv.tm_year = atoi(y.c_str()) - 1900;
        tmv.tm_mon = atoi(mo.c_str()) - 1;
        tmv.tm_mday = atoi(d.c_str());
        tmv.tm_hour = h.empty() ? 0 : atoi(h.c_str());
        tmv.tm_min = mi.empty() ? 0 : atoi(mi.c_str());
        tmv.tm_sec = se.empty() ? 0 : atoi(se.c_str());
        /* The wire value is UTC; timegm avoids reinterpreting it as local. */
        const time_t when = timegm(&tmv);
        if (when > 0) {
            struct timeval tv { when, 0 };
            if (settimeofday(&tv, nullptr) != 0) {
                fprintf(stderr, "[%s] settimeofday denied (not privileged?); "
                                "clock unchanged\n", OSOAP_TAG);
            } else {
                fprintf(stderr, "[%s] clock set by ONVIF client to %s\n",
                    OSOAP_TAG, utc_now().c_str());
            }
        }
    }
    return "<tds:SetSystemDateAndTimeResponse/>";
}

std::string read_firmware()
{
    if (!g_ss.cfg.firmware.empty()) return g_ss.cfg.firmware;
    FILE* f = fopen("/etc/version", "r");
    if (f == nullptr) f = fopen("/etc/sscma_version", "r");
    if (f == nullptr) return "unknown";
    char b[128] = { 0 };
    if (fgets(b, sizeof(b), f) == nullptr) b[0] = '\0';
    fclose(f);
    std::string v(b);
    while (!v.empty() && (v.back() == '\n' || v.back() == '\r')) v.pop_back();
    return v.empty() ? "unknown" : v;
}

std::string op_get_device_information()
{
    std::string x = "<tds:GetDeviceInformationResponse>";
    x += "<tds:Manufacturer>" + xml_escape(g_ss.cfg.manufacturer) + "</tds:Manufacturer>";
    x += "<tds:Model>" + xml_escape(g_ss.cfg.model) + "</tds:Model>";
    x += "<tds:FirmwareVersion>" + xml_escape(read_firmware()) + "</tds:FirmwareVersion>";
    x += "<tds:SerialNumber>" + xml_escape(g_ss.cfg.serial) + "</tds:SerialNumber>";
    x += "<tds:HardwareId>" + xml_escape(g_ss.cfg.hardware) + "</tds:HardwareId>";
    x += "</tds:GetDeviceInformationResponse>";
    return x;
}

std::string op_get_scopes()
{
    std::string x = "<tds:GetScopesResponse>";
    auto add = [&x](const std::string& s) {
        x += "<tds:Scopes><tt:ScopeDef>Fixed</tt:ScopeDef>"
             "<tt:ScopeItem>" + xml_escape(s) + "</tt:ScopeItem></tds:Scopes>";
    };
    add("onvif://www.onvif.org/type/NetworkVideoTransmitter");
    if (!g_ss.cfg.device_name.empty())
        add("onvif://www.onvif.org/name/" + g_ss.cfg.device_name);
    if (!g_ss.cfg.hardware.empty())
        add("onvif://www.onvif.org/hardware/" + g_ss.cfg.hardware);
    if (!g_ss.cfg.location.empty())
        add("onvif://www.onvif.org/location/" + g_ss.cfg.location);
    /* No Profile scope. Advertising one is a conformance claim; see
     * onvif_discovery.cpp and the implementation spec 8.1. */
    x += "</tds:GetScopesResponse>";
    return x;
}

/*
 * GetCapabilities -- the Media1-era discovery call.
 *
 * Section 5.6 of the spec lists only Media2, but nearly every client, ONVIF
 * Device Manager included, calls this first and treats a fault as "device
 * unusable". Answering it is the difference between being found and being
 * usable, so it is implemented even though the media service itself is ver20.
 */
std::string op_get_capabilities(const std::string& ip)
{
    const std::string url = device_service_url(ip);
    std::string x = "<tds:GetCapabilitiesResponse><tds:Capabilities>";
    x += "<tt:Device><tt:XAddr>" + xml_escape(url) + "</tt:XAddr>";
    x += "<tt:Network><tt:IPFilter>false</tt:IPFilter>"
         "<tt:ZeroConfiguration>false</tt:ZeroConfiguration>"
         "<tt:IPVersion6>false</tt:IPVersion6>"
         "<tt:DynDNS>false</tt:DynDNS></tt:Network>";
    x += "<tt:System><tt:DiscoveryResolve>false</tt:DiscoveryResolve>"
         "<tt:DiscoveryBye>true</tt:DiscoveryBye>"
         "<tt:RemoteDiscovery>false</tt:RemoteDiscovery>"
         "<tt:SystemBackup>false</tt:SystemBackup>"
         "<tt:SystemLogging>false</tt:SystemLogging>"
         "<tt:FirmwareUpgrade>false</tt:FirmwareUpgrade></tt:System>";
    x += "<tt:Security><tt:TLS1.1>false</tt:TLS1.1><tt:TLS1.2>false</tt:TLS1.2>"
         "<tt:OnboardKeyGeneration>false</tt:OnboardKeyGeneration>"
         "<tt:AccessPolicyConfig>false</tt:AccessPolicyConfig>"
         "<tt:X.509Token>false</tt:X.509Token><tt:SAMLToken>false</tt:SAMLToken>"
         "<tt:KerberosToken>false</tt:KerberosToken>"
         "<tt:RELToken>false</tt:RELToken></tt:Security>";
    x += "</tt:Device>";
    x += "<tt:Media><tt:XAddr>" + xml_escape(url) + "</tt:XAddr>";
    x += "<tt:StreamingCapabilities><tt:RTPMulticast>false</tt:RTPMulticast>"
         "<tt:RTP_TCP>true</tt:RTP_TCP>"
         "<tt:RTP_RTSP_TCP>true</tt:RTP_RTSP_TCP></tt:StreamingCapabilities>";
    x += "</tt:Media>";
    x += "</tds:Capabilities></tds:GetCapabilitiesResponse>";
    return x;
}

std::string op_get_services(const std::string& ip, bool include_capability)
{
    const std::string url = xml_escape(device_service_url(ip));
    std::string x = "<tds:GetServicesResponse>";

    x += "<tds:Service>";
    x += "<tds:Namespace>http://www.onvif.org/ver10/device/wsdl</tds:Namespace>";
    x += "<tds:XAddr>" + url + "</tds:XAddr>";
    if (include_capability) {
        x += "<tds:Capabilities><tds:Capabilities xmlns:tds=\"http://www.onvif.org/ver10/device/wsdl\">"
             "<tds:Network IPFilter=\"false\" ZeroConfiguration=\"false\" IPVersion6=\"false\" DynDNS=\"false\"/>"
             "<tds:System DiscoveryResolve=\"false\" DiscoveryBye=\"true\" RemoteDiscovery=\"false\""
             " SystemBackup=\"false\" SystemLogging=\"false\" FirmwareUpgrade=\"false\"/>"
             "</tds:Capabilities></tds:Capabilities>";
    }
    x += "<tds:Version><tt:Major>2</tt:Major><tt:Minor>50</tt:Minor></tds:Version>";
    x += "</tds:Service>";

    /* Media1 remains the compatibility service used by onvif-zeep and many
     * established VMS clients. It points at the same endpoint as Media2; the
     * dispatcher selects the response shape from the request namespace. */
    x += "<tds:Service>";
    x += "<tds:Namespace>http://www.onvif.org/ver10/media/wsdl</tds:Namespace>";
    x += "<tds:XAddr>" + url + "</tds:XAddr>";
    x += "<tds:Version><tt:Major>2</tt:Major><tt:Minor>50</tt:Minor></tds:Version>";
    x += "</tds:Service>";

    x += "<tds:Service>";
    x += "<tds:Namespace>http://www.onvif.org/ver20/media/wsdl</tds:Namespace>";
    x += "<tds:XAddr>" + url + "</tds:XAddr>";
    if (include_capability) {
        x += "<tds:Capabilities><tr2:Capabilities SnapshotUri=\""
             + std::string(g_ss.cfg.snapshot_port > 0 ? "true" : "false")
             + "\" Rotation=\"false\" VideoSourceMode=\"false\" OSD=\"false\">"
               "<tr2:ProfileCapabilities MaximumNumberOfProfiles=\""
             + std::to_string(profile_count()) + "\"/>"
               "<tr2:StreamingCapabilities RTPMulticast=\"false\" RTP_RTSP_TCP=\"true\"/>"
               "</tr2:Capabilities></tds:Capabilities>";
    }
    x += "<tds:Version><tt:Major>2</tt:Major><tt:Minor>50</tt:Minor></tds:Version>";
    x += "</tds:Service>";

    x += "</tds:GetServicesResponse>";
    return x;
}

std::string op_get_service_capabilities()
{
    std::string x = "<tds:GetServiceCapabilitiesResponse><tds:Capabilities>";
    x += "<tds:Network IPFilter=\"false\" ZeroConfiguration=\"false\" IPVersion6=\"false\""
         " DynDNS=\"false\" Dot11Configuration=\"false\" HostnameFromDHCP=\"false\"/>";
    x += "<tds:Security TLS1.0=\"false\" TLS1.1=\"false\" TLS1.2=\"false\""
         " OnboardKeyGeneration=\"false\" AccessPolicyConfig=\"false\""
         " DefaultAccessPolicy=\"false\" Dot1X=\"false\" RemoteUserHandling=\"false\""
         " X.509Token=\"false\" SAMLToken=\"false\" KerberosToken=\"false\""
         " UsernameToken=\"false\" HttpDigest=\"";
    x += (!g_ss.cfg.username.empty() ? "true" : "false");
    x += "\" RELToken=\"false\"/>";
    x += "<tds:System DiscoveryResolve=\"false\" DiscoveryBye=\"true\""
         " RemoteDiscovery=\"false\" SystemBackup=\"false\" SystemLogging=\"false\""
         " FirmwareUpgrade=\"false\" HttpFirmwareUpgrade=\"false\""
         " HttpSystemBackup=\"false\" HttpSystemLogging=\"false\""
         " HttpSupportInformation=\"false\"/>";
    x += "</tds:Capabilities></tds:GetServiceCapabilitiesResponse>";
    return x;
}

std::string op_get_network_interfaces()
{
    std::string x = "<tds:GetNetworkInterfacesResponse>";

    struct ifaddrs* ifs = nullptr;
    if (getifaddrs(&ifs) == 0) {
        for (struct ifaddrs* it = ifs; it != nullptr; it = it->ifa_next) {
            if (it->ifa_addr == nullptr) continue;
            if (it->ifa_addr->sa_family != AF_INET) continue;
            if ((it->ifa_flags & IFF_LOOPBACK) != 0) continue;

            char ip[INET_ADDRSTRLEN] = { 0 };
            inet_ntop(AF_INET, &((struct sockaddr_in*)it->ifa_addr)->sin_addr,
                ip, sizeof(ip));

            int prefix = 24;
            if (it->ifa_netmask != nullptr) {
                const uint32_t m = ntohl(((struct sockaddr_in*)it->ifa_netmask)->sin_addr.s_addr);
                prefix = 0;
                for (int b = 31; b >= 0 && ((m >> b) & 1u) != 0u; --b) ++prefix;
            }

            const std::string name = it->ifa_name != nullptr ? it->ifa_name : "eth0";
            x += "<tds:NetworkInterfaces token=\"" + xml_escape(name) + "\">";
            x += "<tt:Enabled>true</tt:Enabled>";
            x += "<tt:Info><tt:Name>" + xml_escape(name) + "</tt:Name>"
                 "<tt:HwAddress>00:00:00:00:00:00</tt:HwAddress>"
                 "<tt:MTU>1500</tt:MTU></tt:Info>";
            x += "<tt:IPv4><tt:Enabled>true</tt:Enabled><tt:Config>";
            x += "<tt:Manual><tt:Address>" + std::string(ip) + "</tt:Address>"
                 "<tt:PrefixLength>" + std::to_string(prefix) + "</tt:PrefixLength></tt:Manual>";
            x += "<tt:DHCP>false</tt:DHCP>";
            x += "</tt:Config></tt:IPv4>";
            x += "</tds:NetworkInterfaces>";
        }
        freeifaddrs(ifs);
    }

    x += "</tds:GetNetworkInterfacesResponse>";
    return x;
}

/* ------------------------------------------------------------------------ */
/* Media1 / Media2 services                                                   */
/* ------------------------------------------------------------------------ */

std::string op_get_profiles(const std::string& token_filter)
{
    std::string x = "<tr2:GetProfilesResponse>";
    const int n = profile_count();
    for (int i = 0; i < n; ++i) {
        const std::string tok = profile_token(i);
        if (!token_filter.empty() && token_filter != tok) continue;
        x += "<tr2:Profiles token=\"" + xml_escape(tok) + "\" fixed=\"true\">";
        x += "<tr2:Name>" + xml_escape(tok) + "</tr2:Name>";
        x += "<tr2:Configurations>";
        x += video_config_xml(tok, i);
        x += metadata_config_xml(tok);
        x += "</tr2:Configurations>";
        x += "</tr2:Profiles>";
    }
    x += "</tr2:GetProfilesResponse>";
    return x;
}

/* Media1 uses direct configuration children and wraps the URI in MediaUri.
 * Frigate's onvif-zeep client intentionally uses this older, still ubiquitous
 * service, so returning a Media2-shaped body to a Media1 request makes the
 * profile look empty even though the XML itself is well formed. */
std::string op_get_profiles_media1(const std::string& token_filter)
{
    std::string x = "<trt:GetProfilesResponse>";
    const int n = profile_count();
    for (int i = 0; i < n; ++i) {
        const std::string tok = profile_token(i);
        if (!token_filter.empty() && token_filter != tok) continue;
        x += "<trt:Profiles token=\"" + xml_escape(tok) + "\" fixed=\"true\">";
        x += "<tt:Name>" + xml_escape(tok) + "</tt:Name>";
        x += video_config_xml(tok, i);
        x += metadata_config_xml(tok);
        x += "</trt:Profiles>";
    }
    x += "</trt:GetProfilesResponse>";
    return x;
}

/* -1 when the token names no profile. Falling back to profile 0 instead would
 * hand the client a URI for a stream it did not ask for, which reads as success
 * and is only noticed later as the wrong resolution or channel. */
int profile_index(const std::string& token)
{
    const int n = profile_count();
    for (int i = 0; i < n; ++i) {
        if (profile_token(i) == token) return i;
    }
    return -1;
}

/* Media2's GetStreamUri returns a bare Uri element, unlike Media1 which wrapped
 * it in MediaUri with timeout and invalid-after fields. Sending the Media1
 * shape here parses as an empty URI on strict clients. */
std::string op_get_stream_uri(const std::string& ip, const std::string& token)
{
    const int idx = profile_index(token);
    if (idx < 0) return "";
    const std::string uri = stream_uri(ip, idx);
    if (uri.empty()) return "";
    return "<tr2:GetStreamUriResponse><tr2:Uri>" + xml_escape(uri) +
           "</tr2:Uri></tr2:GetStreamUriResponse>";
}

std::string op_get_stream_uri_media1(const std::string& ip,
    const std::string& token)
{
    const int idx = profile_index(token);
    if (idx < 0) return "";
    const std::string uri = stream_uri(ip, idx);
    if (uri.empty()) return "";
    return "<trt:GetStreamUriResponse><trt:MediaUri><tt:Uri>" +
           xml_escape(uri) +
           "</tt:Uri><tt:InvalidAfterConnect>false</tt:InvalidAfterConnect>"
           "<tt:InvalidAfterReboot>false</tt:InvalidAfterReboot>"
           "<tt:Timeout>PT60S</tt:Timeout></trt:MediaUri>"
           "</trt:GetStreamUriResponse>";
}

std::string op_get_snapshot_uri(const std::string& ip)
{
    const std::string uri = snapshot_uri(ip);
    if (uri.empty()) return "";
    return "<tr2:GetSnapshotUriResponse><tr2:Uri>" + xml_escape(uri) +
           "</tr2:Uri></tr2:GetSnapshotUriResponse>";
}

std::string op_get_video_source_configurations()
{
    const int width = rtsp_server_width(0) > 0 ? rtsp_server_width(0) : 1920;
    const int height = rtsp_server_height(0) > 0 ? rtsp_server_height(0) : 1080;
    std::string x = "<tr2:GetVideoSourceConfigurationsResponse>";
    x += "<tr2:Configurations token=\"vsc0\">";
    x += "<tt:Name>VideoSource</tt:Name><tt:UseCount>1</tt:UseCount>";
    x += "<tt:SourceToken>vs0</tt:SourceToken>";
    x += "<tt:Bounds x=\"0\" y=\"0\" width=\"" + std::to_string(width) +
         "\" height=\"" + std::to_string(height) + "\"/>";
    x += "</tr2:Configurations>";
    x += "</tr2:GetVideoSourceConfigurationsResponse>";
    return x;
}

std::string op_get_video_encoder_configurations()
{
    std::string x = "<tr2:GetVideoEncoderConfigurationsResponse>";
    const int n = profile_count();
    for (int i = 0; i < n; ++i) {
        const std::string tok = profile_token(i);
        const int width = rtsp_server_width(i) > 0 ? rtsp_server_width(i) : 1920;
        const int height = rtsp_server_height(i) > 0 ? rtsp_server_height(i) : 1080;
        const int frame_rate = rtsp_server_frame_rate(i) > 0
            ? rtsp_server_frame_rate(i) : 30;
        const int bitrate = rtsp_server_encoder_bitrate(i) > 0
            ? rtsp_server_encoder_bitrate(i) : 4096;
        x += "<tr2:Configurations token=\"vec_" + xml_escape(tok) + "\">";
        x += "<tt:Name>" + xml_escape(tok) + "</tt:Name><tt:UseCount>1</tt:UseCount>";
        x += "<tt:Encoding>H264</tt:Encoding>";
        x += "<tt:Resolution><tt:Width>" + std::to_string(width) +
             "</tt:Width><tt:Height>" + std::to_string(height) +
             "</tt:Height></tt:Resolution>";
        x += "<tt:RateControl><tt:FrameRateLimit>" + std::to_string(frame_rate) +
             "</tt:FrameRateLimit><tt:BitrateLimit>" + std::to_string(bitrate) +
             "</tt:BitrateLimit></tt:RateControl>";
        x += "</tr2:Configurations>";
    }
    x += "</tr2:GetVideoEncoderConfigurationsResponse>";
    return x;
}

/* ------------------------------------------------------------------------ */
/* Dispatch                                                                  */
/* ------------------------------------------------------------------------ */

/*
 * Operations reachable without credentials even when Digest is configured.
 * GetSystemDateAndTime because a client cannot compute a Digest response before
 * it knows the device clock (ONVIF Core requires it anonymous); GetCapabilities
 * because VMS probe with it and read 401 as "unusable" rather than "needs
 * login". See the implementation spec 14.9.1 D3.
 */
bool is_anonymous_op(const std::string& body)
{
    return has_op(body, "GetSystemDateAndTime") || has_op(body, "GetCapabilities");
}

/* Digest is checked only for its presence, not verified, when no credentials
 * are configured -- which is the default. When they are, absence of an
 * Authorization header produces 401 with a challenge and the client retries. */
bool auth_ok(struct lws* wsi, const std::string& body)
{
    if (g_ss.cfg.username.empty()) return true;
    if (is_anonymous_op(body)) return true;
    const int n = lws_hdr_total_length(wsi, WSI_TOKEN_HTTP_AUTHORIZATION);
    return n > 0;
}

std::string dispatch(const std::string& body, const std::string& ip, int* http_status)
{
    *http_status = 200;

    /* Device */
    if (has_op(body, "GetSystemDateAndTime")) return envelope(op_get_system_date_and_time());
    if (has_op(body, "SetSystemDateAndTime")) return envelope(op_set_system_date_and_time(body));
    if (has_op(body, "GetDeviceInformation"))  return envelope(op_get_device_information());
    if (has_op(body, "GetServiceCapabilities")) return envelope(op_get_service_capabilities());
    if (has_op(body, "GetCapabilities"))       return envelope(op_get_capabilities(ip));
    if (has_op(body, "GetServices")) {
        const std::string inc = tag_value(body, "IncludeCapability");
        return envelope(op_get_services(ip, inc == "true" || inc == "1"));
    }
    if (has_op(body, "GetScopes"))             return envelope(op_get_scopes());
    if (has_op(body, "GetNetworkInterfaces"))  return envelope(op_get_network_interfaces());

    /* Media1 and Media2 share this HTTP endpoint. Namespace declarations are
     * the one reliable discriminator because clients may choose any prefix. */
    const bool media1 = body.find("http://www.onvif.org/ver10/media/wsdl") !=
                        std::string::npos;
    if (has_op(body, "GetProfiles")) {
        const std::string token = tag_value(body, "Token");
        return envelope(media1 ? op_get_profiles_media1(token)
                               : op_get_profiles(token));
    }
    if (has_op(body, "GetStreamUri")) {
        std::string tok = tag_value(body, "ProfileToken");
        if (tok.empty()) tok = tag_value(body, "Token");
        if (tok.empty() && profile_count() > 0) tok = profile_token(0);
        const std::string r = media1 ? op_get_stream_uri_media1(ip, tok)
                                     : op_get_stream_uri(ip, tok);
        if (r.empty()) {
            *http_status = 400;
            return soap_fault("Sender", "InvalidArgVal", "no such profile");
        }
        return envelope(r);
    }
    if (has_op(body, "GetSnapshotUri")) {
        const std::string r = op_get_snapshot_uri(ip);
        if (r.empty()) {
            *http_status = 400;
            return soap_fault("Sender", "ActionNotSupported", "no snapshot endpoint");
        }
        return envelope(r);
    }
    if (has_op(body, "GetVideoSourceConfigurations"))
        return envelope(op_get_video_source_configurations());
    if (has_op(body, "GetVideoEncoderConfigurations"))
        return envelope(op_get_video_encoder_configurations());

    /*
     * Anything else. Counted separately because "the VMS found the device and
     * then showed it offline" almost always means it asked for something not
     * implemented, and this counter is the only place that says which.
     */
    g_ss.unknown.fetch_add(1, std::memory_order_relaxed);
    *http_status = 400;
    return soap_fault("Sender", "ActionNotSupported", "operation not implemented");
}

/* ------------------------------------------------------------------------ */
/* lws glue                                                                  */
/* ------------------------------------------------------------------------ */

int stage_reply(struct lws* wsi, PerSession* ps, int status,
    const char* ctype, const std::string& payload, bool challenge)
{
    if (ps == nullptr) return -1;
    delete ps->reply;
    ps->reply = new (std::nothrow) std::string(payload);
    if (ps->reply == nullptr) return -1;
    ps->sent = 0;

    uint8_t hdr[LWS_PRE + 640];
    uint8_t* p = hdr + LWS_PRE;
    uint8_t* end = hdr + sizeof(hdr) - 1;

    if (lws_add_http_common_headers(wsi, static_cast<unsigned int>(status), ctype,
            static_cast<lws_filepos_t>(ps->reply->size()), &p, end))
        return -1;
    if (challenge) {
        const std::string ch = "Digest realm=\"" + g_ss.cfg.device_name +
            "\", qop=\"auth\", nonce=\"" + std::to_string(::time(nullptr)) + "\"";
        if (lws_add_http_header_by_name(wsi,
                reinterpret_cast<const unsigned char*>("www-authenticate:"),
                reinterpret_cast<const unsigned char*>(ch.c_str()),
                static_cast<int>(ch.size()), &p, end))
            return -1;
    }
    if (lws_finalize_write_http_header(wsi, hdr + LWS_PRE, &p, end))
        return -1;

    lws_callback_on_writable(wsi);
    return 0;
}

int soap_callback(struct lws* wsi, enum lws_callback_reasons reason,
    void* user, void* in, size_t len)
{
    PerSession* ps = static_cast<PerSession*>(user);

    switch (reason) {

    case LWS_CALLBACK_HTTP: {
        /*
         * A GET on the service path returns a short plain-text description
         * rather than a fault. It is not part of ONVIF, but it is the first
         * thing anyone does when a VMS cannot connect -- open the URL in a
         * browser -- and answering "this is the ONVIF endpoint, POST here" is
         * worth ten lines.
         */
        if (lws_hdr_total_length(wsi, WSI_TOKEN_POST_URI) == 0) {
            static const char kInfo[] =
                "reCamera ONVIF device service.\n"
                "This endpoint accepts SOAP 1.2 over HTTP POST.\n";
            return stage_reply(wsi, ps, 200, "text/plain", kInfo, false) < 0 ? -1 : 0;
        }
        if (ps != nullptr) {
            delete ps->body;
            ps->body = new (std::nothrow) std::string();
            if (ps->body == nullptr) return -1;
            ps->body->reserve(2048);
        }
        return 0;
    }

    case LWS_CALLBACK_HTTP_BODY: {
        if (ps == nullptr || ps->body == nullptr) return 0;
        /* Bounded: an ONVIF request is a few kilobytes. Anything larger is
         * either a broken client or someone probing, and neither deserves
         * unbounded memory on a 180 MB device. */
        if (ps->body->size() + len > (256u * 1024u)) return -1;
        ps->body->append(static_cast<const char*>(in), len);
        return 0;
    }

    case LWS_CALLBACK_HTTP_BODY_COMPLETION: {
        if (ps == nullptr) return -1;
        const std::string body = ps->body != nullptr ? *ps->body : std::string();
        delete ps->body;
        ps->body = nullptr;

        if (!auth_ok(wsi, body)) {
            return stage_reply(wsi, ps, 401, "text/plain", "unauthorized\n", true) < 0 ? -1 : 0;
        }

        int status = 200;
        const std::string reply = dispatch(body, local_ip_of(wsi), &status);
        g_ss.served.fetch_add(1, std::memory_order_relaxed);
        return stage_reply(wsi, ps, status, "application/soap+xml; charset=utf-8",
                   reply, false) < 0 ? -1 : 0;
    }

    case LWS_CALLBACK_HTTP_WRITEABLE: {
        if (ps == nullptr || ps->reply == nullptr) return 0;
        std::string& r = *ps->reply;
        if (ps->sent >= r.size()) {
            return lws_http_transaction_completed(wsi) ? -1 : 0;
        }
        const size_t chunk = 4096;
        const size_t n = r.size() - ps->sent < chunk ? r.size() - ps->sent : chunk;
        std::vector<uint8_t> out(LWS_PRE + n);
        memcpy(out.data() + LWS_PRE, r.data() + ps->sent, n);
        const int w = lws_write(wsi, out.data() + LWS_PRE, n,
            (ps->sent + n >= r.size()) ? LWS_WRITE_HTTP_FINAL : LWS_WRITE_HTTP);
        if (w < static_cast<int>(n)) return -1;
        ps->sent += n;
        if (ps->sent >= r.size()) {
            return lws_http_transaction_completed(wsi) ? -1 : 0;
        }
        lws_callback_on_writable(wsi);
        return 0;
    }

    case LWS_CALLBACK_HTTP_DROP_PROTOCOL:
    case LWS_CALLBACK_CLOSED_HTTP: {
        if (ps != nullptr) {
            delete ps->body;
            delete ps->reply;
            ps->body = nullptr;
            ps->reply = nullptr;
            ps->sent = 0;
        }
        return 0;
    }

    default:
        break;
    }
    return 0;
}

struct lws_protocols g_protocols[] = {
    { "onvif", soap_callback, sizeof(PerSession), 0, 0, nullptr, 0 },
    /* Explicit terminator: LWS_PROTOCOL_LIST_TERM uses designated initialisers
     * and does not compile as C++. */
    { nullptr, nullptr, 0, 0, 0, nullptr, 0 }
};

void soap_loop()
{
    while (g_ss.running.load(std::memory_order_acquire)) {
        lws_service(g_ss.ctx, 0);
    }
}

} // namespace

/* -------------------------------------------------------------------------- */

int onvif_soap_start(const onvif_service_config* cfg, const std::string& uuid)
{
    if (cfg == nullptr) return -1;
    if (g_ss.running.load()) return 0;

    g_ss.cfg = *cfg;
    g_ss.uuid = uuid;

    /* Mount "/" onto the callback protocol. Without a mount lws answers plain
     * HTTP itself with its own error page and the callback never sees the
     * request; both origin and protocol must be set or context creation
     * dereferences null. Both traps are documented in ws_transport_lws.cpp. */
    static struct lws_http_mount mount;
    memset(&mount, 0, sizeof(mount));
    mount.mountpoint = "/";
    mount.mountpoint_len = 1;
    mount.origin_protocol = LWSMPRO_CALLBACK;
    mount.origin = "onvif";
    mount.protocol = "onvif";

    struct lws_context_creation_info info;
    memset(&info, 0, sizeof(info));
    info.port = g_ss.cfg.service_port;
    info.protocols = g_protocols;
    info.mounts = &mount;
    info.extensions = nullptr;
    info.gid = -1;
    info.uid = -1;
    info.keepalive_timeout = 30;

    g_ss.ctx = lws_create_context(&info);
    if (g_ss.ctx == nullptr) {
        /* Loud, because discovery may still be running and will keep telling
         * clients to connect to an address that refuses them. */
        fprintf(stderr, "[%s] cannot listen on tcp/%d -- discovery will advertise "
                        "an endpoint that is not there\n",
            OSOAP_TAG, g_ss.cfg.service_port);
        return -1;
    }

    g_ss.running.store(true, std::memory_order_release);
    g_ss.worker = std::thread(soap_loop);
    fprintf(stderr, "[%s] device service on http://<ip>:%d%s, %d profile(s)%s\n",
        OSOAP_TAG, g_ss.cfg.service_port, g_ss.cfg.service_path.c_str(),
        profile_count(),
        g_ss.cfg.snapshot_port > 0 ? ", snapshot advertised" : ", no snapshot");
    return 0;
}

void onvif_soap_stop(void)
{
    if (!g_ss.running.load()) return;
    g_ss.running.store(false, std::memory_order_release);
    if (g_ss.ctx != nullptr) lws_cancel_service(g_ss.ctx);
    if (g_ss.worker.joinable()) g_ss.worker.join();
    if (g_ss.ctx != nullptr) {
        lws_context_destroy(g_ss.ctx);
        g_ss.ctx = nullptr;
    }
}

bool onvif_service_soap_running(void) { return g_ss.running.load(); }

unsigned long onvif_service_soap_count(void)
{
    return g_ss.served.load(std::memory_order_relaxed);
}

unsigned long onvif_service_soap_unknown_count(void)
{
    return g_ss.unknown.load(std::memory_order_relaxed);
}
