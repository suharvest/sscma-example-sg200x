/*
 * WS-Discovery responder (ONVIF Core Spec / OASIS WS-Discovery 1.0).
 *
 * Pure UDP multicast: no HTTP library, no SOAP toolkit, no third-party
 * dependency at all. That matters here beyond taste -- mongoose is GPL-2.0
 * against an Apache-2.0 repository (see docs/onvif-implementation-spec.md
 * 0.5-B) and gSOAP's generated stubs are GPLv2, so the parts of ONVIF that can
 * be built without either are the parts that can ship today.
 *
 * Protocol, in the amount actually needed:
 *   - listen on UDP 3702, joined to the multicast group 239.255.255.250
 *   - a client multicasts a Probe; reply with a UNICAST ProbeMatches whose
 *     RelatesTo carries the Probe's MessageID
 *   - announce Hello at startup and Bye at shutdown, both multicast
 *
 * The XML is assembled from string templates rather than parsed and rebuilt.
 * The messages are fixed in shape and this avoids pulling in an XML library
 * for four documents; the same approach is planned for the SOAP services.
 * Probe parsing extracts exactly one thing -- the MessageID -- and treats
 * everything else as opaque, which is what keeps that decision safe.
 */

#include "onvif_service.h"

#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>   /* struct timeval for SO_RCVTIMEO */
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#define OS_TAG "onvif_service"

#define WSD_GROUP "239.255.255.250"
#define WSD_PORT 3702

namespace {

struct DiscoveryState {
    int sock = -1;
    std::thread worker;
    std::atomic<bool> running { false };
    std::atomic<unsigned long> probes { 0 };
    onvif_service_config cfg;
    std::string uuid; /* stable urn:uuid for this device */
};

DiscoveryState g_ds;

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
 * Extract the value of the first <...:MessageID> element. A ProbeMatches whose
 * RelatesTo does not match the Probe is ignored by clients, so this is the one
 * field that has to be read; everything else in the Probe is left alone.
 *
 * Deliberately namespace-prefix agnostic: clients use wsa:, a:, s: and bare
 * MessageID interchangeably, and rejecting an unfamiliar prefix would show up
 * as "works with ODM, invisible to Milestone".
 */
std::string extract_message_id(const std::string& msg)
{
    size_t pos = 0;
    while ((pos = msg.find("MessageID", pos)) != std::string::npos) {
        /* Must be a tag name, i.e. preceded by '<' or a prefix separator. */
        size_t lt = msg.rfind('<', pos);
        if (lt == std::string::npos) { pos += 9; continue; }
        bool is_open_tag = true;
        for (size_t k = lt + 1; k < pos; ++k) {
            const char c = msg[k];
            if (!(isalnum(static_cast<unsigned char>(c)) || c == ':' || c == '_' || c == '-')) {
                is_open_tag = false;
                break;
            }
        }
        if (!is_open_tag) { pos += 9; continue; }

        const size_t gt = msg.find('>', pos);
        if (gt == std::string::npos) return "";
        const size_t end = msg.find('<', gt + 1);
        if (end == std::string::npos) return "";

        std::string val = msg.substr(gt + 1, end - gt - 1);
        /* trim */
        const size_t b = val.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) return "";
        const size_t e = val.find_last_not_of(" \t\r\n");
        return val.substr(b, e - b + 1);
    }
    return "";
}

bool is_probe(const std::string& msg)
{
    return msg.find("Probe") != std::string::npos &&
           msg.find("Envelope") != std::string::npos;
}

/*
 * Scopes.
 *
 * Only factual ones. In particular NO onvif://www.onvif.org/Profile/... entry:
 * advertising a profile scope is a conformance claim, and ONVIF asks products
 * that are not conformant to stop making them (see the implementation spec,
 * 8.1). The device genuinely is a NetworkVideoTransmitter, so that scope is
 * accurate and is what clients filter on to populate their camera lists.
 */
std::string build_scopes()
{
    std::string s = "onvif://www.onvif.org/type/NetworkVideoTransmitter";
    if (!g_ds.cfg.device_name.empty()) {
        s += " onvif://www.onvif.org/name/" + xml_escape(g_ds.cfg.device_name);
    }
    if (!g_ds.cfg.hardware.empty()) {
        s += " onvif://www.onvif.org/hardware/" + xml_escape(g_ds.cfg.hardware);
    }
    if (!g_ds.cfg.location.empty()) {
        s += " onvif://www.onvif.org/location/" + xml_escape(g_ds.cfg.location);
    }
    return s;
}

/* The address a client should talk SOAP to. Filled per reply with the local
 * address the Probe arrived on, so a device reachable over both usb0 and eth0
 * hands each client an address it can actually route to. */
std::string build_xaddrs(const std::string& local_ip)
{
    char buf[160];
    snprintf(buf, sizeof(buf), "http://%s:%d%s", local_ip.c_str(),
        g_ds.cfg.service_port, g_ds.cfg.service_path.c_str());
    return buf;
}

std::string build_probe_matches(const std::string& relates_to, const std::string& local_ip)
{
    static std::atomic<unsigned long> seq { 0 };
    char msgid[80];
    snprintf(msgid, sizeof(msgid), "urn:uuid:%s-%lu", g_ds.uuid.c_str(),
        seq.fetch_add(1));

    std::string x;
    x.reserve(1400);
    x += "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";
    x += "<s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\""
         " xmlns:a=\"http://schemas.xmlsoap.org/ws/2004/08/addressing\""
         " xmlns:d=\"http://schemas.xmlsoap.org/ws/2005/04/discovery\""
         " xmlns:dn=\"http://www.onvif.org/ver10/network/wsdl\">";
    x += "<s:Header>";
    x += "<a:MessageID>" + std::string(msgid) + "</a:MessageID>";
    if (!relates_to.empty()) {
        x += "<a:RelatesTo>" + xml_escape(relates_to) + "</a:RelatesTo>";
    }
    x += "<a:To>http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous</a:To>";
    x += "<a:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/ProbeMatches</a:Action>";
    x += "</s:Header>";
    x += "<s:Body><d:ProbeMatches><d:ProbeMatch>";
    x += "<a:EndpointReference><a:Address>urn:uuid:" + g_ds.uuid + "</a:Address></a:EndpointReference>";
    x += "<d:Types>dn:NetworkVideoTransmitter</d:Types>";
    x += "<d:Scopes>" + build_scopes() + "</d:Scopes>";
    x += "<d:XAddrs>" + build_xaddrs(local_ip) + "</d:XAddrs>";
    x += "<d:MetadataVersion>1</d:MetadataVersion>";
    x += "</d:ProbeMatch></d:ProbeMatches></s:Body></s:Envelope>";
    return x;
}

/* Best-effort: the address a packet from `peer` would be answered from. Used
 * only to fill XAddrs, so a wrong guess degrades to "client cannot reach the
 * SOAP service", not to a crash. */
std::string local_ip_for(const struct sockaddr_in& peer)
{
    int s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0) return "127.0.0.1";
    struct sockaddr_in dst = peer;
    dst.sin_port = htons(9);
    std::string out = "127.0.0.1";
    if (connect(s, (struct sockaddr*)&dst, sizeof(dst)) == 0) {
        struct sockaddr_in me {};
        socklen_t len = sizeof(me);
        if (getsockname(s, (struct sockaddr*)&me, &len) == 0) {
            char buf[INET_ADDRSTRLEN];
            if (inet_ntop(AF_INET, &me.sin_addr, buf, sizeof(buf))) out = buf;
        }
    }
    close(s);
    return out;
}

void send_multicast(const std::string& payload)
{
    if (g_ds.sock < 0) return;
    struct sockaddr_in grp {};
    grp.sin_family = AF_INET;
    grp.sin_port = htons(WSD_PORT);
    inet_pton(AF_INET, WSD_GROUP, &grp.sin_addr);
    sendto(g_ds.sock, payload.data(), payload.size(), 0,
        (struct sockaddr*)&grp, sizeof(grp));
}

std::string build_announce(const char* action)
{
    static std::atomic<unsigned long> seq { 1000 };
    char msgid[80];
    snprintf(msgid, sizeof(msgid), "urn:uuid:%s-%lu", g_ds.uuid.c_str(),
        seq.fetch_add(1));
    std::string x;
    x += "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";
    x += "<s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\""
         " xmlns:a=\"http://schemas.xmlsoap.org/ws/2004/08/addressing\""
         " xmlns:d=\"http://schemas.xmlsoap.org/ws/2005/04/discovery\""
         " xmlns:dn=\"http://www.onvif.org/ver10/network/wsdl\">";
    x += "<s:Header>";
    x += "<a:MessageID>" + std::string(msgid) + "</a:MessageID>";
    x += "<a:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</a:To>";
    x += "<a:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/" + std::string(action) + "</a:Action>";
    x += "</s:Header><s:Body><d:" + std::string(action) + ">";
    x += "<a:EndpointReference><a:Address>urn:uuid:" + g_ds.uuid + "</a:Address></a:EndpointReference>";
    if (strcmp(action, "Hello") == 0) {
        x += "<d:Types>dn:NetworkVideoTransmitter</d:Types>";
        x += "<d:Scopes>" + build_scopes() + "</d:Scopes>";
        x += "<d:XAddrs></d:XAddrs>";
    }
    x += "<d:MetadataVersion>1</d:MetadataVersion>";
    x += "</d:" + std::string(action) + "></s:Body></s:Envelope>";
    return x;
}

void discovery_loop()
{
    char buf[8192];
    while (g_ds.running.load(std::memory_order_acquire)) {
        struct sockaddr_in peer {};
        socklen_t plen = sizeof(peer);
        const ssize_t n = recvfrom(g_ds.sock, buf, sizeof(buf) - 1, 0,
            (struct sockaddr*)&peer, &plen);
        if (n <= 0) continue;  /* timeout or error; the flag decides the loop */
        buf[n] = '\0';

        const std::string msg(buf, static_cast<size_t>(n));
        if (!is_probe(msg)) continue;

        const std::string reply = build_probe_matches(extract_message_id(msg),
            local_ip_for(peer));
        /* Unicast back to the prober, per WS-Discovery. */
        sendto(g_ds.sock, reply.data(), reply.size(), 0,
            (struct sockaddr*)&peer, plen);
        g_ds.probes.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace

/* -------------------------------------------------------------------------- */

void onvif_service_config_init(onvif_service_config* cfg)
{
    if (cfg == nullptr) return;
    *cfg = onvif_service_config {};
}

int onvif_service_start(const onvif_service_config* cfg)
{
    if (g_ds.running.load()) return 0;

    onvif_service_config def;
    onvif_service_config_init(&def);
    g_ds.cfg = cfg ? *cfg : def;

    /* A stable identity across restarts: clients key their device list on it,
     * and a UUID that changed every boot would show up as a new camera each
     * time. Derived from the serial number for that reason. Computed before the
     * discovery_enabled check because the SOAP service uses it too. */
    g_ds.uuid = g_ds.cfg.serial.empty() ? "recamera-unknown" : g_ds.cfg.serial;

    /*
     * SOAP first. Starting it before announcing Hello means the endpoint is
     * accepting connections by the time anything is told about it -- the
     * reverse order leaves a window where an eager client connects to a port
     * that is not open yet and gives up.
     *
     * A SOAP failure is not fatal to discovery: a device that is discoverable
     * but whose service port is taken is a diagnosable state, and the log line
     * in onvif_soap_start says so explicitly.
     */
    if (g_ds.cfg.soap_enabled) {
        onvif_soap_start(&g_ds.cfg, g_ds.uuid);
    }

    if (!g_ds.cfg.discovery_enabled) {
        fprintf(stderr, "[%s] discovery disabled by configuration\n", OS_TAG);
        return 0;
    }

    g_ds.sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (g_ds.sock < 0) {
        fprintf(stderr, "[%s] socket failed\n", OS_TAG);
        return -1;
    }

    int on = 1;
    setsockopt(g_ds.sock, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    struct sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(WSD_PORT);
    if (bind(g_ds.sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[%s] bind :%d failed (another ONVIF service running?)\n",
            OS_TAG, WSD_PORT);
        close(g_ds.sock);
        g_ds.sock = -1;
        return -1;
    }

    /* Join on every interface: the device is reachable over usb0, eth0 and
     * wlan0 depending on deployment, and which one a client sits on is not
     * knowable here. Failure is logged, not fatal -- being undiscoverable
     * beats refusing to start. */
    struct ip_mreq mreq {};
    inet_pton(AF_INET, WSD_GROUP, &mreq.imr_multiaddr);
    mreq.imr_interface.s_addr = htonl(INADDR_ANY);
    if (setsockopt(g_ds.sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        fprintf(stderr, "[%s] IP_ADD_MEMBERSHIP failed; discovery will not work\n", OS_TAG);
    }

    /* Bounded receive so the loop can observe the stop flag. */
    struct timeval tv { 1, 0 };
    setsockopt(g_ds.sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    g_ds.running.store(true, std::memory_order_release);
    g_ds.worker = std::thread(discovery_loop);

    send_multicast(build_announce("Hello"));
    fprintf(stderr, "[%s] discovery on udp/%d, XAddrs http://<ip>:%d%s\n",
        OS_TAG, WSD_PORT, g_ds.cfg.service_port, g_ds.cfg.service_path.c_str());
    return 0;
}

void onvif_service_stop(void)
{
    /* Unconditional, and before the discovery check: with discovery disabled
     * the discovery thread never ran, so an early return keyed on it would
     * leave the SOAP listener holding its port after stop() claimed to have
     * released everything. */
    onvif_soap_stop();

    if (!g_ds.running.load()) return;
    send_multicast(build_announce("Bye"));
    g_ds.running.store(false, std::memory_order_release);
    if (g_ds.worker.joinable()) g_ds.worker.join();
    if (g_ds.sock >= 0) {
        close(g_ds.sock);
        g_ds.sock = -1;
    }
}

bool onvif_service_discovery_running(void) { return g_ds.running.load(); }

unsigned long onvif_service_probe_count(void)
{
    return g_ds.probes.load(std::memory_order_relaxed);
}
