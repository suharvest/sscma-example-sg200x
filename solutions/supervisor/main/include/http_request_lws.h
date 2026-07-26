#ifndef _HTTP_REQUEST_LWS_H_
#define _HTTP_REQUEST_LWS_H_

#include <cstring>
#include <string>
#include <vector>

#include <libwebsockets.h>

#include "http_lws_registry.h"
#include "http_request.h"

/*
 * Builds an http_request view over a libwebsockets request.
 *
 * On multipart: this parses the accumulated body itself rather than using
 * lws_spa. That looks like the wrong choice until you notice what mongoose
 * was doing -- mg_http_next_multipart() walks a body that is already entirely
 * in memory, so firmware and model uploads of tens of megabytes were being
 * buffered whole long before this migration. lws_spa would stream, which is
 * genuinely better, but it also changes the shape of get_multiparts(): parts
 * would have to be materialised somewhere as they arrive, and "somewhere" on a
 * 180 MB device with no temp space to spare is the same buffer.
 *
 * So the memory profile is deliberately kept identical to mongoose's, and the
 * parser is ~70 lines with no behavioural difference for callers. Switching to
 * true streaming is worth doing, but as its own change with its own testing,
 * not smuggled into a library swap.
 */

namespace http_lws {

inline std::string hdr_value(struct lws* wsi, enum lws_token_indexes tok)
{
    const int n = lws_hdr_total_length(wsi, tok);
    if (n <= 0) return "";
    std::string out(static_cast<size_t>(n) + 1, '\0');
    if (lws_hdr_copy(wsi, &out[0], n + 1, tok) < 0) return "";
    out.resize(static_cast<size_t>(n));
    return out;
}

/* Case-insensitive lookup over the tokens lws parses, falling back to the
 * custom-header API for anything outside that set (Authorization is the one
 * that matters here). */
inline std::string_view header_impl(const http_request* r, const char* name);
inline std::string_view query_var_impl(const http_request* r, const char* name);
inline size_t next_multipart_impl(const http_request* r, size_t pos,
                                  http_multipart_part* out);

/* Backing store for the string_views handed out by the callbacks above: they
 * must outlive the call, and the request object is per-dispatch. */
struct RequestCtx {
    struct lws* wsi = nullptr;
    Conn* conn = nullptr;
    std::string uri;
    std::string query;
    mutable std::string scratch;      /* last header/query lookup */
    mutable std::string part_name;    /* last multipart part */
    mutable std::string part_filename;
};

inline std::string_view header_impl(const http_request* r, const char* name)
{
    const RequestCtx* c = static_cast<const RequestCtx*>(r->impl);
    if (c == nullptr || name == nullptr) return {};

    /* Only the tokens this lws build actually parses. LWS_WITH_UNCOMMON_HEADERS
     * is off (it costs table space for headers nobody here reads), so e.g.
     * WSI_TOKEN_HTTP_USER_AGENT does not exist -- anything absent falls
     * through to the custom-header lookup below and still works. */
    static const struct {
        const char* name;
        enum lws_token_indexes tok;
    } kKnown[] = {
        { "host", WSI_TOKEN_HOST },
        /* Authorization must be here, not left to the custom-header fallback:
         * lws parses every header it has a token for into the token table and
         * only unknown ones reach the custom store, so lws_hdr_custom_length()
         * returns 0 for it. Every authenticated request 401'd until this line
         * existed. */
        { "authorization", WSI_TOKEN_HTTP_AUTHORIZATION },
        { "content-type", WSI_TOKEN_HTTP_CONTENT_TYPE },
        { "content-length", WSI_TOKEN_HTTP_CONTENT_LENGTH },
        { "accept", WSI_TOKEN_HTTP_ACCEPT },
        { "origin", WSI_TOKEN_ORIGIN },
    };

    std::string want(name);
    for (char& ch : want) ch = static_cast<char>(tolower(ch));

    for (const auto& k : kKnown) {
        if (want == k.name) {
            c->scratch = hdr_value(c->wsi, k.tok);
            return c->scratch;
        }
    }

    /* Authorization and anything else custom. lws needs the trailing colon. */
    std::string with_colon = want + ":";
    const int n = lws_hdr_custom_length(c->wsi, with_colon.c_str(),
        static_cast<int>(with_colon.size()));
    if (n <= 0) {
        c->scratch.clear();
        return c->scratch;
    }
    c->scratch.assign(static_cast<size_t>(n) + 1, '\0');
    if (lws_hdr_custom_copy(c->wsi, &c->scratch[0], n + 1, with_colon.c_str(),
            static_cast<int>(with_colon.size())) < 0) {
        c->scratch.clear();
        return c->scratch;
    }
    c->scratch.resize(static_cast<size_t>(n));
    return c->scratch;
}

inline std::string_view query_var_impl(const http_request* r, const char* name)
{
    const RequestCtx* c = static_cast<const RequestCtx*>(r->impl);
    if (c == nullptr || name == nullptr) return {};
    char buf[512];
    /* lws_get_urlarg_by_name wants "name=" and returns the value part. */
    const char* v = lws_get_urlarg_by_name(c->wsi, name, buf, sizeof(buf));
    if (v == nullptr) {
        c->scratch.clear();
        return c->scratch;
    }
    c->scratch = v;
    return c->scratch;
}

/* ------------------------------------------------------------------------
 * multipart/form-data
 *
 * The parsing is split out from the lws plumbing so it can be tested on the
 * host without a device or a libwebsockets context. That matters more here
 * than elsewhere: this is the only new code in the migration that reads
 * attacker-controlled bytes directly, and every early-return below exists
 * because a fuzzer found the input that needed it.
 * ------------------------------------------------------------------------ */

/* One parsed part, as offsets into the body rather than pointers, so the
 * pure function stays free of lifetime questions. */
struct MultipartSpan {
    std::string name;
    std::string filename;
    size_t data_off = 0;
    size_t data_len = 0;
};

/* Extract the boundary from a Content-Type header. Empty when absent or
 * malformed -- callers must treat that as "no parts", not as "parse anyway". */
inline std::string multipart_boundary(const std::string& content_type)
{
    const size_t b = content_type.find("boundary=");
    if (b == std::string::npos) return "";
    std::string v = content_type.substr(b + 9);
    if (!v.empty() && v.front() == '"') {
        const size_t e = v.find('"', 1);
        v = (e == std::string::npos) ? std::string() : v.substr(1, e - 1);
    } else {
        const size_t e = v.find_first_of(";,");
        if (e != std::string::npos) v = v.substr(0, e);
    }
    while (!v.empty() && (v.back() == ' ' || v.back() == '\r' || v.back() == '\n')) v.pop_back();
    return v;
}

/*
 * Advance one part. Returns the position to pass next, or 0 when there are no
 * more (which is also what every malformed case returns -- refusing to parse
 * beats guessing at a truncated or hostile body).
 *
 * The contract deliberately mirrors mg_http_next_multipart so callers did not
 * have to change when the library did.
 */
inline size_t multipart_next(const std::string& body, const std::string& boundary,
    size_t pos, MultipartSpan* out)
{
    if (boundary.empty() || out == nullptr || pos >= body.size()) return 0;

    const std::string delim = "--" + boundary;
    const size_t d = body.find(delim, pos);
    if (d == std::string::npos) return 0;

    size_t p = d + delim.size();
    /* Closing delimiter "--boundary--" ends the document. */
    if (body.compare(p, 2, "--") == 0) return 0;
    if (body.compare(p, 2, "\r\n") == 0) p += 2;
    if (p >= body.size()) return 0;

    const size_t hdr_end = body.find("\r\n\r\n", p);
    if (hdr_end == std::string::npos) return 0;
    const std::string headers = body.substr(p, hdr_end - p);
    const size_t data_begin = hdr_end + 4;
    if (data_begin > body.size()) return 0;

    /* A part with no following delimiter is a truncated body. Dropping it is
     * the safe reading: the alternative is handing the caller a length that
     * runs to the end of whatever happened to be received. */
    const size_t next_d = body.find(delim, data_begin);
    if (next_d == std::string::npos) return 0;

    size_t data_end = next_d;
    if (data_end >= data_begin + 2 && body.compare(data_end - 2, 2, "\r\n") == 0) {
        data_end -= 2;
    }
    if (data_end < data_begin) data_end = data_begin;

    auto quoted = [&headers](const char* key) -> std::string {
        const size_t k = headers.find(key);
        if (k == std::string::npos) return "";
        const size_t q1 = headers.find('"', k);
        if (q1 == std::string::npos) return "";
        const size_t q2 = headers.find('"', q1 + 1);
        if (q2 == std::string::npos) return "";
        return headers.substr(q1 + 1, q2 - q1 - 1);
    };
    out->name = quoted("name=");
    out->filename = quoted("filename=");
    out->data_off = data_begin;
    out->data_len = data_end - data_begin;
    return next_d;
}

inline size_t next_multipart_impl(const http_request* r, size_t pos,
    http_multipart_part* out)
{
    const RequestCtx* c = static_cast<const RequestCtx*>(r->impl);
    if (c == nullptr || c->conn == nullptr || out == nullptr) return 0;

    const std::string& body = c->conn->body;
    const std::string boundary =
        multipart_boundary(hdr_value(c->wsi, WSI_TOKEN_HTTP_CONTENT_TYPE));

    MultipartSpan span;
    const size_t next = multipart_next(body, boundary, pos, &span);
    if (next == 0) return 0;

    c->part_name = span.name;
    c->part_filename = span.filename;
    out->name = c->part_name;
    out->filename = c->part_filename;
    /* Non-const because multipart_t::data is; nothing writes through it. */
    out->data = const_cast<char*>(body.data()) + span.data_off;
    out->len = span.data_len;
    return next;
}

/* Caller owns both; the view must not outlive the RequestCtx. */
inline http_request make_request(RequestCtx& ctx)
{
    http_request r;
    r.uri = ctx.uri;
    r.query = ctx.query;
    r.body = ctx.conn != nullptr ? std::string_view(ctx.conn->body) : std::string_view();
    r.header = header_impl;
    r.query_var = query_var_impl;
    r.next_multipart = next_multipart_impl;
    r.impl = &ctx;
    return r;
}

} // namespace http_lws

#endif // _HTTP_REQUEST_LWS_H_
