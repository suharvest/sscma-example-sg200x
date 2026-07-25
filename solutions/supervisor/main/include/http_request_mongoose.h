#ifndef _HTTP_REQUEST_MONGOOSE_H_
#define _HTTP_REQUEST_MONGOOSE_H_

#include "http_request.h"
#include "mongoose.h"

/*
 * Builds an http_request view over a mongoose mg_http_message.
 *
 * Zero copying: every field and every callback result points into mongoose's
 * own buffer, so the view costs one small stack struct per request and is
 * valid for exactly as long as the mg_http_message is.
 *
 * For a libwebsockets backend the same three callbacks map onto
 * lws_hdr_copy / lws_get_urlarg_by_name / lws_spa_* respectively. Note lws has
 * no equivalent of mongoose's "parse the whole multipart body in place" walk:
 * its lws_spa API is a streaming callback parser, so that backend will have to
 * drive lws_spa during the body-receive phase and record part boundaries,
 * then replay them here. Everything above this header stays unchanged.
 */

inline std::string_view mgsv(const struct mg_str& s)
{
    return std::string_view(s.buf, s.len);
}

inline std::string_view http_request_mg_header(const http_request* r, const char* name)
{
    auto* hm = static_cast<const struct mg_http_message*>(r->impl);
    struct mg_str* h = mg_http_get_header(const_cast<struct mg_http_message*>(hm), name);
    return h == nullptr ? std::string_view() : mgsv(*h);
}

inline std::string_view http_request_mg_query_var(const http_request* r, const char* name)
{
    auto* hm = static_cast<const struct mg_http_message*>(r->impl);
    struct mg_str v = mg_http_var(hm->query, mg_str(name));
    return v.buf == nullptr ? std::string_view() : mgsv(v);
}

inline size_t http_request_mg_next_multipart(const http_request* r, size_t pos,
    http_multipart_part* out)
{
    auto* hm = static_cast<const struct mg_http_message*>(r->impl);
    struct mg_http_part part;
    size_t next = mg_http_next_multipart(hm->body, pos, &part);
    if (next > 0 && out != nullptr) {
        out->name = mgsv(part.name);
        out->filename = mgsv(part.filename);
        out->data = part.body.buf;
        out->len = part.body.len;
    }
    return next;
}

/* Caller owns the returned value; it must not outlive `hm`. */
inline http_request http_request_from_mg(const struct mg_http_message* hm)
{
    http_request r;
    r.uri = mgsv(hm->uri);
    r.query = mgsv(hm->query);
    r.body = mgsv(hm->body);
    r.header = http_request_mg_header;
    r.query_var = http_request_mg_query_var;
    r.next_multipart = http_request_mg_next_multipart;
    r.impl = hm;
    return r;
}

#endif // _HTTP_REQUEST_MONGOOSE_H_
