#ifndef _HTTP_REQUEST_H_
#define _HTTP_REQUEST_H_

#include <cstddef>
#include <string_view>

/*
 * http_request: a library-neutral view of one in-flight HTTP request.
 *
 * The third and last seam between the supervisor and its HTTP library, after
 * http_dispatch.h (replies from the async path) and
 * components/debug_stream/src/ws_transport.h (WebSocket fan-out). mongoose is
 * GPL-2.0-only against this repository's Apache-2.0 license, so every place
 * that touches it has to be confined to a swappable backend.
 *
 * Design note: this is a *view*, not a copy. uri/query/body point straight
 * into the library's own receive buffer and are valid only for the duration of
 * the dispatch that produced them -- exactly the lifetime mg_http_message
 * already had, so nothing changes for callers. Header and multipart lookups
 * stay as backend function pointers rather than being eagerly materialised:
 * firmware and model uploads arrive as multipart bodies of tens of megabytes,
 * and copying those into a neutral container would be a serious regression.
 *
 * Adding a backend means writing the three callbacks plus a builder; see
 * http_request_mongoose.h.
 */

struct http_request;

/* One part of a multipart/form-data body. `data` points into the request
 * buffer and is not NUL-terminated. */
struct http_multipart_part {
    std::string_view name;
    std::string_view filename;
    char* data = nullptr;
    size_t len = 0;
};

struct http_request {
    std::string_view uri;   /* path only, no query string */
    std::string_view query; /* raw query string, no leading '?' */
    std::string_view body;  /* raw body bytes */

    /* Case-insensitive header lookup. Empty view when absent. */
    std::string_view (*header)(const http_request* r, const char* name) = nullptr;

    /* Query-string variable lookup. Empty view when absent. */
    std::string_view (*query_var)(const http_request* r, const char* name) = nullptr;

    /* Walk multipart/form-data. Pass 0 to start; returns the position to pass
     * next, or 0 when there are no more parts. */
    size_t (*next_multipart)(const http_request* r, size_t pos,
                             http_multipart_part* out) = nullptr;

    /* Backend-private handle (mongoose: the mg_http_message*). */
    const void* impl = nullptr;
};

#endif // _HTTP_REQUEST_H_
