#ifndef HTTP_SERVER_H
#define HTTP_SERVER_H

/*
 * The supervisor's HTTP server.
 *
 * The implementation lives in http_server_lws.h; this header exists so
 * main.cpp names one type regardless of what is underneath. It used to select
 * between a mongoose and a libwebsockets build during the migration away from
 * mongoose (GPL-2.0-only, incompatible with this repository's Apache-2.0
 * licence -- see docs/onvif-implementation-spec.md 0.5-B). mongoose is gone;
 * the indirection stays because it costs nothing and the request, reply and
 * transport seams beneath it are what made that migration a matter of adding
 * files rather than rewriting the API layer.
 */
#include "http_server_lws.h"

using http_server = http_server_lws;

#endif // HTTP_SERVER_H
