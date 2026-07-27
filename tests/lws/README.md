# libwebsockets migration tests

Written for the mongoose → libwebsockets migration
(`docs/lws-migration-progress.md`). They are equally useful as a regression
suite for the HTTP/WebSocket layer generally, since they assert on behaviour
the API layer depends on rather than on either library.

## multipart_fuzz.cpp — host-side, no device needed

The multipart parser is the only code in the migration that reads
attacker-controlled bytes directly, so it is a pure function
(`multipart_boundary` / `multipart_next` in `http_request_lws.h`) specifically
to be testable here.

```sh
python3 - <<'X'          # extract the pure functions
import pathlib
src = pathlib.Path("solutions/supervisor/main/include/http_request_lws.h").read_text()
b = src.index("struct MultipartSpan {"); e = src.rindex("inline size_t next_multipart_impl(")
pathlib.Path("mp.hpp").write_text("#include <string>\n#include <cstddef>\n\n" + src[b:e])
X
clang++ -std=c++17 -g -fsanitize=address,undefined multipart_fuzz.cpp -o fuzz && ./fuzz
```

400k cases: 20 hand-written malformed inputs, 200k random, 200k mutations of a
valid body. Three invariants, and they are the point:

  - every returned span lies inside the body (catches out-of-bounds)
  - each call advances the position (catches infinite loops)
  - no crash under ASan/UBSan

## ws_verify.py — device, WebSocket behaviour

16 assertions against `ws://<device>:8001`. The one that matters is T6:
starve a slow client for 20 seconds and check a healthy peer keeps streaming.
That is where a backend swap breaks things quietly.

**Close every browser tab on the Live page first.** The client limit is two per
kind, and a forgotten console counts against it -- an earlier run failed six
assertions for that reason before anyone suspected the environment.

## soak.py — device, leak hunting

Exercises HTTP, WebSocket connect/disconnect, snapshot and multipart in a loop,
sampling RSS/FD/threads every three minutes.

```sh
scp probe.sh recamera@<device>:/tmp/ && ssh recamera@<device> 'chmod +x /tmp/probe.sh'
python3 soak.py 900
```

Watch the FD count above all: connection-registry entries are attached on
LWS_CALLBACK_HTTP and detached on CLOSED_HTTP/DROP_PROTOCOL, so a path that
skips those shows up here and nowhere else. A 15-minute run at 772 iterations
(~3800 connections) held FD at 8 and 61 with RSS flat.

## Known behaviour, not a defect

A request whose Content-Length understates the body it sends leaves the
connection hanging until it times out, rather than being answered with 400.
lws reads exactly Content-Length bytes and treats the remainder as the start of
another request. Nothing leaks and no other connection is affected; correcting
it would mean extra protocol-level validation for little gain.
