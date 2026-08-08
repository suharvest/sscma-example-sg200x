#!/bin/sh
# The caller provides CODEX_SUDO_PASS in the environment after consuming one
# text line from SSH stdin. Keeping the helper password-free lets the remaining
# stdin bytes stay available for the raw RGB stream.
printf '%s\n' "$CODEX_SUDO_PASS"
