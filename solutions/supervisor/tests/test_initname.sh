#!/bin/sh
# test_initname.sh: test init script name resolution and reconciliation
# Runs under standard POSIX sh (dash/ash/busybox).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUPERVISOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
S93_SCRIPT="$SUPERVISOR_DIR/rootfs/etc/init.d/S93sscma-supervisor"
MAIN_SH="$SUPERVISOR_DIR/rootfs/usr/share/supervisor/scripts/main.sh"

TMPDIR="${TMPDIR:-/tmp}"
TEST_ROOT="$(mktemp -d "$TMPDIR/test_initname.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT INT TERM

PASSED=0
FAILED=0

assert_eq() {
    expected="$1"
    actual="$2"
    desc="$3"
    if [ "$expected" = "$actual" ]; then
        PASSED=$((PASSED + 1))
        echo "  [PASS] $desc"
    else
        FAILED=$((FAILED + 1))
        echo "  [FAIL] $desc: expected '$expected', got '$actual'"
    fi
}

assert_file_exists() {
    file="$1"
    desc="$2"
    if [ -f "$file" ]; then
        PASSED=$((PASSED + 1))
        echo "  [PASS] $desc: exists ($file)"
    else
        FAILED=$((FAILED + 1))
        echo "  [FAIL] $desc: missing ($file)"
    fi
}

assert_file_not_exists() {
    file="$1"
    desc="$2"
    if [ ! -f "$file" ]; then
        PASSED=$((PASSED + 1))
        echo "  [PASS] $desc: does not exist ($file)"
    else
        FAILED=$((FAILED + 1))
        echo "  [FAIL] $desc: unexpectedly exists ($file)"
    fi
}

assert_log_contains() {
    log="$1"
    expected="$2"
    desc="$3"
    if [ -f "$log" ] && grep -Fq "$expected" "$log"; then
        PASSED=$((PASSED + 1))
        echo "  [PASS] $desc: log contains '$expected'"
    else
        FAILED=$((FAILED + 1))
        echo "  [FAIL] $desc: log '$log' does not contain '$expected'"
    fi
}

assert_log_not_contains() {
    log="$1"
    expected="$2"
    desc="$3"
    if [ ! -f "$log" ] || ! grep -Fq "$expected" "$log"; then
        PASSED=$((PASSED + 1))
        echo "  [PASS] $desc: log does not contain '$expected'"
    else
        FAILED=$((FAILED + 1))
        echo "  [FAIL] $desc: log '$log' unexpectedly contains '$expected'"
    fi
}

make_fake_init_script() {
    target_path="$1"
    log_path="$2"
    cat > "$target_path" <<EOF
#!/bin/sh
echo "\$(basename "\$0") \$1" >> "$log_path"
case "\$1" in
    status)
        exit 1
        ;;
    *)
        exit 0
        ;;
esac
EOF
    chmod +x "$target_path"
}

run_s93_reconcile() {
    init_dir="$1"
    apps_dir="$2"
    mode="$3"
    mkdir -p "$apps_dir"
    printf '%s\n' "$mode" > "$apps_dir/mode"
    SV_INIT_D="$init_dir" SV_APPS_DIR="$apps_dir" sh "$S93_SCRIPT" reconcile
}

echo "=== Test Suite: Supervisor Init Script Resolution ==="

# ----------------------------------------------------------------------------
# Layout 1: {S03node-red, S91sscma-node} (stock 03 / 91 layout)
# ----------------------------------------------------------------------------
echo ""
echo "--- Testing Layout 1: {S03node-red, S91sscma-node} ---"
L1_DIR="$TEST_ROOT/layout1/init.d"
L1_APPS="$TEST_ROOT/layout1/apps"
L1_LOG="$TEST_ROOT/layout1/calls.log"
mkdir -p "$L1_DIR" "$L1_APPS"
make_fake_init_script "$L1_DIR/S03node-red" "$L1_LOG"
make_fake_init_script "$L1_DIR/S91sscma-node" "$L1_LOG"

# 1. Resolver checks
res_nr=$(SV_INIT_D="$L1_DIR" bash "$MAIN_SH" _svc_base "node-red" "03node-red")
assert_eq "03node-red" "$res_nr" "Layout 1 resolver for node-red"
res_sscma=$(SV_INIT_D="$L1_DIR" bash "$MAIN_SH" _svc_base "sscma-node" "91sscma-node")
assert_eq "91sscma-node" "$res_sscma" "Layout 1 resolver for sscma-node"

# 2. Park (console mode) via S93 reconcile
run_s93_reconcile "$L1_DIR" "$L1_APPS" "console"
assert_file_exists "$L1_DIR/K03node-red" "Layout 1 parked K03node-red"
assert_file_not_exists "$L1_DIR/S03node-red" "Layout 1 S03node-red gone"
assert_file_exists "$L1_DIR/K91sscma-node" "Layout 1 parked K91sscma-node"
assert_file_not_exists "$L1_DIR/S91sscma-node" "Layout 1 S91sscma-node gone"
assert_log_contains "$L1_LOG" "S03node-red stop" "Layout 1 S03node-red stop called"
assert_log_contains "$L1_LOG" "S91sscma-node stop" "Layout 1 S91sscma-node stop called"

# 3. Arm (nodered mode) via S93 reconcile
run_s93_reconcile "$L1_DIR" "$L1_APPS" "nodered"
assert_file_exists "$L1_DIR/S03node-red" "Layout 1 armed S03node-red"
assert_file_not_exists "$L1_DIR/K03node-red" "Layout 1 K03node-red moved"
assert_file_exists "$L1_DIR/S91sscma-node" "Layout 1 armed S91sscma-node"
assert_file_not_exists "$L1_DIR/K91sscma-node" "Layout 1 K91sscma-node moved"

# ----------------------------------------------------------------------------
# Layout 2: {S92node-red, S91sscma-node} (device image with S92node-red)
# ----------------------------------------------------------------------------
echo ""
echo "--- Testing Layout 2: {S92node-red, S91sscma-node} ---"
L2_DIR="$TEST_ROOT/layout2/init.d"
L2_APPS="$TEST_ROOT/layout2/apps"
L2_LOG="$TEST_ROOT/layout2/calls.log"
mkdir -p "$L2_DIR" "$L2_APPS"
make_fake_init_script "$L2_DIR/S92node-red" "$L2_LOG"
make_fake_init_script "$L2_DIR/S91sscma-node" "$L2_LOG"

# 1. Resolver checks
res_nr=$(SV_INIT_D="$L2_DIR" bash "$MAIN_SH" _svc_base "node-red" "03node-red")
assert_eq "92node-red" "$res_nr" "Layout 2 resolver for node-red finds 92node-red"
res_sscma=$(SV_INIT_D="$L2_DIR" bash "$MAIN_SH" _svc_base "sscma-node" "91sscma-node")
assert_eq "91sscma-node" "$res_sscma" "Layout 2 resolver for sscma-node finds 91sscma-node"

# 2. Park (console mode) via S93 reconcile
run_s93_reconcile "$L2_DIR" "$L2_APPS" "console"
assert_file_exists "$L2_DIR/K92node-red" "Layout 2 parked K92node-red"
assert_file_not_exists "$L2_DIR/S92node-red" "Layout 2 S92node-red gone"
assert_file_exists "$L2_DIR/K91sscma-node" "Layout 2 parked K91sscma-node"
assert_file_not_exists "$L2_DIR/S91sscma-node" "Layout 2 S91sscma-node gone"
assert_log_contains "$L2_LOG" "S92node-red stop" "Layout 2 S92node-red stop called"
assert_log_contains "$L2_LOG" "S91sscma-node stop" "Layout 2 S91sscma-node stop called"

# 3. Arm (nodered mode) via S93 reconcile
run_s93_reconcile "$L2_DIR" "$L2_APPS" "nodered"
assert_file_exists "$L2_DIR/S92node-red" "Layout 2 armed S92node-red"
assert_file_not_exists "$L2_DIR/K92node-red" "Layout 2 K92node-red moved"
assert_file_exists "$L2_DIR/S91sscma-node" "Layout 2 armed S91sscma-node"
assert_file_not_exists "$L2_DIR/K91sscma-node" "Layout 2 K91sscma-node moved"

# ----------------------------------------------------------------------------
# Layout 3: {K92node-red, K91sscma-node} (parked state with 92/91)
# ----------------------------------------------------------------------------
echo ""
echo "--- Testing Layout 3: {K92node-red, K91sscma-node} ---"
L3_DIR="$TEST_ROOT/layout3/init.d"
L3_APPS="$TEST_ROOT/layout3/apps"
L3_LOG="$TEST_ROOT/layout3/calls.log"
mkdir -p "$L3_DIR" "$L3_APPS"
make_fake_init_script "$L3_DIR/K92node-red" "$L3_LOG"
make_fake_init_script "$L3_DIR/K91sscma-node" "$L3_LOG"

# 1. Resolver checks when only K-prefixed files exist
res_nr=$(SV_INIT_D="$L3_DIR" bash "$MAIN_SH" _svc_base "node-red" "03node-red")
assert_eq "92node-red" "$res_nr" "Layout 3 resolver for node-red finds 92node-red from K"
res_sscma=$(SV_INIT_D="$L3_DIR" bash "$MAIN_SH" _svc_base "sscma-node" "91sscma-node")
assert_eq "91sscma-node" "$res_sscma" "Layout 3 resolver for sscma-node finds 91sscma-node from K"

# 2. Park (console mode) via S93 reconcile (idempotent no-op)
run_s93_reconcile "$L3_DIR" "$L3_APPS" "console"
assert_file_exists "$L3_DIR/K92node-red" "Layout 3 K92node-red remains"
assert_file_not_exists "$L3_DIR/S92node-red" "Layout 3 S92node-red not created"
assert_file_exists "$L3_DIR/K91sscma-node" "Layout 3 K91sscma-node remains"
assert_file_not_exists "$L3_DIR/S91sscma-node" "Layout 3 S91sscma-node not created"
# stop should NOT be called because they were already parked
assert_log_not_contains "$L3_LOG" "stop" "Layout 3 idempotent park did not call stop"

# 3. Arm (nodered mode) via S93 reconcile
run_s93_reconcile "$L3_DIR" "$L3_APPS" "nodered"
assert_file_exists "$L3_DIR/S92node-red" "Layout 3 armed S92node-red"
assert_file_not_exists "$L3_DIR/K92node-red" "Layout 3 K92node-red moved"
assert_file_exists "$L3_DIR/S91sscma-node" "Layout 3 armed S91sscma-node"
assert_file_not_exists "$L3_DIR/K91sscma-node" "Layout 3 K91sscma-node moved"

# ----------------------------------------------------------------------------
# main.sh helpers: _svc_enable, _svc_disable, _svc_park_now
# ----------------------------------------------------------------------------
echo ""
echo "--- Testing main.sh Helpers with Layout 2 (92node-red) ---"
HELP_DIR="$TEST_ROOT/helpers/init.d"
HELP_LOG="$TEST_ROOT/helpers/calls.log"
mkdir -p "$HELP_DIR"
make_fake_init_script "$HELP_DIR/S92node-red" "$HELP_LOG"

# Test _svc_disable: stop S92node-red and rename to K92node-red
SV_INIT_D="$HELP_DIR" bash "$MAIN_SH" _svc_disable "92node-red"
assert_file_exists "$HELP_DIR/K92node-red" "main.sh _svc_disable renamed S92 to K92"
assert_file_not_exists "$HELP_DIR/S92node-red" "main.sh _svc_disable removed S92"
assert_log_contains "$HELP_LOG" "S92node-red stop" "main.sh _svc_disable invoked stop"

# Test _svc_enable: rename K92node-red back to S92node-red
en_out=$(SV_INIT_D="$HELP_DIR" bash "$MAIN_SH" _svc_enable "92node-red")
assert_eq "$HELP_DIR/S92node-red" "$en_out" "main.sh _svc_enable printed S92 path"
assert_file_exists "$HELP_DIR/S92node-red" "main.sh _svc_enable renamed K92 to S92"
assert_file_not_exists "$HELP_DIR/K92node-red" "main.sh _svc_enable removed K92"

# Test _svc_park_now: fast rename S92node-red -> K92node-red
SV_INIT_D="$HELP_DIR" bash "$MAIN_SH" _svc_park_now "92node-red"
assert_file_exists "$HELP_DIR/K92node-red" "main.sh _svc_park_now renamed S92 to K92"
assert_file_not_exists "$HELP_DIR/S92node-red" "main.sh _svc_park_now removed S92"

# ----------------------------------------------------------------------------
# Edge cases: prefer S over K with different numbers; fallback to default
# ----------------------------------------------------------------------------
echo ""
echo "--- Testing Edge Cases ---"
EDGE_DIR="$TEST_ROOT/edge/init.d"
EDGE_LOG="$TEST_ROOT/edge/calls.log"
mkdir -p "$EDGE_DIR"

# 1. Empty dir: returns default
res_def_nr=$(SV_INIT_D="$EDGE_DIR" bash "$MAIN_SH" _svc_base "node-red" "03node-red")
assert_eq "03node-red" "$res_def_nr" "Fallback to default 03node-red when empty"
res_def_sscma=$(SV_INIT_D="$EDGE_DIR" bash "$MAIN_SH" _svc_base "sscma-node" "91sscma-node")
assert_eq "91sscma-node" "$res_def_sscma" "Fallback to default 91sscma-node when empty"

# 2. Both S and K with different numbers: prefer S
make_fake_init_script "$EDGE_DIR/S92node-red" "$EDGE_LOG"
make_fake_init_script "$EDGE_DIR/K03node-red" "$EDGE_LOG"
res_pref=$(SV_INIT_D="$EDGE_DIR" bash "$MAIN_SH" _svc_base "node-red" "03node-red")
assert_eq "92node-red" "$res_pref" "Prefer S92 over K03 when both exist"

echo ""
echo "=== Test Results: $PASSED passed, $FAILED failed ==="
if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
exit 0
