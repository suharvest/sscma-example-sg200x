#!/usr/bin/env python3
"""Publish an application package to the CDN and point the ecosystem at it.

Why this exists
---------------
Uploading the .deb was a purely manual step that no script owned and no check
covered. On 2026-07-29 that cost a release: supervisor 0.5.0, 0.5.1 and 0.5.2
were each built, installed on a test device over scp, and verified — while the
CDN still served 0.4.1. Everyone deploying from the SenseCraft App got a
console without any of the three versions' features, and nothing anywhere said
so, because the ecosystem YAML and the CDN agreed with each other perfectly:
both said 0.4.1. They were consistently stale.

So the check that matters is not "does the URL resolve" (it did) or "does the
checksum match" (it did) — it is **"is the version the ecosystem ships the
version we actually build?"** That question spans two repositories, which is
exactly why nothing was asking it.

    scripts/release-app.py --check              # every app: built vs published
    scripts/release-app.py fitness-trainer      # publish one, update the YAML

--check is the guard; run it before a release and after bumping any version.
Publishing verifies the upload by downloading it back and comparing sha256 —
`ossutil` reporting success is not the same as the bytes being retrievable.

The solutions checkout is found via $SENSECRAFT_SOLUTIONS, falling back to the
usual location next to this one.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOLUTIONS_DIR = REPO / "solutions"
DEFAULT_SOLUTIONS_REPO = Path(
    os.environ.get("SENSECRAFT_SOLUTIONS", Path.home() / "project" / "sensecraft-solutions")
)
ECOSYSTEM = "solutions/recamera_ecosystem/devices"
CDN_BASE = "https://sensecraft-statics.seeed.cc/solution-app/recamera_ecosystem/packages"
OSS_BASE = "oss://sensecraft-statics/solution-app/recamera_ecosystem/packages"

_VERSION_RE = re.compile(r"project\(\s*([A-Za-z0-9_-]+)\s+VERSION\s+([0-9][0-9.]*)")
# The .deb URL and the sha256 that follows it inside the same deb_package block.
_DEB_LINE_RE = re.compile(r"^(\s*path:\s*)(\S*/packages/([A-Za-z0-9._+-]+)_([0-9][0-9.]*)_riscv64\.deb)\s*$", re.M)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def built_version(solution: str) -> str | None:
    cml = SOLUTIONS_DIR / solution / "CMakeLists.txt"
    if not cml.exists():
        return None
    for m in _VERSION_RE.finditer(cml.read_text(encoding="utf-8")):
        if m.group(1) == solution:
            return m.group(2)
    return None


def device_yamls(solutions_repo: Path) -> list[Path]:
    d = solutions_repo / ECOSYSTEM
    return sorted(d.glob("*.yaml")) if d.is_dir() else []


def published(solutions_repo: Path) -> dict[str, tuple[str, Path]]:
    """package name -> (version, device yaml) as the ecosystem currently ships."""
    out: dict[str, tuple[str, Path]] = {}
    for y in device_yamls(solutions_repo):
        for m in _DEB_LINE_RE.finditer(y.read_text(encoding="utf-8")):
            out[m.group(3)] = (m.group(4), y)
    return out


def cdn_head(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status
    except Exception:
        return None


def cmd_check(solutions_repo: Path) -> int:
    pub = published(solutions_repo)
    rows, stale = [], 0
    for sol in sorted(p.name for p in SOLUTIONS_DIR.iterdir() if p.is_dir()):
        bv = built_version(sol)
        if bv is None:
            continue
        pv = pub.get(sol, (None, None))[0]
        if pv is None:
            rows.append((sol, bv, "-", "not in ecosystem"))
            continue
        if pv != bv:
            rows.append((sol, bv, pv, "<-- STALE"))
            stale += 1
        else:
            rows.append((sol, bv, pv, ""))

    print(f"{'APP':22} {'BUILT':10} {'PUBLISHED':10}")
    print("-" * 60)
    for sol, bv, pv, note in rows:
        print(f"{sol:22} {bv:10} {pv:10} {note}")

    if stale:
        print(f"\nFAIL: {stale} app(s) build a version the ecosystem does not ship.")
        print("      Run 'scripts/release-app.py <app>' for each, or bump the")
        print("      ecosystem on purpose if the older package is the intended one.")
        return 1
    print("\nOK: every app's built version is the one the ecosystem ships.")
    return 0


def cmd_publish(solution: str, solutions_repo: Path, dry_run: bool, publish_content: bool) -> int:
    version = built_version(solution)
    if not version:
        print(f"error: no project({solution} VERSION ...) in its CMakeLists.txt", file=sys.stderr)
        return 1

    deb = SOLUTIONS_DIR / solution / "build" / f"{solution}_{version}_riscv64.deb"
    if not deb.exists():
        print(f"error: {deb} not found — build it first:", file=sys.stderr)
        print(f"  cd solutions/{solution} && cmake --build build -j4 && (cd build && cpack)", file=sys.stderr)
        return 1

    digest = sha256_file(deb)
    url = f"{CDN_BASE}/{deb.name}"
    print(f"{solution} {version}")
    print(f"  package : {deb.name}  ({deb.stat().st_size} bytes)")
    print(f"  sha256  : {digest}")

    pub = published(solutions_repo)
    entry = pub.get(solution)
    if entry is None:
        print(f"  warning: no ecosystem device YAML references package '{solution}';")
        print(f"           uploading anyway, but nothing will point at it.")
    elif entry[0] == version and cdn_head(url) in (200, 206):
        print(f"  already published and referenced by {entry[1].name}; nothing to do.")
        return 0

    if dry_run:
        print("  [dry-run] would upload and rewrite the device YAML")
        return 0

    print(f"  uploading -> {OSS_BASE}/")
    subprocess.run(["ossutil", "cp", "-f", str(deb), f"{OSS_BASE}/"], check=True, capture_output=True)

    # Download it back. ossutil printing success is not the same as the bytes
    # being retrievable, and a half-published package fails on a user's device
    # instead of here.
    print("  verifying from CDN ...")
    with urllib.request.urlopen(url, timeout=180) as r:
        got = hashlib.sha256(r.read()).hexdigest()
    if got != digest:
        print(f"  FAIL: CDN returned sha256 {got}", file=sys.stderr)
        return 1
    print("  verified: CDN copy is byte-identical")

    if entry is None:
        return 0

    old_version, yaml_path = entry
    text = yaml_path.read_text(encoding="utf-8")
    new_text, n = _DEB_LINE_RE.subn(
        lambda m: (m.group(1) + f"{CDN_BASE}/{deb.name}") if m.group(3) == solution else m.group(0),
        text,
    )
    # The sha256 belonging to this package is the first one after its path.
    idx = new_text.find(deb.name)
    sha_m = re.compile(r"(sha256:\s*)([0-9a-f]{64})").search(new_text, idx)
    if not sha_m:
        print(f"  error: no sha256 found after the package path in {yaml_path.name}", file=sys.stderr)
        return 1
    new_text = new_text[: sha_m.start(2)] + digest + new_text[sha_m.end(2) :]
    yaml_path.write_text(new_text, encoding="utf-8")
    print(f"  {yaml_path.name}: {old_version} -> {version} (+ sha256)")

    if not publish_content:
        print("\nNext (or re-run with --publish-content to chain these):")
        print(f"  cd {solutions_repo}")
        print("  uv run --package sensecraft-solutionctl solutionctl validate solutions/recamera_ecosystem --spec-dir spec --check-urls")
        print("  uv run python scripts/generate_recamera_catalog.py          # console install catalog")
        print("  uv run python scripts/generate_solution_manifest.py         # OTA content + bundled_hashes")
        print("  git add -p && git commit && open a PR")
        return 0

    return run_content_publish(solutions_repo)


def run_content_publish(solutions_repo: Path) -> int:
    """Validate, then regenerate both published artefacts derived from the YAML.

    The dependency runs one way only: this script edits the device YAML, so the
    solution zip and the install catalog are downstream of it and must be
    rebuilt after. Neither generator can call this one — they would be
    republishing content that had not been produced yet.

    The OTA content publish is deliberately NOT chained here. Its generator
    refuses to run while solutions/ has uncommitted paths, because its zips are
    built from the working tree — publishing before the commit would ship
    content that is not in git. This script has just rewritten the device YAML,
    so that guard would fire every time. The correct order is:

        release-app.py <app> --publish-content   # upload, YAML, validate, catalog
        git commit                               # the version bump
        generate_solution_manifest.py            # OTA content, from committed state

    The catalog is chained because it is derived from the YAML this script just
    changed and has no such guard.
    """
    steps = [
        (["uv", "run", "--package", "sensecraft-solutionctl", "solutionctl", "validate",
          "solutions/recamera_ecosystem", "--spec-dir", "spec", "--check-urls"], "validate"),
        (["uv", "run", "python", "scripts/generate_recamera_catalog.py"], "install catalog"),
    ]
    # Strip the variables that hijack `uv run`'s interpreter choice. An active
    # conda/venv in the calling shell leaks through subprocess and makes uv
    # resolve to that interpreter instead of the solutions project's, which
    # fails in confusing ways far from the cause (an argparse TypeError from a
    # different Python, in the observed case).
    env = {k: v for k, v in os.environ.items()
           if k not in ("VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONHOME", "PYTHONPATH")}

    for cmd, label in steps:
        print(f"\n=== {label} ===")
        r = subprocess.run(cmd, cwd=solutions_repo, env=env)
        if r.returncode != 0:
            print(f"\n{label} failed (exit {r.returncode}).", file=sys.stderr)
            return r.returncode

    print("\nUploaded, YAML updated, catalog republished. Still to do, in order:")
    print(f"  cd {solutions_repo}")
    print("  git add -p && git commit                              # the version bump")
    print("  uv run python scripts/generate_solution_manifest.py   # OTA content, from the commit")
    print("  git add solutions/bundled_hashes.json && git commit && open a PR")
    print("\nThe OTA step is after the commit on purpose: its zips come from the")
    print("working tree, so publishing first would ship content that is not in git.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("app", nargs="?", help="solution to publish (omit with --check)")
    ap.add_argument("--check", action="store_true", help="report built-vs-published drift for every app")
    ap.add_argument("--dry-run", action="store_true", help="publish: show what would happen")
    ap.add_argument("--publish-content", action="store_true",
                    help="after updating the YAML, also validate and regenerate the "
                         "install catalog (the OTA content publish stays manual: it "
                         "must run from a committed tree)")
    ap.add_argument("--solutions-repo", type=Path, default=DEFAULT_SOLUTIONS_REPO)
    args = ap.parse_args()

    repo = args.solutions_repo
    if not (repo / ECOSYSTEM).is_dir():
        print(f"error: no {ECOSYSTEM} under {repo}", file=sys.stderr)
        print("       set $SENSECRAFT_SOLUTIONS or pass --solutions-repo", file=sys.stderr)
        return 1

    if args.check or not args.app:
        return cmd_check(repo)
    return cmd_publish(args.app, repo, args.dry_run, args.publish_content)


if __name__ == "__main__":
    raise SystemExit(main())
