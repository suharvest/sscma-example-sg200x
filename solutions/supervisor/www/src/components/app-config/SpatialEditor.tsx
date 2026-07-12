import { useEffect, useMemo, useRef, useState } from "react";
import { Button, message } from "antd";
import { useTranslation } from "react-i18next";
import {
  IConfigItem,
  ConfigValue,
  ILineValue,
  NormPoint,
  LineDirection,
} from "@/api/app/app";
import { pickLocalizedText } from "@/utils/appLocale";

/**
 * On-video zone/line editor. Rendered INSIDE the live player container
 * (absolute inset-0) so drawings land exactly on the displayed frame.
 *
 * Coordinates are stored normalized [0,1] relative to the video CONTENT
 * area (object-fit: contain), which matches the inference frame — the
 * same convention the backend validates and yolo-detector consumes.
 *
 * When the debug stream is off we deliberately still allow "blind"
 * drawing on a reference grid (field devices are often offline); a hint
 * suggests enabling the stream for on-scene calibration. Without a video
 * frame the aspect ratio is unknown, so the grid spans the full player
 * box — normalized values stay correct either way.
 */

interface ContentRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface SpatialEditorProps {
  item: IConfigItem; // type "zone" | "line"
  value: ConfigValue | undefined;
  /** Displayed video content area (object-contain), from the live page. */
  contentRect: ContentRect;
  /** True while the debug stream is actually rendering frames. */
  streaming: boolean;
  onDone: (value: NormPoint[] | ILineValue | null) => void;
  onCancel: () => void;
}

const ACCENT = "#8fc31f";
const HIT_RADIUS_PX = 14;

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/** Drop trailing vertices that duplicate their predecessor (dblclick fires
 *  two pointerdowns at ~the same spot before the dblclick event). */
function dedupeTail(points: NormPoint[]): NormPoint[] {
  const out = [...points];
  while (out.length > 1) {
    const [x1, y1] = out[out.length - 1];
    const [x2, y2] = out[out.length - 2];
    if (Math.hypot(x1 - x2, y1 - y2) < 0.015) out.pop();
    else break;
  }
  return out;
}

/* ---------------- geometry validation ----------------
 * Thresholds MUST stay identical to the backend (config_schema.hpp) so the
 * UI rejects exactly the degenerate shapes the device would reject. All in
 * normalized [0,1] coordinates.
 */
const LINE_MIN_LEN = 0.02; // ~2% of the frame
const ZONE_MIN_AREA = 0.005; // normalized polygon area

/** Remove consecutive duplicate vertices (and a closing vertex that repeats
 *  the first) so area / self-intersection tests see a clean ring. */
function dedupeRing(points: NormPoint[]): NormPoint[] {
  const out: NormPoint[] = [];
  for (const p of points) {
    const last = out[out.length - 1];
    if (last && Math.hypot(p[0] - last[0], p[1] - last[1]) < 0.015) continue;
    out.push(p);
  }
  while (out.length > 1) {
    const first = out[0];
    const last = out[out.length - 1];
    if (Math.hypot(first[0] - last[0], first[1] - last[1]) < 0.015) out.pop();
    else break;
  }
  return out;
}

/** Shoelace area of a closed polygon (absolute, normalized units). */
function polygonArea(pts: NormPoint[]): number {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[(i + 1) % pts.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

/** Proper intersection test for two segments (shared endpoints excluded by
 *  the caller via adjacency skipping). */
function segmentsCross(
  p1: NormPoint,
  p2: NormPoint,
  p3: NormPoint,
  p4: NormPoint
): boolean {
  const cross = (a: NormPoint, b: NormPoint, c: NormPoint) =>
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const d1 = cross(p3, p4, p1);
  const d2 = cross(p3, p4, p2);
  const d3 = cross(p1, p2, p3);
  const d4 = cross(p1, p2, p4);
  return (
    ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
    ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))
  );
}

/** Any pair of non-adjacent edges crossing => self-intersecting. O(n^2),
 *  fine for the maxPoints (<=8) polygons this editor produces. */
function isSelfIntersecting(pts: NormPoint[]): boolean {
  const n = pts.length;
  if (n < 4) return false;
  for (let i = 0; i < n; i++) {
    const a1 = pts[i];
    const a2 = pts[(i + 1) % n];
    for (let j = i + 1; j < n; j++) {
      // Skip edges that share a vertex (adjacent, incl. the wrap-around pair).
      if ((i + 1) % n === j || (j + 1) % n === i) continue;
      if (segmentsCross(a1, a2, pts[j], pts[(j + 1) % n])) return true;
    }
  }
  return false;
}

const SpatialEditor = ({
  item,
  value,
  contentRect,
  streaming,
  onDone,
  onCancel,
}: SpatialEditorProps) => {
  const { t } = useTranslation();
  const wrapRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [ownSize, setOwnSize] = useState({ width: 0, height: 0 });
  const isZone = item.type === "zone";
  const maxPoints = item.maxPoints ?? 8;

  // ---- editing state (component is remounted per item via key=) ----
  const [points, setPoints] = useState<NormPoint[]>(() =>
    isZone && Array.isArray(value)
      ? (value as NormPoint[]).map((p) => [p[0], p[1]] as NormPoint)
      : []
  );
  const initialLine =
    !isZone && value && typeof value === "object" && !Array.isArray(value)
      ? (value as ILineValue)
      : null;
  const [ptA, setPtA] = useState<NormPoint | null>(
    initialLine ? [...initialLine.a] : null
  );
  const [ptB, setPtB] = useState<NormPoint | null>(
    initialLine ? [...initialLine.b] : null
  );
  const [direction, setDirection] = useState<LineDirection>(
    initialLine?.direction === "ab_out" ? "ab_out" : "ab_in"
  );
  const dragIdxRef = useRef<number | null>(null); // zone: vertex idx; line: 0=a 1=b

  // Track our own box so blind-grid mode (no video frame) still has a rect.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const update = () =>
      setOwnSize({ width: el.clientWidth, height: el.clientHeight });
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Escape cancels the edit.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  // Drawing surface: the video content rect while streaming, otherwise the
  // whole player box (blind grid).
  const rect: ContentRect =
    streaming && contentRect.width > 0 && contentRect.height > 0
      ? contentRect
      : { left: 0, top: 0, width: ownSize.width, height: ownSize.height };
  const W = rect.width;
  const H = rect.height;

  /* ---------------- coordinate conversion ----------------
   * screen (pointer clientX/Y) -> normalized [0,1]:
   *   nx = (clientX - svgBox.left) / svgBox.width
   *   ny = (clientY - svgBox.top)  / svgBox.height
   * (the SVG element IS the content rect, so its bounding box is the
   *  normalization frame; values are clamped into 0..1)
   * normalized -> SVG pixel space: px = nx * W ; py = ny * H
   */
  const toNorm = (e: React.PointerEvent): NormPoint => {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box || !box.width || !box.height) return [0, 0];
    return [
      clamp01((e.clientX - box.left) / box.width),
      clamp01((e.clientY - box.top) / box.height),
    ];
  };
  const toPx = (p: NormPoint): [number, number] => [p[0] * W, p[1] * H];

  /** Index of the vertex within grab distance of the pointer, or -1. */
  const hitVertex = (norm: NormPoint, verts: (NormPoint | null)[]): number => {
    for (let i = 0; i < verts.length; i++) {
      const v = verts[i];
      if (!v) continue;
      const dx = (v[0] - norm[0]) * W;
      const dy = (v[1] - norm[1]) * H;
      if (Math.hypot(dx, dy) <= HIT_RADIUS_PX) return i;
    }
    return -1;
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (!W || !H) return;
    e.preventDefault();
    const norm = toNorm(e);
    const verts = isZone ? points : [ptA, ptB];
    const hit = hitVertex(norm, verts);
    if (hit >= 0) {
      dragIdxRef.current = hit;
    } else if (isZone) {
      if (points.length >= maxPoints) {
        message.warning(
          t("config.zoneHint", { max: maxPoints })
        );
        return;
      }
      setPoints((prev) => [...prev, norm]);
      dragIdxRef.current = points.length; // drag the fresh vertex right away
    } else if (!ptA) {
      setPtA(norm);
      dragIdxRef.current = 0;
    } else if (!ptB) {
      setPtB(norm);
      dragIdxRef.current = 1;
    } else {
      return; // both endpoints placed — only dragging is allowed
    }
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const idx = dragIdxRef.current;
    if (idx === null) return;
    const norm = toNorm(e);
    if (isZone) {
      setPoints((prev) => prev.map((p, i) => (i === idx ? norm : p)));
    } else if (idx === 0) {
      setPtA(norm);
    } else {
      setPtB(norm);
    }
  };

  const onPointerUp = () => {
    dragIdxRef.current = null;
  };

  const zoneComplete = () => {
    const pts = dedupeTail(points);
    if (pts.length < 3) {
      message.warning(t("config.zoneNeedPoints"));
      return;
    }
    // Reject degenerate polygons (too small / self-intersecting) so the count
    // can't silently fail on-device — same thresholds as the backend.
    const ring = dedupeRing(pts);
    if (
      ring.length < 3 ||
      polygonArea(ring) < ZONE_MIN_AREA ||
      isSelfIntersecting(ring)
    ) {
      message.error(t("config.zoneInvalid"));
      return;
    }
    onDone(pts);
  };

  const onDoubleClick = () => {
    if (isZone) zoneComplete();
  };

  const finish = () => {
    if (isZone) {
      zoneComplete();
      return;
    }
    if (!ptA || !ptB) return;
    // Reject a degenerate (too-short / zero-length) line — same threshold as
    // the backend, so crossings can actually be detected on-device.
    if (Math.hypot(ptB[0] - ptA[0], ptB[1] - ptA[1]) < LINE_MIN_LEN) {
      message.error(t("config.lineTooShort"));
      return;
    }
    const line: ILineValue = { a: ptA, b: ptB };
    if (item.directional) line.direction = direction;
    onDone(line);
  };

  const doneDisabled = isZone
    ? dedupeTail(points).length < 3
    : !ptA || !ptB;

  /* ---------------- line arrow geometry ----------------
   * Backend convention (yolo-detector geometry.h / person_tracker.cpp):
   * cross(p) = (bx-ax)(py-ay) - (by-ay)(px-ax); a crossing that moves from
   * the cross>0 side to the cross<0 side is "+1" and counts as IN when
   * direction == "ab_in". In PIXEL space the normal [vy, -vx] always points
   * into the cross<0 side (any aspect ratio), so the arrow marks where the
   * "in" traffic is heading.
   */
  const arrow = useMemo(() => {
    if (isZone || !ptA || !ptB || !W || !H) return null;
    const [ax, ay] = [ptA[0] * W, ptA[1] * H];
    const [bx, by] = [ptB[0] * W, ptB[1] * H];
    const mx = (ax + bx) / 2;
    const my = (ay + by) / 2;
    let vx = bx - ax;
    let vy = by - ay;
    const len = Math.hypot(vx, vy);
    if (len < 1) return null;
    vx /= len;
    vy /= len;
    let nx = vy; // points into the "in" destination side for ab_in
    let ny = -vx;
    if (direction === "ab_out") {
      nx = -nx;
      ny = -ny;
    }
    const L = Math.max(24, Math.min(40, len * 0.35));
    const tx = mx + nx * L;
    const ty = my + ny * L;
    return {
      mx,
      my,
      tx,
      ty,
      head1: [tx - nx * 10 + vx * 6, ty - ny * 10 + vy * 6],
      head2: [tx - nx * 10 - vx * 6, ty - ny * 10 - vy * 6],
      labelX: tx + nx * 12,
      labelY: ty + ny * 12,
    };
  }, [isZone, ptA, ptB, direction, W, H]);

  const gridLines = useMemo(() => {
    if (streaming || !W || !H) return null;
    const lines = [];
    for (let i = 1; i < 10; i++) {
      lines.push(
        <line
          key={`v${i}`}
          x1={(W * i) / 10}
          y1={0}
          x2={(W * i) / 10}
          y2={H}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth={1}
        />
      );
      lines.push(
        <line
          key={`h${i}`}
          x1={0}
          y1={(H * i) / 10}
          x2={W}
          y2={(H * i) / 10}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth={1}
        />
      );
    }
    return lines;
  }, [streaming, W, H]);

  const title = pickLocalizedText(item.title, item.title_zh) || item.key;
  const vertexDots = (verts: NormPoint[], labels?: string[]) =>
    verts.map((p, i) => {
      const [cx, cy] = toPx(p);
      return (
        <g key={i}>
          <circle
            cx={cx}
            cy={cy}
            r={6}
            fill="#ffffff"
            stroke={ACCENT}
            strokeWidth={2.5}
            style={{ cursor: "grab" }}
          />
          {labels?.[i] && (
            <text
              x={cx + 10}
              y={cy - 8}
              fill="#ffffff"
              fontSize={12}
              fontFamily="SFMono-Regular, Menlo, monospace"
              paintOrder="stroke"
              stroke="rgba(0,0,0,0.7)"
              strokeWidth={3}
            >
              {labels[i]}
            </text>
          )}
        </g>
      );
    });

  return (
    <div
      ref={wrapRef}
      className="absolute inset-0 z-10"
      style={{ background: streaming ? "rgba(0,0,0,0.25)" : "rgba(0,0,0,0.85)" }}
    >
      {/* drawing surface (spans the video content rect exactly) */}
      {W > 0 && H > 0 && (
        <svg
          ref={svgRef}
          className="absolute"
          style={{
            left: rect.left,
            top: rect.top,
            width: W,
            height: H,
            cursor: "crosshair",
            touchAction: "none",
          }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onDoubleClick={onDoubleClick}
        >
          {gridLines}
          {!streaming && (
            <rect
              x={0.5}
              y={0.5}
              width={W - 1}
              height={H - 1}
              fill="none"
              stroke="rgba(255,255,255,0.25)"
              strokeDasharray="6 4"
            />
          )}

          {isZone ? (
            <>
              {points.length >= 2 && (
                <polygon
                  points={points.map((p) => toPx(p).join(",")).join(" ")}
                  fill="rgba(143,195,31,0.18)"
                  stroke={ACCENT}
                  strokeWidth={2}
                  strokeLinejoin="round"
                />
              )}
              {vertexDots(points)}
            </>
          ) : (
            <>
              {ptA && ptB && (
                <line
                  x1={ptA[0] * W}
                  y1={ptA[1] * H}
                  x2={ptB[0] * W}
                  y2={ptB[1] * H}
                  stroke={ACCENT}
                  strokeWidth={3}
                />
              )}
              {arrow && (
                <g
                  stroke={ACCENT}
                  strokeWidth={2.5}
                  strokeLinecap="round"
                  fill="none"
                >
                  <line x1={arrow.mx} y1={arrow.my} x2={arrow.tx} y2={arrow.ty} />
                  <line
                    x1={arrow.head1[0]}
                    y1={arrow.head1[1]}
                    x2={arrow.tx}
                    y2={arrow.ty}
                  />
                  <line
                    x1={arrow.head2[0]}
                    y1={arrow.head2[1]}
                    x2={arrow.tx}
                    y2={arrow.ty}
                  />
                  <text
                    x={arrow.labelX}
                    y={arrow.labelY}
                    fill={ACCENT}
                    stroke="rgba(0,0,0,0.7)"
                    strokeWidth={3}
                    paintOrder="stroke"
                    fontSize={12}
                    fontFamily="SFMono-Regular, Menlo, monospace"
                    textAnchor="middle"
                  >
                    {t("config.inLabel")}
                  </text>
                </g>
              )}
              {vertexDots(
                [ptA, ptB].filter(Boolean) as NormPoint[],
                ["A", "B"]
              )}
            </>
          )}
        </svg>
      )}

      {/* editing toolbar (top of the player) */}
      <div className="absolute top-0 inset-x-0 flex items-center justify-between gap-8 flex-wrap px-12 py-8 bg-black bg-opacity-70">
        <span className="rc-mono text-12 text-white">
          {t("config.editing", { name: title })}
          {isZone && (
            <span className="opacity-60"> · {points.length}/{maxPoints}</span>
          )}
        </span>
        <div className="flex items-center gap-8">
          {isZone && (
            <Button
              size="small"
              disabled={!points.length}
              onClick={() => setPoints((prev) => prev.slice(0, -1))}
            >
              {t("config.undo")}
            </Button>
          )}
          {!isZone && item.directional && (
            <Button
              size="small"
              disabled={!ptA || !ptB}
              onClick={() =>
                setDirection((d) => (d === "ab_in" ? "ab_out" : "ab_in"))
              }
            >
              {t("config.flipDirection")}
            </Button>
          )}
          <Button size="small" onClick={() => onDone(null)}>
            {t("config.clear")}
          </Button>
          <Button size="small" onClick={onCancel}>
            {t("common.cancel")}
          </Button>
          <Button
            size="small"
            type="primary"
            disabled={doneDisabled}
            onClick={finish}
          >
            {t("config.done")}
          </Button>
        </div>
      </div>

      {/* hints (bottom of the player) */}
      <div className="absolute bottom-8 left-8 right-8 pointer-events-none">
        <div className="text-11 text-white opacity-75 bg-black bg-opacity-50 rounded-6 px-8 py-4 inline-block">
          {isZone
            ? t("config.zoneHint", { max: maxPoints })
            : t("config.lineHint")}
          {!streaming && <> — {t("config.gridHint")}</>}
        </div>
      </div>
    </div>
  );
};

export default SpatialEditor;
