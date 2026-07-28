/* eslint-disable */
// @ts-nocheck
//
// VENDORED — DO NOT EDIT HERE. Single source of truth:
//   app_collaboration/frontend/src/overlay/renderer.js
//
// Manual sync procedure (renderer is a self-contained, zero-dep ESM):
//   1. cd app_collaboration/frontend && node src/overlay/build-overlay.mjs
//   2. cp src/overlay/renderer.js \
//        <this-repo>/solutions/supervisor/www/src/vendor/recamera-overlay.js
//   3. re-add this header block on top.
// Imported by www/src/views/live/index.tsx via `@/vendor/recamera-overlay`.
// See renderer README/overlay-contract.md for the full model contract.

/**
 * RecameraOverlay — unified overlay renderer for the reCamera ecosystem.
 *
 * SINGLE SOURCE OF TRUTH. This module is consumed in three places:
 *   1. app_collaboration preview (Tauri) — imported by modules/preview.js and
 *      registered on `window.RecameraOverlay` so draw_*.js adapter scripts
 *      (run via `new Function('ctx','data','canvas','img', script)`) can call it.
 *   2. sensecraft-solutions draw_*.js — each script maps its own MQTT payload to
 *      a `model` and calls `window.RecameraOverlay.render(...)`.
 *   3. supervisor (device Console) www — a copy of the built artifact lives at
 *      www/src/vendor/recamera-overlay.js (see build-overlay.mjs + that file's
 *      header for the manual sync procedure).
 *
 * The renderer knows NOTHING about any specific app. It only understands the
 * `model` — a flat list of primitive layers. All app-specific knowledge
 * (which MQTT field means what) lives in the adapters, never here.
 *
 * Public API:
 *   RecameraOverlay.render(ctx, model, { width, height, theme, state })
 *
 * `width`/`height` are the pixel dimensions of the video CONTENT area
 * (letterbox already resolved by the host). All normalized coords in the model
 * are multiplied by these. Callers that size their canvas to the content area
 * simply pass canvas.width / canvas.height.
 *
 * See README/overlay-contract.md for the full model contract.
 *
 * Zero dependencies. Pure Canvas2D.
 */

// ============================================================
// Theme / palette
// ============================================================

const BRAND = "#8fc31f"; // reCamera theme green — the single accent color.

// Per-category box colors. `boxes` items may override with an explicit color;
// otherwise the label (or index) is hashed to one of these for stable coloring.
const CATEGORY_COLORS = [
  "#8fc31f", // brand green
  "#4A90D9", // blue
  "#F5A623", // amber
  "#E0518A", // pink
  "#7DD3FC", // sky
  "#B07CE8", // purple
  "#38BDF8", // cyan
  "#F97316", // orange
];

// status card tone → header color
const TONE_COLORS = {
  ok: "#22C55E",
  warn: "#EAB308",
  alert: "#EF4444",
};

const CARD_BG = "rgba(0, 0, 0, 0.65)";
const CARD_RADIUS = 10;
const PILL_BG = "rgba(0, 0, 0, 0.72)";
const TEXT = "#FFFFFF";

function hashColor(key, i) {
  if (typeof key === "string" && key.length) {
    let h = 0;
    for (let k = 0; k < key.length; k++) h = (h * 31 + key.charCodeAt(k)) | 0;
    return CATEGORY_COLORS[Math.abs(h) % CATEGORY_COLORS.length];
  }
  return CATEGORY_COLORS[(i || 0) % CATEGORY_COLORS.length];
}

// ============================================================
// Canvas helpers
// ============================================================

function ensureRoundRect(ctx) {
  if (typeof ctx.roundRect === "function") return;
  ctx.roundRect = function (x, y, w, h, radii) {
    const r = typeof radii === "number" ? radii : Array.isArray(radii) ? radii[0] : 0;
    const rr = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
    this.moveTo(x + rr, y);
    this.arcTo(x + w, y, x + w, y + h, rr);
    this.arcTo(x + w, y + h, x, y + h, rr);
    this.arcTo(x, y + h, x, y, rr);
    this.arcTo(x, y, x + w, y, rr);
    this.closePath();
  };
}

function fillRoundRect(ctx, x, y, w, h, r, fill) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  ctx.fillStyle = fill;
  ctx.fill();
}

/** Deep-black pill with white text — the unified label chrome. Returns width. */
function drawPill(ctx, x, y, text, opts = {}) {
  const font = opts.font || "600 12px Inter, system-ui, sans-serif";
  const padX = 6;
  const padY = 3;
  ctx.font = font;
  const metrics = ctx.measureText(text);
  const h = (opts.size || 12) + padY * 2 + 2;
  const w = metrics.width + padX * 2;
  // Anchor above (default) or below the reference y.
  const py = opts.below ? y : y - h;
  fillRoundRect(ctx, x, py, w, h, 4, opts.bg || PILL_BG);
  ctx.fillStyle = opts.color || TEXT;
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + padX, py + h / 2 + 0.5);
  ctx.textBaseline = "alphabetic";
  return w;
}

// ============================================================
// Primitive layer handlers
// ============================================================

function drawBoxes(ctx, layer, W, H) {
  const items = layer.items || [];
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    const bb = it.bbox;
    if (!bb) continue;
    // Normalized top-left origin (contract). Multiply by content area.
    const x = bb.x * W;
    const y = bb.y * H;
    const w = bb.w * W;
    const h = bb.h * H;
    const color = it.color || hashColor(it.label != null ? it.label : it.trackId, i);

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 6);
    ctx.stroke();
    ctx.restore();

    // Label pill (label + optional confidence + optional trackId).
    let text = "";
    if (it.label != null) text += String(it.label);
    if (it.confidence != null) {
      const pct = it.confidence <= 1 ? it.confidence * 100 : it.confidence;
      text += (text ? " " : "") + pct.toFixed(0) + "%";
    }
    if (it.trackId != null) text = `#${it.trackId} ${text}`.trim();
    if (text) {
      const py = y > 22 ? y - 2 : y + h + 20; // flip below box if near top
      drawPill(ctx, x, py, text, { color: TEXT, bg: PILL_BG });
    }
  }
}

function drawKeypoints(ctx, layer, W, H) {
  const items = layer.items || [];
  for (let g = 0; g < items.length; g++) {
    const grp = items[g];
    const pts = grp.points || [];
    const color = grp.color || BRAND;
    // Weight scales with density, like the dot radius below. A 468-point
    // facemesh wants a faint hairline or it turns into a solid blob; a
    // 17-joint body skeleton drawn that way is nearly invisible over live
    // video, which is what a single shared default produced before.
    // `lineWidth` / `alpha` on the group override the heuristic.
    const dense = pts.length > 100;
    const mid = !dense && pts.length > 30;
    const lineWidth = grp.lineWidth != null ? grp.lineWidth : dense ? 1 : mid ? 1.5 : 3;
    const alpha = grp.alpha != null ? grp.alpha : dense ? 0.5 : mid ? 0.7 : 0.95;

    // Edges first (skeleton / mesh wireframe) so dots sit on top.
    if (Array.isArray(grp.edges) && grp.edges.length) {
      const trace = () => {
        ctx.beginPath();
        for (const e of grp.edges) {
          const a = pts[e[0]];
          const b = pts[e[1]];
          if (!a || !b) continue;
          ctx.moveTo(a[0] * W, a[1] * H);
          ctx.lineTo(b[0] * W, b[1] * H);
        }
        ctx.stroke();
      };
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      // Sparse skeletons get a dark halo underneath: the accent green vanishes
      // against a pale wall or bare skin, and the overlay has no control over
      // what the camera is pointed at. Skipped for dense meshes, where the
      // halo would fill the gaps between strokes.
      if (!dense) {
        ctx.strokeStyle = "rgba(0,0,0,0.45)";
        ctx.globalAlpha = alpha;
        ctx.lineWidth = lineWidth + 2;
        trace();
      }
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = lineWidth;
      trace();
      ctx.restore();
    }
    // Dot radius shrinks for dense meshes (e.g. 468-pt facemesh).
    const r = dense ? 0.9 : mid ? 1.4 : 3.5;
    ctx.save();
    ctx.fillStyle = color;
    ctx.globalAlpha = dense ? 0.75 : 1;
    // Same reasoning as the edge halo: outline the joints of a sparse skeleton
    // so they stay readable on a bright background.
    if (!dense) {
      ctx.strokeStyle = "rgba(0,0,0,0.45)";
      ctx.lineWidth = 1.5;
    }
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p[0] * W, p[1] * H, r, 0, Math.PI * 2);
      if (!dense) ctx.stroke();
      ctx.fill();
    }
    ctx.restore();
  }
}

function drawPolygons(ctx, layer, W, H) {
  const items = layer.items || [];
  for (const poly of items) {
    const pts = poly.points || [];
    if (pts.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(pts[0][0] * W, pts[0][1] * H);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0] * W, pts[i][1] * H);
    ctx.closePath();
    if (poly.fill) {
      ctx.fillStyle = poly.fill;
      ctx.fill();
    }
    ctx.strokeStyle = poly.stroke || BRAND;
    ctx.lineWidth = 2;
    ctx.stroke();
    if (poly.label) {
      drawPill(ctx, pts[0][0] * W, pts[0][1] * H - 2, String(poly.label));
    }
  }
}

function drawPath(ctx, layer, W, H) {
  const items = layer.items || [];
  for (const p of items) {
    const pts = p.points || [];
    if (pts.length < 2) continue;
    ctx.save();
    ctx.strokeStyle = p.color || BRAND;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(pts[0][0] * W, pts[0][1] * H);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0] * W, pts[i][1] * H);
    ctx.stroke();
    ctx.restore();
  }
}

function drawHeatmap(ctx, layer, W, H) {
  const points = layer.points || [];
  if (!points.length) return;
  ctx.save();
  for (const p of points) {
    const x = p[0] * W;
    const y = p[1] * H;
    const weight = p[2] != null ? p[2] : 1;
    const radius = Math.max(12, 40 * weight);
    const grad = ctx.createRadialGradient(x, y, 0, x, y, radius);
    grad.addColorStop(0, `rgba(143, 195, 31, ${Math.min(0.55, 0.25 + weight * 0.4)})`);
    grad.addColorStop(1, "rgba(143, 195, 31, 0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawText(ctx, layer, W, H) {
  const items = layer.items || [];
  ctx.save();
  ctx.font = "600 13px Inter, system-ui, sans-serif";
  for (const it of items) {
    ctx.fillStyle = it.color || TEXT;
    ctx.fillText(String(it.text), it.x * W, it.y * H);
  }
  ctx.restore();
}

// ---- Cards -------------------------------------------------

function cardOrigin(anchor, W, cardW) {
  const margin = 12;
  if (anchor === "tr") return { x: W - cardW - margin, y: margin };
  return { x: margin, y: margin }; // tl default
}

function drawClassificationCard(ctx, data, W) {
  const cardW = Math.min(240, W - 24);
  const { x, y } = cardOrigin(data.anchor, W, cardW);
  const scores = Object.entries(data.data.scores || {})
    .map(([k, v]) => ({ k, v: Number(v) }))
    .filter((s) => Number.isFinite(s.v))
    .sort((a, b) => b.v - a.v)
    .slice(0, 5);
  const rows = scores.length;
  const cardH = 54 + rows * 16 + 10;

  fillRoundRect(ctx, x, y, cardW, cardH, CARD_RADIUS, CARD_BG);

  // Big label + confidence
  ctx.font = "600 20px Inter, system-ui, sans-serif";
  ctx.fillStyle = BRAND;
  const label = String(data.data.label ?? "—");
  ctx.fillText(label, x + 14, y + 30);
  if (data.data.confidence != null) {
    const pct = data.data.confidence <= 1 ? data.data.confidence * 100 : data.data.confidence;
    ctx.font = "500 13px Inter, system-ui, sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.font = "600 20px Inter, system-ui, sans-serif";
    const bigW = ctx.measureText(label).width;
    ctx.font = "500 13px Inter, system-ui, sans-serif";
    ctx.fillText(`${pct.toFixed(0)}%`, x + 14 + bigW + 8, y + 30);
  }

  // Score bars
  let by = y + 50;
  const barX = x + 14;
  const labelW = 70;
  const barMax = cardW - 28 - labelW - 34;
  ctx.font = "11px Inter, system-ui, sans-serif";
  for (const s of scores) {
    ctx.fillStyle = "rgba(255,255,255,0.8)";
    ctx.fillText(s.k.length > 10 ? s.k.slice(0, 10) : s.k, barX, by + 8);
    const trackX = barX + labelW;
    fillRoundRect(ctx, trackX, by + 2, barMax, 4, 2, "rgba(255,255,255,0.2)");
    const isTop = s.k === data.data.label;
    fillRoundRect(ctx, trackX, by + 2, Math.max(2, s.v * barMax), 4, 2, isTop ? BRAND : "rgba(255,255,255,0.45)");
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.fillText(`${(s.v * 100).toFixed(0)}%`, trackX + barMax + 6, by + 8);
    by += 16;
  }
}

function drawStatusCard(ctx, data, W) {
  const d = data.data;
  const cardW = Math.min(230, W - 24);
  const metrics = d.metrics || [];
  const bannerH = d.banner ? 24 : 0;
  const cardH = 40 + metrics.length * 18 + bannerH + 10;
  const { x, y } = cardOrigin(data.anchor, W, cardW);

  fillRoundRect(ctx, x, y, cardW, cardH, CARD_RADIUS, CARD_BG);

  // Title in tone color
  ctx.font = "600 18px Inter, system-ui, sans-serif";
  ctx.fillStyle = TONE_COLORS[d.tone] || BRAND;
  ctx.fillText(String(d.title ?? "—"), x + 14, y + 28);

  // Metric rows: key left (muted), value right (white)
  ctx.font = "12px Inter, system-ui, sans-serif";
  let my = y + 46;
  for (const m of metrics) {
    ctx.fillStyle = "rgba(255,255,255,0.65)";
    ctx.fillText(String(m.k), x + 14, my + 8);
    ctx.fillStyle = TEXT;
    const vw = ctx.measureText(String(m.v)).width;
    ctx.fillText(String(m.v), x + cardW - 14 - vw, my + 8);
    my += 18;
  }

  // Alert banner (red bar)
  if (d.banner) {
    fillRoundRect(ctx, x + 10, my + 2, cardW - 20, 20, 5, "rgba(239,68,68,0.9)");
    ctx.fillStyle = TEXT;
    ctx.font = "600 12px Inter, system-ui, sans-serif";
    ctx.fillText(String(d.banner), x + 18, my + 16);
  }
}

function drawMetricsCard(ctx, data, W) {
  const d = data.data;
  const cardW = Math.min(210, W - 24);
  const metrics = d.metrics || [];
  const hasTitle = d.title != null;
  const cardH = (hasTitle ? 34 : 12) + metrics.length * 18 + 8;
  const { x, y } = cardOrigin(data.anchor || "tr", W, cardW);

  fillRoundRect(ctx, x, y, cardW, cardH, CARD_RADIUS, CARD_BG);
  let my = y + 14;
  if (hasTitle) {
    ctx.font = "600 13px Inter, system-ui, sans-serif";
    ctx.fillStyle = TEXT;
    ctx.fillText(String(d.title), x + 12, my + 8);
    my += 26;
  }
  ctx.font = "12px Inter, system-ui, sans-serif";
  for (const m of metrics) {
    ctx.fillStyle = "rgba(255,255,255,0.65)";
    ctx.fillText(String(m.k), x + 12, my + 8);
    ctx.fillStyle = TEXT;
    const vw = ctx.measureText(String(m.v)).width;
    ctx.fillText(String(m.v), x + cardW - 12 - vw, my + 8);
    my += 18;
  }
}

function drawCard(ctx, layer, W) {
  if (!layer.data) return;
  // layer.data is {label,...}; anchor lives on the layer, expose to helpers.
  const payload = { data: layer.data, anchor: layer.anchor };
  if (layer.variant === "classification") return drawClassificationCard(ctx, payload, W);
  if (layer.variant === "status") return drawStatusCard(ctx, payload, W);
  if (layer.variant === "metrics") return drawMetricsCard(ctx, payload, W);
}

// ============================================================
// Dispatch
// ============================================================

const HANDLERS = {
  boxes: drawBoxes,
  keypoints: drawKeypoints,
  polygons: drawPolygons,
  path: drawPath,
  heatmap: drawHeatmap,
  text: drawText,
  // card & raster handled specially (need W only / reserved)
};

/**
 * Render a model onto a 2D context.
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} model  { layers: [...] }
 * @param {object} opts   { width, height, theme, state }
 */
export function render(ctx, model, opts = {}) {
  if (!ctx || !model || !Array.isArray(model.layers)) return;
  const W = opts.width || (ctx.canvas ? ctx.canvas.width : 0);
  const H = opts.height || (ctx.canvas ? ctx.canvas.height : 0);
  if (!W || !H) return;
  ensureRoundRect(ctx);
  ctx.textBaseline = "alphabetic";

  for (const layer of model.layers) {
    if (!layer || !layer.type) continue;
    try {
      if (layer.type === "card") {
        drawCard(ctx, layer, W);
      } else if (layer.type === "raster") {
        // Reserved extension point — see overlay-contract.md. Not yet drawn.
        continue;
      } else {
        const fn = HANDLERS[layer.type];
        if (fn) fn(ctx, layer, W, H);
      }
    } catch (e) {
      // A malformed layer must never break the rest of the overlay.
      if (typeof console !== "undefined") console.warn("[RecameraOverlay] layer failed:", layer.type, e);
    }
  }
}

export const RecameraOverlay = { render, version: "1.0.0", BRAND, CATEGORY_COLORS };

export default RecameraOverlay;

// Auto-register on window so draw_*.js adapter scripts (executed via
// `new Function('ctx','data','canvas','img', script)`) can reach it without
// an import. Idempotent — a pre-existing global wins so the host controls it.
if (typeof window !== "undefined") {
  window.RecameraOverlay = window.RecameraOverlay || RecameraOverlay;
}
