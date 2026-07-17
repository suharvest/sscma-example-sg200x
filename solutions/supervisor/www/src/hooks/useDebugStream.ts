import { useCallback, useEffect, useRef, useState } from "react";
import JMuxer from "jmuxer";

/**
 * Debug stream hook: H.264-over-WebSocket video (fed into JMuxer / MSE)
 * plus a second WebSocket carrying inference result JSON for overlay.
 *
 * Frame format: raw H.264 access units with an 8-byte little-endian uint64
 * millisecond timestamp appended at the tail. The tail MUST be stripped
 * before feeding JMuxer (the legacy overview hook fed it through, which
 * was a bug — do not replicate).
 *
 * High-frequency data (per-frame delay, per-message results) is buffered in
 * refs and flushed to React state on timers — a 30fps setState would
 * re-render the whole page on every video frame.
 */

export type DebugStreamStatus =
  | "idle" // switch off
  | "connecting"
  | "streaming"
  | "error"; // failed to connect / dropped

export interface IResultBox {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  target: number | string;
  /** Rich display label from the parallel `labels[]` array (e.g. an app that
   *  annotates a face box with "male 20-29 neutral"). Falls back to target. */
  label?: string;
}

export interface IResultMessage {
  /** local receive time, preformatted for display (toLocaleTimeString) */
  receivedAt: string;
  /** raw JSON text (shown in the message list) */
  raw: string;
  /** parsed payload, null when not valid JSON */
  data: Record<string, unknown> | null;
}

export interface IOverlayFrame {
  boxes: IResultBox[];
  /** inference resolution the coordinates are relative to */
  resW: number;
  resH: number;
}

interface UseDebugStreamOptions {
  enabled: boolean;
  /** ws://host:port/ video path — null disables video */
  wsUrl: string | null;
  /** ws://host:port/results — null disables results */
  resultsUrl: string | null;
  /** target video element */
  videoRef: React.RefObject<HTMLVideoElement>;
  /** connect timeout, ms */
  connectTimeout?: number;
  maxMessages?: number;
}

const DEFAULT_RESOLUTION = 640;
/** results/overlay ref-buffer flush cadence */
const RESULTS_FLUSH_MS = 150;
/** frame delay/timestamp flush cadence */
const DELAY_FLUSH_MS = 1000;
/** Live-edge keeper: JMuxer/MediaSource accumulates a decode buffer so the
 *  video playhead drifts behind real time; the (now fresh) overlay then leads
 *  the video. Periodically snap the video back to the live edge when it lags. */
const LIVE_EDGE_CHECK_MS = 500;
const LIVE_EDGE_MAX_LAG = 0.6; // seconds behind buffered end before snapping
const LIVE_EDGE_TARGET = 0.15; // seconds from the edge to land on after a snap

/** Parse a results JSON message into an overlay frame (defensive). */
function parseOverlayFrame(
  data: Record<string, unknown> | null
): IOverlayFrame | null {
  if (!data) return null;
  const rawBoxes = data.boxes;
  if (!Array.isArray(rawBoxes)) return null;

  let resW = DEFAULT_RESOLUTION;
  let resH = DEFAULT_RESOLUTION;
  const resolution = data.resolution as unknown;
  if (Array.isArray(resolution) && resolution.length >= 2) {
    const w = Number(resolution[0]);
    const h = Number(resolution[1]);
    if (w > 0 && h > 0) {
      resW = w;
      resH = h;
    }
  } else if (typeof resolution === "number" && resolution > 0) {
    resW = resolution;
    resH = resolution;
  } else if (resolution && typeof resolution === "object") {
    const r = resolution as { width?: number; height?: number };
    if (r.width && r.height) {
      resW = r.width;
      resH = r.height;
    }
  }

  // Optional parallel labels[] array: labels[i] is a rich, already-formatted
  // display string for boxes[i] (e.g. face-analysis emits "male 20-29 neutral"
  // while the box's own target is just "face").
  const rawLabels = Array.isArray(data.labels)
    ? (data.labels as unknown[])
    : null;
  const labelAt = (i: number): string | undefined => {
    const v = rawLabels ? rawLabels[i] : undefined;
    return typeof v === "string" && v ? v : undefined;
  };

  const boxes: IResultBox[] = [];
  for (let i = 0; i < rawBoxes.length; i++) {
    const b = rawBoxes[i];
    if (Array.isArray(b) && b.length >= 4) {
      boxes.push({
        x: Number(b[0]) || 0,
        y: Number(b[1]) || 0,
        w: Number(b[2]) || 0,
        h: Number(b[3]) || 0,
        score: Number(b[4]) || 0,
        target: b[5] as number | string,
        label: labelAt(i),
      });
    } else if (b && typeof b === "object") {
      const o = b as Record<string, number>;
      boxes.push({
        x: Number(o.x) || 0,
        y: Number(o.y) || 0,
        w: Number(o.w) || 0,
        h: Number(o.h) || 0,
        score: Number(o.score) || 0,
        target: o.target,
        label: labelAt(i),
      });
    }
  }
  return { boxes, resW, resH };
}

export default function useDebugStream({
  enabled,
  wsUrl,
  resultsUrl,
  videoRef,
  connectTimeout = 8000,
  maxMessages = 100,
}: UseDebugStreamOptions) {
  const [status, setStatus] = useState<DebugStreamStatus>("idle");
  const [messages, setMessages] = useState<IResultMessage[]>([]);
  const [overlay, setOverlay] = useState<IOverlayFrame | null>(null);
  const [lastFrameDelay, setLastFrameDelay] = useState<number | null>(null);
  const [lastFrameTs, setLastFrameTs] = useState<number | null>(null);

  const videoWsRef = useRef<WebSocket | null>(null);
  const resultsWsRef = useRef<WebSocket | null>(null);
  const jmuxerRef = useRef<JMuxer | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Independent connect timeout for the results-only WS (video owns timeoutRef).
  const resultsTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // High-frequency buffers, flushed to state on timers (see effect below).
  const frameDelayRef = useRef<number | null>(null);
  const frameTsRef = useRef<number | null>(null);
  const pendingMsgsRef = useRef<IResultMessage[]>([]);
  const pendingOverlayRef = useRef<IOverlayFrame | null>(null);

  const cleanup = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (resultsTimeoutRef.current) {
      clearTimeout(resultsTimeoutRef.current);
      resultsTimeoutRef.current = null;
    }
    for (const ref of [videoWsRef, resultsWsRef]) {
      const ws = ref.current;
      if (ws) {
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        try {
          ws.close();
        } catch (e) {
          /* noop */
        }
        ref.current = null;
      }
    }
    // Destroy JMuxer completely; a reconnect must rebuild it.
    if (jmuxerRef.current) {
      try {
        jmuxerRef.current.destroy();
      } catch (e) {
        /* noop */
      }
      jmuxerRef.current = null;
    }
    const video = videoRef.current;
    if (video) {
      try {
        video.pause();
        video.srcObject = null;
        video.removeAttribute("src");
        video.load();
      } catch (e) {
        /* noop */
      }
    }
  }, [videoRef]);

  useEffect(() => {
    if (!enabled) {
      cleanup();
      setStatus("idle");
      setOverlay(null);
      setLastFrameDelay(null);
      setLastFrameTs(null);
      return;
    }
    if (!wsUrl && !resultsUrl) {
      setStatus("error");
      return;
    }

    let disposed = false;
    setStatus("connecting");
    setMessages([]);
    setOverlay(null);
    frameDelayRef.current = null;
    frameTsRef.current = null;
    pendingMsgsRef.current = [];
    pendingOverlayRef.current = null;

    const delayTimer = setInterval(() => {
      setLastFrameDelay(frameDelayRef.current);
      setLastFrameTs(frameTsRef.current);
    }, DELAY_FLUSH_MS);
    const flushTimer = setInterval(() => {
      const pending = pendingMsgsRef.current;
      if (pending.length) {
        pendingMsgsRef.current = [];
        // Newest first, like the arrival order the list renders.
        setMessages((prev) =>
          [...pending.reverse(), ...prev].slice(0, maxMessages)
        );
        // Results-only apps (no video ws) still count as "streaming".
        if (!wsUrl) {
          setStatus("streaming");
        }
      }
      if (pendingOverlayRef.current) {
        setOverlay(pendingOverlayRef.current);
        pendingOverlayRef.current = null;
      }
    }, RESULTS_FLUSH_MS);
    // Keep the MSE video near the live edge so it doesn't trail the overlay.
    const liveEdgeTimer = setInterval(() => {
      const v = videoRef.current;
      if (!v || v.seeking || !v.buffered || v.buffered.length === 0) return;
      const end = v.buffered.end(v.buffered.length - 1);
      if (end - v.currentTime > LIVE_EDGE_MAX_LAG) {
        try {
          v.currentTime = end - LIVE_EDGE_TARGET;
        } catch (e) {
          /* transient: buffer being updated */
        }
      }
    }, LIVE_EDGE_CHECK_MS);

    const dispose = () => {
      disposed = true;
      clearInterval(delayTimer);
      clearInterval(flushTimer);
      clearInterval(liveEdgeTimer);
      cleanup();
    };

    // ---- video path ----
    if (wsUrl && videoRef.current) {
      jmuxerRef.current = new JMuxer({
        node: videoRef.current,
        mode: "video",
        flushingTime: 0,
        fps: 30,
        clearBuffer: true,
        debug: false,
      });

      let ws: WebSocket;
      try {
        ws = new WebSocket(wsUrl);
      } catch (e) {
        setStatus("error");
        return dispose;
      }
      ws.binaryType = "arraybuffer";
      videoWsRef.current = ws;

      // No infinite loading: fail hard if nothing arrives in time.
      timeoutRef.current = setTimeout(() => {
        if (!disposed) {
          cleanup();
          setStatus("error");
        }
      }, connectTimeout);

      ws.onmessage = (evt: MessageEvent) => {
        if (disposed || !(evt.data instanceof ArrayBuffer)) return;
        const buffer = new Uint8Array(evt.data);
        if (buffer.length <= 8) return;
        // Trailing 8-byte little-endian uint64 ms timestamp.
        const view = new DataView(evt.data, buffer.length - 8, 8);
        const low = view.getUint32(0, true);
        const high = view.getUint32(4, true);
        const ts = Number((BigInt(high) << BigInt(32)) | BigInt(low));
        // The timestamp is the device's wall clock. If the device clock is
        // not synced (common on offline field units), the difference is
        // meaningless — only report the delay when it looks sane (<1h skew).
        const delay = Date.now() - ts;
        frameTsRef.current = ts > 0 ? ts : null;
        frameDelayRef.current =
          ts > 0 && Math.abs(delay) < 3600_000 ? delay : null;
        // Strip the timestamp tail before feeding JMuxer (zero-copy view).
        const video = buffer.subarray(0, buffer.length - 8);
        jmuxerRef.current?.feed({ video });
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
          timeoutRef.current = null;
        }
        setStatus("streaming");
      };
      ws.onerror = () => {
        if (!disposed) {
          cleanup();
          setStatus("error");
        }
      };
      ws.onclose = () => {
        if (!disposed) {
          cleanup();
          setStatus("error");
        }
      };
    }

    // ---- results path ----
    if (resultsUrl) {
      try {
        const rws = new WebSocket(resultsUrl);
        resultsWsRef.current = rws;
        // Results-only mode (no video ws): give the results channel its own
        // connect timeout so it can't hang in "connecting" forever. When a
        // video ws is present it owns the connection status, so skip this.
        if (!wsUrl) {
          resultsTimeoutRef.current = setTimeout(() => {
            if (!disposed) {
              cleanup();
              setStatus("error");
            }
          }, connectTimeout);
        }
        rws.onopen = () => {
          if (disposed) return;
          if (resultsTimeoutRef.current) {
            clearTimeout(resultsTimeoutRef.current);
            resultsTimeoutRef.current = null;
          }
          // Results-only: an open socket is enough to be "streaming" — don't
          // wait for the first business message. With video present, video
          // drives the status (leave it alone).
          if (!wsUrl) {
            setStatus("streaming");
          }
        };
        rws.onmessage = (evt: MessageEvent) => {
          if (disposed || typeof evt.data !== "string") return;
          let data: Record<string, unknown> | null = null;
          try {
            data = JSON.parse(evt.data);
          } catch (e) {
            data = null;
          }
          pendingMsgsRef.current.push({
            receivedAt: new Date().toLocaleTimeString(),
            raw: evt.data,
            data,
          });
          if (pendingMsgsRef.current.length > maxMessages) {
            pendingMsgsRef.current.shift(); // cap the buffer between flushes
          }
          const frame = parseOverlayFrame(data);
          if (frame) {
            pendingOverlayRef.current = frame;
          }
        };
        // Results channel failures are non-fatal when video works.
        rws.onerror = () => {
          if (!disposed && !wsUrl) {
            cleanup();
            setStatus("error");
          }
        };
        // Drop after connecting: tear down and (results-only) surface the
        // error, unless we're disposing intentionally.
        rws.onclose = () => {
          if (!disposed && !wsUrl) {
            cleanup();
            setStatus("error");
          }
        };
      } catch (e) {
        if (!wsUrl) {
          setStatus("error");
        }
      }
    }

    return dispose;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, wsUrl, resultsUrl]);

  return { status, messages, overlay, lastFrameDelay, lastFrameTs };
}
