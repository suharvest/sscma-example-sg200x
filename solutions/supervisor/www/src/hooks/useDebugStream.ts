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
}

export interface IResultMessage {
  /** local receive time (ms) */
  receivedAt: number;
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

/** Parse a results JSON message into an overlay frame (defensive). */
export function parseOverlayFrame(
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

  const boxes: IResultBox[] = [];
  for (const b of rawBoxes) {
    if (Array.isArray(b) && b.length >= 4) {
      boxes.push({
        x: Number(b[0]) || 0,
        y: Number(b[1]) || 0,
        w: Number(b[2]) || 0,
        h: Number(b[3]) || 0,
        score: Number(b[4]) || 0,
        target: b[5] as number | string,
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

  const videoWsRef = useRef<WebSocket | null>(null);
  const resultsWsRef = useRef<WebSocket | null>(null);
  const jmuxerRef = useRef<JMuxer | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cleanup = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
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
        return () => {
          disposed = true;
          cleanup();
        };
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
        setLastFrameDelay(ts > 0 && Math.abs(delay) < 3600_000 ? delay : null);
        // Strip the timestamp tail before feeding JMuxer.
        const video = buffer.slice(0, -8);
        jmuxerRef.current?.feed({ video });
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
          timeoutRef.current = null;
        }
        setStatus((prev) => (prev === "streaming" ? prev : "streaming"));
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
          setStatus((prev) => (prev === "error" ? prev : "error"));
        }
      };
    }

    // ---- results path ----
    if (resultsUrl) {
      try {
        const rws = new WebSocket(resultsUrl);
        resultsWsRef.current = rws;
        rws.onmessage = (evt: MessageEvent) => {
          if (disposed || typeof evt.data !== "string") return;
          let data: Record<string, unknown> | null = null;
          try {
            data = JSON.parse(evt.data);
          } catch (e) {
            data = null;
          }
          const msg: IResultMessage = {
            receivedAt: Date.now(),
            raw: evt.data,
            data,
          };
          setMessages((prev) => {
            const next = [msg, ...prev];
            return next.length > maxMessages
              ? next.slice(0, maxMessages)
              : next;
          });
          const frame = parseOverlayFrame(data);
          if (frame) {
            setOverlay(frame);
          }
          // Results-only apps (no video ws) still count as "streaming".
          if (!wsUrl) {
            setStatus("streaming");
          }
        };
        // Results channel failures are non-fatal when video works.
        rws.onerror = () => {
          if (!disposed && !wsUrl) {
            setStatus("error");
          }
        };
      } catch (e) {
        if (!wsUrl) {
          setStatus("error");
        }
      }
    }

    return () => {
      disposed = true;
      cleanup();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, wsUrl, resultsUrl]);

  return { status, messages, overlay, lastFrameDelay };
}
