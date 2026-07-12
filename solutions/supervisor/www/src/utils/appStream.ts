import { baseIP } from "@/utils/supervisorRequest";
import { IAppManifest, IDebugWsInfo } from "@/api/app/app";

/** Device hostname the browser should use to reach camera services. */
export function getDeviceHost(): string {
  try {
    return new URL(baseIP).hostname;
  } catch (e) {
    return window.location.hostname;
  }
}

/** Fill the `{host}` placeholder in manifest rtsp_url. */
export function resolveRtspUrl(app?: IAppManifest | null): string {
  if (!app?.rtsp_url) return "";
  return app.rtsp_url.replace(/\{host\}/g, getDeviceHost());
}

function normalizePath(path: string | undefined, fallback: string): string {
  const p = path || fallback;
  return p.startsWith("/") ? p : `/${p}`;
}

/** ws:// URL for the debug H.264 video stream, null when unsupported. */
export function resolveDebugVideoUrl(app?: IAppManifest | null): string | null {
  const dbg: IDebugWsInfo | undefined = app?.debug_ws;
  if (!dbg?.port) return null;
  return `ws://${getDeviceHost()}:${dbg.port}${normalizePath(
    dbg.video_path,
    "/"
  )}`;
}

/** ws:// URL for the inference results channel, null when unsupported. */
export function resolveDebugResultsUrl(
  app?: IAppManifest | null
): string | null {
  const dbg: IDebugWsInfo | undefined = app?.debug_ws;
  if (!dbg?.port) return null;
  return `ws://${getDeviceHost()}:${dbg.port}${normalizePath(
    dbg.results_path,
    "/results"
  )}`;
}
