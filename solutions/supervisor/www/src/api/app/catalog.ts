/**
 * Install-from-cloud catalog.
 *
 * The camera has no route to the internet in the common setup: over USB its
 * only interface is usb0 and the default route points back at itself. So the
 * device cannot fetch its own packages, and appMgr/installApp deliberately
 * only ever installs a .deb that is already under /userdata.
 *
 * The browser, however, does have internet — it is the machine the user is
 * sitting at. So it downloads on the device's behalf and pushes the bytes over
 * the existing chunked-upload API. Nothing on the device changes.
 *
 * The catalog itself is generated from the SenseCraft ecosystem solution
 * (scripts/generate_recamera_catalog.py in sensecraft-solutions) and published
 * next to the packages. It is never hand-written: package URLs and checksums
 * have exactly one source of truth, the solution's device YAMLs.
 */

const CATALOG_URL =
  "https://sensecraft-statics.seeed.cc/solution-app/recamera_ecosystem/catalog.json";

export interface ICatalogFile {
  url: string;
  filename: string;
  sha256?: string;
  size?: number | null;
  /** models only: absolute directory on the device */
  target_path?: string;
}

export interface ICatalogApp {
  /** Package name; equals the app's gallery manifest id. */
  id: string;
  preset: string;
  name: string;
  name_zh?: string;
  description?: string;
  description_zh?: string;
  package: ICatalogFile;
  models: ICatalogFile[];
}

export interface ICatalog {
  schema: number;
  source?: string;
  apps: ICatalogApp[];
}

/** Total bytes the browser will move for one app (package + models). */
export const appDownloadSize = (app: ICatalogApp): number =>
  (app.package.size || 0) +
  app.models.reduce((sum, m) => sum + (m.size || 0), 0);

/**
 * Fetch the catalog. Rejects on any failure — the caller shows the manual
 * upload path instead, which is the only thing that works on a machine with no
 * internet either.
 *
 * `no-store`: the catalog changes whenever a package is republished, and a
 * browser-cached copy would offer a version that no longer exists.
 */
export const fetchCatalog = async (signal?: AbortSignal): Promise<ICatalog> => {
  const res = await fetch(CATALOG_URL, { cache: "no-store", signal });
  if (!res.ok) {
    throw new Error(`catalog HTTP ${res.status}`);
  }
  const data = (await res.json()) as ICatalog;
  if (!data || !Array.isArray(data.apps)) {
    throw new Error("catalog is malformed");
  }
  return data;
};

/**
 * Download one catalog file into memory as a File, ready for uploadFiles().
 *
 * Progress comes from the streaming reader rather than Content-Length alone,
 * so a proxy that strips the header still shows movement — it just cannot show
 * a percentage, which the caller renders as indeterminate.
 */
export const downloadToFile = async (
  entry: ICatalogFile,
  onProgress?: (loaded: number, total: number | null) => void,
  signal?: AbortSignal
): Promise<File> => {
  const res = await fetch(entry.url, { cache: "no-store", signal });
  if (!res.ok) {
    throw new Error(`${entry.filename}: HTTP ${res.status}`);
  }

  const declared = Number(res.headers.get("Content-Length")) || entry.size || 0;
  const total = declared > 0 ? declared : null;

  if (!res.body) {
    // No streaming support: fall back to a single blob, no progress.
    const blob = await res.blob();
    onProgress?.(blob.size, blob.size);
    return new File([blob], entry.filename);
  }

  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loaded += value.length;
      onProgress?.(loaded, total);
    }
  }
  return new File([new Blob(chunks as BlobPart[])], entry.filename);
};

/**
 * SHA-256 of the downloaded bytes, hex.
 *
 * Verified before anything is pushed to the device: a truncated or
 * proxy-mangled download would otherwise be installed by opkg and fail in a
 * far less obvious place. Returns "" where WebCrypto is unavailable — the
 * console is served over plain HTTP, and crypto.subtle only exists in secure
 * contexts, which for http:// means localhost only. The caller treats an empty
 * digest as "cannot verify" rather than as a mismatch.
 */
export const sha256Hex = async (file: File): Promise<string> => {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return "";
  const digest = await subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
};
