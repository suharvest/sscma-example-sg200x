/**
 * appMgr privacy blur API types.
 *
 * The camera applications mask detected subjects in the video before it is
 * encoded, so the masking is present in the recording and in every stream —
 * it is not a console-side overlay that a viewer could switch off.
 */

export interface IBlurConfig {
  /** Master switch for masking detected subjects. */
  enabled: boolean;
  /**
   * Masking implementation:
   *  - `pixelate` : block colours sampled from the picture, composited in
   *                 hardware, effectively free at runtime
   *  - `coverex`  : solid colour blocks, at most 4 regions
   *  - `mosaic`   : the stock vendor effect, which renders as black/white noise
   */
  backend: "mosaic" | "coverex" | "pixelate";
  /** Mosaic cell size in pixels; the hardware grid supports only 8 or 16. */
  block_px: number;
  /** How many subjects may be masked at once, 1-8. */
  max_regions: number;
  /**
   * Opacity of the mask, 0-255. 255 hides the subject completely; anything
   * lower lets the original picture through, and whatever shows through is
   * recognisable again — a half-transparent mask over a face can leave that
   * face identifiable, at which point it is no longer a privacy measure.
   */
  alpha: number;
}

/** getBlurConfig -> data */
export type IGetBlurConfigResult = IBlurConfig;

/** setBlurConfig -> data */
export interface ISetBlurConfigResult {
  /** False in Node-RED mode (note: "nodered_mode") — config stored, not applied. */
  restarted: boolean;
  note?: string;
}

/** Any subset of the config fields; omitted fields keep their stored value. */
export interface ISetBlurConfigParams {
  enabled?: boolean;
  backend?: IBlurConfig["backend"];
  block_px?: number;
  max_regions?: number;
  /** 0-255. Out-of-range values are rejected by the backend, not clamped. */
  alpha?: number;
}
