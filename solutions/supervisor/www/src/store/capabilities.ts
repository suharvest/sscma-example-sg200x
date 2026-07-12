import { create } from "zustand";
import { ICapabilities } from "@/api/device/device";
import { getCapabilitiesApi } from "@/api/device";
import { isOk } from "@/utils/api";

/**
 * Hardware capability store (Phase 5).
 *
 * `capabilities` stays undefined until deviceMgr/getCapabilities answers.
 * Older firmware without the endpoint keeps it undefined forever — every
 * consumer must treat undefined as "unknown, keep the legacy behavior"
 * (same graceful-degradation pattern as the audio card's `supported`).
 */
type CapabilitiesStoreType = {
  capabilities: ICapabilities | undefined;
  loading: boolean;
  /** Fetch once at app init; `refresh` forces a backend reprobe. */
  fetchCapabilities: (refresh?: boolean) => Promise<void>;
};

const useCapabilitiesStore = create<CapabilitiesStoreType>((set, get) => ({
  capabilities: undefined,
  loading: false,
  fetchCapabilities: async (refresh = false) => {
    if (get().loading) return;
    set({ loading: true });
    try {
      const res = await getCapabilitiesApi(refresh);
      if (isOk(res) && res.data && typeof res.data.device_type === "string") {
        set({ capabilities: res.data });
      }
    } catch (e) {
      // 404 on older firmware / transient network error: keep undefined so
      // capability-driven UI falls back to the legacy static rendering.
    } finally {
      set({ loading: false });
    }
  },
}));

export default useCapabilitiesStore;
