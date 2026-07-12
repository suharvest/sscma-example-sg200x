import { supervisorRequest } from "@/utils/request";

/**
 * LED control — backend route: /api/led/<name>/on | /api/led/<name>/off
 * (writes /sys/class/leds/<name>/brightness). A non-zero code means the
 * LED does not exist on this device.
 */
export const setLedApi = async (name: string, on: boolean) =>
  supervisorRequest(
    {
      url: `api/led/${encodeURIComponent(name)}/${on ? "on" : "off"}`,
      method: "get",
      timeout: 5000,
    },
    {
      catchs: true,
    }
  );
