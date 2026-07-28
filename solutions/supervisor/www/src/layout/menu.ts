/**
 * Shared sidebar menu definition (PC + mobile).
 * Gallery mode hides the Node-RED–backed Workspace/Dashboard entries;
 * Node-RED mode symmetrically hides the Console group (P4-D). Routes stay
 * registered either way — direct links land on pages that render their own
 * "wrong mode" hints (e.g. the Applications banner).
 *
 * `label` / `title` are i18n keys — render with t(item.label) / t(section.title).
 */

export interface MenuItem {
  key: string;
  /** i18n key, e.g. "menu.applications" */
  label: string;
  route: string;
}

export interface MenuSection {
  /** i18n key, e.g. "menu.sectionConsole" */
  title: string;
  items: MenuItem[];
}

interface MenuOptions {
  galleryMode: boolean;
  /**
   * False until queryDeviceInfo has populated the store. While unknown we
   * keep the pre-P4-D rendering (Console section visible) to avoid the
   * sidebar flashing on every load in the common console-mode case.
   */
  modeKnown?: boolean;
}

export function getMenuSections({
  galleryMode,
  modeKnown = true,
}: MenuOptions): MenuSection[] {
  const sections: MenuSection[] = [];

  // Console group (Applications/Live) is gallery-mode only: in Node-RED mode
  // the C++ app stack is stopped, so those two entries have nothing to show.
  // Device is NOT part of this gate — its contents (time/timezone/NTP, audio,
  // battery, power, privacy-blur driver) all go through supervisor's own APIs
  // and stay useful in either mode, so it lives in Configuration below.
  if (galleryMode || !modeKnown) {
    sections.push({
      title: "menu.sectionConsole",
      items: [
        { key: "applications", label: "menu.applications", route: "/" },
        { key: "live", label: "menu.live", route: "/live" },
      ],
    });
  }

  // Overview is the bare camera preview. It is Node-RED mode only: gallery mode
  // has Live, which does everything Overview does and more (overlay, RTSP
  // fallback, per-app binding), so showing both there is just a worse duplicate.
  // Both entries used to be gated on `isReCamera`, a field the C++ backend has
  // never emitted (zero hits for it across all history in *.cpp/*.h, and absent
  // from queryDeviceInfo on-device) — so the flag was permanently false and
  // Overview had never once appeared in the sidebar. The gate is gone rather
  // than repaired: every device running this firmware is a reCamera.
  if (!galleryMode) {
    sections.push({
      title: "menu.sectionWorkspace",
      items: [
        { key: "overview", label: "menu.overview", route: "/overview" },
        { key: "dashboard", label: "menu.dashboard", route: "/dashboard" },
        { key: "workspace", label: "menu.workspace", route: "/workspace" },
      ],
    });
  }

  sections.push(
    {
      title: "menu.sectionConfiguration",
      items: [
        { key: "device", label: "menu.device", route: "/device" },
        { key: "files", label: "menu.files", route: "/files" },
        { key: "security", label: "menu.security", route: "/security" },
        { key: "network", label: "menu.network", route: "/network" },
        {
          key: "integrations",
          label: "menu.integrations",
          route: "/integrations",
        },
      ],
    },
    {
      title: "menu.sectionSystem",
      items: [
        { key: "terminal", label: "menu.terminal", route: "/terminal" },
        { key: "system", label: "menu.system", route: "/system" },
      ],
    }
  );

  return sections;
}
