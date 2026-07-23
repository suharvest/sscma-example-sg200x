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
  isReCamera: boolean;
  /**
   * False until queryDeviceInfo has populated the store. While unknown we
   * keep the pre-P4-D rendering (Console section visible) to avoid the
   * sidebar flashing on every load in the common console-mode case.
   */
  modeKnown?: boolean;
}

export function getMenuSections({
  galleryMode,
  isReCamera,
  modeKnown = true,
}: MenuOptions): MenuSection[] {
  const sections: MenuSection[] = [];

  // Console group (Applications/Live/Device) is gallery-mode only: in
  // Node-RED mode the C++ app stack is stopped, so the entries are hidden.
  if (galleryMode || !modeKnown) {
    sections.push({
      title: "menu.sectionConsole",
      items: [
        { key: "applications", label: "menu.applications", route: "/" },
        { key: "live", label: "menu.live", route: "/live" },
        { key: "device", label: "menu.device", route: "/device" },
      ],
    });
  }

  if (!galleryMode) {
    const workspaceItems: MenuItem[] = [];
    if (isReCamera) {
      workspaceItems.push({
        key: "overview",
        label: "menu.overview",
        route: "/overview",
      });
    }
    workspaceItems.push(
      { key: "dashboard", label: "menu.dashboard", route: "/dashboard" },
      { key: "workspace", label: "menu.workspace", route: "/workspace" }
    );
    sections.push({ title: "menu.sectionWorkspace", items: workspaceItems });
  } else if (isReCamera) {
    // Overview (camera preview) stays available in gallery mode.
    sections.push({
      title: "menu.sectionWorkspace",
      items: [{ key: "overview", label: "menu.overview", route: "/overview" }],
    });
  }

  sections.push(
    {
      title: "menu.sectionConfiguration",
      items: [
        { key: "files", label: "menu.files", route: "/files" },
        { key: "security", label: "menu.security", route: "/security" },
        { key: "network", label: "menu.network", route: "/network" },
        {
          key: "home-assistant",
          label: "menu.homeAssistant",
          route: "/home-assistant",
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
