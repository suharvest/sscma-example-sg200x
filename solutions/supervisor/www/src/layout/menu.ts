/**
 * Shared sidebar menu definition (PC + mobile).
 * Gallery mode hides the Node-RED–backed Workspace/Dashboard entries.
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
}

export function getMenuSections({
  galleryMode,
  isReCamera,
}: MenuOptions): MenuSection[] {
  const sections: MenuSection[] = [
    {
      title: "menu.sectionConsole",
      items: [
        { key: "applications", label: "menu.applications", route: "/" },
        { key: "live", label: "menu.live", route: "/live" },
        { key: "device", label: "menu.device", route: "/device" },
      ],
    },
  ];

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
      ],
    },
    {
      title: "menu.sectionSystem",
      items: [
        { key: "terminal", label: "menu.terminal", route: "/terminal" },
        { key: "system", label: "menu.system", route: "/system" },
        { key: "power", label: "menu.power", route: "/power" },
      ],
    }
  );

  return sections;
}
