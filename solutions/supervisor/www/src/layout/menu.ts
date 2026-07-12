/**
 * Shared sidebar menu definition (PC + mobile).
 * Gallery mode hides the Node-RED–backed Workspace/Dashboard entries.
 */

export interface MenuItem {
  key: string;
  label: string;
  route: string;
}

export interface MenuSection {
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
      title: "Console",
      items: [
        { key: "applications", label: "Applications", route: "/" },
        { key: "live", label: "Live", route: "/live" },
        { key: "device", label: "Device", route: "/device" },
      ],
    },
  ];

  if (!galleryMode) {
    const workspaceItems: MenuItem[] = [];
    if (isReCamera) {
      workspaceItems.push({
        key: "overview",
        label: "Overview",
        route: "/overview",
      });
    }
    workspaceItems.push(
      { key: "dashboard", label: "Dashboard", route: "/dashboard" },
      { key: "workspace", label: "Workspace", route: "/workspace" }
    );
    sections.push({ title: "Workspace", items: workspaceItems });
  } else if (isReCamera) {
    // Overview (camera preview) stays available in gallery mode.
    sections.push({
      title: "Workspace",
      items: [{ key: "overview", label: "Overview", route: "/overview" }],
    });
  }

  sections.push(
    {
      title: "Configuration",
      items: [
        { key: "files", label: "Files", route: "/files" },
        { key: "security", label: "Security", route: "/security" },
        { key: "network", label: "Network", route: "/network" },
      ],
    },
    {
      title: "System",
      items: [
        { key: "terminal", label: "Terminal", route: "/terminal" },
        { key: "system", label: "System", route: "/system" },
        { key: "power", label: "Power", route: "/power" },
      ],
    }
  );

  return sections;
}
