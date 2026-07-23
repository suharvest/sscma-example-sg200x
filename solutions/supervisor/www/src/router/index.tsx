import { Outlet } from "react-router-dom";
import Network from "@/views/network";
import Init from "@/views/init";
import Overview from "@/views/overview";
import Security from "@/views/security";
import WebShell from "@/views/terminal";
import System from "@/views/system";
import Power from "@/views/power";
import Workspace from "@/views/workspace";
import Dashboard from "@/views/dashboard";
import Files from "@/views/files";
import Applications from "@/views/applications";
import Live from "@/views/live";
import DeviceTools from "@/views/device";
import HomeAssistant from "@/views/home-assistant";
import ConfigLayout from "@/layout/config";
import MainLayout from "@/layout/main";

const Routes = [
  // ConfigLayout group must come first: both groups share path "/", and the
  // bare "/" URL must resolve to the Applications index route below, not to
  // the MainLayout group (which has no index child and would render blank).
  {
    path: "/",
    element: (
      <ConfigLayout>
        <Outlet />
      </ConfigLayout>
    ),
    children: [
      // Applications gallery is the default landing page.
      {
        index: true,
        element: <Applications />,
      },
      {
        path: "applications",
        element: <Applications />,
      },
      {
        path: "live",
        element: <Live />,
      },
      {
        path: "device",
        element: <DeviceTools />,
      },
      {
        path: "init",
        element: <Init />,
      },
      {
        path: "files",
        element: <Files />,
      },
      {
        path: "network",
        element: <Network />,
      },
      {
        path: "overview",
        element: <Overview />,
      },
      {
        path: "security",
        element: <Security />,
      },
      {
        path: "home-assistant",
        element: <HomeAssistant />,
      },
      {
        path: "terminal",
        element: <WebShell />,
      },
      {
        path: "system",
        element: <System />,
      },
      {
        path: "power",
        element: <Power />,
      },
    ],
  },
  {
    path: "/",
    element: (
      <MainLayout>
        <Outlet />
      </MainLayout>
    ),
    children: [
      // Kept for compatibility; hidden from menus in gallery mode.
      {
        path: "/dashboard",
        element: <Dashboard />,
      },
      {
        path: "/workspace",
        element: <Workspace />,
      },
    ],
  },
];

export default Routes;
