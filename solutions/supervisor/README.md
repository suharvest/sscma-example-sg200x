# Supervisor Solution  

## Overview  

**Supervisor** is a built-in service for **ReCamera-OS** that provides:  

- An **HTTP service** for remote communication.  
- A **Web UI** for device management and monitoring.  
- **System status monitoring** to ensure stable device operation.  

Beyond the stock system service, this build also owns which application runs on
the camera and how the device presents itself to the outside world. Those parts
are described in [What this build adds](#what-this-build-adds) below.


![](../../images/recam_OS_structure.png)

The foundational system service providing:
- System Services:
    - Device management: Identify and configure connected devices, storage devices, etc.
    - User Management: Manage user accounts, credentials, and SSH keys.
    - Network configuration: Configure wired and wireless network connections.
    - File system operations: Manage device files.
    - Device Discovery: Uses mDNS to broadcast device information. The device hostname is recamera.local.When a web interface sends a request, the recamera device scans the local network for other recamera devices via mDNS, generates a list of discovered devices, formats the data, and returns it to the web interface. (Note: Currently, only one device’s information is returned.)

- Update Service:
    - Package/firmware download management
    - Security verification
    - Installation automation

- Daemon Service:
    - System health monitoring
    - Automatic application recovery

- Logging Service:
    - Runtime status tracking
    - Error diagnostics

- Application Service:
    - Application Deployment
    - Application Packaging


## Getting Started  

Before building this solution, ensure that you have set up the **ReCamera-OS** environment as described in the main project documentation:  

🔗 **[SSCMA Example for SG200X - Main README](../../README.md)**  

This includes:  

- Setting up **ReCamera-OS**  
- Configuring the SDK path  
- Preparing the necessary toolchain  

If you haven't completed these steps, follow the instructions in the main project README before proceeding.

## Building & Installing  

### 1. Navigate to the `supervisor` Solution  

```bash
cd solutions/supervisor
```

### 2. Build the Application  

By default, the application is built **without** the Web UI.  

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build
```

#### ⚙️ Enabling Web UI  

If you want to include the Web UI, enable the `WEB` option before building. This will:  

1. Recompile the front-end project in `www/`.  
2. Copy the output to `rootfs/usr/share/supervisor/www/`.  

To enable Web UI:  

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DWEB=ON .
cmake --build build
```

**Note:** The Web UI build process requires **Node.js** to be installed.

### 3. Package the Application  

```bash
cd build && cpack
```

This will generate an `.ipk` package for installation.

## Deploying & Running  

### 1. Transfer the Package to Your Device  

Copy the package to the ReCamera device using `scp`:  

```bash
scp build/supervisor-1.0.0-1.ipk recamera@192.168.42.1:/tmp/
```

Replace `recamera@192.168.42.1` with your device's IP address.

### 2. Install the Package  

SSH into the device and install the package:  

```bash
ssh recamera@192.168.42.1
sudo opkg install /tmp/supervisor-1.0.0-1.ipk
```

### 3. Run the Supervisor Service  

Once installed, start the service:  

```bash
sudo supervisor
```

If running correctly, the HTTP server should be accessible.

### 4. Access the Web UI  

If built with Web UI enabled, open a browser and visit:  

```
http://<device-ip>:<port>
```

Replace `<device-ip>` with the actual IP of your ReCamera device.

## Disabling the Supervisor Service  

If you want to **disable** the `supervisor` service from running automatically on startup, remove or move the init script:  

```bash
mv /etc/init.d/S93sscma-supervisor /etc/init.d/S93sscma-supervisor.bak
```

To **reenable** it, move it back:  

```bash
mv /etc/init.d/S93sscma-supervisor.bak /etc/init.d/S93sscma-supervisor
```

## Directory Structure  

```
supervisor/
├── CMakeLists.txt    # CMake build configuration
├── control           # OPKG packaging script
├── main              # Source code directory
├── README.md         # This README file
├── rootfs            # Resource files (installed to system root)
│   └── usr/share/supervisor/www  # Web UI files (if enabled)
└── www               # Web UI source code (requires Node.js)
```
---

## What this build adds

The sections above describe the stock supervisor. This build additionally owns
application selection and three device-wide features, all reachable from the Web
UI.

### Application gallery

The console lists the applications installed on the device and switches between
them. Only one may run at a time — the camera pipeline is exclusive, and two
applications opening VPSS at once produces anything from a stream that never
comes up to a wedge that needs a power cycle.

Consequences worth knowing before packaging an application:

- **Gallery applications ship their init script disabled** (`K92<name>`, not
  `S92<name>`). Autostart-on-boot would race whatever the console started. The
  console starts the chosen application by invoking the script directly and
  records the selection in `/userdata/local/apps/state.json`; `app_restore`
  brings that one back at boot.
- **An application registers by dropping its manifest into
  `/userdata/local/apps/`**, normally from its own `postinst`. A `.deb` that
  installs a binary but no manifest works fine — it simply does not appear in
  the gallery.
- The manifest may declare `"privacy_blur": true`, which is what puts the
  masking shortcut on that application's debug page. An application that does
  not apply the mask should not declare it: a switch that changes nothing on the
  picture beside it reads as a broken feature.

Manifests bundled with this package live in `rootfs/usr/share/supervisor/apps/`;
each may carry `<id>.md` / `<id>.zh.md`, which the console renders as that
application's integration guide.

### Privacy masking

Device-wide, stored in `/userdata/local/blur.conf`, configured on the **Device**
page with a shortcut (switch plus opacity) on the debug page. When on,
applications that support it conceal what they detect **before the frame is
encoded**, so the RTSP stream, the console preview and `/snapshot.jpg` are all
masked. Off by default.

`enabled`, `alpha` and the block count apply live; changing the backend, block
size or region count rebuilds the hardware regions and restarts the running
application. The API says which happened, so the console only warns about a
restart when one actually occurs.

The same page installs and restores the patched kernel modules that let the
camera hardware composite the mask; without them the masking is done on the CPU
at roughly 38 ms per frame. See the [main README](../../README.md#privacy-masking-and-the-patched-kernel-modules)
for what the patch does and why. The modules are build artefacts and are not
tracked in git, so a package built from a fresh clone reports `not_packaged` and
greys the button out — that is expected.

### ONVIF

WS-Discovery, Device and Media2 services, and analytics metadata, so a VMS can
find the camera and pull its stream without per-vendor configuration. Settings
live on the Integrations page. Design notes and the deliberate omissions are in
[`docs/onvif-implementation-spec.md`](../../docs/onvif-implementation-spec.md).

### Integrations page

RTSP, Home Assistant and ONVIF, each collapsible. RTSP is separate from Home
Assistant because that URL has nothing to do with Home Assistant — VLC, a VMS
and ffmpeg all use the same one.

## Note on the HTTP layer

The HTTP and WebSocket implementation is **libwebsockets** (MIT), reached
through the `http_dispatch` / `http_request` / `ws_transport` seams rather than
called directly. It replaced mongoose, which is GPL-2.0-only and therefore
incompatible with this Apache-2.0 tree. Keep application code on the seam: the
point of the abstraction is that a third library change touches implementation
files only.
