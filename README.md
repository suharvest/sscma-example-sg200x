# SSCMA Example for SG200X  

This repository provides a compilation framework for developing and running applications on the **ReCamera** platform. It includes setup instructions, compilation steps, and installation guidelines.  

## Project Directory Structure  

```bash
.
├── cmake         # Build scripts
├── components    # Functional components
├── docs          # Documentation
├── images        # Images
├── scripts       # Scripts
├── solutions     # Applications
├── test          # Tests
└── tools         # Tools
```

## Prerequisites  

### 1. Clone and Set Up **ReCamera-OS**  

This project depends on **ReCamera-OS**, which provides the necessary toolchain, SDK, and runtime environment. Ensure you have cloned and set up **ReCamera-OS** from the following repository:  

🔗 [ReCamera-OS GitHub Repository](https://github.com/Seeed-Studio/reCamera-OS)  

```bash
git clone https://github.com/Seeed-Studio/reCamera-OS.git
cd reCamera-OS
# Follow the setup instructions in the repository
```  
Setup Environment Variables:

   ```bash
   export SG200X_SDK_PATH=<PATH_TO_RECAMERA-OS>/output/sg2002_recamera_emmc/
   export PATH=<PATH_TO_RECAMERA-OS>/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH
   ```  


### 2. Use a Prebuilt SDK (Optional)  

If you do not wish to build **ReCamera-OS** manually, you can download a prebuilt SDK package:  

1. Visit [ReCamera-OS Releases](https://github.com/Seeed-Studio/reCamera-OS/releases).  
2. Download the latest **reCamera_OS_SDK_x.x.x.tar.gz** package.  
3. Extract the package and set the SDK path:  

   ```bash
   export SG200X_SDK_PATH=<PATH_TO_RECAMERA-OS-SDK>/sg2002_recamera_emmc/
   ```  

## Compilation Guide  

Follow these steps to set up the environment, compile the project, and generate the necessary application package.  

### 1. Clone This Repository  

```bash
git clone https://github.com/Seeed-Studio/sscma-example-sg200x
cd sscma-example-sg200x
git submodule update --init
```  

### 2. Build the Application  

Navigate to the project directory and compile:  

```bash
cd solutions/helloworld
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build
```  

If the build process completes successfully, the executable binary should be available in the `build` directory.  

### 3. Package the Application  

To prepare the application for distribution, package it using `cpack`:  

```bash
cd build && cpack
```  

This will generate a **.deb** package, which can be installed on the device.  

## Deploying the Application  

### 1. Transfer the Package to the Device  

Use **scp** or other file transfer methods to copy the package to the ReCamera device:  

```bash
scp build/helloworld-1.0.0-1.deb recamera@192.168.42.1:/tmp/
```  

Replace `recamera@192.168.42.1` with the actual username and IP address of your device.  

### 2. Install the Application  

Log into the device via SSH and install the package using `opkg`:  

```bash
ssh recamera@192.168.42.1
sudo opkg install /tmp/helloworld-1.0.0-1.deb
```  

**Note**: sudo password is the same as the WEB UI password. default is `recamera`.

### 3. Run the Application  

Once installed, you can run the application:  

```bash
helloworld
Hello, ReCamera!
```  

For more information, go to the specific solution's README.
---

## Privacy Masking and the Patched Kernel Modules

Several solutions here (`face-analysis`, `facemesh-reader`, `retail-vision`,
`detection-blur`) can conceal the people they detect before the frame is
encoded, so the RTSP stream, the console's debug video and the ONVIF snapshot
all carry the mask. It is a device-wide setting, configured in the supervisor
console and stored in `/userdata/local/blur.conf`.

**This works on stock firmware.** Nothing below is required to build or run
anything in this repository.

### Why a patched kernel is offered at all

The CV181x region engine can composite a mask in hardware, but its MOSAIC mode
fills the mask with values from `get_random_u32()` — it renders noise over the
subject, not a pixelated version of them. The colour table that would make it
pixelate is not exposed to userspace.

The patch adds one ioctl so the application can supply that table, and one
module parameter (`mask_force_alpha`) that controls whether the gaps between
mask cells are painted black. With it, masking is composited by the camera
hardware; without it, the same masking is done on the CPU at roughly 38 ms per
frame.

The application detects this at runtime. On a stock kernel you will see:

```
kernel has no mosaic colour table; using software compositing
```

and everything keeps working, more slowly.

### Building the modules

The modules are build artefacts and are **not tracked in git**, so a fresh
clone packages a supervisor `.deb` whose "deploy driver" button is greyed out
and reports `not_packaged`. That is expected.

To build them you need the forked kernel tree — not the stock SDK tree, and not
this repository:

- `github.com/suharvest/reCamera-OS` (branch `sg200x-reCamera`)
- `github.com/suharvest/osdrv` (branch `feat/mosaic-colour-lut`), a submodule of
  the above

See [`docs/kernel-build.md`](docs/kernel-build.md) for the build procedure and
its two traps: the modules must be built from the fork rather than from the
prebuilt SDK, and `vermagic` must match the running kernel exactly or the camera
will fail to come up after the next reboot. The console refuses to install
modules whose `vermagic` disagrees with the device, so a mismatch is reported
rather than bricking the camera — but check it anyway.

Installing or restoring the modules is done from the console's Device page, and
takes effect after a reboot. The stock modules are backed up to
`/userdata/ko-backup/` before the first install and can be put back from the
same page.
