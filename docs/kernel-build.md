# 内核模块构建

## 三个目录，各自是什么

| 目录 | 性质 | 在 git 里吗 |
|---|---|---|
| `reCamera-OS/` | fork of Seeed-Studio/reCamera-OS，分支 `sg200x-reCamera` | ✅ `suharvest/reCamera-OS` |
| `reCamera-OS/osdrv/` | 子模块，fork of sophgo/osdrv，分支 `feat/mosaic-colour-lut` | ✅ `suharvest/osdrv` |
| `sg2002_recamera_emmc/` | **构建产物**，1.1 GB | ❌ 也不该在 |
| `linux_5.10/` | 内核源码树 | ✅ 自带 git（有厂商修改） |

`sg2002_recamera_emmc/` 是 reCamera-OS 在 CI 里编出来的产物打包分发的 —— 证据是里面的
`libcviruntime.so` 还留着构建路径 `/home/runner/work/reCamera-OS/reCamera-OS/output/...`。
它可以重新生成，不进 git 是对的；应用交叉编译只用它的头文件和库。

## 内核模块必须从 fork 编，不是从 SDK 树

两棵树都有 osdrv 源码。**从 fork 编**，这样"编出来的 ko"和"提交的代码"在结构上不可能对不上。

```bash
docker exec ubuntu_dev_x86 bash -c '
  cd /workspace/linux_5.10
  touch .scmversion
  rm -f include/generated/utsrelease.h
  export PATH=/workspace/host-tools/gcc/riscv64-linux-musl-x86_64/bin:$PATH
  make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-musl- include/generated/utsrelease.h

  export ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-musl-
  export KERNEL_DIR=/workspace/linux_5.10 CVIARCH=CV181X CVIARCH_L=cv181x
  cd /workspace/reCamera-OS/osdrv/interdrv/vpss && make clean && make
  cd /workspace/reCamera-OS/osdrv/interdrv/rgn  && make clean && make
'
```

编译时间与从 SDK 树编相同：同样的源文件、同样的编译器，且流程本来就是全量 `make clean && make`。

构建产物落在 fork 里，已通过 `.git/modules/osdrv/info/exclude` 屏蔽（不用 `.gitignore`，
免得提 PR 时多带一个上游没有的文件）。

## vermagic 陷阱（错了设备会拒绝加载）

- `linux_5.10` 是脏的 git 树，`scripts/setlocalversion` 会给 vermagic 追加 `+`。
  `touch .scmversion` 是必须的。
- `include/generated/utsrelease.h` 会缓存旧值，**必须先删再重新生成**，否则新 ko 里还是带 `+`。
- 目标：`vermagic=5.10.4-tag- preempt mod_unload riscv`，**结尾无 `+`**。
  用 `strings <ko> | grep vermagic` 验证。
- 预期的无害警告：`Symbol info of vmlinux is missing`、`Module.symvers is missing`
  （只做了 modules_prepare 没全量编内核）。

## 部署

`/` 是只读 ext4，`/mnt/system` 在根分区上且 overlay 不覆盖 `/mnt`：

```bash
mount -o remount,rw /
cp cv181x_{rgn,vpss}.ko /mnt/system/ko/
sync
mount -o remount,ro /
reboot
```

**不 remount 的话 `cp` 会失败但脚本可能继续跑**，变成"报告成功、实际没换"。
拷完必须比对 md5。原厂驱动备份在设备 `/userdata/ko-backup/`。

## 防分叉

改完内核跑一下：

```bash
sscma-example-sg200x/scripts/check-osdrv-sync.sh
```

它 diff 那四个被改过的文件，SDK 树和 fork 不一致就报错 —— 万一有人手滑改了 SDK 树那份，
能立刻发现，而不是编出一个源码不在版本控制里的 ko。
