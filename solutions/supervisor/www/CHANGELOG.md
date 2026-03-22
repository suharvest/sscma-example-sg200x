# Changelog (0.2.2 -> feature/sensecraft-classify-dashboard)

## English

### Added
- `action=train` end-to-end flow: fetch model info (train API v2), download/upload model, create cloud app, deploy dashboard flow, and auto-redirect when ready.
- Train robustness improvements: device type guard, upload progress/slow warning/cancel, Node-RED startup gating, dashboard readiness polling, and reload-and-replay confirm on failure.
- Default dashboard flow extracted to `src/utils/flowDefaults.ts` (re-exported via `src/utils/index.ts`).
- Workspace refactor: flow lifecycle helpers moved to `src/views/workspace/services/flowService.ts`; train orchestration moved to `src/views/workspace/services/trainActionService.tsx`.
- One-click deploy script: `scripts/deploy.sh`.

### Changed
- `action=train` creates cloud app `classify_<model>`, deploys flow, and redirects in-tab to dashboard after readiness checks.
- Train replay metadata is now embedded in Node-RED `flow_data` (`comment` node `__train_meta__`) instead of extending cloud `model_data` schema.
- `action=model` selects dashboard flow only for `task=classify` + `model_format=cvimodel`, updates `model` node fields before deploy, and conditionally auto-opens dashboard.
- Redirect/session handling improved (`redirect_url` encoding + action cache/cleanup); `sensecraftRequest` retries on `code=401` refresh path.

### Fixed
- `sendFlow` no longer silently succeeds when Node-RED `/flows` rejects writes (`revision` empty is treated as failure).
- Flow deploy failures can prompt reload/retry.
- App switching can recover train-delivered models without cloud model URL by parsing `__train_meta__` and replaying train download/upload.
- UI polish: app name overflow no longer hides edit/delete buttons; dashboard layout is responsive.

## 中文

### 新增
- `action=train` 全流程：训练平台 v2 获取模型信息、下载并上传设备、创建云端应用、部署 Dashboard Flow，并在就绪后自动跳转。
- 训练流程稳健性增强：设备类型校验、上传进度/慢上传提醒/取消上传、Node-RED 启动门控、Dashboard 就绪轮询、失败后重载重放确认。
- 默认 Dashboard Flow 拆分到 `src/utils/flowDefaults.ts`（由 `src/utils/index.ts` 重新导出）。
- Workspace 重构：Flow 生命周期能力迁移到 `src/views/workspace/services/flowService.ts`，`train` 编排迁移到 `src/views/workspace/services/trainActionService.tsx`。
- 一键部署脚本：`scripts/deploy.sh`。

### 变更
- `action=train` 改为创建云端应用 `classify_<model>`，完成下发后基于就绪检测在当前页跳转 Dashboard。
- 训练回放元数据改为写入 Node-RED `flow_data`（`comment` 节点 `__train_meta__`），不扩展云端 `model_data` 协议。
- `action=model` 仅在 `task=classify` 且 `model_format=cvimodel` 时使用 Dashboard Flow，部署前更新 `model` 节点字段，并按条件自动跳转 Dashboard。
- 跳转与会话处理优化（`redirect_url` 编码 + action 缓存/清理）；`sensecraftRequest` 在 `code=401` 刷新链路可重试。

### 修复
- `sendFlow` 对 Node-RED `/flows` 写入失败不再静默成功（`revision` 为空视为失败）。
- Flow 下发失败可提示重载并重试。
- 应用切换时，若云端无模型下载地址，可从 `__train_meta__` 回放下载/上传 train 模型。
- UI 细节修复：长应用名不再遮挡编辑/删除按钮，Dashboard 布局保持响应式。
