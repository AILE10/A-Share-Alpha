# A-Share Alpha Studio — Antigravity Plugin

标准 Antigravity 插件包，用于在任意 Antigravity 环境中一键复用 **A-Share Alpha Studio** 的量化开发红线、一键部署运维技能与多因子/LightGBM 投研工作流。

## 包含内容
- `rules/AGENTS.md`：5 条 A 股量化开发与风控红线（严禁 Mock 数据、严守 A 股 T+1/涨跌停/百股/真实费用、严防未来函数与 5 日 Embargo、合规 Adapter 与统一前置风控、真实运行与浏览器验证）。
- `skills/alpha-studio-ops/SKILL.md`：环境初始化、真实 A 股数据同步、回测与模拟盘冷启动、服务启动与排障 Runbook。
- `skills/alpha-quant-research/SKILL.md`：因子研发、Walk-Forward 回测优化、周度复盘与模型快照迭代标准流程。
- `assets/logo.svg`：128×128 矢量高对比度图标。

## 如何安装到另一台 Antigravity

### 方式一：随项目工作区自动发现（最简单）
直接解压完整便携包 `a-share-alpha-studio-portable-full.zip` 并用 Antigravity 打开根目录，Antigravity 会自动发现 `.agents/plugins/a-share-alpha-studio/` 并启用全部规则与技能。

### 方式二：安装为全局插件（所有工作区生效）
将本目录复制到目标机器的 `~/.gemini/config/plugins/a-share-alpha-studio/`（或在解压插件包后运行附带的 `install-plugin.ps1` / `install-plugin.sh`），也可以通过 CLI 安装：
```bash
agy plugin install ./a-share-alpha-studio
```
安装后可在 Antigravity 的 Customizations 页面一键开启或关闭，无需重启。
