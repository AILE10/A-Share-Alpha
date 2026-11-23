# 在任意新电脑 / Antigravity 中一键复用与部署清单

当你把 **A-Share Alpha Studio** 克隆到任意新电脑并用 **Antigravity** 打开后，Antigravity 会自动发现并加载仓库内置的 `.agents/plugins/a-share-alpha-studio/` 插件与 `.agents/rules/` 全局规则。

## 一键部署脚本（无需手动逐条执行）

- **Windows**：
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1
  ```
- **macOS / Linux**：
  ```bash
  chmod +x scripts/deploy.sh && ./scripts/deploy.sh
  ```
- **Docker Compose**：
  ```bash
  docker compose up -d --build
  ```

## 发给新电脑 Antigravity 的一键引导指令

```text
请读取本项目自带的 alpha-studio-ops 技能，按标准流程在本机完成环境初始化、依赖安装、同步真实 A 股行情与回测初始化，运行全量 pytest 验收，然后启动前后端工作台并用浏览器验证页面正常渲染。
```

## 全市场历史扩展（可选，用于 2025-01 / 2024-08 起严格样本外全市场回测）

```powershell
# 1. 全市场活跃股增量同步与模型重训（支持断点续传）
backend\.venv\Scripts\python.exe -u scripts\sync_full_market.py --retrain

# 2. 将全市场历史日线与 PIT 财报向前扩展至 2023-06-01
backend\.venv\Scripts\python.exe -u scripts\extend_history.py --start 2023-06-01 --workers 8

# 3. 同步全市场主力资金流
backend\.venv\Scripts\python.exe -u scripts\sync_capital_flow.py --workers 4

# 4. 运行全量 53 项回归测试
cd backend && .venv\Scripts\python.exe -m pytest -q
```

## 启用本地 Git 隐私保护与自动推送钩子（可选）

```powershell
backend\.venv\Scripts\python.exe scripts\install_git_hooks.py
```
