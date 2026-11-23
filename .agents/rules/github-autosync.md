# GitHub 自动隐私审计与同步规则

1. **严禁上传个人隐私与敏感数据**：提交与推送前严禁包含 `backend/.env`、本地行情与模拟盘数据库（`data/*.duckdb`、`data/*.db`、`data/models/`）、压缩包（`dist/*.zip`）、API Token、Webhook 密钥或本机用户目录绝对路径；仓库本地 Git 提交身份必须保持匿名/脱敏（如 GitHub noreply 邮箱）。
2. **每次代码更新验收后自动同步 GitHub**：在本工作区完成功能开发、修复或配置变更，并且通过对应测试与构建验证（如 `pytest` / `npm run build`）后，若本地已配置 `origin` 远程仓库，必须自动执行隐私审计并同步推送到 GitHub：
   ```powershell
   backend\.venv\Scripts\python.exe scripts\sync_github.py -m "<简明中文或英文变更说明>"
   ```
   （如沙箱内无法访问外网推送，按终端规则对 `git push` 或 `sync_github.py` 申请网络放行执行。）
3. **仅在有实际代码/文档变更时提交**：纯查询问答、回测跑数或仅产生本地数据库变化（无受追踪代码改动）时无需空提交。
