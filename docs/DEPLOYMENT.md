# A-Share Alpha Studio 全流程部署与运维文档

## 1. 系统架构与存储说明

- **后端**：Python 3.11+ / FastAPI / APScheduler / LightGBM / DuckDB / SQLite
- **前端**：React 18 + TypeScript + Vite + Tailwind CSS + Ant Design + ECharts + TradingView Lightweight Charts
- **双引擎本地存储**（均位于 `data/` 目录）：
  - `data/market_analytics.duckdb`：真实 A 股日线（未复权 + 后复权因子）、PIT 公告日财报、指数日线、申万一级行业、四维 17 因子与 IC 时序。
  - `data/studio_state.db`：模拟实盘账本/持仓/成交流水、推荐归因、回测历史、周度复盘、模型版本快照、风控配置与审计日志。
  - `data/models/`：LightGBM 模型版本快照文件（支持一键重训与无损回滚）。

## 2. 本地一键启动（Windows / macOS / Linux）

### 2.1 配置环境变量（可选）
复制 `backend/.env.example` 为 `backend/.env`，按需配置：
- `TUSHARE_TOKEN`（可选，默认使用免 Token 的 AKShare 主 + Baostock 备）
- `WECOM_WEBHOOK_URL` / `DINGTALK_WEBHOOK_URL`（半自动辅助下单群推送）
- `QMT_USERDATA_PATH` / `QMT_ACCOUNT_ID`（仅已开通合规券商权限账户填写）

### 2.2 首次同步真实 A 股数据并初始化模型
```powershell
# 1. 拉取真实 A 股历史日线、中证1000/沪深300、申万行业与 PIT 财报并完成因子检验
backend\.venv\Scripts\python.exe -u scripts\m1_verify_real_data.py --n 60 --days 730

# 2. 运行 Purged Walk-Forward LightGBM 与精确回测验证
backend\.venv\Scripts\python.exe -u scripts\m2_verify_backtest.py

# 3. 初始化模型快照并跑通模拟盘闭环
backend\.venv\Scripts\python.exe -u scripts\m3_verify_paper_loop.py
```

### 2.3 一键启动前后端工作台
- **Windows**：
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\start_dev.ps1
  ```
- **Linux / macOS**：
  ```bash
  chmod +x scripts/start_dev.sh && ./scripts/start_dev.sh
  ```
- 浏览器访问 `http://localhost:5173`，后端 Swagger 文档 `http://localhost:8000/docs`。

## 3. Docker Compose 容器化部署

```bash
docker compose up -d --build
```
容器启动后：
- 前端（Nginx 反向代理 `/api` 与 WebSocket）：`http://localhost:5173`
- 后端 FastAPI：`http://localhost:8000/api/v1/health`
- 数据卷 `./data:/app/data` 持久保存 DuckDB、SQLite 与模型快照。

## 4. 定时任务调度说明（北京时间 Asia/Shanghai）

| 触发时间 | 任务 ID | 功能 |
| :--- | :--- | :--- |
| 工作日 `09:26` | `pre_market_unlock` | 盘前 T+1 持仓可卖股数解锁 |
| 交易时段每 120s | `intraday_top100_refresh` | 仅对持仓 + Top 100 候选拉取真实快照，盯市与盘中止损监控 |
| 工作日 `14:40` | `rebalance_1440` | 生成调仓信号、推送可解释推荐，经 RiskMiddleware 路由至当前券商模式 |
| 工作日 `15:30` | `post_market_sync` | 盘后增量日线落库与模拟盘日净值对齐 |
| 周五 `16:30` | `weekly_review` | 生成周度复盘、跟踪误差归因与近 20 日因子 IC 衰减预警 |

## 5. 全量自动化测试回归

```powershell
cd backend
.venv\Scripts\python.exe -m pytest
```
