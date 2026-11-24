# A-Share Alpha Studio

面向个人投资者的 **A 股全栈实时量化投研、Purged Walk-Forward 回测优化、模拟实盘与合规辅助交易系统**。

> **免责声明**：本项目仅供个人量化研究与技术学习使用，不构成任何投资建议；内置所有券商接入均采用合规抽象 Adapter（模拟盘 / 人工确认辅助单 / 券商正规授权 QMT），严禁任何外挂或违规抓包。

---

## 核心策略与系统亮点

- **全真实 A 股数据源（零 Mock）**：免费免 Token 接入 AKShare 主源（腾讯日线 + 东财 PIT 财报与主力资金流 + 新浪日历/指数）+ Baostock 备用，预留 Tushare Pro 适配器，全部落盘本地 DuckDB 与 SQLite。
- **动态股票池与四维 20 因子库**：剔除 ST/*ST、停牌、次新（< 120 交易日）、低流动性与后 10% 微盘股；覆盖量价动量反转（含日内-隔夜聪明钱价差 `smart_intra_minus_on_20d`）、流动性与主力资金流（含 20 日换手变异系数 `turn_cv_20d`、5/20 日对数换手比 `turn_ratio_5_20`）、波动率与筹码韧性（含距 120 日新高距离 `high_120d_prox`）、严格按真实公告日 `ann_date < trade_date` 对齐的 PIT 基本面，经 MAD 去极值 → 申万一级行业 + 对数市值 OLS 中性化 → 横截面 Z-Score。
- **严防未来函数的 Purged Walk-Forward LightGBM**：默认采用 **10 日超额收益 `Huber 回归` + `3-Seed Bagging` 截面集成**（消除 LambdaRank 尾部过拟合，同时保留 `LambdaRank` 对照模式），强制 252 日滚动训练 + 10 日标签清洗（Purge）+ 10 日隔离期（Embargo），内置线性精确 **TreeSHAP** 归因，配合两级水位缓存支持秒级参数对照与区间切片。
- **最优动态持有 + 防追高过滤 + 风险平价 + 非对称金叉择时与防自锁风控**：
  - 事件驱动动态持有（10 日最小持有期与 10 日预测周期对齐 + Top 35 缓冲带 + 浮盈趋势续持 + 深度掉队 3 日确认淘汰 + 利润保护移动止盈），配合防追高过滤（屏蔽 3 日>20% 过热与 MACD 顶背离，不误杀低位反转机会）；
  - 逆波动率风险平价（Risk Parity）与等权对比，强约束单票 ≤ 12%、单申万行业 ≤ 25%；
  - **金叉/死叉顶部保护非对称择时**（金叉强牛 `98%` / 金叉续持 `75%` / 跌破 `MA20×0.985` 预警 `45%` / 死叉反弹压仓 `30%` / 确认熊市 `10%` / 深度熊市或急跌 `0%` 空仓）+ 全市场 4,100+ 股盘中实时状态 WebSocket 推流 + 统一 `RiskMiddleware` 与 **90 日滚动高点防自锁回撤熔断**（`-8% WARNING 0.5x` / `-12% HALTED`）。
- **高精度 A 股撮合与模拟实盘闭环**：强制 T+1、100 股整数倍、主板 10% / ST 5% / 创科 20% / 北交 30% 涨跌停封板顺延、真实佣金/印花税/过户费/滑点，内置 APScheduler 交易日全自动调度与周度归因复盘。

---

## 新电脑一键部署（3 种方式任选）

克隆仓库后无需任何付费 Token，**一条命令即可自动完成环境配置、依赖安装、真实 A 股初始化与前后端启动**：

### 方式 1：Windows 一键部署（推荐）
```powershell
git clone <你的仓库地址> a-share-alpha-studio
cd a-share-alpha-studio

# 快速部署（首次自动装依赖 + 拉取 60 只代表性真实 A 股 + 训练模型 + 启动工作台）
powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1

# 如需全市场 3,000+ 活跃 A 股 + 回补至 2023-06-01（支持断点续传）
powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1 -Mode full

# 日常仅启动前后端服务
powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1 -Mode start
```

### 方式 2：macOS / Linux 一键部署
```bash
git clone <你的仓库地址> a-share-alpha-studio
cd a-share-alpha-studio
chmod +x scripts/deploy.sh

# 快速部署并启动
./scripts/deploy.sh

# 或全市场深度初始化
./scripts/deploy.sh full
```

### 方式 3：Docker Compose 容器化一键部署
```bash
git clone <你的仓库地址> a-share-alpha-studio
cd a-share-alpha-studio
docker compose up -d --build
```

启动完成后直接访问：
- **量化工作台前端**：`http://localhost:5173`
- **FastAPI 接口文档**：`http://localhost:8000/docs`

> **新电脑使用 Antigravity 打开本项目**：仓库内置 `.agents/plugins/a-share-alpha-studio/` 与 `.agents/rules/`，在新电脑用 Antigravity 打开后可自动加载全套量化红线与运维技能，详见 [BOOTSTRAP_ANTIGRAVITY.md](BOOTSTRAP_ANTIGRAVITY.md)。

---

## 测试与文档

```powershell
# 运行全量 56 项自动化测试
cd backend
.venv\Scripts\python.exe -m pytest -q
```

- [全流程部署与运维文档](docs/DEPLOYMENT.md)
- [合规券商 QMT/PTrade 接入指南](docs/COMPLIANT_QMT_GUIDE.md)
- [Antigravity 新机一键复用清单](BOOTSTRAP_ANTIGRAVITY.md)
