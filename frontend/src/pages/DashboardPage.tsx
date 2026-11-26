import React, { useEffect, useMemo, useState } from 'react';
import { Button, Popconfirm, Progress, Spin, Table, Tag, message } from 'antd';
import { CloudDownloadOutlined, DatabaseOutlined, ReloadOutlined, SyncOutlined } from '@ant-design/icons';
import { aShareColor, api, fmtNum, fmtPct } from '../services/api';
import { useAlphaWebSocket } from '../services/ws';
import { useAsyncTask } from '../hooks/useAsyncTask';
import { TaskProgressBanner } from '../components/common/TaskProgressBanner';
import { DailyWorkflowCard } from '../components/common/DailyWorkflowCard';
import { EChartsBox } from '../components/charts/EChartsBox';

export const DashboardPage: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [dataStatus, setDataStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [reloading, setReloading] = useState(false);

  const load = async (showSpinner = false) => {
    if (showSpinner) setReloading(true);
    try {
      const [sum, ds] = await Promise.all([api.getDashboardSummary(), api.getDataStatus()]);
      setData(sum);
      setDataStatus(ds);
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '仪表盘加载失败');
    } finally {
      setLoading(false);
      if (showSpinner) setReloading(false);
    }
  };

  const dashTask = useAsyncTask({
    kinds: [
      'data_sync_active_market',
      'data_sync_incremental',
      'data_sync_sample_60',
      'data_sync_top100_snapshot',
      'paper_intraday',
      'regime_refresh',
    ],
    onDone: async (task) => {
      await load(false);
      if (task.kind === 'paper_intraday') {
        message.success('已完成 Top-100 盘中实时盯市与全市场大盘刷新');
      } else if (task.kind === 'data_sync_incremental') {
        message.success('已完成最新收盘日线增量同步、因子与大盘状态更新！');
      } else if (task.kind === 'data_sync_active_market') {
        message.success('全市场活跃 A 股同步与因子重算完成！');
      } else if (task.kind !== 'regime_refresh') {
        message.success(`「${task.title}」执行完成`);
      }
    },
    onError: (task, errMsg) => {
      message.error(`「${task.title}」失败: ${errMsg}`);
    },
  });

  useAlphaWebSocket((evt) => {
    if (evt.channel === 'market_regime' && evt.payload?.regime) {
      setData((prev: any) => {
        if (!prev) return prev;
        const p = evt.payload;
        const hist = Array.isArray(prev.regime?.history) ? [...prev.regime.history] : [];
        if (p.trade_date && p.csi1000_close !== undefined && p.breadth_above_ma20 !== undefined) {
          const pt = {
            date: p.trade_date,
            csi1000_close: p.csi1000_close,
            breadth: p.breadth_above_ma20,
            regime: p.regime,
            target_exposure: p.target_exposure,
          };
          if (hist.length && hist[hist.length - 1].date === pt.date) {
            hist[hist.length - 1] = pt;
          } else {
            hist.push(pt);
          }
        }
        return {
          ...prev,
          regime: {
            ...(prev.regime || {}),
            ...p,
            history: hist.slice(-60),
          },
        };
      });
    }
  });

  useEffect(() => {
    load(false);
  }, []);

  const handleIntradayRefresh = async () => {
    try {
      await dashTask.startTask(() => api.triggerIntradayRefreshAsync());
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '盘中刷新启动失败');
    }
  };

  const handleIncrementalSync = async () => {
    try {
      await dashTask.startTask(() => api.triggerDataSync('incremental', { workers: 10 }));
      message.info('已启动收盘日线与大盘状态增量同步，进度实时展示');
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '启动增量同步失败');
    }
  };

  const handleFullMarketSync = async () => {
    try {
      await dashTask.startTask(() => api.triggerDataSync('active_market', { workers: 4, days: 730 }));
      message.info('已启动后台全市场活跃股断点续传同步（支持随时中断续传，进度实时推送）');
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '启动全市场同步失败');
    }
  };

  const navOption = useMemo(() => {
    const paperCurve = data?.paper_nav_curve || [];
    const btCurve = data?.backtest_reference_curve || [];
    const usePaper = paperCurve.length >= 2;
    const src = usePaper ? paperCurve : btCurve;
    const dates = src.map((x: any) => x.trade_date || x.date);
    return {
      tooltip: { trigger: 'axis' as const },
      legend: {
        top: 0,
        textStyle: { color: '#CBD5E1', fontSize: 12 },
        data: [usePaper ? '模拟实盘净值' : '策略参考净值(样本外)', '中证1000', '沪深300'],
      },
      grid: { left: 48, right: 20, top: 34, bottom: 26 },
      xAxis: { type: 'category' as const, data: dates, axisLine: { lineStyle: { color: '#334155' } } },
      yAxis: {
        type: 'value' as const,
        scale: true,
        splitLine: { lineStyle: { color: '#1E293B' } },
      },
      series: [
        {
          name: usePaper ? '模拟实盘净值' : '策略参考净值(样本外)',
          type: 'line' as const,
          smooth: false,
          showSymbol: false,
          lineStyle: { color: '#F43F5E', width: 2.5 },
          areaStyle: { color: 'rgba(244,63,94,0.10)' },
          data: src.map((x: any) => x.nav),
        },
        {
          name: '中证1000',
          type: 'line' as const,
          showSymbol: false,
          lineStyle: { color: '#38BDF8', width: 1.8 },
          data: src.map((x: any) => x.csi1000_nav),
        },
        {
          name: '沪深300',
          type: 'line' as const,
          showSymbol: false,
          lineStyle: { color: '#94A3B8', width: 1.5, type: 'dashed' as const },
          data: src.map((x: any) => x.csi300_nav),
        },
      ],
    };
  }, [data]);

  const regimeOption = useMemo(() => {
    const hist = data?.regime?.history || [];
    return {
      tooltip: { trigger: 'axis' as const },
      legend: { top: 0, textStyle: { color: '#CBD5E1', fontSize: 11 }, data: ['中证1000', '站上MA20宽度(%)'] },
      grid: { left: 52, right: 48, top: 32, bottom: 24 },
      xAxis: { type: 'category' as const, data: hist.map((x: any) => x.date), axisLine: { lineStyle: { color: '#334155' } } },
      yAxis: [
        { type: 'value' as const, scale: true, splitLine: { lineStyle: { color: '#1E293B' } } },
        { type: 'value' as const, min: 0, max: 100, splitLine: { show: false }, axisLabel: { formatter: '{value}%' } },
      ],
      series: [
        {
          name: '中证1000',
          type: 'line' as const,
          showSymbol: false,
          lineStyle: { color: '#38BDF8', width: 2 },
          data: hist.map((x: any) => x.csi1000_close),
        },
        {
          name: '站上MA20宽度(%)',
          type: 'bar' as const,
          yAxisIndex: 1,
          itemStyle: { color: 'rgba(245,158,11,0.35)' },
          data: hist.map((x: any) => Number((x.breadth * 100).toFixed(1))),
        },
      ],
    };
  }, [data]);

  if (loading) {
    return (
      <div className="h-96 flex flex-col items-center justify-center gap-3">
        <Spin size="large" />
        <div className="text-xs text-slate-400">正在加载策略总览、大盘实时状态与每日流程...</div>
      </div>
    );
  }

  const acct = data?.account || {};
  const reg = data?.regime || {};
  const wm = data?.risk_watermarks || {};
  const positions = data?.positions || [];

  const regimeCn =
    reg.regime === 'OFFENSIVE'
      ? '强势进攻'
      : reg.regime === 'DEFENSIVE'
        ? '弱势防守'
        : reg.regime === 'NEUTRAL'
          ? '震荡中性'
          : '待同步';

  const regimeColorClass =
    reg.regime === 'OFFENSIVE'
      ? 'text-rose-400'
      : reg.regime === 'DEFENSIVE'
        ? 'text-emerald-400'
        : 'text-amber-400';

  const posColumns = [
    {
      title: '代码 / 名称',
      dataIndex: 'ts_code',
      render: (_: any, r: any) => (
        <div>
          <span className="num font-semibold text-slate-100">{r.ts_code}</span>
          <span className="ml-2 text-slate-300">{r.name}</span>
          <Tag className="ml-2 text-[11px]" color="geekblue">{r.sw_l1_name}</Tag>
        </div>
      ),
    },
    {
      title: '持仓 / 可卖(T+1)',
      dataIndex: 'total_shares',
      align: 'right' as const,
      render: (_: any, r: any) => (
        <span className="num">
          {r.total_shares} / <span className="text-sky-400">{r.sellable_shares}</span>
        </span>
      ),
    },
    {
      title: '成本 / 现价',
      align: 'right' as const,
      render: (_: any, r: any) => (
        <span className="num">
          {fmtNum(r.avg_cost)} / <span className="text-slate-100 font-semibold">{fmtNum(r.current_price)}</span>
        </span>
      ),
    },
    {
      title: '止损位',
      dataIndex: 'stop_loss_price',
      align: 'right' as const,
      render: (v: number) => <span className="num text-emerald-400">¥{fmtNum(v)}</span>,
    },
    {
      title: '仓位',
      dataIndex: 'weight',
      align: 'right' as const,
      render: (v: number) => <span className="num">{(v * 100).toFixed(2)}%</span>,
    },
    {
      title: '浮动盈亏',
      dataIndex: 'unrealized_pnl',
      align: 'right' as const,
      render: (v: number, r: any) => (
        <span className={`num font-semibold ${aShareColor(v)}`}>
          {v >= 0 ? '+' : ''}
          {fmtNum(v)} ({fmtPct(r.unrealized_pnl_pct)})
        </span>
      ),
    },
  ];

  return (
    <div className="space-y-4">
      {/* Header Row */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-slate-100 m-0">策略总览与大盘状态仪表盘</h1>
          <div className="text-xs text-slate-400 mt-0.5 flex flex-wrap items-center gap-2">
            <span>
              默认策略：20 因子增强 + Huber 10日 6-Seed Bagging + 强牛120日新高突破叠加 + 抗锁仓11%/12%回撤风控 + 双指数二八分化择时 · DuckDB 完整日线 {dataStatus?.stocks_synced ?? 0} 只真股 ({dataStatus?.date_range?.[0]} ~ {dataStatus?.date_range?.[1]})
            </span>
            {dataStatus?.partial_trailing_date && (
              <Tag color="warning" className="m-0 text-[11px]">
                {dataStatus.partial_trailing_date} 仅同步 {dataStatus.partial_trailing_stocks} 股（建议点击收盘增量同步）
              </Tag>
            )}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Popconfirm
            title="启动收盘日线与大盘状态增量同步？"
            description="将补齐最新完整交易日全市场日线、中证1000/沪深300指数、资金流并重算因子与大盘择时状态。"
            onConfirm={handleIncrementalSync}
            okText="开始增量同步"
            cancelText="取消"
          >
            <Button
              icon={<DatabaseOutlined />}
              loading={dashTask.runningKind === 'data_sync_incremental'}
              disabled={dashTask.isRunning}
            >
              收盘增量同步
            </Button>
          </Popconfirm>
          <Popconfirm
            title="启动免 Token 全市场活跃 A 股断点续传同步？"
            description="将扫描全部 4,600+ 只 A 股，自动保留成交额 ≥ 4000 万的活跃股并发拉取近 2 年日线与全市场 PIT 财报（已同步股票自动跳过）。"
            onConfirm={handleFullMarketSync}
            okText="开始全市场同步"
            cancelText="取消"
          >
            <Button
              icon={<CloudDownloadOutlined />}
              loading={dashTask.runningKind === 'data_sync_active_market'}
              disabled={dashTask.isRunning}
            >
              全市场活跃股同步
            </Button>
          </Popconfirm>
          <Button
            icon={<SyncOutlined spin={dashTask.runningKind === 'paper_intraday'} />}
            type="primary"
            onClick={handleIntradayRefresh}
            loading={dashTask.runningKind === 'paper_intraday'}
            disabled={dashTask.isRunning && dashTask.runningKind !== 'paper_intraday'}
          >
            盘中快照盯市刷新
          </Button>
          <Button icon={<ReloadOutlined spin={reloading} />} loading={reloading} onClick={() => load(true)}>
            重新加载
          </Button>
        </div>
      </div>

      <TaskProgressBanner task={dashTask.activeTask} />
      {reloading && !dashTask.isRunning && (
        <div className="quant-card px-3 py-2 border border-sky-500/30 bg-sky-950/20">
          <div className="text-xs text-sky-300 mb-1">正在重新加载总览数据与行情状态...</div>
          <Progress percent={70} size="small" status="active" showInfo={false} strokeColor="#38BDF8" trailColor="#1E293B" />
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="quant-card p-4">
          <div className="text-xs text-slate-400">模拟实盘总资产 (CNY)</div>
          <div className="num text-2xl font-bold text-slate-100 mt-1">¥{fmtNum(acct.total_equity)}</div>
          <div className="mt-2 flex items-center justify-between text-xs">
            <span className="text-slate-400">累计收益率</span>
            <span className={`num font-semibold ${aShareColor(acct.total_return)}`}>{fmtPct(acct.total_return)}</span>
          </div>
        </div>

        <div className="quant-card p-4">
          <div className="text-xs text-slate-400">持仓市值 / 可用现金</div>
          <div className="num text-2xl font-bold text-sky-400 mt-1">¥{fmtNum(acct.market_value)}</div>
          <div className="mt-2 flex items-center justify-between text-xs">
            <span className="text-slate-400">现金 ¥{fmtNum(acct.cash)}</span>
            <span className="num text-slate-300">{positions.length} 只持股</span>
          </div>
          {acct.total_equity > 0 && acct.cash / acct.total_equity >= 0.2 && (
            <div className="mt-1 text-[11px] text-amber-300/90">
              闲置现金 {((acct.cash / acct.total_equity) * 100).toFixed(0)}% · 实盘可停泊货币ETF/逆回购增厚
            </div>
          )}
        </div>

        <div className="quant-card p-4">
          <div className="flex items-center justify-between gap-1 text-xs text-slate-400">
            <span>大盘 Regime 择时 ({reg.trade_date || '-'})</span>
            <div className="flex items-center gap-1">
              {reg.style_mode === 'LARGE_CAP_RESCUE' && (
                <Tag color="purple" className="m-0 text-[10px] leading-4 px-1.5">
                  二八分化·切大中盘
                </Tag>
              )}
              <Tag color={reg.is_realtime ? 'processing' : 'default'} className="m-0 text-[10px] leading-4 px-1.5">
                {reg.is_realtime ? `盘中实时 ${reg.quote_time || ''}` : '收盘确认'}
              </Tag>
            </div>
          </div>
          <div className={`text-2xl font-bold mt-1 ${regimeColorClass}`}>
            {regimeCn} · 目标 {Math.round((reg.target_exposure || 0) * 100)}%
          </div>
          <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
            <span>
              中证1000 <b className="num text-slate-200">{fmtNum(reg.csi1000_close)}</b>
              {reg.csi1000_pct_chg !== undefined && reg.csi1000_pct_chg !== null && (
                <span className={`num ml-1 ${aShareColor(reg.csi1000_pct_chg)}`}>
                  {reg.csi1000_pct_chg >= 0 ? '+' : ''}
                  {Number(reg.csi1000_pct_chg).toFixed(2)}%
                </span>
              )}
            </span>
            <span>
              MA20宽度 <b className="num text-amber-400">{reg.breadth_above_ma20 !== null && reg.breadth_above_ma20 !== undefined ? `${(reg.breadth_above_ma20 * 100).toFixed(2)}%` : '-'}</b>
            </span>
          </div>
          {reg.advancers !== undefined && reg.decliners !== undefined && (
            <div className="mt-1 flex items-center justify-between text-[11px] text-slate-500">
              <span>
                涨 <b className="num text-rose-400">{reg.advancers}</b> / 跌 <b className="num text-emerald-400">{reg.decliners}</b>
              </span>
              <span>已采样 {reg.quoted_count ?? '-'} 股</span>
            </div>
          )}
        </div>

        <div className="quant-card p-4">
          <div className="text-xs text-slate-400">组合当前回撤 & 风控熔断</div>
          <div className={`num text-2xl font-bold mt-1 ${aShareColor(acct.current_drawdown)}`}>
            {fmtPct(acct.current_drawdown)}
          </div>
          <div className="mt-2 flex items-center justify-between text-xs">
            <span className="text-slate-400">预警 {fmtPct(wm.drawdown_warn, 0)} / 熔断 {fmtPct(wm.drawdown_halt, 0)}</span>
            <Tag color={wm.fuse_status === 'HALTED' ? 'error' : wm.fuse_status === 'WARNING' ? 'warning' : 'cyan'} className="m-0">
              {wm.fuse_status || 'NORMAL'}
            </Tag>
          </div>
        </div>
      </div>

      {/* Daily Fixed Trading Workflow Flowchart & Live Execution Status */}
      <DailyWorkflowCard
        initialWorkflow={data?.daily_workflow || null}
        onStepCompleted={() => load(false)}
      />

      {/* Main Charts */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="quant-card p-4 xl:col-span-2">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold text-slate-200">
              净值走势 vs 基准 (中证1000 / 沪深300)
            </span>
            <span className="text-xs text-slate-400">
              {(data?.paper_nav_curve?.length || 0) >= 2 ? '来源：实时模拟盘日净值' : '来源：最新 Walk-Forward 样本外参考曲线'}
            </span>
          </div>
          <EChartsBox option={navOption} height={310} />
        </div>

        <div className="quant-card p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold text-slate-200">中证1000 与全市场宽度 (近60日)</span>
          </div>
          <EChartsBox option={regimeOption} height={310} />
        </div>
      </div>

      {/* Holdings & Risk Watermarks */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="quant-card p-4 xl:col-span-2">
          <div className="text-sm font-semibold text-slate-200 mb-3">当日持仓明细与止损水位</div>
          <Table
            rowKey="ts_code"
            size="small"
            pagination={false}
            dataSource={positions}
            columns={posColumns}
            locale={{ emptyText: '当前暂无持仓，可在「模拟实盘与复盘」页触发 14:40 调仓' }}
          />
        </div>

        <div className="quant-card p-4 space-y-4">
          <div className="text-sm font-semibold text-slate-200">前置风控实时水位</div>
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">单票最高仓位</span>
              <span className="num text-slate-200">
                {((wm.max_single_weight || 0) * 100).toFixed(2)}% / {((wm.single_limit || 0.12) * 100).toFixed(0)}%
              </span>
            </div>
            <Progress
              percent={Math.min(100, Math.round(((wm.max_single_weight || 0) / (wm.single_limit || 0.12)) * 100))}
              strokeColor="#38BDF8"
              trailColor="#1E293B"
              showInfo={false}
            />
          </div>
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">单申万一级行业最高 ({wm.max_industry_name || '无'})</span>
              <span className="num text-slate-200">
                {((wm.max_industry_weight || 0) * 100).toFixed(2)}% / {((wm.industry_limit || 0.25) * 100).toFixed(0)}%
              </span>
            </div>
            <Progress
              percent={Math.min(100, Math.round(((wm.max_industry_weight || 0) / (wm.industry_limit || 0.25)) * 100))}
              strokeColor="#F59E0B"
              trailColor="#1E293B"
              showInfo={false}
            />
          </div>
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">组合回撤熔断消耗</span>
              <span className="num text-slate-200">
                {Math.abs((wm.current_drawdown || 0) * 100).toFixed(2)}% / {Math.abs((wm.drawdown_halt || -0.12) * 100).toFixed(0)}%
              </span>
            </div>
            <Progress
              percent={Math.min(100, Math.round((Math.abs(wm.current_drawdown || 0) / Math.abs(wm.drawdown_halt || 0.12)) * 100))}
              strokeColor="#F43F5E"
              trailColor="#1E293B"
              showInfo={false}
            />
          </div>
          <div className="pt-2 border-t border-slate-800 text-xs text-slate-400 space-y-1">
            <div>DuckDB 真实日线: <span className="num text-slate-200">{fmtNum(dataStatus?.daily_bar_rows, 0)}</span> 行</div>
            <div>PIT 财报公告库: <span className="num text-slate-200">{fmtNum(dataStatus?.financial_pit_rows, 0)}</span> 条</div>
            <div>
              主力资金流状态:{' '}
              <Tag color={dataStatus?.capital_flow_status === 'OK' ? 'success' : 'warning'} className="m-0">
                {dataStatus?.capital_flow_status || '-'}
              </Tag>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
