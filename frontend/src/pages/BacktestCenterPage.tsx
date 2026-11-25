import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Button,
  Form,
  Input,
  InputNumber,
  Popconfirm,
  Progress,
  Select,
  Spin,
  Switch,
  Table,
  Tag,
  Tooltip,
  message,
} from 'antd';
import {
  ClearOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import { aShareColor, api, fmtNum, fmtPct } from '../services/api';
import { useAlphaWebSocket } from '../services/ws';
import { EChartsBox } from '../components/charts/EChartsBox';

interface RunningJobState {
  jobId: string;
  name: string;
  stage: string;
  pct: number;
  elapsedSec: number;
}

const PRESETS: Array<{ label: string; values: Record<string, any> }> = [
  {
    label: '最优动态+Huber10日Bag6+强牛动量(推荐)',
    values: {
      name: '最优动态持有-Huber10日Bag6+强牛动量倾斜',
      model_mode: 'regression',
      ensemble_seeds_count: 6,
      weight_method: 'risk_parity',
      holding_mode: 'dynamic',
      min_hold_days: 10,
      buffer_rank: 35,
      deep_exit_rank: 150,
      exit_confirm_days: 3,
      swap_score_gap: 0.7,
      profit_ride_pct: 0.1,
      profit_protect_pct: 0.15,
      stop_cooldown_days: 5,
      target_holdings: 10,
      hard_stop_pct: 0.08,
      atr_stop_mult: 2.5,
      slippage_bps: 10,
      bull_mom_h120_weight: 0.1,
      bull_mom_ret20_weight: 0.08,
      enable_regime_timing: true,
      enable_technical_filter: true,
      include_heatmap: true,
    },
  },
  {
    label: '对照:纯模型(关闭强牛动量)',
    values: {
      name: '对照实验-Huber10日Bag6纯模型(关强牛动量)',
      model_mode: 'regression',
      ensemble_seeds_count: 6,
      weight_method: 'risk_parity',
      holding_mode: 'dynamic',
      min_hold_days: 10,
      buffer_rank: 35,
      deep_exit_rank: 150,
      exit_confirm_days: 3,
      swap_score_gap: 0.7,
      profit_ride_pct: 0.1,
      profit_protect_pct: 0.15,
      stop_cooldown_days: 5,
      target_holdings: 10,
      hard_stop_pct: 0.08,
      atr_stop_mult: 2.5,
      slippage_bps: 10,
      bull_mom_h120_weight: 0,
      bull_mom_ret20_weight: 0,
      enable_regime_timing: true,
      enable_technical_filter: true,
      include_heatmap: false,
    },
  },
  {
    label: '对照:固定10日',
    values: {
      name: '对照实验-固定10日调仓',
      holding_mode: 'fixed',
      rebalance_days: 10,
      buffer_rank: 25,
      include_heatmap: false,
    },
  },
  {
    label: '对照:固定5日',
    values: {
      name: '对照实验-固定5日调仓',
      holding_mode: 'fixed',
      rebalance_days: 5,
      buffer_rank: 25,
      include_heatmap: false,
    },
  },
  {
    label: '消融:关择时',
    values: {
      name: '消融实验-动态持有(关大盘择时)',
      holding_mode: 'dynamic',
      enable_regime_timing: false,
      enable_technical_filter: true,
      include_heatmap: false,
    },
  },
  {
    label: '压力:20bps滑点',
    values: {
      name: '压力测试-动态持有(20bps高滑点)',
      holding_mode: 'dynamic',
      slippage_bps: 20,
      include_heatmap: false,
    },
  },
];

export const BacktestCenterPage: React.FC = () => {
  const [runs, setRuns] = useState<any[]>([]);
  const [activeRun, setActiveRun] = useState<any>(null);
  const [cacheStatus, setCacheStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [job, setJob] = useState<RunningJobState | null>(null);
  const [form] = Form.useForm();
  const holdingMode = Form.useWatch('holding_mode', form) || 'dynamic';
  const jobRef = useRef<RunningJobState | null>(null);
  jobRef.current = job;

  const refreshCache = async () => {
    try {
      const cs = await api.getBacktestCacheStatus();
      setCacheStatus(cs);
    } catch {
      // ignore
    }
  };

  const loadRuns = async (pickId?: string) => {
    try {
      const list = await api.listBacktestRuns();
      setRuns(list);
      const doneList = list.filter((r: any) => r.status === 'DONE');
      const id = pickId || doneList[0]?.id || list[0]?.id;
      if (id) {
        const detail = await api.getBacktestRun(id);
        if (detail.status === 'DONE' || !activeRun) {
          setActiveRun(detail);
        }
      } else {
        setActiveRun(null);
      }
    } catch {
      // ignore initial empty
    } finally {
      setLoading(false);
    }
  };

  const finishJob = async (runId: string, elapsed?: number, cacheInfo?: any) => {
    setJob(null);
    await Promise.all([loadRuns(runId), refreshCache()]);
    const hitMsg = cacheInfo?.ml_cache_hit
      ? '（命中两级缓存 · 秒级完成）'
      : cacheInfo?.factor_cache_hit
        ? '（命中全市场因子缓存）'
        : '';
    const secMsg = elapsed ? `，耗时 ${elapsed}s` : '';
    message.success(`全市场 Walk-Forward 精确回测完成${secMsg}${hitMsg}`);
  };

  // Real-time WebSocket progress updates
  useAlphaWebSocket((evt) => {
    if (evt.channel !== 'task_progress' || evt.payload?.mode !== 'backtest') return;
    const p = evt.payload;
    if (p.stage === 'done') {
      finishJob(p.run_id || p.task_id, p.elapsed_sec, p.cache_info);
    } else if (p.stage === 'error') {
      setJob(null);
      message.error(`回测执行失败: ${p.error || '未知错误'}`);
      loadRuns();
    } else {
      setJob({
        jobId: p.run_id || p.task_id,
        name: p.name || jobRef.current?.name || '全市场回测',
        stage: p.stage || '运行中...',
        pct: Math.round((p.pct || 0) * 100),
        elapsedSec: Number(p.elapsed_sec || 0),
      });
    }
  });

  // Initial load + recover any in-progress background backtest job
  useEffect(() => {
    loadRuns();
    refreshCache();
    api
      .listBacktestJobs()
      .then((jobs: any[]) => {
        const active = (jobs || []).find((j) => j.status === 'RUNNING');
        if (active) {
          setJob({
            jobId: active.job_id,
            name: active.name || '全市场回测',
            stage: active.stage || '后台计算中...',
            pct: Math.round((active.pct || 0) * 100),
            elapsedSec: Number(active.elapsed_sec || 0),
          });
        }
      })
      .catch(() => {});
  }, []);

  // Reliable 2.5s polling fallback while a job is active (guarantees zero missed completions)
  useEffect(() => {
    if (!job?.jobId) return;
    const timer = window.setInterval(async () => {
      try {
        const st = await api.getBacktestJob(job.jobId);
        if (st.status === 'DONE') {
          window.clearInterval(timer);
          finishJob(st.run_id || job.jobId, st.elapsed_sec, st.cache_info);
        } else if (st.status === 'FAILED') {
          window.clearInterval(timer);
          setJob(null);
          message.error(`回测执行失败: ${st.error || '未知错误'}`);
          loadRuns();
        } else {
          setJob({
            jobId: st.job_id,
            name: st.name || job.name,
            stage: st.stage || '后台计算中...',
            pct: Math.round((st.pct || 0) * 100),
            elapsedSec: Number(st.elapsed_sec || 0),
          });
        }
      } catch {
        // ignore transient poll glitch
      }
    }, 2500);
    return () => window.clearInterval(timer);
  }, [job?.jobId]);

  const handleRun = async () => {
    try {
      const vals = await form.validateFields();
      const payload = {
        ...vals,
        start_date: vals.start_date || null,
        end_date: vals.end_date || null,
        use_cache: true,
      };
      const res = await api.runBacktestAsync(payload);
      setJob({
        jobId: res.job_id,
        name: vals.name || '全市场回测',
        stage: res.stage || '正在启动全市场回测任务...',
        pct: Math.round((res.pct || 0.02) * 100),
        elapsedSec: 0,
      });
      message.info('已启动后台异步回测任务（零超时，进度实时推送）');
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '启动回测失败');
    }
  };

  const handleClearCache = async () => {
    try {
      await api.clearBacktestCache();
      await refreshCache();
      message.success('已清空研究缓存，下次运行将重新计算全市场因子与模型');
    } catch {
      message.error('清空缓存失败');
    }
  };

  const handleDeleteRun = async (id: string) => {
    try {
      await api.deleteBacktestRun(id);
      message.success('已删除该回测记录');
      await loadRuns(activeRun?.id === id ? undefined : activeRun?.id);
    } catch {
      message.error('删除记录失败');
    }
  };

  const equityOption = useMemo(() => {
    const eq = activeRun?.equity_curve || [];
    const compEq = activeRun?.metrics?.comparison?.equity_curve || [];
    const compLabel =
      activeRun?.metrics?.comparison?.method === 'equal_weight' ? '对照: 等权组合' : '对照: 风险平价';
    return {
      tooltip: { trigger: 'axis' as const },
      legend: {
        top: 0,
        textStyle: { color: '#CBD5E1', fontSize: 11 },
        data: ['当前策略净值', compLabel, '中证1000', '沪深300'],
      },
      grid: { left: 48, right: 20, top: 34, bottom: 26 },
      xAxis: { type: 'category' as const, data: eq.map((x: any) => x.date), axisLine: { lineStyle: { color: '#334155' } } },
      yAxis: { type: 'value' as const, scale: true, splitLine: { lineStyle: { color: '#1E293B' } } },
      series: [
        {
          name: '当前策略净值',
          type: 'line' as const,
          showSymbol: false,
          lineStyle: { color: '#F43F5E', width: 2.5 },
          data: eq.map((x: any) => x.nav),
        },
        ...(compEq.length
          ? [
              {
                name: compLabel,
                type: 'line' as const,
                showSymbol: false,
                lineStyle: { color: '#F59E0B', width: 1.8, type: 'dashed' as const },
                data: compEq.map((x: any) => x.nav),
              },
            ]
          : []),
        {
          name: '中证1000',
          type: 'line' as const,
          showSymbol: false,
          lineStyle: { color: '#38BDF8', width: 1.8 },
          data: eq.map((x: any) => x.csi1000_nav),
        },
        {
          name: '沪深300',
          type: 'line' as const,
          showSymbol: false,
          lineStyle: { color: '#94A3B8', width: 1.4, type: 'dotted' as const },
          data: eq.map((x: any) => x.csi300_nav),
        },
      ],
    };
  }, [activeRun]);

  const drawdownOption = useMemo(() => {
    const dd = activeRun?.drawdown_curve || [];
    const eq = activeRun?.equity_curve || [];
    return {
      tooltip: { trigger: 'axis' as const },
      legend: { top: 0, textStyle: { color: '#CBD5E1', fontSize: 11 }, data: ['策略动态回撤(%)', '相对中证1000超额(%)'] },
      grid: { left: 48, right: 48, top: 30, bottom: 24 },
      xAxis: { type: 'category' as const, data: dd.map((x: any) => x.date), axisLine: { lineStyle: { color: '#334155' } } },
      yAxis: [
        { type: 'value' as const, splitLine: { lineStyle: { color: '#1E293B' } }, axisLabel: { formatter: '{value}%' } },
        { type: 'value' as const, splitLine: { show: false }, axisLabel: { formatter: '{value}%' } },
      ],
      series: [
        {
          name: '策略动态回撤(%)',
          type: 'line' as const,
          showSymbol: false,
          lineStyle: { color: '#10B981', width: 1.5 },
          areaStyle: { color: 'rgba(16,185,129,0.22)' },
          data: dd.map((x: any) => Number((x.drawdown * 100).toFixed(2))),
        },
        {
          name: '相对中证1000超额(%)',
          type: 'line' as const,
          yAxisIndex: 1,
          showSymbol: false,
          lineStyle: { color: '#38BDF8', width: 1.8 },
          data: eq.map((x: any) => Number(((x.excess_1000 || 0) * 100).toFixed(2))),
        },
      ],
    };
  }, [activeRun]);

  const heatmapOption = useMemo(() => {
    const hm = activeRun?.wf_heatmap;
    if (!hm?.cells?.length) return null;
    const xs = hm.x_values.map((v: any) => `${hm.x_param}=${v}`);
    const ys = hm.y_values.map((v: any) => `${hm.y_param}=${v}`);
    const vals = hm.cells.map((c: any) => c.value);
    return {
      tooltip: {
        formatter: (p: any) => {
          const cell = hm.cells.find((c: any) => c.x_index === p.value[0] && c.y_index === p.value[1]);
          return `${xs[p.value[0]]} / ${ys[p.value[1]]}<br/>夏普: <b>${cell?.sharpe_ratio}</b><br/>年化: ${fmtPct(cell?.annualized_return)}<br/>最大回撤: ${fmtPct(cell?.max_drawdown)}`;
        },
      },
      grid: { left: 120, right: 30, top: 20, bottom: 45 },
      xAxis: { type: 'category' as const, data: xs },
      yAxis: { type: 'category' as const, data: ys },
      visualMap: {
        min: Math.min(...vals),
        max: Math.max(...vals),
        orient: 'horizontal' as const,
        left: 'center',
        bottom: 0,
        inRange: { color: ['#0F766E', '#1E293B', '#E11D48'] },
        textStyle: { color: '#94A3B8', fontSize: 10 },
      },
      series: [
        {
          type: 'heatmap' as const,
          data: hm.cells.map((c: any) => [c.x_index, c.y_index, c.value]),
          label: { show: true, color: '#F8FAFC', fontSize: 12 },
        },
      ],
    };
  }, [activeRun]);

  const fiOption = useMemo(() => {
    const fi = activeRun?.metrics?.feature_importance || {};
    const entries = Object.entries(fi)
      .map(([k, v]: [string, any]) => ({ name: k, shap: Number(((v.shap || 0) * 100).toFixed(2)), gain: Number(((v.gain || 0) * 100).toFixed(2)) }))
      .sort((a, b) => a.shap - b.shap)
      .slice(-10);
    return {
      tooltip: { trigger: 'axis' as const },
      legend: { top: 0, textStyle: { color: '#CBD5E1', fontSize: 11 }, data: ['TreeSHAP(%)', 'Gain(%)'] },
      grid: { left: 12, right: 20, top: 28, bottom: 20, containLabel: true },
      xAxis: { type: 'value' as const, splitLine: { lineStyle: { color: '#1E293B' } } },
      yAxis: { type: 'category' as const, data: entries.map((e) => e.name), axisLabel: { fontSize: 11 } },
      series: [
        { name: 'TreeSHAP(%)', type: 'bar' as const, itemStyle: { color: '#F43F5E' }, data: entries.map((e) => e.shap) },
        { name: 'Gain(%)', type: 'bar' as const, itemStyle: { color: '#38BDF8' }, data: entries.map((e) => e.gain) },
      ],
    };
  }, [activeRun]);

  const m = activeRun?.metrics || {};
  const doneRuns = runs.filter((r) => r.status === 'DONE');

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-slate-100 m-0">回测与样本外优化中心</h1>
          <div className="text-xs text-slate-400 mt-0.5">
            后台异步零超时 · 两级全市场研究缓存（首次滚动训练后，调参/消融实验秒级出结果）· A股 T+1 / 涨跌停 / 百股 / 真实费用滑点
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {cacheStatus && (
            <div className="flex items-center gap-1.5 text-xs bg-[#0F1624] border border-slate-800 px-2.5 py-1 rounded">
              <ThunderboltOutlined className={cacheStatus.cached_ml_modes?.length ? 'text-amber-400' : 'text-slate-500'} />
              <span className="text-slate-300">
                全库 <b className="num text-sky-400">{cacheStatus.stocks_count}</b> 股
              </span>
              {cacheStatus.factor_panel_cached ? (
                <Tag color="cyan" className="m-0">因子已缓存</Tag>
              ) : (
                <Tag color="default" className="m-0">未预热</Tag>
              )}
              {cacheStatus.cached_ml_modes?.map((mode: string) => (
                <Tag key={mode} color="gold" className="m-0">
                  {mode} 秒级就绪
                </Tag>
              ))}
              {(cacheStatus.factor_panel_cached || cacheStatus.cached_ml_modes?.length > 0) && (
                <Tooltip title="清空内存缓存（DuckDB 更新行情后会自动失效）">
                  <Button type="text" size="small" icon={<ClearOutlined />} onClick={handleClearCache} />
                </Tooltip>
              )}
            </div>
          )}
          {doneRuns.length > 0 && (
            <Select
              style={{ width: 290 }}
              value={activeRun?.id}
              onChange={(id) => loadRuns(id)}
              options={doneRuns.map((r) => ({
                value: r.id,
                label: `${r.name} (${r.model_mode}/${r.weight_method}) · 夏普 ${r.sharpe_ratio ?? '-'}`,
              }))}
            />
          )}
        </div>
      </div>

      {/* Live Async Backtest Progress Banner */}
      {job && (
        <div className="quant-card p-3 border border-sky-500/40 bg-sky-950/20">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs mb-1.5">
            <span className="text-sky-300 font-medium">
              正在后台执行「{job.name}」：{job.stage}
            </span>
            <span className="num text-slate-300">
              {job.pct}% · 已耗时 {job.elapsedSec.toFixed(1)}s
            </span>
          </div>
          <Progress percent={job.pct} size="small" status="active" showInfo={false} strokeColor="#38BDF8" />
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
        {/* Parameter Form */}
        <div className="quant-card p-4 xl:col-span-3 space-y-3">
          <div className="text-sm font-semibold text-slate-200">策略与风控参数配置</div>

          {/* 1-Click Experiment Presets */}
          <div>
            <div className="text-[11px] text-slate-400 mb-1.5">一键实验预设（点击自动填参）：</div>
            <div className="flex flex-wrap gap-1">
              {PRESETS.map((p) => (
                <Tag
                  key={p.label}
                  color="geekblue"
                  className="cursor-pointer m-0 hover:opacity-80"
                  onClick={() => form.setFieldsValue(p.values)}
                >
                  {p.label}
                </Tag>
              ))}
            </div>
          </div>

          <Form
            form={form}
            layout="vertical"
            size="small"
            initialValues={{
              name: '最优动态持有-Huber10日Bag6+强牛动量(25年1月至今)',
              start_date: '2025-01-01',
              end_date: '',
              model_mode: 'regression',
              ensemble_seeds_count: 6,
              weight_method: 'risk_parity',
              holding_mode: 'dynamic',
              rebalance_days: 10,
              min_hold_days: 10,
              deep_exit_rank: 150,
              exit_confirm_days: 3,
              swap_score_gap: 0.7,
              profit_ride_pct: 0.1,
              profit_protect_pct: 0.15,
              stop_cooldown_days: 5,
              target_holdings: 10,
              buffer_rank: 35,
              hard_stop_pct: 0.08,
              atr_stop_mult: 2.5,
              slippage_bps: 10,
              bull_mom_h120_weight: 0.1,
              bull_mom_ret20_weight: 0.08,
              cash_yield_rate: 0,
              enable_regime_timing: true,
              enable_technical_filter: true,
              include_heatmap: true,
            }}
          >
            <Form.Item name="name" label="实验名称（便于多轮对比）" rules={[{ required: true, message: '请输入实验名称' }]}>
              <Input placeholder="例如：最优动态持有-Huber回归(25年1月至今)" />
            </Form.Item>
            <div className="mb-2">
              <div className="text-[11px] text-slate-400 mb-1">快捷回测区间：</div>
              <div className="flex flex-wrap gap-1">
                {[
                  { label: '25年1月至今', s: '2025-01-01', e: '' },
                  { label: '全部样本外', s: '', e: '' },
                  { label: '仅2025年', s: '2025-01-01', e: '2025-12-31' },
                  { label: '仅2026年', s: '2026-01-01', e: '' },
                ].map((rg) => (
                  <Tag
                    key={rg.label}
                    color="cyan"
                    className="cursor-pointer m-0 hover:opacity-80"
                    onClick={() => form.setFieldsValue({ start_date: rg.s, end_date: rg.e })}
                  >
                    {rg.label}
                  </Tag>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Form.Item name="start_date" label="回测开始日期">
                <Input type="date" />
              </Form.Item>
              <Form.Item name="end_date" label="结束日期(空=最新)">
                <Input type="date" />
              </Form.Item>
            </div>
            <Form.Item name="holding_mode" label="持有与调仓策略">
              <Select
                options={[
                  { value: 'dynamic', label: '最优动态持有 (事件驱动 · 推荐)' },
                  { value: 'fixed', label: '固定周期调仓 (对照模式)' },
                ]}
              />
            </Form.Item>
            <div className="grid grid-cols-3 gap-2">
              <Form.Item name="model_mode" label="LightGBM 模式 (10日)" className="col-span-2">
                <Select options={[{ value: 'regression', label: 'Huber 回归 (推荐)' }, { value: 'lambdarank', label: 'LambdaRank (对照)' }]} />
              </Form.Item>
              <Form.Item name="ensemble_seeds_count" label="集成种子">
                <Select
                  options={[
                    { value: 6, label: '6-Seed(推荐)' },
                    { value: 3, label: '3-Seed(快)' },
                    { value: 12, label: '12-Seed(极稳)' },
                    { value: 1, label: '1-Seed' },
                  ]}
                />
              </Form.Item>
            </div>
            <Form.Item name="weight_method" label="组合优化权重">
              <Select options={[{ value: 'risk_parity', label: '风险平价 (Risk Parity)' }, { value: 'equal_weight', label: '等权组合 (Equal Weight)' }]} />
            </Form.Item>
            {holdingMode === 'dynamic' ? (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <Form.Item name="min_hold_days" label="最小持有期(日)">
                    <InputNumber min={1} max={30} className="w-full" />
                  </Form.Item>
                  <Form.Item name="target_holdings" label="目标持股数">
                    <InputNumber min={5} max={20} className="w-full" />
                  </Form.Item>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Form.Item name="buffer_rank" label="常规缓冲带排名">
                    <InputNumber min={10} max={80} className="w-full" />
                  </Form.Item>
                  <Form.Item name="deep_exit_rank" label="深度淘汰排名">
                    <InputNumber min={40} max={500} step={10} className="w-full" />
                  </Form.Item>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Form.Item name="exit_confirm_days" label="走弱确认(日)">
                    <InputNumber min={1} max={10} className="w-full" />
                  </Form.Item>
                  <Form.Item name="stop_cooldown_days" label="止损冷静期(日)">
                    <InputNumber min={0} max={15} className="w-full" />
                  </Form.Item>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Form.Item name="swap_score_gap" label="换仓分差(σ)">
                    <InputNumber min={0} max={2} step={0.1} className="w-full" />
                  </Form.Item>
                  <Form.Item name="profit_ride_pct" label="浮盈奔跑阈值">
                    <InputNumber min={0.03} max={0.5} step={0.02} className="w-full" />
                  </Form.Item>
                </div>
              </>
            ) : (
              <div className="grid grid-cols-2 gap-2">
                <Form.Item name="rebalance_days" label="固定调仓周期(日)">
                  <InputNumber min={1} max={20} className="w-full" />
                </Form.Item>
                <Form.Item name="target_holdings" label="目标持股数">
                  <InputNumber min={5} max={20} className="w-full" />
                </Form.Item>
              </div>
            )}
            <div className="grid grid-cols-2 gap-2">
              {holdingMode === 'fixed' ? (
                <Form.Item name="buffer_rank" label="换手缓冲带排名">
                  <InputNumber min={10} max={80} className="w-full" />
                </Form.Item>
              ) : (
                <Form.Item name="profit_protect_pct" label="保本止盈阈值">
                  <InputNumber min={0.05} max={0.5} step={0.02} className="w-full" />
                </Form.Item>
              )}
              <Form.Item name="slippage_bps" label="滑点 (bps)">
                <InputNumber min={0} max={50} className="w-full" />
              </Form.Item>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Form.Item name="hard_stop_pct" label="硬止损比例">
                <InputNumber min={0.03} max={0.2} step={0.01} className="w-full" />
              </Form.Item>
              <Form.Item name="atr_stop_mult" label="ATR止损倍数">
                <InputNumber min={1.0} max={5.0} step={0.5} className="w-full" />
              </Form.Item>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Form.Item
                name="bull_mom_h120_weight"
                label={<Tooltip title="仅在 OFFENSIVE 强牛状态生效：给模型打分温和叠加距120日新高 Z-Score（震荡/熊市=0）">强牛新高倾斜</Tooltip>}
              >
                <InputNumber min={0} max={0.5} step={0.02} className="w-full" />
              </Form.Item>
              <Form.Item
                name="bull_mom_ret20_weight"
                label={<Tooltip title="仅在 OFFENSIVE 强牛状态生效：给模型打分温和叠加20日动量 Z-Score（震荡/熊市=0）">强牛动量倾斜</Tooltip>}
              >
                <InputNumber min={0} max={0.5} step={0.02} className="w-full" />
              </Form.Item>
            </div>
            <Form.Item
              name="cash_yield_rate"
              label={<Tooltip title="默认 0% 仅统计纯股票 Alpha；可切换 1.5%/1.8% 查看空仓期闲置现金配置货币ETF/国债逆回购的真实底薪视角">闲置现金货币底薪</Tooltip>}
            >
              <Select
                options={[
                  { value: 0, label: '0% · 纯股票 Alpha 口径 (默认)' },
                  { value: 0.015, label: '1.5% · 货币 ETF 保守底薪' },
                  { value: 0.018, label: '1.8% · 国债逆回购参考底薪' },
                ]}
              />
            </Form.Item>
            <div className="grid grid-cols-3 gap-1 mb-3">
              <Form.Item name="enable_regime_timing" label="大盘择时" valuePropName="checked" className="mb-0">
                <Switch size="small" />
              </Form.Item>
              <Form.Item
                name="enable_technical_filter"
                label={<Tooltip title="剔除近3日涨幅>20%过热与MACD顶背离，不误杀低位短期回调反转股">防追高过滤</Tooltip>}
                valuePropName="checked"
                className="mb-0"
              >
                <Switch size="small" />
              </Form.Item>
              <Form.Item name="include_heatmap" label="3×3热力图" valuePropName="checked" className="mb-0">
                <Switch size="small" />
              </Form.Item>
            </div>
            <Button type="primary" block icon={<PlayCircleOutlined />} loading={!!job} onClick={handleRun}>
              {job ? `后台计算中 (${job.pct}%)` : '运行全市场 Walk-Forward 回测'}
            </Button>
          </Form>

          <Alert
            type="info"
            showIcon
            className="text-[11px]"
            message="加速提示：首轮全库计算后自动缓存因子与样本外打分；后续调整仓位/滑点/止损/择时/共振开关时，关闭 3×3 热力图仅需约 3~6 秒即可出报告。"
          />
        </div>

        {/* Report & Charts */}
        <div className="xl:col-span-9 space-y-4">
          {loading ? (
            <div className="quant-card h-80 flex items-center justify-center"><Spin /></div>
          ) : !activeRun ? (
            <div className="quant-card p-10 text-center text-slate-400">
              尚未运行回测，请点击左侧「运行全市场 Walk-Forward 回测」基于 DuckDB 真实数据生成报告。
            </div>
          ) : (
            <>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {[
                  { label: '年化收益', val: fmtPct(m.annualized_return), cls: aShareColor(m.annualized_return) },
                  { label: '相对中证1000超额', val: fmtPct(m.excess_return_csi1000), cls: aShareColor(m.excess_return_csi1000) },
                  { label: '夏普 / 卡玛比率', val: `${fmtNum(m.sharpe_ratio)} / ${fmtNum(m.calmar_ratio)}`, cls: 'text-sky-400' },
                  { label: '最大回撤 (修复)', val: `${fmtPct(m.max_drawdown)} (${m.mdd_recovery_days ?? '-'}日)`, cls: 'text-emerald-400' },
                  { label: '胜率 / 盈亏比 / 换手', val: `${((m.win_rate || 0) * 100).toFixed(0)}% / ${fmtNum(m.profit_loss_ratio, 1)} / ${fmtNum(m.annualized_turnover, 1)}x`, cls: 'text-amber-400' },
                ].map((k) => (
                  <div key={k.label} className="quant-card p-3">
                    <div className="text-[11px] text-slate-400">{k.label}</div>
                    <div className={`num text-base font-bold mt-1 ${k.cls}`}>{k.val}</div>
                  </div>
                ))}
              </div>

              <div className="quant-card p-4">
                <div className="flex flex-wrap justify-between items-center gap-2 mb-2 text-xs text-slate-400">
                  <span className="text-sm font-semibold text-slate-200">
                    样本外净值曲线 · <span className="text-sky-400">{activeRun.name}</span>
                  </span>
                  <span>
                    样本外 RankIC: <b className="num text-sky-400">{fmtNum(m.oos_rank_ic, 4)}</b> | 累计费+滑: <b className="num text-amber-400">¥{fmtNum(m.total_cost)}</b>
                    {m.total_cash_interest > 0 ? <> | 货币底薪: <b className="num text-emerald-400">+¥{fmtNum(m.total_cash_interest)}</b></> : null}
                    {m.elapsed_sec ? <> | 耗时: <b className="num text-slate-300">{m.elapsed_sec}s</b></> : null}
                  </span>
                </div>
                <EChartsBox option={equityOption} height={290} />
              </div>

              <div className="quant-card p-4">
                <div className="text-sm font-semibold text-slate-200 mb-2">水下回撤与相对中证1000超额曲线</div>
                <EChartsBox option={drawdownOption} height={210} />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="quant-card p-4">
                  <div className="text-sm font-semibold text-slate-200 mb-2">Walk-Forward 参数敏感性热力图 (样本外夏普)</div>
                  {heatmapOption ? (
                    <EChartsBox option={heatmapOption} height={270} />
                  ) : (
                    <div className="text-slate-500 text-xs py-12 text-center">本次实验未勾选 3×3 参数热力图（可在左侧开启后重跑）</div>
                  )}
                </div>
                <div className="quant-card p-4">
                  <div className="text-sm font-semibold text-slate-200 mb-2">LightGBM 特征重要性 (TreeSHAP & Gain Top 10)</div>
                  <EChartsBox option={fiOption} height={270} />
                </div>
              </div>

              {/* Multi-Round Experiment Comparison Table */}
              <div className="quant-card p-4 overflow-hidden">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-slate-200">多轮回测实验横向对比 ({doneRuns.length} 组已完成 · 点击任意行切换上方图表)</span>
                </div>
                <Table
                  rowKey="id"
                  size="small"
                  scroll={{ x: 'max-content' }}
                  pagination={{ pageSize: 6 }}
                  dataSource={runs}
                  onRow={(r) => ({
                    onClick: () => r.status === 'DONE' && loadRuns(r.id),
                    className: r.id === activeRun?.id ? 'bg-sky-950/50 cursor-pointer' : r.status === 'DONE' ? 'cursor-pointer' : '',
                  })}
                  columns={[
                    {
                      title: '实验名称',
                      dataIndex: 'name',
                      render: (v: string, r: any) => (
                        <div className="flex items-center gap-1.5">
                          <span className="font-semibold text-slate-100">{v}</span>
                          {r.id === activeRun?.id && <Tag color="cyan" className="m-0">当前查看</Tag>}
                          {r.status === 'RUNNING' && <Tag color="processing" className="m-0">运行中</Tag>}
                          {r.status === 'FAILED' && <Tag color="error" className="m-0">失败</Tag>}
                        </div>
                      ),
                    },
                    {
                      title: '回测区间',
                      render: (_: any, r: any) => (
                        <span className="num text-[11px] text-slate-300">
                          {r.start_date || '-'} ~ {r.end_date || '-'}
                        </span>
                      ),
                    },
                    {
                      title: '模型 / 权重',
                      render: (_: any, r: any) => (
                        <span className="text-xs">
                          <Tag color="purple" className="m-0 mr-1">{r.model_mode}</Tag>
                          <Tag color="geekblue" className="m-0">{r.weight_method === 'risk_parity' ? '风险平价' : '等权'}</Tag>
                        </span>
                      ),
                    },
                    {
                      title: '持有 / 参数摘要',
                      render: (_: any, r: any) => (
                        <span className="num text-[11px] text-slate-400">
                          {r.holding_mode === 'dynamic' ? (
                            <Tag color="gold" className="m-0 mr-1">动态(锁{r.min_hold_days ?? 10}日/冷{r.stop_cooldown_days ?? 5}日)</Tag>
                          ) : (
                            <Tag color="default" className="m-0 mr-1">固定{r.rebalance_days}日</Tag>
                          )}
                          {r.target_holdings}股/滑点{r.slippage_bps}bp · 择时{r.enable_regime_timing ? '开' : '关'}
                        </span>
                      ),
                    },
                    {
                      title: '均持',
                      dataIndex: 'avg_hold_days',
                      align: 'right' as const,
                      render: (v: number | null) => <span className="num text-xs text-slate-300">{v ? `${v}日` : '-'}</span>,
                    },
                    {
                      title: '年化收益',
                      dataIndex: 'annualized_return',
                      align: 'right' as const,
                      render: (v: number | null) => <span className={`num font-semibold ${aShareColor(v)}`}>{fmtPct(v)}</span>,
                    },
                    {
                      title: '中证1000超额',
                      dataIndex: 'excess_return_csi1000',
                      align: 'right' as const,
                      render: (v: number | null) => <span className={`num ${aShareColor(v)}`}>{fmtPct(v)}</span>,
                    },
                    {
                      title: '夏普',
                      dataIndex: 'sharpe_ratio',
                      align: 'right' as const,
                      sorter: (a: any, b: any) => (a.sharpe_ratio ?? -99) - (b.sharpe_ratio ?? -99),
                      render: (v: number | null) => <span className="num font-bold text-sky-400">{fmtNum(v)}</span>,
                    },
                    {
                      title: '最大回撤',
                      dataIndex: 'max_drawdown',
                      align: 'right' as const,
                      render: (v: number | null) => <span className="num text-emerald-400">{fmtPct(v)}</span>,
                    },
                    {
                      title: '换手',
                      dataIndex: 'annualized_turnover',
                      align: 'right' as const,
                      render: (v: number | null) => <span className="num text-amber-400">{v !== null && v !== undefined ? `${fmtNum(v, 1)}x` : '-'}</span>,
                    },
                    {
                      title: '耗时',
                      dataIndex: 'elapsed_sec',
                      align: 'right' as const,
                      render: (v: number | null) => <span className="num text-xs text-slate-400">{v ? `${v}s` : '-'}</span>,
                    },
                    {
                      title: '操作',
                      render: (_: any, r: any) => (
                        <Popconfirm
                          title="确认删除该实验记录？"
                          onConfirm={(e) => {
                            e?.stopPropagation();
                            handleDeleteRun(r.id);
                          }}
                          onCancel={(e) => e?.stopPropagation()}
                        >
                          <Button
                            type="text"
                            danger
                            size="small"
                            icon={<DeleteOutlined />}
                            onClick={(e) => e.stopPropagation()}
                          />
                        </Popconfirm>
                      ),
                    },
                  ]}
                />
              </div>

              <div className="quant-card p-4 overflow-hidden">
                <div className="text-sm font-semibold text-slate-200 mb-2">回测成交流水 (最近 50 笔)</div>
                <Table
                  rowKey={(r) => `${r.trade_date}_${r.ts_code}_${r.side}_${r.qty}_${r.price}_${r.amount}`}
                  size="small"
                  scroll={{ x: 'max-content' }}
                  pagination={{ pageSize: 8 }}
                  dataSource={(activeRun.trades || []).slice(-50).reverse()}
                  columns={[
                    { title: '日期', dataIndex: 'trade_date', className: 'num' },
                    { title: '标的', render: (_: any, r: any) => <span className="num">{r.ts_code} {r.name}</span> },
                    { title: '方向', dataIndex: 'side', render: (v: string) => <Tag color={v === 'BUY' ? 'error' : 'success'}>{v}</Tag> },
                    { title: '股数', dataIndex: 'qty', align: 'right' as const, className: 'num' },
                    { title: '成交价', dataIndex: 'price', align: 'right' as const, render: (v: number) => <span className="num">¥{fmtNum(v)}</span> },
                    { title: '总成本(费+滑)', align: 'right' as const, render: (_: any, r: any) => <span className="num text-slate-400">¥{fmtNum((r.commission || 0) + (r.stamp_duty || 0) + (r.transfer_fee || 0) + (r.slippage || 0))}</span> },
                    { title: '已实现盈亏', dataIndex: 'realized_pnl', align: 'right' as const, render: (v: number | null) => v !== null ? <span className={`num ${aShareColor(v)}`}>{v >= 0 ? '+' : ''}{fmtNum(v)}</span> : '-' },
                    { title: '原因', dataIndex: 'reason' },
                  ]}
                />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
