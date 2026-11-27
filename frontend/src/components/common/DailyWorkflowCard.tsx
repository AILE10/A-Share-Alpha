import React, { useCallback, useEffect, useState } from 'react';
import { Button, Progress, Spin, Tag, Tooltip, message } from 'antd';
import {
  ArrowRightOutlined,
  CheckCircleFilled,
  ClockCircleOutlined,
  CloseCircleFilled,
  LoadingOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  ScheduleOutlined,
} from '@ant-design/icons';
import { api } from '../../services/api';
import { useAlphaWebSocket } from '../../services/ws';
import { useAsyncTask } from '../../hooks/useAsyncTask';

export interface DailyWorkflowStep {
  step_key: string;
  order: number;
  schedule_label: string;
  title: string;
  short_desc: string;
  action_kind: string;
  action_label: string;
  frequency: string;
  status: 'PENDING' | 'SCHEDULED' | 'RUNNING' | 'DONE' | 'FAILED' | 'SKIPPED';
  trigger_source: string;
  time_label: string;
  summary: string;
  pct: number;
  is_current_stage: boolean;
}

export interface DailyWorkflowState {
  trade_date: string;
  weekday_cn: string;
  server_time: string;
  is_trading_day: boolean;
  current_stage_key: string;
  phase_hint: string;
  completed_count: number;
  total_due_count: number;
  steps: DailyWorkflowStep[];
}

interface DailyWorkflowCardProps {
  initialWorkflow?: DailyWorkflowState | null;
  onStepCompleted?: () => void;
}

const STEP_KINDS = [
  'pre_market_unlock',
  'paper_intraday',
  'regime_refresh',
  'paper_rebalance',
  'data_sync_incremental',
  'weekly_review',
];

export const DailyWorkflowCard: React.FC<DailyWorkflowCardProps> = ({
  initialWorkflow = null,
  onStepCompleted,
}) => {
  const [wf, setWf] = useState<DailyWorkflowState | null>(initialWorkflow);
  const [loadingInitial, setLoadingInitial] = useState<boolean>(!initialWorkflow);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchWorkflow = useCallback(async (showSpinner = false) => {
    if (showSpinner) setRefreshing(true);
    try {
      const res = await api.getDailyWorkflow();
      if (res && Array.isArray(res.steps) && res.steps.length > 0) {
        setWf(res);
        setLoadError(null);
      } else {
        setLoadError('后端暂未返回每日流程步骤');
      }
    } catch (e: any) {
      setLoadError(e?.response?.data?.detail || '流程状态请求失败，可能后端正在重启或网络抖动');
    } finally {
      setLoadingInitial(false);
      if (showSpinner) setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    if (initialWorkflow && Array.isArray(initialWorkflow.steps) && initialWorkflow.steps.length > 0) {
      setWf(initialWorkflow);
      setLoadError(null);
      setLoadingInitial(false);
    } else {
      fetchWorkflow(false);
    }
  }, [initialWorkflow, fetchWorkflow]);

  // Auto-retry if initial workflow load failed transiently (e.g. backend restarting)
  useEffect(() => {
    if (wf || !loadError) return;
    const timer = window.setTimeout(() => {
      fetchWorkflow(false);
    }, 4000);
    return () => window.clearTimeout(timer);
  }, [wf, loadError, fetchWorkflow]);

  const wfTask = useAsyncTask({
    kinds: STEP_KINDS,
    onDone: async (task) => {
      await fetchWorkflow(false);
      if (onStepCompleted) onStepCompleted();
      if (task.kind === 'pre_market_unlock') {
        message.success('已完成盘前 T+1 持仓解锁');
      }
    },
    onError: async () => {
      await fetchWorkflow(false);
    },
  });

  // Instant WebSocket updates when scheduler or manual triggers update any workflow node
  const { connected: wsConnected } = useAlphaWebSocket((evt) => {
    if (evt.channel === 'workflow_step' && evt.payload?.step_key) {
      const p = evt.payload;
      setWf((prev) => {
        if (!prev) return prev;
        const steps = prev.steps.map((s) =>
          s.step_key === p.step_key
            ? {
                ...s,
                status: p.status,
                trigger_source: p.trigger_source || s.trigger_source,
                time_label: p.time_label || s.time_label,
                summary: p.summary || s.summary,
                pct: p.status === 'DONE' ? 1 : p.status === 'RUNNING' ? Math.max(s.pct, 0.25) : 0,
              }
            : s
        );
        const doneCount = steps.filter(
          (s) => (s.frequency === 'DAILY' || prev.weekday_cn === '周五') && s.status === 'DONE'
        ).length;
        return { ...prev, completed_count: doneCount, steps };
      });
    }
  });

  useEffect(() => {
    if (wsConnected && !wf) {
      fetchWorkflow(false);
    }
  }, [wsConnected, wf, fetchWorkflow]);

  const handleRunStep = async (step: DailyWorkflowStep) => {
    try {
      if (step.step_key === 'pre_market_unlock') {
        await wfTask.startTask(() => api.triggerPreMarketUnlockAsync());
      } else if (step.step_key === 'intraday_monitor') {
        await wfTask.startTask(() => api.triggerIntradayRefreshAsync());
      } else if (step.step_key === 'rebalance_1440') {
        await wfTask.startTask(() => api.triggerPaperRebalanceAsync(true));
      } else if (step.step_key === 'post_market_sync') {
        await wfTask.startTask(() => api.triggerDataSync('incremental', { workers: 10 }));
      } else if (step.step_key === 'weekly_review') {
        await wfTask.startTask(() => api.generateWeeklyReviewAsync());
      }
      await fetchWorkflow(false);
    } catch (e: any) {
      message.error(e?.response?.data?.detail || `启动「${step.title}」失败`);
    }
  };

  if (loadingInitial && !wf) {
    return (
      <div className="quant-card p-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2.5 text-xs text-slate-300">
          <Spin size="small" />
          <span>正在加载每日固定交易流程（自动调度 + 实时状态）...</span>
        </div>
      </div>
    );
  }

  if (!wf || !wf.steps?.length) {
    return (
      <div className="quant-card p-4 border border-amber-500/40 flex flex-wrap items-center justify-between gap-3">
        <div className="space-y-0.5">
          <div className="text-sm font-semibold text-slate-100">
            每日固定交易流程（自动调度 + 实时状态）
          </div>
          <div className="text-xs text-amber-300/90">
            {loadError || '流程状态暂未加载，系统正在自动重试...'}
          </div>
        </div>
        <Button
          size="small"
          type="primary"
          icon={<ReloadOutlined spin={refreshing} />}
          loading={refreshing}
          onClick={() => fetchWorkflow(true)}
        >
          立即重试加载
        </Button>
      </div>
    );
  }

  const activeTask = wfTask.activeTask;
  const activeKindToStep: Record<string, string> = {
    pre_market_unlock: 'pre_market_unlock',
    paper_intraday: 'intraday_monitor',
    regime_refresh: 'intraday_monitor',
    paper_rebalance: 'rebalance_1440',
    data_sync_incremental: 'post_market_sync',
    weekly_review: 'weekly_review',
  };
  const runningStepKey = activeTask ? activeKindToStep[activeTask.kind] || '' : '';

  const renderStatusBadge = (step: DailyWorkflowStep, isRunningNow: boolean) => {
    const effStatus = isRunningNow ? 'RUNNING' : step.status;
    if (effStatus === 'RUNNING') {
      return (
        <Tag color="processing" icon={<LoadingOutlined spin />} className="m-0 text-[11px]">
          执行中
        </Tag>
      );
    }
    if (effStatus === 'DONE') {
      return (
        <Tag color="success" icon={<CheckCircleFilled />} className="m-0 text-[11px]">
          已完成
        </Tag>
      );
    }
    if (effStatus === 'FAILED') {
      return (
        <Tag color="error" icon={<CloseCircleFilled />} className="m-0 text-[11px]">
          执行异常
        </Tag>
      );
    }
    if (effStatus === 'SCHEDULED') {
      return (
        <Tag color="purple" icon={<ScheduleOutlined />} className="m-0 text-[11px]">
          周五自动
        </Tag>
      );
    }
    if (effStatus === 'SKIPPED') {
      return (
        <Tag color="default" className="m-0 text-[11px]">
          休市跳过
        </Tag>
      );
    }
    return (
      <Tag color="warning" icon={<ClockCircleOutlined />} className="m-0 text-[11px]">
        待执行
      </Tag>
    );
  };

  return (
    <div className="quant-card p-4 space-y-3">
      {/* Header Row */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-semibold text-slate-100">
            每日固定交易流程（自动调度 + 实时状态）
          </span>
          <Tag color={wf.is_trading_day ? 'cyan' : 'default'} className="m-0 text-[11px]">
            {wf.trade_date} {wf.weekday_cn} · {wf.is_trading_day ? 'A 股交易日' : '非交易日休市'}
          </Tag>
          <Tag color="geekblue" className="m-0 text-[11px]">
            今日完成 {wf.completed_count} / {wf.total_due_count} 步
          </Tag>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span className="hidden sm:inline text-sky-300/90">{wf.phase_hint}</span>
          <Button
            size="small"
            icon={<ReloadOutlined spin={refreshing} />}
            loading={refreshing}
            onClick={() => fetchWorkflow(true)}
          >
            刷新流程
          </Button>
        </div>
      </div>

      {/* 5-Step Connected Visual Flowchart */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-3 items-stretch">
        {wf.steps.map((step, idx) => {
          const isRunningNow = runningStepKey === step.step_key || step.status === 'RUNNING';
          const effStatus = isRunningNow ? 'RUNNING' : step.status;
          const livePct = isRunningNow
            ? Math.max(8, Math.round((activeTask?.pct ?? step.pct ?? 0.25) * 100))
            : effStatus === 'DONE'
              ? 100
              : 0;
          const liveSummary =
            isRunningNow && activeTask?.stage ? activeTask.stage : step.summary;

          const cardBorder =
            effStatus === 'RUNNING'
              ? 'border-sky-500/80 bg-sky-950/30 shadow-[0_0_16px_rgba(56,189,248,0.18)]'
              : effStatus === 'DONE'
                ? 'border-emerald-500/40 bg-emerald-950/15'
                : effStatus === 'FAILED'
                  ? 'border-rose-500/60 bg-rose-950/20'
                  : step.is_current_stage
                    ? 'border-amber-400/60 bg-amber-950/15'
                    : 'border-slate-800 bg-[#0F1624]';

          const circleClass =
            effStatus === 'DONE'
              ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50'
              : effStatus === 'RUNNING'
                ? 'bg-sky-500/25 text-sky-300 border-sky-400/70'
                : effStatus === 'FAILED'
                  ? 'bg-rose-500/20 text-rose-300 border-rose-500/50'
                  : step.is_current_stage
                    ? 'bg-amber-500/20 text-amber-300 border-amber-400/60'
                    : 'bg-slate-800 text-slate-400 border-slate-700';

          const sourceCn =
            step.trigger_source === 'AUTO'
              ? '自动'
              : step.trigger_source === 'MANUAL'
                ? '手动'
                : '';

          return (
            <div key={step.step_key} className="relative flex items-stretch">
              <div
                className={`w-full rounded-lg border p-3 flex flex-col justify-between gap-2 transition-all ${cardBorder}`}
              >
                {/* Top: Step Order + Clock + Current Indicator */}
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between gap-1">
                    <div className="flex items-center gap-1.5">
                      <span
                        className={`num w-6 h-6 rounded-full border text-[11px] font-bold flex items-center justify-center ${circleClass}`}
                      >
                        0{step.order}
                      </span>
                      <span className="num text-xs font-semibold text-sky-300">
                        {step.schedule_label}
                      </span>
                    </div>
                    {step.is_current_stage && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-400/40">
                        当前时段
                      </span>
                    )}
                  </div>

                  {/* Step Title & Plain-Chinese Explanation */}
                  <div>
                    <Tooltip title={step.short_desc}>
                      <div className="text-xs font-bold text-slate-100 leading-snug">
                        {step.title}
                      </div>
                    </Tooltip>
                    <div className="text-[11px] text-slate-400 leading-relaxed mt-0.5">
                      {step.short_desc}
                    </div>
                  </div>
                </div>

                {/* Middle: Status + Real Execution Summary + Inline Progress */}
                <div className="space-y-1.5 pt-1 border-t border-slate-800/80">
                  <div className="flex items-center justify-between gap-1 flex-wrap">
                    {renderStatusBadge(step, isRunningNow)}
                    {step.time_label && (
                      <span className="num text-[10px] text-slate-400">
                        {step.time_label}
                        {sourceCn ? ` · ${sourceCn}` : ''}
                      </span>
                    )}
                  </div>

                  {isRunningNow && (
                    <Progress
                      percent={livePct}
                      size="small"
                      status="active"
                      strokeColor="#38BDF8"
                      trailColor="#1E293B"
                      className="m-0"
                    />
                  )}

                  <div
                    className={`text-[11px] leading-snug min-h-[30px] ${
                      effStatus === 'FAILED'
                        ? 'text-rose-300'
                        : effStatus === 'DONE'
                          ? 'text-emerald-300/90'
                          : isRunningNow
                            ? 'text-sky-300'
                            : 'text-slate-400'
                    }`}
                  >
                    {liveSummary}
                  </div>
                </div>

                {/* Bottom: One-click Manual Action */}
                <Button
                  size="small"
                  block
                  type={step.is_current_stage && effStatus !== 'DONE' ? 'primary' : 'default'}
                  icon={<PlayCircleOutlined />}
                  loading={isRunningNow}
                  disabled={wfTask.isRunning && !isRunningNow}
                  onClick={() => handleRunStep(step)}
                >
                  {isRunningNow ? '执行中...' : effStatus === 'DONE' ? `重跑 · ${step.action_label}` : step.action_label}
                </Button>
              </div>

              {/* Rightward Flowchart Connector Arrow between steps on desktop */}
              {idx < wf.steps.length - 1 && (
                <div className="hidden md:flex items-center justify-center -mr-2.5 pl-0.5 z-10 pointer-events-none">
                  <ArrowRightOutlined
                    className={`text-xs ${
                      effStatus === 'DONE' ? 'text-emerald-400' : 'text-slate-600'
                    }`}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
