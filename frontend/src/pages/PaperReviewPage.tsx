import React, { useEffect, useState } from 'react';
import { Alert, Button, InputNumber, Modal, Progress, Select, Spin, Table, Tabs, Tag, message } from 'antd';
import {
  ClearOutlined,
  FileTextOutlined,
  HistoryOutlined,
  RetweetOutlined,
  SyncOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import { aShareColor, api, fmtNum, fmtPct } from '../services/api';
import { ModelVersionItem, WeeklyReviewItem } from '../types/quant';
import { useAsyncTask } from '../hooks/useAsyncTask';
import { TaskProgressBanner } from '../components/common/TaskProgressBanner';

export const PaperReviewPage: React.FC = () => {
  const [accountState, setAccountState] = useState<any>(null);
  const [reviews, setReviews] = useState<WeeklyReviewItem[]>([]);
  const [versions, setVersions] = useState<ModelVersionItem[]>([]);
  const [jobs, setJobs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string>('');
  const [retrainMode, setRetrainMode] = useState<string>('regression');
  const [resetOpen, setResetOpen] = useState(false);
  const [resetCash, setResetCash] = useState<number>(300000);

  const loadAll = async () => {
    try {
      const [acc, revs, vers, jbs] = await Promise.all([
        api.getPaperAccount(),
        api.listWeeklyReviews(),
        api.listModelVersions(),
        api.getSchedulerJobs(),
      ]);
      setAccountState(acc);
      setReviews(revs);
      setVersions(vers);
      setJobs(jbs);
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '加载模拟盘数据失败');
    } finally {
      setLoading(false);
    }
  };

  const paperTask = useAsyncTask({
    kinds: ['paper_intraday', 'paper_rebalance', 'weekly_review', 'model_retrain'],
    onDone: async (task) => {
      await loadAll();
      if (task.kind === 'paper_intraday') {
        message.success('已完成 Top-100 盘中实时盯市与全市场大盘刷新');
      } else if (task.kind === 'paper_rebalance') {
        message.success('已执行 14:40 全市场选股调仓信号生成与模拟撮合');
      } else if (task.kind === 'weekly_review') {
        message.success('已生成最新周度复盘与 20 日因子衰减报告');
      } else if (task.kind === 'model_retrain') {
        message.success('新模型版本（6-Seed Bagging）训练完成并已激活');
      }
    },
    onError: (task, errMsg) => {
      message.error(`「${task.title}」执行失败: ${errMsg}`);
    },
  });

  useEffect(() => {
    loadAll();
  }, []);

  const runAction = async (key: string, fn: () => Promise<any>, okMsg: string) => {
    setBusy(key);
    try {
      await fn();
      await loadAll();
      message.success(okMsg);
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '操作执行失败');
    } finally {
      setBusy('');
    }
  };

  const runAsyncAction = async (launcher: () => Promise<any>) => {
    try {
      await paperTask.startTask(launcher);
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '后台任务启动失败');
    }
  };

  if (loading) {
    return (
      <div className="h-96 flex items-center justify-center">
        <Spin size="large" />
      </div>
    );
  }

  const acct = accountState?.account || {};
  const positions = accountState?.positions || [];
  const orders = accountState?.orders || [];
  const latestReview = reviews[0] || null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-slate-100 m-0">模拟实盘与周度复盘迭代中心</h1>
          <div className="text-xs text-slate-400 mt-0.5">
            全自动 APScheduler 交易时段定时任务（{jobs.length} 个任务已挂载）· 卖一/买一+滑点撮合 · 实盘 vs 回测偏差归因 · 模型版本快照一键重训与回滚
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            icon={<SyncOutlined spin={paperTask.runningKind === 'paper_intraday'} />}
            loading={paperTask.runningKind === 'paper_intraday'}
            disabled={paperTask.isRunning && paperTask.runningKind !== 'paper_intraday'}
            onClick={() => runAsyncAction(() => api.triggerIntradayRefreshAsync())}
          >
            盘中盯市刷新
          </Button>
          <Button
            type="primary"
            icon={<ThunderboltOutlined />}
            loading={paperTask.runningKind === 'paper_rebalance'}
            disabled={paperTask.isRunning && paperTask.runningKind !== 'paper_rebalance'}
            onClick={() => runAsyncAction(() => api.triggerPaperRebalanceAsync(true))}
          >
            立即触发 14:40 调仓
          </Button>
          <Button
            icon={<FileTextOutlined />}
            loading={paperTask.runningKind === 'weekly_review'}
            disabled={paperTask.isRunning && paperTask.runningKind !== 'weekly_review'}
            onClick={() => runAsyncAction(() => api.generateWeeklyReviewAsync())}
          >
            生成周度复盘
          </Button>
          <Button
            danger
            icon={<ClearOutlined />}
            loading={busy === 'reset'}
            disabled={paperTask.isRunning}
            onClick={() => setResetOpen(true)}
          >
            重置模拟盘
          </Button>
        </div>
      </div>

      <TaskProgressBanner task={paperTask.activeTask} />
      {busy && !paperTask.isRunning && (
        <div className="quant-card px-3 py-2 border border-sky-500/30 bg-sky-950/20">
          <div className="text-xs text-sky-300 mb-1">
            {busy === 'reset' ? '正在清空并重置模拟盘账本...' : '正在切换并激活目标模型版本...'}
          </div>
          <Progress percent={75} size="small" status="active" showInfo={false} strokeColor="#38BDF8" trailColor="#1E293B" />
        </div>
      )}

      <Modal
        title="清空模拟盘并重置初始本金"
        open={resetOpen}
        okText="确认清空重置"
        okButtonProps={{ danger: true, loading: busy === 'reset' }}
        cancelText="取消"
        onCancel={() => setResetOpen(false)}
        onOk={async () => {
          await runAction(
            'reset',
            () => api.resetPaperAccount(resetCash, true, true, true),
            `模拟盘已清空并重置本金为 ¥${fmtNum(resetCash)}`
          );
          setResetOpen(false);
        }}
      >
        <div className="space-y-3 py-2">
          <Alert
            type="warning"
            showIcon
            message="此操作将清空当前模拟持仓、委托成交、历史净值曲线及旧复盘记录，并从今日重新记账（不影响 DuckDB 历史行情与已训练模型）。"
          />
          <div className="flex items-center gap-3">
            <span className="text-sm text-slate-300">新初始本金 (CNY)：</span>
            <InputNumber
              min={10000}
              max={100000000}
              step={50000}
              value={resetCash}
              onChange={(v) => setResetCash(Number(v || 300000))}
              style={{ width: 200 }}
            />
          </div>
        </div>
      </Modal>

      {/* Account Strip */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="quant-card p-3">
          <div className="text-[11px] text-slate-400">总权益</div>
          <div className="num text-lg font-bold text-slate-100">¥{fmtNum(acct.total_equity)}</div>
        </div>
        <div className="quant-card p-3">
          <div className="text-[11px] text-slate-400">可用现金</div>
          <div className="num text-lg font-bold text-sky-400">¥{fmtNum(acct.cash)}</div>
        </div>
        <div className="quant-card p-3">
          <div className="text-[11px] text-slate-400">持仓市值</div>
          <div className="num text-lg font-bold text-slate-200">¥{fmtNum(acct.market_value)}</div>
        </div>
        <div className="quant-card p-3">
          <div className="text-[11px] text-slate-400">累计收益 / 回撤</div>
          <div className={`num text-lg font-bold ${aShareColor(acct.total_return)}`}>
            {fmtPct(acct.total_return)} / {fmtPct(acct.current_drawdown)}
          </div>
        </div>
        <div className="quant-card p-3">
          <div className="text-[11px] text-slate-400">当前模式 / 熔断</div>
          <div className="mt-1 flex gap-1.5">
            <Tag color="blue">{acct.broker_mode}</Tag>
            <Tag color={acct.fuse_status === 'HALTED' ? 'error' : 'cyan'}>{acct.fuse_status}</Tag>
          </div>
        </div>
      </div>

      <div className="quant-card p-4">
        <Tabs
          items={[
            {
              key: 'ledger',
              label: `持仓与订单流水 (${positions.length} 持仓 / ${orders.length} 订单)`,
              children: (
                <div className="space-y-5">
                  <div>
                    <div className="text-sm font-semibold text-slate-200 mb-2">实时持仓明细</div>
                    <Table
                      rowKey="ts_code"
                      size="small"
                      pagination={false}
                      dataSource={positions}
                      columns={[
                        { title: '代码/名称', render: (_: any, r: any) => <span className="num font-semibold">{r.ts_code} {r.name}</span> },
                        { title: '申万行业', dataIndex: 'sw_l1_name', render: (v: string) => <Tag color="geekblue">{v}</Tag> },
                        { title: '总股数 / 可卖(T+1)', align: 'right' as const, render: (_: any, r: any) => <span className="num">{r.total_shares} / <b className="text-sky-400">{r.sellable_shares}</b></span> },
                        { title: '成本 / 现价', align: 'right' as const, render: (_: any, r: any) => <span className="num">{fmtNum(r.avg_cost)} / {fmtNum(r.current_price)}</span> },
                        { title: '止损位', dataIndex: 'stop_loss_price', align: 'right' as const, render: (v: number) => <span className="num text-emerald-400">¥{fmtNum(v)}</span> },
                        { title: '仓位', dataIndex: 'weight', align: 'right' as const, render: (v: number) => <span className="num">{(v * 100).toFixed(2)}%</span> },
                        { title: '浮动盈亏', dataIndex: 'unrealized_pnl', align: 'right' as const, render: (v: number, r: any) => <span className={`num font-semibold ${aShareColor(v)}`}>{v >= 0 ? '+' : ''}{fmtNum(v)} ({fmtPct(r.unrealized_pnl_pct)})</span> },
                      ]}
                    />
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-slate-200 mb-2">委托与成交流水</div>
                    <Table
                      rowKey="order_id"
                      size="small"
                      pagination={{ pageSize: 10 }}
                      dataSource={orders}
                      columns={[
                        { title: '时间', dataIndex: 'created_at', render: (v: string) => <span className="num text-xs">{v?.slice(0, 19).replace('T', ' ')}</span> },
                        { title: '标的', render: (_: any, r: any) => <span className="num">{r.ts_code} {r.name}</span> },
                        { title: '方向', dataIndex: 'side', render: (v: string) => <Tag color={v === 'BUY' ? 'error' : 'success'}>{v}</Tag> },
                        { title: '状态', dataIndex: 'status', render: (v: string) => <Tag color={v === 'FILLED' ? 'cyan' : v.startsWith('REJECT') ? 'error' : 'gold'}>{v}</Tag> },
                        { title: '委托/成交', align: 'right' as const, render: (_: any, r: any) => <span className="num">{r.request_qty} / {r.filled_qty}</span> },
                        { title: '成交均价', dataIndex: 'avg_fill_price', align: 'right' as const, render: (v: number) => <span className="num">¥{fmtNum(v)}</span> },
                        { title: '费+滑点', align: 'right' as const, render: (_: any, r: any) => <span className="num text-slate-400">¥{fmtNum((r.commission || 0) + (r.stamp_duty || 0) + (r.transfer_fee || 0) + (r.slippage_cost || 0))}</span> },
                        { title: '模式', dataIndex: 'broker_mode', render: (v: string) => <Tag>{v}</Tag> },
                        { title: '说明/拦截原因', render: (_: any, r: any) => <span className="text-xs text-slate-400">{r.reject_reason || r.reason_summary}</span> },
                      ]}
                    />
                  </div>
                </div>
              ),
            },
            {
              key: 'review',
              label: `周度复盘与偏差归因 (${reviews.length})`,
              children: latestReview ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                    <div className="p-3 rounded bg-[#0F1624] border border-slate-800">
                      <div className="text-[11px] text-slate-400">模拟实盘周期收益</div>
                      <div className={`num text-lg font-bold ${aShareColor(latestReview.paper_return)}`}>{fmtPct(latestReview.paper_return)}</div>
                    </div>
                    <div className="p-3 rounded bg-[#0F1624] border border-slate-800">
                      <div className="text-[11px] text-slate-400">理论回测同期收益</div>
                      <div className={`num text-lg font-bold ${aShareColor(latestReview.theory_bt_return)}`}>{fmtPct(latestReview.theory_bt_return)}</div>
                    </div>
                    <div className="p-3 rounded bg-[#0F1624] border border-slate-800">
                      <div className="text-[11px] text-slate-400">实盘-回测跟踪误差</div>
                      <div className={`num text-lg font-bold ${aShareColor(latestReview.tracking_error)}`}>{fmtPct(latestReview.tracking_error)}</div>
                    </div>
                    <div className="p-3 rounded bg-[#0F1624] border border-slate-800">
                      <div className="text-[11px] text-slate-400">累计滑点损耗</div>
                      <div className="num text-lg font-bold text-amber-400">{fmtPct(latestReview.slippage_drag)}</div>
                    </div>
                    <div className="p-3 rounded bg-[#0F1624] border border-slate-800">
                      <div className="text-[11px] text-slate-400">涨跌停漏单影响</div>
                      <div className="num text-lg font-bold text-slate-200">{fmtPct(latestReview.missed_fill_impact)}</div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="text-sm font-semibold text-slate-200">系统迭代优化建议</div>
                    {latestReview.optimization_advice.map((adv, idx) => (
                      <Alert
                        key={idx}
                        type={adv.severity === 'HIGH' ? 'error' : adv.severity === 'MEDIUM' ? 'warning' : 'success'}
                        showIcon
                        message={`${adv.title} [${adv.suggested_action}]`}
                        description={adv.detail}
                      />
                    ))}
                  </div>

                  <div>
                    <div className="text-sm font-semibold text-slate-200 mb-2">
                      近 20 日因子 IC 衰减预警列表 ({latestReview.decayed_factors.length})
                    </div>
                    <Table
                      rowKey="factor_name"
                      size="small"
                      pagination={false}
                      dataSource={latestReview.decayed_factors}
                      columns={[
                        { title: '因子', render: (_: any, r: any) => <span>{r.label} <span className="num text-xs text-slate-400">({r.factor_name})</span></span> },
                        { title: '维度', dataIndex: 'dimension', render: (v: string) => <Tag color="geekblue">{v}</Tag> },
                        { title: '历史 RankIC', dataIndex: 'hist_rank_ic', align: 'right' as const, render: (v: number) => <span className={`num ${aShareColor(v)}`}>{v.toFixed(4)}</span> },
                        { title: '近20日 RankIC', dataIndex: 'recent_20d_rank_ic', align: 'right' as const, render: (v: number) => <span className={`num font-bold ${aShareColor(v)}`}>{v.toFixed(4)}</span> },
                        { title: '状态', dataIndex: 'decay_status', render: (v: string) => <Tag color={v === 'REVERSED' ? 'error' : 'warning'}>{v}</Tag> },
                      ]}
                      locale={{ emptyText: '近 20 日暂无显著衰减或反转因子' }}
                    />
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-slate-400">暂无复盘记录，请点击右上角「生成周度复盘」</div>
              ),
            },
            {
              key: 'models',
              label: `模型版本快照与回滚 (${versions.length})`,
              children: (
                <div className="space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-2 p-3 rounded bg-[#0F1624] border border-slate-800">
                    <div className="text-xs text-slate-300">
                      一键滚动重训将严格执行 <b>Purged Walk-Forward (10日标签 · Embargo ≥ 10日 · 6-Seed Bagging)</b> 并落盘新版本快照，支持随时无损回滚至任意历史版本。
                    </div>
                    <div className="flex items-center gap-2">
                      <Select
                        size="small"
                        value={retrainMode}
                        onChange={setRetrainMode}
                        style={{ width: 175 }}
                        options={[
                          { value: 'regression', label: 'Huber 回归 (推荐)' },
                          { value: 'lambdarank', label: 'LambdaRank 排序' },
                        ]}
                      />
                      <Button
                        type="primary"
                        size="small"
                        icon={<RetweetOutlined />}
                        loading={paperTask.runningKind === 'model_retrain'}
                        disabled={paperTask.isRunning && paperTask.runningKind !== 'model_retrain'}
                        onClick={() =>
                          runAsyncAction(() =>
                            api.retrainModelAsync(retrainMode, `工作台一键滚动重训 (${retrainMode} · 6-Seed Bagging)`)
                          )
                        }
                      >
                        一键滚动重训
                      </Button>
                    </div>
                  </div>
                  <Table
                    rowKey="version_id"
                    size="small"
                    pagination={false}
                    dataSource={versions}
                    columns={[
                      {
                        title: '版本 ID',
                        dataIndex: 'version_id',
                        render: (v: string, r: ModelVersionItem) => (
                          <div className="flex items-center gap-2">
                            <span className="num text-xs text-sky-400">{v}</span>
                            {r.is_active && <Tag color="success">当前激活</Tag>}
                          </div>
                        ),
                      },
                      { title: '模式', dataIndex: 'model_type', render: (v: string) => <Tag color="purple">{v}</Tag> },
                      { title: '训练窗口', render: (_: any, r: ModelVersionItem) => <span className="num text-xs">{r.train_start} ~ {r.train_end}</span> },
                      { title: 'Embargo', dataIndex: 'embargo_days', render: (v: number) => <Tag color="cyan">{v}日</Tag> },
                      { title: '样本外 RankIC', dataIndex: 'oos_rank_ic', align: 'right' as const, render: (v: number) => <span className={`num font-semibold ${aShareColor(v)}`}>{v.toFixed(4)}</span> },
                      { title: '样本外 ICIR', dataIndex: 'oos_icir', align: 'right' as const, render: (v: number) => <span className="num">{v.toFixed(3)}</span> },
                      { title: '备注', dataIndex: 'notes', className: 'text-xs text-slate-400' },
                      {
                        title: '操作',
                        render: (_: any, r: ModelVersionItem) =>
                          r.is_active ? (
                            <span className="text-xs text-slate-500">运行中</span>
                          ) : (
                            <Button
                              size="small"
                              icon={<HistoryOutlined />}
                              loading={busy === r.version_id}
                              onClick={() =>
                                runAction(
                                  r.version_id,
                                  () => api.activateModelVersion(r.version_id),
                                  `已切换回滚至模型版本 ${r.version_id}`
                                )
                              }
                            >
                              一键回滚激活
                            </Button>
                          ),
                      },
                    ]}
                  />
                </div>
              ),
            },
          ]}
        />
      </div>
    </div>
  );
};
