import axios from 'axios';

export const http = axios.create({
  baseURL: '/api/v1',
  timeout: 180_000,
});

export const api = {
  // Unified Background Action Tasks (live progress)
  listActionTasks: (activeOnly = false, kind?: string) =>
    http.get('/tasks', { params: { active_only: activeOnly, ...(kind ? { kind } : {}) } }).then((r) => r.data),
  getActionTask: (taskId: string) =>
    http.get(`/tasks/${encodeURIComponent(taskId)}`).then((r) => r.data),

  // Dashboard & Data
  getDashboardSummary: () => http.get('/dashboard/summary').then((r) => r.data),
  getDailyWorkflow: () => http.get('/dashboard/workflow').then((r) => r.data),
  refreshRealtimeRegime: () => http.post('/dashboard/regime/refresh').then((r) => r.data),
  refreshRealtimeRegimeAsync: () => http.post('/dashboard/regime/refresh-async').then((r) => r.data),
  getDataStatus: () => http.get('/data/status').then((r) => r.data),
  triggerDataSync: (
    mode: 'incremental' | 'active_market' | 'sample_60' | 'top100_snapshot' = 'incremental',
    options?: { max_stocks?: number; days?: number; workers?: number; min_amount?: number }
  ) => http.post('/data/sync', { mode, ...(options || {}) }).then((r) => r.data),
  getHealth: () => http.get('/health').then((r) => r.data),

  // Factors
  getFactorsSummary: (horizon = 5) => http.get('/factors/summary', { params: { horizon } }).then((r) => r.data),
  getFactorDetail: (name: string, horizon = 5) =>
    http.get(`/factors/${name}/detail`, { params: { horizon } }).then((r) => r.data),
  getFactorCorrelation: () => http.get('/factors/correlation').then((r) => r.data),
  recomputeFactors: (horizon = 5) => http.post('/factors/compute', { horizon }).then((r) => r.data),
  recomputeFactorsAsync: (horizon = 5) => http.post('/factors/compute-async', { horizon }).then((r) => r.data),

  // Backtest & Optimization
  listBacktestRuns: () => http.get('/backtest/runs').then((r) => r.data),
  getBacktestRun: (id: string) => http.get(`/backtest/runs/${id}`).then((r) => r.data),
  deleteBacktestRun: (id: string) => http.delete(`/backtest/runs/${encodeURIComponent(id)}`).then((r) => r.data),
  runBacktest: (payload: Record<string, any>) => http.post('/backtest/run', payload).then((r) => r.data),
  runBacktestAsync: (payload: Record<string, any>) => http.post('/backtest/run-async', payload).then((r) => r.data),
  listBacktestJobs: () => http.get('/backtest/jobs').then((r) => r.data),
  getBacktestJob: (jobId: string) => http.get(`/backtest/jobs/${encodeURIComponent(jobId)}`).then((r) => r.data),
  getBacktestCacheStatus: () => http.get('/backtest/cache-status').then((r) => r.data),
  clearBacktestCache: () => http.post('/backtest/cache-clear').then((r) => r.data),
  runWalkForwardOpt: (payload: Record<string, any>) =>
    http.post('/backtest/walk-forward-opt', payload).then((r) => r.data),

  // Signals & K-Line
  getLatestSignals: () => http.get('/signals/latest').then((r) => r.data),
  refreshSignals: (fetchLiveQuotes = true) =>
    http.post('/signals/refresh', null, { params: { fetch_live_quotes: fetchLiveQuotes } }).then((r) => r.data),
  refreshSignalsAsync: (fetchLiveQuotes = true) =>
    http.post('/signals/refresh-async', null, { params: { fetch_live_quotes: fetchLiveQuotes } }).then((r) => r.data),
  getStockKLine: (tsCode: string, limit = 160) =>
    http.get(`/signals/kline/${encodeURIComponent(tsCode)}`, { params: { limit } }).then((r) => r.data),

  // Paper Trading, Weekly Review & Models
  getPaperAccount: () => http.get('/paper/account').then((r) => r.data),
  getPaperNav: () => http.get('/paper/nav').then((r) => r.data),
  triggerPreMarketUnlockAsync: () => http.post('/paper/pre-market-unlock-async').then((r) => r.data),
  triggerIntradayRefresh: () => http.post('/paper/intraday-refresh').then((r) => r.data),
  triggerIntradayRefreshAsync: () => http.post('/paper/intraday-refresh-async').then((r) => r.data),
  triggerPaperRebalance: (fetchLiveQuotes = true) =>
    http.post('/paper/rebalance', { fetch_live_quotes: fetchLiveQuotes }).then((r) => r.data),
  triggerPaperRebalanceAsync: (fetchLiveQuotes = true) =>
    http.post('/paper/rebalance-async', { fetch_live_quotes: fetchLiveQuotes }).then((r) => r.data),
  resetPaperAccount: (initialCash = 300_000, clearReviews = true, clearSignals = true, clearAuditLogs = true) =>
    http
      .post('/paper/reset', {
        initial_cash: initialCash,
        clear_reviews: clearReviews,
        clear_signals: clearSignals,
        clear_audit_logs: clearAuditLogs,
      })
      .then((r) => r.data),
  listWeeklyReviews: () => http.get('/paper/reviews').then((r) => r.data),
  generateWeeklyReview: () => http.post('/paper/reviews/generate').then((r) => r.data),
  generateWeeklyReviewAsync: () => http.post('/paper/reviews/generate-async').then((r) => r.data),
  listModelVersions: () => http.get('/models/versions').then((r) => r.data),
  retrainModel: (modelType = 'regression', notes = '工作台一键滚动重训 (Huber 10d · 6-Seed Bagging)') =>
    http.post('/models/retrain', { model_type: modelType, notes }).then((r) => r.data),
  retrainModelAsync: (modelType = 'regression', notes = '工作台一键滚动重训 (Huber 10d · 6-Seed Bagging)') =>
    http.post('/models/retrain-async', { model_type: modelType, notes }).then((r) => r.data),
  activateModelVersion: (versionId: string) =>
    http.post(`/models/${encodeURIComponent(versionId)}/activate`).then((r) => r.data),
  getSchedulerJobs: () => http.get('/scheduler/jobs').then((r) => r.data),

  // Broker & Risk Control
  getBrokerConfig: () => http.get('/broker/config').then((r) => r.data),
  updateBrokerConfig: (payload: Record<string, any>) => http.put('/broker/config', payload).then((r) => r.data),
  testWebhooks: () => http.post('/broker/webhook/test').then((r) => r.data),
  listAssistedPending: () => http.get('/broker/assisted/pending').then((r) => r.data),
  confirmAssistedFill: (orderId: string, payload: Record<string, any>) =>
    http.post(`/broker/assisted/${encodeURIComponent(orderId)}/confirm-fill`, payload).then((r) => r.data),
  cancelAssistedOrder: (orderId: string) =>
    http.post(`/broker/assisted/${encodeURIComponent(orderId)}/cancel`).then((r) => r.data),
  submitManualOrder: (payload: Record<string, any>) => http.post('/broker/manual-order', payload).then((r) => r.data),
  getRiskConfig: () => http.get('/risk/config').then((r) => r.data),
  updateRiskConfig: (payload: Record<string, any>) => http.put('/risk/config', payload).then((r) => r.data),
  resetDrawdownFuse: () => http.post('/risk/fuse-reset').then((r) => r.data),
  getRiskAuditLogs: (limit = 100) => http.get('/risk/audit-logs', { params: { limit } }).then((r) => r.data),
};

export const fmtPct = (v: number | null | undefined, digits = 2): string => {
  if (v === null || v === undefined || Number.isNaN(v)) return '-';
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`;
};

export const fmtNum = (v: number | null | undefined, digits = 2): string => {
  if (v === null || v === undefined || Number.isNaN(v)) return '-';
  return Number(v).toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits });
};

/** Authentic A-share coloring: positive = crimson red (#F43F5E), negative = emerald green (#10B981) */
export const aShareColor = (v: number | null | undefined): string => {
  if (v === null || v === undefined || Number.isNaN(v) || v === 0) return 'text-slate-300';
  return v > 0 ? 'text-rose-400' : 'text-emerald-400';
};
