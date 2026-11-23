export interface RegimeCard {
  trade_date: string;
  regime: 'OFFENSIVE' | 'NEUTRAL' | 'DEFENSIVE' | 'UNKNOWN';
  target_exposure: number;
  csi1000_close: number | null;
  csi1000_ma20: number | null;
  csi1000_ma60: number | null;
  breadth_above_ma20: number | null;
  history: Array<{
    date: string;
    csi1000_close: number;
    breadth: number;
    regime: string;
    target_exposure: number;
  }>;
}

export interface PaperAccountInfo {
  account_id: string;
  name: string;
  broker_mode: 'PAPER' | 'ASSISTED' | 'QMT';
  initial_cash: number;
  cash: number;
  total_equity: number;
  market_value: number;
  total_return: number;
  peak_equity: number;
  current_drawdown: number;
  fuse_status: 'NORMAL' | 'WARNING' | 'HALTED';
  updated_at: string;
}

export interface PaperPositionItem {
  ts_code: string;
  name: string;
  sw_l1_name: string;
  total_shares: number;
  sellable_shares: number;
  avg_cost: number;
  entry_date: string;
  highest_price_since_entry: number;
  current_price: number;
  stop_loss_price: number;
  atr_stop_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  weight: number;
}

export interface PaperOrderItem {
  order_id: string;
  ts_code: string;
  name: string;
  sw_l1_name: string;
  side: 'BUY' | 'SELL';
  request_qty: number;
  filled_qty: number;
  limit_price: number;
  avg_fill_price: number;
  commission: number;
  stamp_duty: number;
  transfer_fee: number;
  slippage_cost: number;
  status: string;
  broker_mode: string;
  reject_reason: string;
  reason_summary: string;
  created_at: string;
}

export interface FactorSummaryItem {
  factor_name: string;
  label: string;
  dimension: string;
  ic_mean: number | null;
  rank_ic_mean: number | null;
  icir: number | null;
  rank_icir: number | null;
  t_stat: number | null;
  ic_win_rate: number | null;
  recent_rank_ic_20d: number | null;
  ls_annual_return: number | null;
  decay_status: 'STABLE' | 'DECAYING' | 'REVERSED' | 'INSUFFICIENT';
  n_periods: number;
  data_status: 'OK' | 'PARTIAL_DEGRADED' | 'DEGRADED_MISSING';
}

export interface FactorAttribution {
  name: string;
  label: string;
  dimension: string;
  raw_value: number | null;
  std_value: number | null;
  shap: number;
}

export interface SignalRecommendationItem {
  signal_id: string;
  trade_date: string;
  ts_code: string;
  name: string;
  sw_l1_name: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  composite_rank: number;
  model_score: number;
  suggested_weight: number;
  ref_price: number;
  stop_loss_price: number;
  take_profit_price: number;
  top_factors: FactorAttribution[];
  tech_patterns: string[];
  rationale: string;
  status: string;
}

export interface KLineCandle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  amount: number;
  ma5: number | null;
  ma10: number | null;
  ma20: number | null;
  ma60: number | null;
  macd_dif: number | null;
  macd_dea: number | null;
  macd_hist: number | null;
  atr14: number | null;
}

export interface ModelVersionItem {
  version_id: string;
  created_at: string;
  model_type: string;
  train_start: string;
  train_end: string;
  embargo_days: number;
  oos_ic: number;
  oos_rank_ic: number;
  oos_icir: number;
  feature_importance: Record<string, { gain: number; shap: number }>;
  is_active: boolean;
  notes: string;
}

export interface WeeklyReviewItem {
  review_id: string;
  week_start: string;
  week_end: string;
  created_at: string;
  paper_return: number;
  theory_bt_return: number;
  tracking_error: number;
  slippage_drag: number;
  missed_fill_impact: number;
  decayed_factors: Array<{
    factor_name: string;
    label: string;
    dimension: string;
    hist_rank_ic: number;
    recent_20d_rank_ic: number;
    decay_status: string;
  }>;
  optimization_advice: Array<{
    type: string;
    severity: 'HIGH' | 'MEDIUM' | 'LOW';
    title: string;
    detail: string;
    suggested_action: string;
  }>;
  model_version_id: string;
}

export interface RiskConfigData {
  max_single_weight: number;
  max_industry_weight: number;
  stock_hard_stop_pct: number;
  stock_atr_mult: number;
  drawdown_warn_pct: number;
  drawdown_halt_pct: number;
  max_orders_per_min: number;
  max_orders_per_day: number;
}

export interface BrokerConfigData {
  active_mode: 'PAPER' | 'ASSISTED' | 'QMT';
  wecom_webhook_url: string;
  dingtalk_webhook_url: string;
  dingtalk_secret: string;
  has_dingtalk_secret: boolean;
  qmt_userdata_path: string;
  qmt_account_id: string;
  qmt_dry_run: boolean;
  tushare_token_configured: boolean;
  adapters_health: Record<string, any>;
}
