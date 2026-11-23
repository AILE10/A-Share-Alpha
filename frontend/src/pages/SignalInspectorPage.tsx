import React, { useEffect, useState } from 'react';
import { Button, Spin, Tag, message } from 'antd';
import { SyncOutlined, ThunderboltOutlined } from '@ant-design/icons';
import { api, fmtNum } from '../services/api';
import { SignalRecommendationItem } from '../types/quant';
import { KLineChart } from '../components/charts/KLineChart';

const ACTION_TAG: Record<string, { color: string; text: string }> = {
  BUY: { color: 'error', text: '建议买入' },
  SELL: { color: 'success', text: '建议调出' },
  HOLD: { color: 'processing', text: '缓冲带持有' },
};

export const SignalInspectorPage: React.FC = () => {
  const [signals, setSignals] = useState<SignalRecommendationItem[]>([]);
  const [meta, setMeta] = useState<{ regime?: any; model_version_id?: string }>({});
  const [selectedCode, setSelectedCode] = useState<string>('');
  const [kline, setKline] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [klineLoading, setKlineLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [ordering, setOrdering] = useState(false);

  const loadSignals = async () => {
    try {
      const res = await api.getLatestSignals();
      const items: SignalRecommendationItem[] = res.items || [];
      setSignals(items);
      setMeta({ regime: res.regime, model_version_id: res.model_version_id });
      if (items.length && !items.some((x) => x.ts_code === selectedCode)) {
        setSelectedCode(items[0].ts_code);
      }
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '获取推荐清单失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSignals();
  }, []);

  useEffect(() => {
    if (!selectedCode) return;
    setKlineLoading(true);
    api
      .getStockKLine(selectedCode, 160)
      .then(setKline)
      .catch(() => setKline(null))
      .finally(() => setKlineLoading(false));
  }, [selectedCode]);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      const res = await api.refreshSignals(true);
      const items: SignalRecommendationItem[] = res.items || [];
      setSignals(items);
      setMeta({ regime: res.regime, model_version_id: res.model_version_id });
      if (items.length) setSelectedCode(items[0].ts_code);
      message.success(`已刷新实时打分，共 ${items.length} 条可解释推荐信号（已广播至 WebSocket）`);
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '刷新信号失败');
    } finally {
      setRefreshing(false);
    }
  };

  const selectedSignal = signals.find((s) => s.ts_code === selectedCode) || null;

  const handleQuickOrder = async () => {
    if (!selectedSignal || selectedSignal.action === 'HOLD') return;
    setOrdering(true);
    try {
      const qty = Math.max(100, Math.floor((80_000 / Math.max(selectedSignal.ref_price, 1)) / 100) * 100);
      const receipt = await api.submitManualOrder({
        ts_code: selectedSignal.ts_code,
        side: selectedSignal.action,
        qty,
        ref_price: selectedSignal.ref_price,
        reason_summary: selectedSignal.rationale,
      });
      if (receipt.status === 'REJECTED_RISK' || receipt.status === 'REJECTED_LIMIT') {
        message.warning(`下单被拦截: ${receipt.reject_reason}`);
      } else {
        message.success(`委托提交成功 [${receipt.broker_mode}] 状态: ${receipt.status} (${receipt.order_id})`);
      }
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '下单失败');
    } finally {
      setOrdering(false);
    }
  };

  if (loading) {
    return (
      <div className="h-96 flex items-center justify-center">
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-slate-100 m-0">实时选股推荐与 K 线透视</h1>
          <div className="text-xs text-slate-400 mt-0.5">
            激活模型: <span className="num text-sky-400">{meta.model_version_id || '-'}</span> · 每条推荐附带综合排名、Top 3 TreeSHAP 因子分值、技术共振形态与参考止损/止盈位
          </div>
        </div>
        <Button type="primary" icon={<SyncOutlined spin={refreshing} />} loading={refreshing} onClick={handleRefresh}>
          实时快照重算推荐清单
        </Button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
        {/* Left: Recommendation List */}
        <div className="quant-card p-3 xl:col-span-3 max-h-[680px] overflow-y-auto space-y-2">
          <div className="text-xs font-semibold text-slate-400 px-1 flex justify-between">
            <span>买卖/持有清单 ({signals.length})</span>
            <span>排名 / 仓位</span>
          </div>
          {signals.map((s) => {
            const act = ACTION_TAG[s.action] || ACTION_TAG.HOLD;
            const active = s.ts_code === selectedCode;
            return (
              <div
                key={s.signal_id}
                onClick={() => setSelectedCode(s.ts_code)}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  active
                    ? 'bg-sky-950/60 border-sky-500/80 shadow-md'
                    : 'bg-[#0F1624] border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <Tag color={act.color} className="m-0 text-[11px]">{act.text}</Tag>
                    <span className="font-semibold text-slate-100 text-sm">{s.name}</span>
                  </div>
                  <span className="num text-xs text-amber-400 font-bold">#{s.composite_rank}</span>
                </div>
                <div className="flex items-center justify-between mt-1.5 text-xs text-slate-400">
                  <span className="num">{s.ts_code} · {s.sw_l1_name}</span>
                  <span className="num text-slate-200">
                    ¥{fmtNum(s.ref_price)} · <b className="text-sky-400">{(s.suggested_weight * 100).toFixed(1)}%</b>
                  </span>
                </div>
              </div>
            );
          })}
          {!signals.length && <div className="text-slate-500 text-xs text-center py-8">暂无推荐信号</div>}
        </div>

        {/* Center: TradingView Lightweight Charts K-Line */}
        <div className="quant-card p-4 xl:col-span-6">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
            <div>
              <span className="text-lg font-bold text-slate-100">{kline?.name || selectedSignal?.name || '-'}</span>
              <span className="num ml-2 text-sm text-sky-400">{kline?.ts_code || selectedCode}</span>
              {kline?.sw_l1_name && <Tag color="geekblue" className="ml-2">{kline.sw_l1_name}</Tag>}
              {kline?.board && <Tag className="ml-1">{kline.board}</Tag>}
            </div>
            {selectedSignal && (
              <div className="text-xs text-slate-400">
                止损参考 <b className="num text-emerald-400">¥{fmtNum(selectedSignal.stop_loss_price)}</b> · 止盈参考{' '}
                <b className="num text-rose-400">¥{fmtNum(selectedSignal.take_profit_price)}</b>
              </div>
            )}
          </div>
          {klineLoading ? (
            <div className="h-[520px] flex items-center justify-center"><Spin /></div>
          ) : kline?.candles?.length ? (
            <KLineChart
              candles={kline.candles}
              markers={kline.markers || []}
              stopLossPrice={selectedSignal?.stop_loss_price}
              takeProfitPrice={selectedSignal?.take_profit_price}
              refPrice={selectedSignal?.ref_price}
              height={390}
            />
          ) : (
            <div className="h-80 flex items-center justify-center text-slate-500">暂无 K 线数据</div>
          )}
        </div>

        {/* Right: Structured Explainability Card */}
        <div className="quant-card p-4 xl:col-span-3 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-slate-100">结构化推荐归因卡片</span>
            {selectedSignal && (
              <Tag color={(ACTION_TAG[selectedSignal.action] || ACTION_TAG.HOLD).color} className="m-0">
                {(ACTION_TAG[selectedSignal.action] || ACTION_TAG.HOLD).text}
              </Tag>
            )}
          </div>

          {selectedSignal ? (
            <>
              <div className="grid grid-cols-2 gap-2">
                <div className="p-2.5 rounded bg-[#0F1624] border border-slate-800">
                  <div className="text-[11px] text-slate-400">模型综合排名</div>
                  <div className="num text-lg font-bold text-amber-400">#{selectedSignal.composite_rank}</div>
                </div>
                <div className="p-2.5 rounded bg-[#0F1624] border border-slate-800">
                  <div className="text-[11px] text-slate-400">建议目标仓位</div>
                  <div className="num text-lg font-bold text-sky-400">
                    {(selectedSignal.suggested_weight * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="p-2.5 rounded bg-[#0F1624] border border-slate-800">
                  <div className="text-[11px] text-slate-400">参考止损价 (8%/2.5ATR)</div>
                  <div className="num text-base font-bold text-emerald-400">¥{fmtNum(selectedSignal.stop_loss_price)}</div>
                </div>
                <div className="p-2.5 rounded bg-[#0F1624] border border-slate-800">
                  <div className="text-[11px] text-slate-400">参考止盈目标</div>
                  <div className="num text-base font-bold text-rose-400">
                    {selectedSignal.take_profit_price > 0 ? `¥${fmtNum(selectedSignal.take_profit_price)}` : '-'}
                  </div>
                </div>
              </div>

              {/* Top 3 Contributing Factors */}
              <div className="space-y-2">
                <div className="text-xs font-semibold text-slate-300">Top 3 贡献因子 (LightGBM TreeSHAP)</div>
                {(selectedSignal.top_factors || []).map((f) => {
                  const pos = f.shap >= 0;
                  const barPct = Math.min(100, Math.round(Math.abs(f.shap) * 180));
                  return (
                    <div key={f.name} className="p-2.5 rounded bg-[#0F1624] border border-slate-800 space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="font-medium text-slate-200">{f.label}</span>
                        <span className={`num font-semibold ${pos ? 'text-rose-400' : 'text-emerald-400'}`}>
                          SHAP {f.shap >= 0 ? '+' : ''}{f.shap.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex justify-between text-[11px] text-slate-400 num">
                        <span>原始值: {f.raw_value !== null ? f.raw_value.toFixed(4) : '-'}</span>
                        <span>Z-Score: {f.std_value !== null ? f.std_value.toFixed(2) : '-'}</span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-800 rounded overflow-hidden">
                        <div
                          className={pos ? 'h-full bg-rose-500' : 'h-full bg-emerald-500'}
                          style={{ width: `${Math.max(8, barPct)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Technical Resonance Patterns */}
              <div className="space-y-1.5">
                <div className="text-xs font-semibold text-slate-300">触发技术面共振形态</div>
                <div className="flex flex-wrap gap-1.5">
                  {(selectedSignal.tech_patterns || []).length > 0 ? (
                    selectedSignal.tech_patterns.map((p) => (
                      <Tag key={p} color={p.startsWith('风险') ? 'error' : 'cyan'} className="m-0">
                        {p}
                      </Tag>
                    ))
                  ) : (
                    <span className="text-xs text-slate-500">常规趋势持仓</span>
                  )}
                </div>
              </div>

              <div className="p-2.5 rounded bg-slate-900/90 border border-slate-800 text-xs text-slate-300 leading-relaxed">
                {selectedSignal.rationale}
              </div>

              {selectedSignal.action !== 'HOLD' && (
                <Button
                  type="primary"
                  danger={selectedSignal.action === 'BUY'}
                  block
                  icon={<ThunderboltOutlined />}
                  loading={ordering}
                  onClick={handleQuickOrder}
                >
                  经前置风控提交 {selectedSignal.action} 委托
                </Button>
              )}
            </>
          ) : (
            <div className="text-slate-500 text-xs py-8 text-center">请选择左侧推荐标的查看归因</div>
          )}
        </div>
      </div>
    </div>
  );
};
