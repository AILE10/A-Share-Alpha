import React, { useEffect, useState } from 'react';
import { Layout, Menu, Tag, Badge, Modal, Form, InputNumber, Button, message, Tooltip } from 'antd';
import {
  DashboardOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  RadarChartOutlined,
  FundProjectionScreenOutlined,
  SafetyCertificateOutlined,
  SyncOutlined,
  WifiOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import { useLocation, useNavigate } from 'react-router-dom';
import { api } from '../../services/api';
import { useAlphaWebSocket } from '../../services/ws';

const { Header, Sider, Content } = Layout;

interface Props {
  children: React.ReactNode;
}

const REGIME_META: Record<string, { label: string; color: string; dot: string }> = {
  OFFENSIVE: { label: '强势进攻', color: 'error', dot: 'bg-rose-500' },
  NEUTRAL: { label: '震荡中性', color: 'warning', dot: 'bg-amber-400' },
  DEFENSIVE: { label: '弱势防守', color: 'success', dot: 'bg-emerald-500' },
  UNKNOWN: { label: '待同步', color: 'default', dot: 'bg-slate-500' },
};

const BROKER_META: Record<string, { label: string; color: string }> = {
  PAPER: { label: '高精度模拟盘', color: 'blue' },
  ASSISTED: { label: '半自动辅助', color: 'gold' },
  QMT: { label: '合规QMT桥接', color: 'purple' },
};

export const AppShell: React.FC<Props> = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const [regime, setRegime] = useState<string>('NEUTRAL');
  const [targetExp, setTargetExp] = useState<number>(0.55);
  const [regimeDetail, setRegimeDetail] = useState<any>(null);
  const [refreshingRegime, setRefreshingRegime] = useState(false);
  const [brokerMode, setBrokerMode] = useState<string>('PAPER');
  const [fuseStatus, setFuseStatus] = useState<string>('NORMAL');
  const [stagedOrder, setStagedOrder] = useState<any | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [form] = Form.useForm();

  const { connected } = useAlphaWebSocket((evt) => {
    if (evt.channel === 'market_regime' && evt.payload?.regime) {
      setRegime(evt.payload.regime);
      if (evt.payload.target_exposure !== undefined) setTargetExp(evt.payload.target_exposure);
      setRegimeDetail((prev: any) => ({ ...(prev || {}), ...evt.payload }));
    }
    if (evt.channel === 'paper_nav' && evt.payload?.fuse_status) {
      setFuseStatus(evt.payload.fuse_status);
    }
    if (evt.channel === 'assisted_order' && evt.payload?.order_id) {
      setStagedOrder(evt.payload);
      form.setFieldsValue({
        filled_qty: evt.payload.request_qty,
        avg_fill_price: evt.payload.limit_price,
      });
    }
    if (evt.channel === 'risk_alert') {
      message.warning(`风控告警: ${evt.payload?.type || '触发风险规则'}`);
    }
  });

  useEffect(() => {
    api
      .getDashboardSummary()
      .then((d) => {
        if (d.regime?.regime) setRegime(d.regime.regime);
        if (d.regime?.target_exposure !== undefined) setTargetExp(d.regime.target_exposure);
        if (d.regime) setRegimeDetail(d.regime);
        if (d.broker_mode) setBrokerMode(d.broker_mode);
        if (d.risk_watermarks?.fuse_status) setFuseStatus(d.risk_watermarks.fuse_status);
      })
      .catch(() => {});
  }, [location.pathname]);

  const handleRefreshRegime = async () => {
    if (refreshingRegime) return;
    setRefreshingRegime(true);
    try {
      const snap = await api.refreshRealtimeRegime();
      if (snap?.regime) setRegime(snap.regime);
      if (snap?.target_exposure !== undefined) setTargetExp(snap.target_exposure);
      setRegimeDetail(snap);
      message.success(
        `实时大盘已刷新 (${snap.quote_time || snap.trade_date}) · 中证1000 ${snap.csi1000_close} · MA20宽度 ${(snap.breadth_above_ma20 * 100).toFixed(1)}%`
      );
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '实时大盘状态刷新失败');
    } finally {
      setRefreshingRegime(false);
    }
  };

  const menuItems = [
    { key: '/', icon: <DashboardOutlined />, label: '策略总览与大盘' },
    { key: '/factors', icon: <ExperimentOutlined />, label: '因子实验室' },
    { key: '/backtest', icon: <LineChartOutlined />, label: '回测与样本外优化' },
    { key: '/signals', icon: <RadarChartOutlined />, label: '实时选股与K线' },
    { key: '/paper', icon: <FundProjectionScreenOutlined />, label: '模拟实盘与复盘' },
    { key: '/broker-risk', icon: <SafetyCertificateOutlined />, label: '券商网关与风控' },
  ];

  const regInfo = REGIME_META[regime] || REGIME_META.UNKNOWN;
  const brkInfo = BROKER_META[brokerMode] || BROKER_META.PAPER;

  const handleConfirmFill = async () => {
    if (!stagedOrder) return;
    try {
      const vals = await form.validateFields();
      setConfirming(true);
      await api.confirmAssistedFill(stagedOrder.order_id, vals);
      message.success(`已回写实盘成交 ${stagedOrder.ts_code}`);
      setStagedOrder(null);
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '成交回写失败');
    } finally {
      setConfirming(false);
    }
  };

  return (
    <Layout className="min-h-screen bg-[#0B0F19]">
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        width={220}
        className="border-r border-slate-800/80"
        style={{ background: '#0E1422' }}
      >
        <div className="h-14 flex items-center gap-2.5 px-4 border-b border-slate-800/80">
          <div className="w-7 h-7 rounded-md bg-gradient-to-br from-rose-500 to-sky-500 flex items-center justify-center shrink-0">
            <ThunderboltOutlined className="text-white text-sm" />
          </div>
          {!collapsed && (
            <div className="overflow-hidden">
              <div className="text-sm font-bold tracking-wide text-slate-100 leading-tight">
                A-Share Alpha
              </div>
              <div className="text-[10px] text-sky-400 tracking-widest uppercase">QUANT STUDIO</div>
            </div>
          )}
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{ background: 'transparent', borderRight: 0, marginTop: 8 }}
        />
      </Sider>

      <Layout className="bg-[#0B0F19]">
        <Header
          className="px-5 flex items-center justify-between border-b border-slate-800/80"
          style={{ background: '#0F1524', height: 56, lineHeight: '56px' }}
        >
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-xs">
              <span className={`w-2.5 h-2.5 rounded-full ${regInfo.dot} shadow-sm`} />
              <span className="text-slate-400">大盘状态:</span>
              <Tooltip
                title={
                  regimeDetail ? (
                    <div className="text-xs space-y-0.5">
                      <div>
                        时间: {regimeDetail.updated_at || regimeDetail.trade_date || '-'}{' '}
                        {regimeDetail.is_realtime ? '(盘中全市场实时)' : '(收盘确认)'}
                      </div>
                      <div>
                        中证1000: {regimeDetail.csi1000_close ?? '-'}
                        {regimeDetail.csi1000_pct_chg !== undefined && regimeDetail.csi1000_pct_chg !== null
                          ? ` (${regimeDetail.csi1000_pct_chg >= 0 ? '+' : ''}${Number(regimeDetail.csi1000_pct_chg).toFixed(2)}%)`
                          : ''}
                        {' · '}MA20 {regimeDetail.csi1000_ma20 ?? '-'} / MA60 {regimeDetail.csi1000_ma60 ?? '-'}
                      </div>
                      {regimeDetail.csi300_close !== undefined && regimeDetail.csi300_close !== null && (
                        <div>
                          沪深300: {regimeDetail.csi300_close}
                          {regimeDetail.csi300_pct_chg !== undefined && regimeDetail.csi300_pct_chg !== null
                            ? ` (${regimeDetail.csi300_pct_chg >= 0 ? '+' : ''}${Number(regimeDetail.csi300_pct_chg).toFixed(2)}%)`
                            : ''}
                          {' · '}
                          {regimeDetail.style_mode === 'LARGE_CAP_RESCUE' ? '二八分化 → 切前50%大中盘' : '全市场选股'}
                        </div>
                      )}
                      <div>
                        全市场站上MA20宽度:{' '}
                        {regimeDetail.breadth_above_ma20 !== undefined && regimeDetail.breadth_above_ma20 !== null
                          ? `${(Number(regimeDetail.breadth_above_ma20) * 100).toFixed(2)}%`
                          : '-'}
                        {regimeDetail.quoted_count ? ` (${regimeDetail.quoted_count}股)` : ''}
                      </div>
                      {regimeDetail.advancers !== undefined && (
                        <div>
                          盘中涨跌: 涨 {regimeDetail.advancers} / 跌 {regimeDetail.decliners}
                        </div>
                      )}
                    </div>
                  ) : (
                    '点击右侧图标立即刷新盘中全市场实时大盘状态'
                  )
                }
              >
                <Tag color={regInfo.color} className="m-0 font-semibold cursor-help">
                  {regInfo.label} ({Math.round(targetExp * 100)}%)
                  {regimeDetail?.is_realtime && regimeDetail?.quote_time
                    ? ` · 实时 ${String(regimeDetail.quote_time).slice(0, 5)}`
                    : regimeDetail?.trade_date
                      ? ` · ${String(regimeDetail.trade_date).slice(5)}`
                      : ''}
                </Tag>
              </Tooltip>
              <Tooltip title="立即拉取全市场 4,100+ 股实时行情刷新大盘状态 (~2秒)">
                <Button
                  type="text"
                  size="small"
                  icon={<SyncOutlined spin={refreshingRegime} className="text-sky-400" />}
                  onClick={handleRefreshRegime}
                  className="px-1 h-6 flex items-center"
                />
              </Tooltip>
            </div>
            <div className="hidden md:flex items-center gap-2 text-xs">
              <span className="text-slate-400">执行模式:</span>
              <Tag color={brkInfo.color} className="m-0">
                {brkInfo.label}
              </Tag>
            </div>
            <div className="hidden lg:flex items-center gap-2 text-xs">
              <span className="text-slate-400">风控水位:</span>
              <Tag
                color={fuseStatus === 'HALTED' ? 'error' : fuseStatus === 'WARNING' ? 'warning' : 'cyan'}
                className="m-0"
              >
                {fuseStatus === 'HALTED' ? '回撤熔断' : fuseStatus === 'WARNING' ? '回撤预警' : '风控正常'}
              </Tag>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Tooltip title={connected ? 'FastAPI WebSocket 实时推流已连接' : '正在重连 WebSocket...'}>
              <div className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800">
                <Badge status={connected ? 'processing' : 'error'} />
                <WifiOutlined className={connected ? 'text-emerald-400' : 'text-slate-500'} />
                <span className="text-slate-300">{connected ? '实时推流在线' : '断线重连中'}</span>
              </div>
            </Tooltip>
          </div>
        </Header>

        <Content className="p-5 overflow-auto">{children}</Content>
      </Layout>

      {/* Global Assisted Order Confirmation Modal triggered via WebSocket */}
      <Modal
        open={Boolean(stagedOrder)}
        title={`半自动辅助交易单确认 · ${stagedOrder?.side} ${stagedOrder?.ts_code} ${stagedOrder?.name || ''}`}
        onCancel={() => setStagedOrder(null)}
        footer={[
          <Button key="close" onClick={() => setStagedOrder(null)}>
            稍后在风控页处理
          </Button>,
          <Button key="ok" type="primary" loading={confirming} onClick={handleConfirmFill}>
            确认已实盘成交并回写持仓
          </Button>,
        ]}
      >
        {stagedOrder && (
          <div className="space-y-3">
            <div className="p-3 rounded bg-slate-900 border border-slate-800 text-xs space-y-1">
              <div>单号: <span className="num text-sky-400">{stagedOrder.order_id}</span></div>
              <div>建议数量: <span className="num font-bold">{stagedOrder.request_qty} 股</span> | 参考价: <span className="num">¥{stagedOrder.limit_price}</span></div>
              <div>触发原因: {stagedOrder.reason_summary}</div>
            </div>
            <Form form={form} layout="vertical">
              <Form.Item name="filled_qty" label="实盘成交股数" rules={[{ required: true }]}>
                <InputNumber min={100} step={100} className="w-full" />
              </Form.Item>
              <Form.Item name="avg_fill_price" label="实盘成交均价 (¥)" rules={[{ required: true }]}>
                <InputNumber min={0.01} step={0.01} className="w-full" />
              </Form.Item>
            </Form>
          </div>
        )}
      </Modal>
    </Layout>
  );
};
