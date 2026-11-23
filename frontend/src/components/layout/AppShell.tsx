import React, { useEffect, useState } from 'react';
import { Layout, Menu, Tag, Badge, Modal, Form, InputNumber, Button, message, Tooltip } from 'antd';
import {
  DashboardOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  RadarChartOutlined,
  FundProjectionScreenOutlined,
  SafetyCertificateOutlined,
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
  const [brokerMode, setBrokerMode] = useState<string>('PAPER');
  const [fuseStatus, setFuseStatus] = useState<string>('NORMAL');
  const [stagedOrder, setStagedOrder] = useState<any | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [form] = Form.useForm();

  const { connected } = useAlphaWebSocket((evt) => {
    if (evt.channel === 'market_regime' && evt.payload?.regime) {
      setRegime(evt.payload.regime);
      if (evt.payload.target_exposure !== undefined) setTargetExp(evt.payload.target_exposure);
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
        if (d.broker_mode) setBrokerMode(d.broker_mode);
        if (d.risk_watermarks?.fuse_status) setFuseStatus(d.risk_watermarks.fuse_status);
      })
      .catch(() => {});
  }, [location.pathname]);

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
              <Tag color={regInfo.color} className="m-0 font-semibold">
                {regInfo.label} ({Math.round(targetExp * 100)}%)
              </Tag>
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
