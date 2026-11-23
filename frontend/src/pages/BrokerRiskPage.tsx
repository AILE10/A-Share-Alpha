import React, { useEffect, useState } from 'react';
import {
  Alert,
  Button,
  Form,
  Input,
  InputNumber,
  Modal,
  Radio,
  Select,
  Spin,
  Switch,
  Table,
  Tag,
  message,
} from 'antd';
import {
  BellOutlined,
  CheckCircleOutlined,
  SafetyOutlined,
  SaveOutlined,
  UndoOutlined,
} from '@ant-design/icons';
import { api, fmtNum } from '../services/api';
import { BrokerConfigData, RiskConfigData } from '../types/quant';

export const BrokerRiskPage: React.FC = () => {
  const [brokerCfg, setBrokerCfg] = useState<BrokerConfigData | null>(null);
  const [riskCfg, setRiskCfg] = useState<RiskConfigData | null>(null);
  const [pending, setPending] = useState<any[]>([]);
  const [auditLogs, setAuditLogs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [savingBroker, setSavingBroker] = useState(false);
  const [savingRisk, setSavingRisk] = useState(false);
  const [fillTarget, setFillTarget] = useState<any | null>(null);

  const [brokerForm] = Form.useForm();
  const [riskForm] = Form.useForm();
  const [fillForm] = Form.useForm();
  const [manualForm] = Form.useForm();

  const loadAll = async () => {
    try {
      const [bc, rc, pend, logs] = await Promise.all([
        api.getBrokerConfig(),
        api.getRiskConfig(),
        api.listAssistedPending(),
        api.getRiskAuditLogs(80),
      ]);
      setBrokerCfg(bc);
      setRiskCfg(rc);
      setPending(pend);
      setAuditLogs(logs);
      brokerForm.setFieldsValue(bc);
      riskForm.setFieldsValue(rc);
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '风控与券商配置加载失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAll();
  }, []);

  const handleSaveBroker = async () => {
    setSavingBroker(true);
    try {
      const vals = await brokerForm.validateFields();
      const updated = await api.updateBrokerConfig(vals);
      setBrokerCfg(updated);
      message.success(`已保存券商网关配置（当前模式: ${updated.active_mode}）`);
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '保存失败');
    } finally {
      setSavingBroker(false);
    }
  };

  const handleSaveRisk = async () => {
    setSavingRisk(true);
    try {
      const vals = await riskForm.validateFields();
      const updated = await api.updateRiskConfig(vals);
      setRiskCfg(updated);
      message.success('前置风控中间件阈值已生效');
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '保存失败');
    } finally {
      setSavingRisk(false);
    }
  };

  const handleTestWebhook = async () => {
    try {
      const res = await api.testWebhooks();
      const summary = (res.results || []).map((r: any) => `${r.channel}: ${r.sent ? '成功' : r.reason || r.error}`).join(' | ');
      message.info(summary);
    } catch {
      message.error('Webhook 测试请求失败');
    }
  };

  const handleConfirmFill = async () => {
    if (!fillTarget) return;
    try {
      const vals = await fillForm.validateFields();
      await api.confirmAssistedFill(fillTarget.order_id, vals);
      message.success(`实盘成交已回写持仓账本 (${fillTarget.ts_code})`);
      setFillTarget(null);
      loadAll();
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '回写失败');
    }
  };

  const handleManualTestOrder = async () => {
    try {
      const vals = await manualForm.validateFields();
      const res = await api.submitManualOrder(vals);
      if (res.status === 'REJECTED_RISK' || res.status === 'REJECTED_LIMIT') {
        message.warning(`前置风控已拦截: ${res.reject_reason}`);
      } else {
        message.success(`委托状态: ${res.status} (${res.order_id})`);
      }
      loadAll();
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '测试委托失败');
    }
  };

  if (loading) {
    return (
      <div className="h-96 flex items-center justify-center">
        <Spin size="large" />
      </div>
    );
  }

  const qmtHealth = brokerCfg?.adapters_health?.QMT || {};

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-slate-100 m-0">券商网关与前置风控设置</h1>
        <div className="text-xs text-slate-400 mt-0.5">
          统一 BaseBrokerAdapter 适配器模式 · 任何信号进入执行层前强制经过 RiskMiddleware 拦截与审计 · 严禁任何逆向抓包或模拟点击外挂
        </div>
      </div>

      <Alert
        type="info"
        showIcon
        message="合规底线声明"
        description="本系统仅提供本地模拟盘（Paper）、半自动人工辅助下单与 Webhook 提醒（Assisted），以及面向已开通券商官方程序化权限账户的迅投 MiniQMT / 恒生 PTrade 合规桥接骨架（默认 Dry-Run 安全模式）。"
      />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {/* Broker Mode & Webhook / QMT Config */}
        <div className="quant-card p-4 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-slate-100">券商执行模式与通知通道</span>
            <Button size="small" icon={<BellOutlined />} onClick={handleTestWebhook}>
              测试 Webhook 推送
            </Button>
          </div>

          <Form form={brokerForm} layout="vertical" size="small">
            <Form.Item name="active_mode" label="当前激活执行适配器">
              <Radio.Group optionType="button" buttonStyle="solid">
                <Radio.Button value="PAPER">PAPER 模拟盘</Radio.Button>
                <Radio.Button value="ASSISTED">ASSISTED 半自动辅助</Radio.Button>
                <Radio.Button value="QMT">QMT 合规桥接</Radio.Button>
              </Radio.Group>
            </Form.Item>

            <Form.Item name="wecom_webhook_url" label="企业微信机器人 Webhook URL">
              <Input placeholder="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=..." />
            </Form.Item>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <Form.Item name="dingtalk_webhook_url" label="钉钉机器人 Webhook URL">
                <Input placeholder="https://oapi.dingtalk.com/robot/send?access_token=..." />
              </Form.Item>
              <Form.Item name="dingtalk_secret" label="钉钉加签 Secret (可选)">
                <Input.Password placeholder="SEC..." />
              </Form.Item>
            </div>

            <div className="p-3 rounded bg-[#0F1624] border border-slate-800 space-y-2 mb-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-violet-300">合规迅投 MiniQMT / PTrade 接口配置</span>
                <Tag color={qmtHealth.ready_for_live ? 'success' : 'purple'}>
                  {qmtHealth.ready_for_live ? '实盘就绪' : 'Dry-Run 演练态'}
                </Tag>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                <Form.Item name="qmt_userdata_path" label="MiniQMT userdata_mini 路径" className="mb-1">
                  <Input placeholder="C:\国金QMT交易端\userdata_mini" />
                </Form.Item>
                <Form.Item name="qmt_account_id" label="券商资金账号" className="mb-1">
                  <Input placeholder="仅已开通程序化权限账户填写" />
                </Form.Item>
              </div>
              <Form.Item name="qmt_dry_run" label="Dry-Run 安全演练模式（开启后仅校验不向券商实盘发单）" valuePropName="checked" className="mb-1">
                <Switch />
              </Form.Item>
              {qmtHealth.issues?.length > 0 && (
                <div className="text-[11px] text-slate-400">
                  环境自检: {qmtHealth.issues.join('；')}
                </div>
              )}
            </div>

            <Button type="primary" icon={<SaveOutlined />} loading={savingBroker} onClick={handleSaveBroker}>
              保存券商网关配置
            </Button>
          </Form>
        </div>

        {/* Pre-trade RiskMiddleware Thresholds */}
        <div className="quant-card p-4 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-slate-100">
              <SafetyOutlined className="mr-1.5 text-sky-400" />
              前置风控中间件阈值 (RiskMiddleware)
            </span>
            <Button
              size="small"
              icon={<UndoOutlined />}
              onClick={async () => {
                await api.resetDrawdownFuse();
                message.success('已手动复位组合回撤熔断状态为 NORMAL');
                loadAll();
              }}
            >
              复位回撤熔断
            </Button>
          </div>

          <Form form={riskForm} layout="vertical" size="small">
            <div className="grid grid-cols-2 gap-3">
              <Form.Item name="max_single_weight" label="单票仓位上限 (默认 0.12)">
                <InputNumber min={0.02} max={0.5} step={0.01} className="w-full" />
              </Form.Item>
              <Form.Item name="max_industry_weight" label="单申万行业上限 (默认 0.25)">
                <InputNumber min={0.05} max={0.8} step={0.01} className="w-full" />
              </Form.Item>
              <Form.Item name="stock_hard_stop_pct" label="个股硬止损跌幅 (默认 0.08)">
                <InputNumber min={0.02} max={0.3} step={0.01} className="w-full" />
              </Form.Item>
              <Form.Item name="stock_atr_mult" label="ATR 移动止损倍数 (默认 2.5)">
                <InputNumber min={0.5} max={6.0} step={0.5} className="w-full" />
              </Form.Item>
              <Form.Item name="drawdown_warn_pct" label="组合回撤预警线 (减半仓)">
                <InputNumber min={0.02} max={0.3} step={0.01} className="w-full" />
              </Form.Item>
              <Form.Item name="drawdown_halt_pct" label="组合回撤熔断线 (禁买)">
                <InputNumber min={0.05} max={0.4} step={0.01} className="w-full" />
              </Form.Item>
              <Form.Item name="max_orders_per_min" label="每分钟报撤单频率上限">
                <InputNumber min={1} max={200} className="w-full" />
              </Form.Item>
              <Form.Item name="max_orders_per_day" label="每日报撤单总次数上限">
                <InputNumber min={1} max={2000} className="w-full" />
              </Form.Item>
            </div>
            <Button type="primary" icon={<SaveOutlined />} loading={savingRisk} onClick={handleSaveRisk}>
              保存风控红线阈值
            </Button>
          </Form>

          {/* Quick RiskMiddleware Verification Order Form */}
          <div className="pt-3 border-t border-slate-800">
            <div className="text-xs font-semibold text-slate-300 mb-2">实时验证委托（经当前激活模式与风控中间件）</div>
            <Form
              form={manualForm}
              layout="inline"
              size="small"
              initialValues={{ ts_code: '688187.SH', side: 'BUY', qty: 100 }}
              className="flex flex-wrap gap-2"
            >
              <Form.Item name="ts_code" rules={[{ required: true }]}>
                <Input style={{ width: 110 }} placeholder="代码" />
              </Form.Item>
              <Form.Item name="side">
                <Select style={{ width: 80 }} options={[{ value: 'BUY', label: '买入' }, { value: 'SELL', label: '卖出' }]} />
              </Form.Item>
              <Form.Item name="qty" rules={[{ required: true }]}>
                <InputNumber min={100} step={100} style={{ width: 95 }} />
              </Form.Item>
              <Form.Item name="ref_price">
                <InputNumber min={0.01} step={0.5} style={{ width: 95 }} placeholder="参考价(可选)" />
              </Form.Item>
              <Button size="small" type="default" onClick={handleManualTestOrder}>
                测试提交
              </Button>
            </Form>
          </div>
        </div>
      </div>

      {/* Assisted Pending Queue */}
      <div className="quant-card p-4">
        <div className="text-sm font-semibold text-slate-200 mb-2">
          半自动辅助待确认队列 ({pending.length})
        </div>
        <Table
          rowKey="order_id"
          size="small"
          pagination={false}
          dataSource={pending}
          columns={[
            { title: '单号', dataIndex: 'order_id', className: 'num text-xs text-sky-400' },
            { title: '标的', render: (_: any, r: any) => <span className="num">{r.ts_code} {r.name}</span> },
            { title: '方向', dataIndex: 'side', render: (v: string) => <Tag color={v === 'BUY' ? 'error' : 'success'}>{v}</Tag> },
            { title: '建议股数', dataIndex: 'request_qty', align: 'right' as const, className: 'num' },
            { title: '参考价', dataIndex: 'limit_price', align: 'right' as const, render: (v: number) => <span className="num">¥{fmtNum(v)}</span> },
            { title: '原因', dataIndex: 'reason_summary', className: 'text-xs text-slate-400' },
            {
              title: '人工成交回写',
              render: (_: any, r: any) => (
                <div className="flex gap-2">
                  <Button
                    size="small"
                    type="primary"
                    icon={<CheckCircleOutlined />}
                    onClick={() => {
                      setFillTarget(r);
                      fillForm.setFieldsValue({ filled_qty: r.request_qty, avg_fill_price: r.limit_price });
                    }}
                  >
                    录入实盘成交
                  </Button>
                  <Button
                    size="small"
                    danger
                    onClick={async () => {
                      await api.cancelAssistedOrder(r.order_id);
                      message.info('已撤销待确认辅助单');
                      loadAll();
                    }}
                  >
                    撤单
                  </Button>
                </div>
              ),
            },
          ]}
          locale={{ emptyText: '暂无待确认辅助交易单（切换至 ASSISTED 模式后触发调仓或提交测试单将显示于此）' }}
        />
      </div>

      {/* Risk Audit Logs */}
      <div className="quant-card p-4">
        <div className="text-sm font-semibold text-slate-200 mb-2">前置风控拦截与放行审计日志</div>
        <Table
          rowKey="id"
          size="small"
          pagination={{ pageSize: 8 }}
          dataSource={auditLogs}
          columns={[
            { title: '时间', dataIndex: 'timestamp', render: (v: string) => <span className="num text-xs">{v?.slice(0, 19).replace('T', ' ')}</span> },
            { title: '标的', dataIndex: 'ts_code', className: 'num' },
            { title: '方向', dataIndex: 'side', render: (v: string) => <Tag color={v === 'BUY' ? 'error' : 'success'}>{v}</Tag> },
            { title: '风控规则', dataIndex: 'rule_name', className: 'num text-xs' },
            { title: '判定', dataIndex: 'decision', render: (v: string) => <Tag color={v === 'PASS' ? 'cyan' : 'error'}>{v}</Tag> },
            { title: '详情', render: (_: any, r: any) => <span className="text-xs text-slate-400">{r.detail?.reason || JSON.stringify(r.detail)}</span> },
          ]}
        />
      </div>

      <Modal
        open={Boolean(fillTarget)}
        title={`录入实盘成交回写 · ${fillTarget?.ts_code} ${fillTarget?.name || ''}`}
        onCancel={() => setFillTarget(null)}
        onOk={handleConfirmFill}
        okText="确认回写持仓"
      >
        <Form form={fillForm} layout="vertical">
          <Form.Item name="filled_qty" label="实盘成交股数" rules={[{ required: true }]}>
            <InputNumber min={100} step={100} className="w-full" />
          </Form.Item>
          <Form.Item name="avg_fill_price" label="实盘成交均价 (¥)" rules={[{ required: true }]}>
            <InputNumber min={0.01} step={0.01} className="w-full" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};
