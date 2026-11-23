import React, { useEffect, useMemo, useState } from 'react';
import { Alert, Button, Spin, Table, Tag, message } from 'antd';
import { CalculatorOutlined } from '@ant-design/icons';
import { aShareColor, api, fmtPct } from '../services/api';
import { FactorSummaryItem } from '../types/quant';
import { EChartsBox } from '../components/charts/EChartsBox';

const DECAY_TAG: Record<string, { color: string; text: string }> = {
  STABLE: { color: 'cyan', text: '稳定有效' },
  DECAYING: { color: 'warning', text: '近20日衰减' },
  REVERSED: { color: 'error', text: 'IC反转预警' },
  INSUFFICIENT: { color: 'default', text: '样本不足' },
};

export const FactorLabPage: React.FC = () => {
  const [factors, setFactors] = useState<FactorSummaryItem[]>([]);
  const [selected, setSelected] = useState<string>('ret_20d');
  const [detail, setDetail] = useState<any>(null);
  const [corr, setCorr] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [computing, setComputing] = useState(false);

  const loadAll = async () => {
    setLoading(true);
    try {
      const [sum, c] = await Promise.all([api.getFactorsSummary(5), api.getFactorCorrelation()]);
      const list: FactorSummaryItem[] = sum.factors || [];
      setFactors(list);
      setCorr(c);
      if (list.length && !list.some((x) => x.factor_name === selected)) {
        setSelected(list[0].factor_name);
      }
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '因子数据加载失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAll();
  }, []);

  useEffect(() => {
    if (!selected) return;
    api
      .getFactorDetail(selected, 5)
      .then(setDetail)
      .catch(() => {});
  }, [selected]);

  const handleRecompute = async () => {
    setComputing(true);
    try {
      await api.recomputeFactors(5);
      await loadAll();
      const d = await api.getFactorDetail(selected, 5);
      setDetail(d);
      message.success('已完成四维 17 因子重算与单因子检验');
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '重算失败');
    } finally {
      setComputing(false);
    }
  };

  const quintileOption = useMemo(() => {
    const q = detail?.quintile_nav || [];
    const dates = q.map((r: any) => r.date);
    const colors = ['#10B981', '#34D399', '#94A3B8', '#FB7185', '#F43F5E', '#38BDF8'];
    const keys = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'LS'];
    return {
      tooltip: { trigger: 'axis' as const },
      legend: { top: 0, textStyle: { color: '#CBD5E1', fontSize: 11 }, data: keys },
      grid: { left: 46, right: 16, top: 32, bottom: 24 },
      xAxis: { type: 'category' as const, data: dates, axisLine: { lineStyle: { color: '#334155' } } },
      yAxis: { type: 'value' as const, scale: true, splitLine: { lineStyle: { color: '#1E293B' } } },
      series: keys.map((k, idx) => ({
        name: k,
        type: 'line' as const,
        showSymbol: false,
        lineStyle: { color: colors[idx], width: k === 'LS' || k === 'Q5' ? 2.2 : 1.4, type: k === 'LS' ? ('dashed' as const) : ('solid' as const) },
        data: q.map((r: any) => r[k]),
      })),
    };
  }, [detail]);

  const icOption = useMemo(() => {
    const s = detail?.ic_series || [];
    return {
      tooltip: { trigger: 'axis' as const },
      legend: { top: 0, textStyle: { color: '#CBD5E1', fontSize: 11 }, data: ['日RankIC', '累计RankIC', '20日滚动IC'] },
      grid: { left: 46, right: 46, top: 32, bottom: 24 },
      xAxis: { type: 'category' as const, data: s.map((x: any) => x.date), axisLine: { lineStyle: { color: '#334155' } } },
      yAxis: [
        { type: 'value' as const, splitLine: { lineStyle: { color: '#1E293B' } } },
        { type: 'value' as const, splitLine: { show: false } },
      ],
      series: [
        {
          name: '日RankIC',
          type: 'bar' as const,
          data: s.map((x: any) => ({
            value: x.rank_ic ?? 0,
            itemStyle: { color: (x.rank_ic ?? 0) >= 0 ? 'rgba(244,63,94,0.45)' : 'rgba(16,185,129,0.45)' },
          })),
        },
        {
          name: '20日滚动IC',
          type: 'line' as const,
          showSymbol: false,
          lineStyle: { color: '#F59E0B', width: 1.8 },
          data: s.map((x: any) => x.rolling_20d_ic),
        },
        {
          name: '累计RankIC',
          type: 'line' as const,
          yAxisIndex: 1,
          showSymbol: false,
          lineStyle: { color: '#38BDF8', width: 2.2 },
          data: s.map((x: any) => x.cum_rank_ic),
        },
      ],
    };
  }, [detail]);

  const heatmapOption = useMemo(() => {
    const labels: string[] = corr?.labels || [];
    const mat: number[][] = corr?.matrix || [];
    const pts: Array<[number, number, number]> = [];
    for (let i = 0; i < mat.length; i++) {
      for (let j = 0; j < mat[i].length; j++) {
        pts.push([j, i, Number((mat[i][j] ?? 0).toFixed(2))]);
      }
    }
    return {
      tooltip: {
        position: 'top' as const,
        formatter: (p: any) => `${labels[p.value[1]]} × ${labels[p.value[0]]}: <b>${p.value[2]}</b>`,
      },
      grid: { left: 120, right: 50, top: 16, bottom: 95 },
      xAxis: {
        type: 'category' as const,
        data: labels,
        axisLabel: { rotate: 45, fontSize: 10, color: '#94A3B8' },
      },
      yAxis: {
        type: 'category' as const,
        data: labels,
        axisLabel: { fontSize: 10, color: '#94A3B8' },
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: false,
        orient: 'vertical' as const,
        right: 4,
        top: 'center',
        inRange: { color: ['#10B981', '#0F172A', '#F43F5E'] },
        textStyle: { color: '#94A3B8', fontSize: 10 },
      },
      series: [
        {
          type: 'heatmap' as const,
          data: pts,
          label: { show: false },
        },
      ],
    };
  }, [corr]);

  const degradedList = factors.filter((f) => f.data_status !== 'OK');

  const columns = [
    {
      title: '因子名称',
      dataIndex: 'label',
      render: (_: any, r: FactorSummaryItem) => (
        <div>
          <div className="font-semibold text-slate-100">{r.label}</div>
          <div className="num text-[11px] text-slate-400">{r.factor_name}</div>
        </div>
      ),
    },
    {
      title: '维度',
      dataIndex: 'dimension',
      render: (v: string) => <Tag color="geekblue">{v}</Tag>,
    },
    {
      title: 'RankIC均值',
      dataIndex: 'rank_ic_mean',
      align: 'right' as const,
      sorter: (a: FactorSummaryItem, b: FactorSummaryItem) => (a.rank_ic_mean ?? 0) - (b.rank_ic_mean ?? 0),
      render: (v: number | null) => <span className={`num font-semibold ${aShareColor(v)}`}>{v !== null ? v.toFixed(4) : '-'}</span>,
    },
    {
      title: 'RankICIR',
      dataIndex: 'rank_icir',
      align: 'right' as const,
      sorter: (a: FactorSummaryItem, b: FactorSummaryItem) => Math.abs(a.rank_icir ?? 0) - Math.abs(b.rank_icir ?? 0),
      render: (v: number | null) => <span className={`num font-bold ${aShareColor(v)}`}>{v !== null ? v.toFixed(3) : '-'}</span>,
    },
    {
      title: 'IC胜率',
      dataIndex: 'ic_win_rate',
      align: 'right' as const,
      render: (v: number | null) => <span className="num">{v !== null ? `${(v * 100).toFixed(1)}%` : '-'}</span>,
    },
    {
      title: '多空年化',
      dataIndex: 'ls_annual_return',
      align: 'right' as const,
      render: (v: number | null) => <span className={`num ${aShareColor(v)}`}>{fmtPct(v)}</span>,
    },
    {
      title: '近20日IC',
      dataIndex: 'recent_rank_ic_20d',
      align: 'right' as const,
      render: (v: number | null) => <span className={`num ${aShareColor(v)}`}>{v !== null ? v.toFixed(4) : '-'}</span>,
    },
    {
      title: '衰退监控',
      dataIndex: 'decay_status',
      render: (v: string, r: FactorSummaryItem) => {
        const t = DECAY_TAG[v] || DECAY_TAG.INSUFFICIENT;
        return (
          <div className="flex items-center gap-1">
            <Tag color={t.color} className="m-0">{t.text}</Tag>
            {r.data_status !== 'OK' && <Tag color="orange" className="m-0">降级</Tag>}
          </div>
        );
      },
    },
  ];

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
          <h1 className="text-xl font-bold text-slate-100 m-0">因子实验室 · 四维正交因子检验</h1>
          <div className="text-xs text-slate-400 mt-0.5">
            中位数填充 → MAD去极值 → 申万行业+对数市值OLS中性化 → 横截面Z-Score · 点击任意因子行切换右侧 5 分层净值与 IC 时序
          </div>
        </div>
        <Button type="primary" icon={<CalculatorOutlined />} loading={computing} onClick={handleRecompute}>
          重算全量四维因子
        </Button>
      </div>

      {degradedList.length > 0 && (
        <Alert
          type="warning"
          showIcon
          message={
            degradedList.some((d) => d.data_status === 'DEGRADED_MISSING')
              ? `数据源优雅降级提示：${degradedList.map((d) => d.label).join('、')} 当前接口缺失，已打标 DEGRADED_MISSING 并以中性值安全处理（严禁使用随机数替代）`
              : `数据源部分覆盖提示：${degradedList.map((d) => d.label).join('、')} 近 60 日覆盖尚不完整（PARTIAL_DEGRADED），已有真实日期正常参与截面计算，历史缺失日期以中性值安全处理`
          }
        />
      )}

      {/* Table + Selected Factor Charts */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
        <div className="quant-card p-4 xl:col-span-7 overflow-hidden">
          <div className="text-sm font-semibold text-slate-200 mb-3">单因子 RankIC / ICIR 排行榜 (未来5日调仓周期)</div>
          <Table
            rowKey="factor_name"
            size="small"
            pagination={false}
            scroll={{ x: 'max-content' }}
            dataSource={factors}
            columns={columns}
            onRow={(record) => ({
              onClick: () => setSelected(record.factor_name),
              className: record.factor_name === selected ? 'bg-sky-950/50 cursor-pointer' : 'cursor-pointer',
            })}
          />
        </div>

        <div className="xl:col-span-5 space-y-4">
          <div className="quant-card p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-slate-200">
                5 分层净值曲线 (Q1~Q5 & 多空 LS) · <span className="text-sky-400">{detail?.meta?.label || selected}</span>
              </span>
            </div>
            <EChartsBox option={quintileOption} height={240} />
          </div>

          <div className="quant-card p-4">
            <div className="text-sm font-semibold text-slate-200 mb-2">
              日度 RankIC、20日滚动与累计 IC · <span className="text-sky-400">{detail?.meta?.label || selected}</span>
            </div>
            <EChartsBox option={icOption} height={220} />
          </div>
        </div>
      </div>

      {/* Correlation Heatmap */}
      <div className="quant-card p-4">
        <div className="text-sm font-semibold text-slate-200 mb-2">17×17 因子横截面 Spearman 相关性热力图</div>
        <EChartsBox option={heatmapOption} height={420} />
      </div>
    </div>
  );
};
