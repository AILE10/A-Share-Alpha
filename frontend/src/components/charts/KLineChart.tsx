import React, { useEffect, useMemo, useRef } from 'react';
import { ColorType, createChart, CrosshairMode, LineStyle, SeriesMarker, Time } from 'lightweight-charts';
import { KLineCandle } from '../../types/quant';
import { EChartsBox } from './EChartsBox';

interface MarkerSpec {
  time: string;
  position: 'aboveBar' | 'belowBar' | 'inBar';
  color: string;
  shape: 'arrowUp' | 'arrowDown' | 'circle' | 'square';
  text: string;
}

interface Props {
  candles: KLineCandle[];
  markers?: MarkerSpec[];
  stopLossPrice?: number;
  takeProfitPrice?: number;
  refPrice?: number;
  height?: number;
}

export const KLineChart: React.FC<Props> = ({
  candles,
  markers = [],
  stopLossPrice,
  takeProfitPrice,
  refPrice,
  height = 380,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || !candles.length) return;

    const chart = createChart(el, {
      width: el.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: '#0F1624' },
        textColor: '#94A3B8',
        fontFamily: 'JetBrains Mono, Consolas, monospace',
      },
      grid: {
        vertLines: { color: '#1E293B' },
        horzLines: { color: '#1E293B' },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#1E293B' },
      timeScale: { borderColor: '#1E293B', timeVisible: false },
    });

    // Authentic A-share red-up / green-down candlestick styling
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#F43F5E',
      downColor: '#10B981',
      borderUpColor: '#F43F5E',
      borderDownColor: '#10B981',
      wickUpColor: '#F43F5E',
      wickDownColor: '#10B981',
    });

    candleSeries.setData(
      candles.map((c) => ({
        time: c.time as Time,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }))
    );

    if (markers.length > 0) {
      const sorted: SeriesMarker<Time>[] = [...markers]
        .sort((a, b) => a.time.localeCompare(b.time))
        .map((m) => ({
          time: m.time as Time,
          position: m.position,
          color: m.color,
          shape: m.shape,
          text: m.text,
        }));
      candleSeries.setMarkers(sorted);
    }

    if (stopLossPrice && stopLossPrice > 0) {
      candleSeries.createPriceLine({
        price: stopLossPrice,
        color: '#10B981',
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: `止损 ¥${stopLossPrice.toFixed(2)}`,
      });
    }
    if (takeProfitPrice && takeProfitPrice > 0) {
      candleSeries.createPriceLine({
        price: takeProfitPrice,
        color: '#F43F5E',
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: `止盈 ¥${takeProfitPrice.toFixed(2)}`,
      });
    }
    if (refPrice && refPrice > 0) {
      candleSeries.createPriceLine({
        price: refPrice,
        color: '#38BDF8',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: true,
        title: `参考 ¥${refPrice.toFixed(2)}`,
      });
    }

    // Moving Averages overlay (MA5, MA10, MA20, MA60)
    const addMa = (key: 'ma5' | 'ma10' | 'ma20' | 'ma60', color: string, width = 1) => {
      const s = chart.addLineSeries({
        color,
        lineWidth: width as 1 | 2,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      s.setData(
        candles
          .filter((c) => c[key] !== null)
          .map((c) => ({ time: c.time as Time, value: Number(c[key]) }))
      );
    };
    addMa('ma5', '#F59E0B', 1);
    addMa('ma10', '#A78BFA', 1);
    addMa('ma20', '#38BDF8', 2);
    addMa('ma60', '#64748B', 1);

    // Volume histogram at bottom 18%
    const volSeries = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    });
    volSeries.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    volSeries.setData(
      candles.map((c) => ({
        time: c.time as Time,
        value: c.volume,
        color: c.close >= c.open ? 'rgba(244,63,94,0.35)' : 'rgba(16,185,129,0.35)',
      }))
    );

    chart.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
      if (el) chart.applyOptions({ width: el.clientWidth });
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.remove();
    };
  }, [candles, markers, stopLossPrice, takeProfitPrice, refPrice, height]);

  const macdOption = useMemo(() => {
    const dates = candles.map((c) => c.time);
    return {
      tooltip: { trigger: 'axis' as const },
      grid: { left: 48, right: 20, top: 22, bottom: 24 },
      xAxis: { type: 'category' as const, data: dates, axisLine: { lineStyle: { color: '#334155' } } },
      yAxis: {
        type: 'value' as const,
        splitLine: { lineStyle: { color: '#1E293B' } },
      },
      legend: {
        top: 0,
        right: 12,
        textStyle: { color: '#94A3B8', fontSize: 11 },
        data: ['DIF', 'DEA', 'MACD柱'],
      },
      series: [
        {
          name: 'MACD柱',
          type: 'bar' as const,
          data: candles.map((c) => ({
            value: c.macd_hist ?? 0,
            itemStyle: { color: (c.macd_hist ?? 0) >= 0 ? '#F43F5E' : '#10B981' },
          })),
        },
        {
          name: 'DIF',
          type: 'line' as const,
          showSymbol: false,
          lineStyle: { color: '#38BDF8', width: 1.5 },
          data: candles.map((c) => c.macd_dif),
        },
        {
          name: 'DEA',
          type: 'line' as const,
          showSymbol: false,
          lineStyle: { color: '#F59E0B', width: 1.5 },
          data: candles.map((c) => c.macd_dea),
        },
      ],
    };
  }, [candles]);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-4 text-xs text-slate-400 px-1">
        <span className="text-amber-400">MA5</span>
        <span className="text-violet-400">MA10</span>
        <span className="text-sky-400 font-semibold">MA20 (核心共振线)</span>
        <span className="text-slate-400">MA60</span>
        <span className="ml-auto text-slate-500">红涨绿跌 · TradingView Lightweight Charts</span>
      </div>
      <div ref={containerRef} className="w-full rounded-lg overflow-hidden border border-slate-800" />
      <div className="rounded-lg border border-slate-800 bg-[#0F1624] p-2">
        <EChartsBox option={macdOption} height={140} />
      </div>
    </div>
  );
};
