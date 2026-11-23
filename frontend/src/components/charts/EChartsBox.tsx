import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

interface Props {
  option: echarts.EChartsOption;
  height?: number | string;
  className?: string;
}

export const EChartsBox: React.FC<Props> = ({ option, height = 300, className = '' }) => {
  const ref = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const chart = echarts.init(el, undefined, { renderer: 'canvas' });
    chartRef.current = chart;

    const ro = new ResizeObserver(() => {
      chart.resize();
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.setOption(
        {
          backgroundColor: 'transparent',
          textStyle: { color: '#94A3B8', fontFamily: 'JetBrains Mono, Inter, sans-serif' },
          ...option,
        },
        true
      );
    }
  }, [option]);

  return <div ref={ref} style={{ height, width: '100%' }} className={className} />;
};
