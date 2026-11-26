import React from 'react';
import { Progress } from 'antd';
import { SyncOutlined } from '@ant-design/icons';
import { AsyncTaskState } from '../../hooks/useAsyncTask';

interface Props {
  task: AsyncTaskState | null;
  compact?: boolean;
}

export const TaskProgressBanner: React.FC<Props> = ({ task, compact = false }) => {
  if (!task || task.status !== 'RUNNING') return null;
  if (compact) {
    return (
      <div className="flex items-center gap-2 px-2.5 py-1 rounded bg-sky-950/60 border border-sky-500/40 text-xs max-w-xs">
        <SyncOutlined spin className="text-sky-400 shrink-0" />
        <div className="min-w-[110px] flex-1">
          <div className="flex items-center justify-between gap-2 text-[11px] leading-tight mb-0.5">
            <span className="text-sky-200 truncate">{task.stage}</span>
            <span className="num text-sky-300 shrink-0">{task.pct}%</span>
          </div>
          <Progress percent={task.pct} size="small" status="active" showInfo={false} strokeColor="#38BDF8" trailColor="#1E293B" className="m-0 leading-none" />
        </div>
      </div>
    );
  }
  return (
    <div className="quant-card p-3 border border-sky-500/40 bg-sky-950/20">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs mb-1.5">
        <span className="text-sky-300 font-medium flex items-center gap-1.5">
          <SyncOutlined spin className="text-sky-400" />
          <span>
            正在执行「{task.title}」：{task.stage}
          </span>
        </span>
        <span className="num text-slate-300">
          {task.pct}% · 已耗时 {task.elapsedSec.toFixed(1)}s
        </span>
      </div>
      <Progress
        percent={task.pct}
        size="small"
        status="active"
        showInfo={false}
        strokeColor="#38BDF8"
        trailColor="#1E293B"
      />
    </div>
  );
};
