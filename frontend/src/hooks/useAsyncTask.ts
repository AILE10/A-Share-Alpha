import { useEffect, useRef, useState } from 'react';
import { api } from '../services/api';
import { useAlphaWebSocket } from '../services/ws';

export interface AsyncTaskState {
  taskId: string;
  kind: string;
  title: string;
  status: 'RUNNING' | 'DONE' | 'FAILED';
  stage: string;
  pct: number; // 0..100
  elapsedSec: number;
  error?: string;
}

interface UseAsyncTaskOptions {
  /** Task kind(s) this hook watches & recovers on mount (e.g. ['signals_refresh'] or ['paper_rebalance','paper_intraday','weekly_review','model_retrain']) */
  kinds: string[];
  onDone?: (task: AsyncTaskState, result: any) => void;
  onError?: (task: AsyncTaskState, errMsg: string) => void;
}

/**
 * Reusable live-progress hook backed by WebSocket (`action_task`) + 1.2s polling fallback
 * + 0.5s smooth elapsed timer + automatic active-task recovery when returning to a page.
 */
export function useAsyncTask({ kinds, onDone, onError }: UseAsyncTaskOptions) {
  const [activeTask, setActiveTask] = useState<AsyncTaskState | null>(null);
  const taskRef = useRef<AsyncTaskState | null>(null);
  const doneIdsRef = useRef<Set<string>>(new Set());
  const onDoneRef = useRef(onDone);
  const onErrorRef = useRef(onError);
  const kindsKey = kinds.join(',');
  taskRef.current = activeTask;
  onDoneRef.current = onDone;
  onErrorRef.current = onError;

  const toState = (raw: any): AsyncTaskState => ({
    taskId: raw.task_id,
    kind: raw.kind,
    title: raw.title || '后台任务执行中',
    status: raw.status || 'RUNNING',
    stage: raw.stage || '正在执行...',
    pct: Math.min(100, Math.max(0, Math.round((Number(raw.pct) || 0) * 100))),
    elapsedSec: Number(raw.elapsed_sec || 0),
    error: raw.error || '',
  });

  const finalize = async (raw: any) => {
    const tid = raw.task_id;
    if (!tid || doneIdsRef.current.has(tid)) return;
    doneIdsRef.current.add(tid);
    const st = toState(raw);
    if (st.status === 'DONE') {
      let fullResult = raw.result;
      if (fullResult === undefined) {
        try {
          const detail = await api.getActionTask(tid);
          fullResult = detail?.result;
        } catch {
          // ignore
        }
      }
      setActiveTask(null);
      onDoneRef.current?.(st, fullResult);
    } else if (st.status === 'FAILED') {
      setActiveTask(null);
      onErrorRef.current?.(st, st.error || st.stage || '后台任务执行失败');
    }
  };

  // Recover any running task of watched kinds when mounting the page
  useEffect(() => {
    const watched = new Set(kindsKey.split(',').filter(Boolean));
    if (watched.size === 0) return;
    api
      .listActionTasks(true)
      .then((list: any[]) => {
        const hit = (list || []).find((t) => t.status === 'RUNNING' && watched.has(t.kind));
        if (hit) {
          setActiveTask(toState(hit));
        }
      })
      .catch(() => {});
  }, [kindsKey]);

  // Real-time WebSocket updates
  useAlphaWebSocket((evt) => {
    if (evt.channel !== 'action_task' || !evt.payload?.task_id) return;
    const p = evt.payload;
    const watched = new Set(kindsKey.split(',').filter(Boolean));
    const isMine = (taskRef.current && taskRef.current.taskId === p.task_id) || watched.has(p.kind);
    if (!isMine) return;
    if (p.status === 'DONE' || p.status === 'FAILED') {
      void finalize(p);
    } else if (p.status === 'RUNNING') {
      setActiveTask(toState(p));
    }
  });

  // Reliable 1.2s polling fallback while active (guarantees zero missed completions)
  useEffect(() => {
    if (!activeTask?.taskId || activeTask.status !== 'RUNNING') return;
    const tid = activeTask.taskId;
    const poll = window.setInterval(async () => {
      try {
        const st = await api.getActionTask(tid);
        if (st.status === 'DONE' || st.status === 'FAILED') {
          window.clearInterval(poll);
          void finalize(st);
        } else {
          setActiveTask(toState(st));
        }
      } catch {
        // ignore transient poll glitch
      }
    }, 1200);
    return () => window.clearInterval(poll);
  }, [activeTask?.taskId, activeTask?.status]);

  // Smooth 0.5s elapsed-time ticker so UI always feels responsive even during single heavy step
  useEffect(() => {
    if (!activeTask || activeTask.status !== 'RUNNING') return;
    const tick = window.setInterval(() => {
      setActiveTask((prev) => (prev && prev.status === 'RUNNING' ? { ...prev, elapsedSec: +(prev.elapsedSec + 0.5).toFixed(1) } : prev));
    }, 500);
    return () => window.clearInterval(tick);
  }, [activeTask?.taskId, activeTask?.status]);

  const startTask = async (launcher: () => Promise<any>) => {
    const init = await launcher();
    if (init?.task_id) {
      if (init.status === 'DONE' || init.status === 'FAILED') {
        await finalize(init);
      } else {
        setActiveTask(toState(init));
      }
    }
    return init;
  };

  return {
    activeTask,
    isRunning: Boolean(activeTask && activeTask.status === 'RUNNING'),
    runningKind: activeTask?.status === 'RUNNING' ? activeTask.kind : '',
    startTask,
    clearTask: () => setActiveTask(null),
  };
}
