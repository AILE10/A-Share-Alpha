import { useEffect, useRef, useState } from 'react';

export interface WsEvent {
  channel: string;
  timestamp: string;
  payload: any;
}

export function useAlphaWebSocket(onEvent?: (evt: WsEvent) => void) {
  const [connected, setConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<WsEvent | null>(null);
  const cbRef = useRef(onEvent);
  cbRef.current = onEvent;

  useEffect(() => {
    let closed = false;
    let ws: WebSocket | null = null;
    let pingTimer: number | null = null;
    let reconnectTimer: number | null = null;

    const connect = () => {
      if (closed) return;
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = `${proto}//${window.location.host}/api/v1/ws/stream`;
      ws = new WebSocket(url);

      ws.onopen = () => {
        setConnected(true);
        pingTimer = window.setInterval(() => {
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, 25_000);
      };

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === 'pong') return;
          setLastEvent(data);
          cbRef.current?.(data);
        } catch {
          // ignore non-JSON heartbeat frames
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (pingTimer) window.clearInterval(pingTimer);
        if (!closed) {
          reconnectTimer = window.setTimeout(connect, 3000);
        }
      };

      ws.onerror = () => {
        ws?.close();
      };
    };

    connect();
    return () => {
      closed = true;
      if (pingTimer) window.clearInterval(pingTimer);
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, []);

  return { connected, lastEvent };
}
