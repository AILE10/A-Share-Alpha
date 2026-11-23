import React from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { ConfigProvider, theme } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { AppShell } from './components/layout/AppShell';
import { DashboardPage } from './pages/DashboardPage';
import { FactorLabPage } from './pages/FactorLabPage';
import { BacktestCenterPage } from './pages/BacktestCenterPage';
import { SignalInspectorPage } from './pages/SignalInspectorPage';
import { PaperReviewPage } from './pages/PaperReviewPage';
import { BrokerRiskPage } from './pages/BrokerRiskPage';

export const App: React.FC = () => {
  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: '#38BDF8',
          colorBgBase: '#0B0F19',
          colorBgContainer: '#121826',
          colorBorder: '#1E293B',
          borderRadius: 8,
        },
      }}
    >
      <BrowserRouter>
        <AppShell>
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/factors" element={<FactorLabPage />} />
            <Route path="/backtest" element={<BacktestCenterPage />} />
            <Route path="/signals" element={<SignalInspectorPage />} />
            <Route path="/paper" element={<PaperReviewPage />} />
            <Route path="/broker-risk" element={<BrokerRiskPage />} />
          </Routes>
        </AppShell>
      </BrowserRouter>
    </ConfigProvider>
  );
};

export default App;
