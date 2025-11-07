import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getAgentScheduleProfile,
  getCopyTradeScores,
  getWalletPlan,
  getWalletPlanHistory,
  getWeightedNews,
} from "./api";
import { AllocationEntry, PlanSummary, WalletPlan, WeightedNewsEntry } from "./types";

type CopyScoreMap = Record<
  string,
  {
    success_rate: number;
    trade_count: number;
    signal_score?: number;
    last_signal_at?: string | null;
  }
>;

const formatPercent = (value: number, digits = 2) =>
  `${(value * 100).toFixed(digits)}%`;

const formatCurrency = (value: number) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);

const riskBadge = (riskLevel: string) => {
  const normalized = riskLevel.toLowerCase();
  if (normalized === "low") return "badge low";
  if (normalized === "medium") return "badge medium";
  return "badge high";
};

const stopLossText = (entry: AllocationEntry) => {
  const [upper, lower] = entry.stop_loss;
  return `+${(upper * 100).toFixed(1)}% / ${(lower * 100).toFixed(1)}%`;
};

const copyScoreLabel = (score?: number) => {
  if (score === undefined) return "n/a";
  return `${(score * 100).toFixed(1)}%`;
};

const App = () => {
  const [plan, setPlan] = useState<WalletPlan | null>(null);
  const [summary, setSummary] = useState<PlanSummary | null>(null);
  const [history, setHistory] = useState<WalletPlan[]>([]);
  const [copyScores, setCopyScores] = useState<CopyScoreMap>({});
  const [weightedNews, setWeightedNews] = useState<WeightedNewsEntry[]>([]);
  const [profile, setProfile] = useState<string>("minutes");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [planResponse, historyResponse, copyResponse, newsResponse, profileResponse] =
        await Promise.all([
          getWalletPlan(),
          getWalletPlanHistory(12),
          getCopyTradeScores().catch(() => ({ wallets: {} })),
          getWeightedNews().catch(() => ({ entries: [] })),
          getAgentScheduleProfile().catch(() => ({ profile: profile })),
        ]);

      setPlan(planResponse.plan);
      setSummary(planResponse.summary);
      setHistory(historyResponse.history || []);
      setCopyScores(copyResponse.wallets || {});
      setWeightedNews(newsResponse.entries || []);
      if (profileResponse.profile) {
        setProfile(profileResponse.profile);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh dashboard");
    } finally {
      setLoading(false);
    }
  }, [profile]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const allocations = useMemo(() => {
    if (!plan?.allocations) return [] as Array<[string, AllocationEntry]>;
    return Object.entries(plan.allocations).sort(([, a], [, b]) => b.allocation - a.allocation);
  }, [plan]);

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Agentic Trading Dashboard</h1>
          <p className="muted">
            Wallet balancing orchestrated across DQN, chart, sentiment, risk, and copy trade agents.
          </p>
        </div>
        <button className="refresh-button" onClick={() => refresh()} disabled={loading}>
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </header>

      {error && (
        <div className="section" style={{ borderColor: "rgba(239,68,68,0.35)", color: "#fecaca" }}>
          <strong>Unable to load dashboard data:</strong> {error}
        </div>
      )}

      <section className="section">
        <div className="app-header" style={{ alignItems: "baseline" }}>
          <h2>Wallet Allocation Plan</h2>
          <div className="muted">
            Schedule profile: <strong>{profile}</strong>
            {plan?.generated_at && (
              <span style={{ marginLeft: "0.75rem" }}>
                Generated: {new Date(plan.generated_at).toLocaleString()}
              </span>
            )}
          </div>
        </div>

        {plan ? (
          <div style={{ overflowX: "auto" }}>
            <table className="allocation-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Allocation</th>
                  <th>Target (USDC)</th>
                  <th>Confidence</th>
                  <th>Risk</th>
                  <th>Stop Loss</th>
                  <th>Gas Fee</th>
                  <th>PnL Estimate</th>
                </tr>
              </thead>
              <tbody>
                {allocations.map(([ticker, entry]) => (
                  <tr key={ticker}>
                    <td>{ticker}</td>
                    <td>{formatPercent(entry.allocation)}</td>
                    <td>{formatCurrency(entry.target_usdc)}</td>
                    <td>{(entry.confidence * 100).toFixed(1)}%</td>
                    <td>
                      <span className={riskBadge(entry.risk_level)}>{entry.risk_level}</span>
                    </td>
                    <td>{stopLossText(entry)}</td>
                    <td>{formatCurrency(entry.gas_fee_estimate)}</td>
                    <td>{formatCurrency(entry.pnl_estimate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">No wallet plan available yet. Trigger orchestrator cycle to generate one.</p>
        )}
      </section>

      <div className="grid">
        <section className="section">
          <h2>Orchestrator Summary</h2>
          <div className="summary-box">
            {summary?.summary ? summary.summary : "Summary will appear once the workforce orchestrator reports back."}
          </div>
        </section>

        <section className="section">
          <h2>Copy Trade Signals</h2>
          {Object.keys(copyScores).length === 0 ? (
            <p className="muted">No copy trade scores recorded yet.</p>
          ) : (
            <div className="scores-grid">
              {Object.entries(copyScores).map(([address, score]) => (
                <div className="score-card" key={address}>
                  <h4>{address}</h4>
                  <div className="muted">Signal confidence: {copyScoreLabel(score.success_rate)}</div>
                  <div className="muted">Signals observed: {score.trade_count}</div>
                  <div className="muted">Momentum: {copyScoreLabel(score.signal_score)}</div>
                  {score.last_signal_at && (
                    <div className="muted">Last signal: {new Date(score.last_signal_at).toLocaleString()}</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      <div className="grid">
        <section className="section">
          <h2>Plan History</h2>
          {history.length === 0 ? (
            <p className="muted">Once the orchestrator emits plans they will appear here.</p>
          ) : (
            <ul className="history-list">
              {history.map((entry) => (
                <li className="history-card" key={entry.generated_at}>
                  <h4>{new Date(entry.generated_at).toLocaleString()}</h4>
                  <span>Profile: {entry.profile || "n/a"}</span>
                  <span>Total value: {formatCurrency(entry.total_value_usdc)}</span>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="section">
          <h2>News Sentiment (weighted)</h2>
          {weightedNews.length === 0 ? (
            <p className="muted">No news sentiment history cached.</p>
          ) : (
            <div className="news-grid">
              {weightedNews.map((entry, idx) => (
                <div className="news-card" key={`${entry.ticker}-${idx}`}>
                  <div>
                    <strong>{entry.ticker ?? "Market"}</strong>
                    <div className="muted">
                      Updated {entry.last_updated ? new Date(entry.last_updated).toLocaleString() : "recently"}
                    </div>
                  </div>
                  <div>
                    Score: {(entry.weighted_score ?? 0).toFixed(2)}
                    <br />
                    Weight: {(entry.total_weight ?? 0).toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export default App;

