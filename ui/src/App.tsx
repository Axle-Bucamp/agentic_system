import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
  getAgentScheduleProfile,
  getAgentStatus,
  getChatHistory,
  getCopyTradeScores,
  getFactInsight,
  getFusionRecommendations,
  getGraphSnapshot,
  getRecentLogs,
  getTrendAssessment,
  getWalletPlan,
  getWalletPlanHistory,
  getWeightedNews,
  postChatMessage,
  toggleCopyTradeAgent,
  getDashboardSettings,
  updateDashboardSettings,
  triggerReviewRun,
  getForecastTickers,
  runTrendPipeline,
  runFactPipeline,
  runFusionPipeline,
  runPrunePipeline,
  getPortfolioTrades,
  getAgentRewards,
  getPipelineLiveConfig,
  updatePipelineLiveConfig,
  getAiDecisions,
} from "./api";
import {
  AllocationEntry,
  AgentRewardSummary,
  AgentStatusResponse,
  ChatEntry,
  FactInsight,
  FusionRecommendation,
  GraphSnapshotResponse,
  PlanSummary,
  TradeRecord,
  TrendAssessment,
  WalletPlan,
  WeightedNewsEntry,
  DashboardSettings,
  ReviewSnapshot,
  PipelineLiveConfig,
  PipelineLiveEntry,
  AiDecision,
} from "./types";

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

const formatPipelineName = (value: string) =>
  value
    .split("_")
    .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join(" ");

const agentDisplayName = (value: string) => {
  const normalized = value.toUpperCase();
  const labels: Record<string, string> = {
    TREND: "Trend",
    FACT: "Fact",
    DQN: "DQN",
    CHART: "Chart",
    COPYTRADE: "Copy Trade",
    NEWS: "News",
    RISK: "Risk",
  };
  return labels[normalized] ?? formatPipelineName(normalized.toLowerCase());
};

const approximateInterval = (seconds?: number) => {
  if (!seconds || seconds <= 0) return "n/a";
  if (seconds >= 86400) {
    const days = seconds / 86400;
    return `${days % 1 === 0 ? days.toFixed(0) : days.toFixed(1)}d`;
  }
  if (seconds >= 3600) {
    const hours = seconds / 3600;
    return `${hours % 1 === 0 ? hours.toFixed(0) : hours.toFixed(1)}h`;
  }
  const minutes = seconds / 60;
  return `${minutes % 1 === 0 ? Math.max(minutes, 1).toFixed(0) : minutes.toFixed(1)}m`;
};

const copyScoreLabel = (score?: number) => {
  if (score === undefined) return "n/a";
  return `${(score * 100).toFixed(1)}%`;
};

const agenticBadge = (agentic?: boolean) => {
  if (agentic === true) {
    return <span className="badge low">Agentic</span>;
  }
  if (agentic === false) {
    return <span className="badge high">Fallback</span>;
  }
  return <span className="badge medium">Unknown</span>;
};

const App = () => {
  const CHAT_USER_ID = "portfolio-operator";

  const [plan, setPlan] = useState<WalletPlan | null>(null);
  const [summary, setSummary] = useState<PlanSummary | null>(null);
  const [history, setHistory] = useState<WalletPlan[]>([]);
  const [copyScores, setCopyScores] = useState<CopyScoreMap>({});
  const [weightedNews, setWeightedNews] = useState<WeightedNewsEntry[]>([]);
  const [profile, setProfile] = useState<string>("minutes");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [chatHistory, setChatHistory] = useState<ChatEntry[]>([]);
  const [chatInput, setChatInput] = useState<string>("");
  const [chatBusy, setChatBusy] = useState<boolean>(false);
  const [chatError, setChatError] = useState<string | null>(null);

  const [agentStatus, setAgentStatus] = useState<AgentStatusResponse | null>(null);
  const [fusionItems, setFusionItems] = useState<FusionRecommendation[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<"fusion" | "trend" | "fact">("fusion");
  const [selectedTicker, setSelectedTicker] = useState<string>("");
  const [trendDetail, setTrendDetail] = useState<TrendAssessment | null>(null);
  const [factDetail, setFactDetail] = useState<FactInsight | null>(null);
  const [graphSnapshot, setGraphSnapshot] = useState<GraphSnapshotResponse | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [logSource, setLogSource] = useState<string>("trading");
  const [settingsState, setSettingsState] = useState<DashboardSettings | null>(null);
  const [settingsDraft, setSettingsDraft] = useState<DashboardSettings | null>(null);
  const [settingsMessage, setSettingsMessage] = useState<string | null>(null);
  const [settingsBusy, setSettingsBusy] = useState<boolean>(false);
  const [reviewSnapshot, setReviewSnapshot] = useState<ReviewSnapshot | null>(null);
  const [forecastTickers, setForecastTickers] = useState<string[]>([]);
  const [pipelineMessage, setPipelineMessage] = useState<string | null>(null);
  const [pipelineBusy, setPipelineBusy] = useState<boolean>(false);
  const [recentTrades, setRecentTrades] = useState<TradeRecord[]>([]);
  const [agentRewards, setAgentRewards] = useState<Record<string, AgentRewardSummary>>({});
  const [pipelineLiveConfig, setPipelineLiveConfig] = useState<PipelineLiveConfig | null>(null);
  const [pipelineLiveDraft, setPipelineLiveDraft] = useState<PipelineLiveConfig | null>(null);
  const [pipelineIntervals, setPipelineIntervals] = useState<string[]>(["minutes", "hours", "days"]);
  const [pipelineLiveMessage, setPipelineLiveMessage] = useState<string | null>(null);
  const [pipelineLiveBusy, setPipelineLiveBusy] = useState<boolean>(false);
  const [aiDecisions, setAiDecisions] = useState<AiDecision[]>([]);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [
        planResponse,
        historyResponse,
        copyResponse,
        newsResponse,
        profileResponse,
        fusionResponse,
        statusResponse,
        graphResponse,
        logResponse,
        settingsResponse,
        liveConfigResponse,
        forecastResponse,
        tradesResponse,
        rewardsResponse,
        decisionsResponse,
      ] = await Promise.all([
        getWalletPlan(),
        getWalletPlanHistory(12),
        getCopyTradeScores().catch(() => ({ wallets: {} })),
        getWeightedNews().catch(() => ({ entries: [] })),
        getAgentScheduleProfile().catch(() => ({ profile })),
        getFusionRecommendations().catch(() => ({ items: [] })),
        getAgentStatus().catch(() => null),
        getGraphSnapshot(75, 150).catch(() => ({ nodes: [], edges: [] })),
        getRecentLogs(logSource, 120).catch(() => ({
          source: logSource,
          path: "",
          lines: [],
          logfire_enabled: false,
        })),
        getDashboardSettings().catch(() => ({ settings: undefined })),
        getPipelineLiveConfig().catch(() => ({
          pipelines: {},
          intervals: ["minutes", "hours", "days"],
        })),
        getForecastTickers().catch(() => ({ tickers: [], count: 0 })),
        getPortfolioTrades(20).catch(() => ({
          trades: [],
          count: 0,
          limit: 0,
          offset: 0,
          total: 0,
        })),
        getAgentRewards().catch(() => ({ rewards: {} })),
        getAiDecisions().catch(() => ({ decisions: [], count: 0, total: 0 })),
      ]);

      setPlan(planResponse.plan);
      setSummary(planResponse.summary);
      setHistory(historyResponse.history || []);
      setCopyScores(copyResponse.wallets || {});
      setWeightedNews(newsResponse.entries || []);
      if (profileResponse.profile) {
        setProfile(profileResponse.profile);
      }
      setFusionItems(fusionResponse.items || []);
      setAgentStatus(statusResponse);
      setGraphSnapshot(graphResponse);
      setLogs(logResponse.lines || []);
      if (settingsResponse.settings) {
        setSettingsState(settingsResponse.settings);
        setSettingsDraft(settingsResponse.settings);
        setReviewSnapshot(settingsResponse.settings.latest_review_snapshot ?? null);
      }
      if (!settingsResponse.settings) {
        setReviewSnapshot(null);
      }
      if (liveConfigResponse.intervals) {
        setPipelineIntervals(liveConfigResponse.intervals);
      }
      if (liveConfigResponse.pipelines) {
        setPipelineLiveConfig(liveConfigResponse.pipelines);
        setPipelineLiveDraft(
          JSON.parse(JSON.stringify(liveConfigResponse.pipelines)) as PipelineLiveConfig,
        );
      } else {
        setPipelineLiveConfig(null);
        setPipelineLiveDraft(null);
      }
      if (forecastResponse.tickers) {
        const cleaned = forecastResponse.tickers
          .map((ticker: { symbol?: string }) => ticker.symbol ?? "")
          .filter((value: string) => Boolean(value));
        setForecastTickers(cleaned);
      }
      setRecentTrades(tradesResponse.trades || []);
      setAgentRewards(rewardsResponse.rewards || {});
      setAiDecisions(decisionsResponse.decisions || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh dashboard");
    } finally {
      setLoading(false);
    }
  }, [profile, logSource]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    void (async () => {
      try {
        const response = await getChatHistory(CHAT_USER_ID, 40);
        setChatHistory(response.history || []);
      } catch {
        setChatHistory([]);
      }
    })();
  }, []);

  const allocations = useMemo(() => {
    if (!plan?.allocations) return [] as Array<[string, AllocationEntry]>;
    return Object.entries(plan.allocations).sort(([, a], [, b]) => b.allocation - a.allocation);
  }, [plan]);

  const latestTrades = useMemo(() => recentTrades.slice(0, 12), [recentTrades]);

  const agentRewardEntries = useMemo(
    () =>
      Object.entries(agentRewards).sort(
        ([, a], [, b]) => (b.average_reward ?? 0) - (a.average_reward ?? 0),
      ),
    [agentRewards],
  );

  const availableTickers = useMemo(() => {
    const tickers = new Set<string>();
    fusionItems.forEach((item) => tickers.add(item.ticker));
    allocations.forEach(([ticker]) => tickers.add(ticker));
    weightedNews.forEach((item) => {
      if (item.ticker) tickers.add(item.ticker);
    });
    forecastTickers.forEach((ticker) => tickers.add(ticker));
    recentTrades.forEach((trade) => {
      if (trade.ticker) tickers.add(trade.ticker);
    });
    return Array.from(tickers).sort();
  }, [allocations, fusionItems, weightedNews, forecastTickers, recentTrades]);

  useEffect(() => {
    if (!selectedTicker && availableTickers.length > 0) {
      setSelectedTicker(availableTickers[0]);
    }
  }, [availableTickers, selectedTicker]);

  useEffect(() => {
    if (selectedAgent !== "trend" || !selectedTicker) {
      setTrendDetail(null);
      return;
    }
    let cancelled = false;
    void getTrendAssessment(selectedTicker)
      .then((res) => {
        if (!cancelled) {
          setTrendDetail(res.assessment ?? null);
        }
      })
      .catch(() => {
        if (!cancelled) setTrendDetail(null);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedAgent, selectedTicker]);

  useEffect(() => {
    if (selectedAgent !== "fact" || !selectedTicker) {
      setFactDetail(null);
      return;
    }
    let cancelled = false;
    void getFactInsight(selectedTicker)
      .then((res) => {
        if (!cancelled) {
          setFactDetail(res.insight ?? null);
        }
      })
      .catch(() => {
        if (!cancelled) setFactDetail(null);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedAgent, selectedTicker]);

  const fusionTickerMap = useMemo(() => {
    const map = new Map<string, FusionRecommendation>();
    fusionItems.forEach((item) => {
      map.set(item.ticker, item);
    });
    return map;
  }, [fusionItems]);

  const selectedFusion = selectedTicker ? fusionTickerMap.get(selectedTicker) : undefined;
  const copytradeEnabled = Boolean((agentStatus?.copytrade as { enabled?: boolean })?.enabled ?? true);
  const logfireEnabled = Boolean(agentStatus?.logfire_enabled);

  const reviewWeights = useMemo(
    () =>
      reviewSnapshot?.weights
        ? Object.entries(reviewSnapshot.weights).sort(([, a], [, b]) => b - a)
        : ([] as Array<[string, number]>),
    [reviewSnapshot],
  );

  const weightDelta = useCallback(
    (agent: string, weight: number) => {
      const previous = reviewSnapshot?.previous_weights?.[agent];
      if (previous === undefined || previous === null) return "—";
      const diff = weight - previous;
      if (Math.abs(diff) < 0.0005) return "↔︎ 0.0%";
      const arrow = diff > 0 ? "↑" : "↓";
      return `${arrow} ${(Math.abs(diff) * 100).toFixed(1)}%`;
    },
    [reviewSnapshot],
  );

  const reviewMetrics = useMemo(
    () =>
      reviewSnapshot?.metrics
        ? Object.entries(reviewSnapshot.metrics)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 6)
        : ([] as Array<[string, number]>),
    [reviewSnapshot],
  );

  const topFusionRows = useMemo(() => fusionItems.slice(0, 5), [fusionItems]);

  const graphNodeCount = graphSnapshot?.nodes.length ?? 0;
  const graphEdgeCount = graphSnapshot?.edges.length ?? 0;
  const topGraphNodes = useMemo(() => {
    if (!graphSnapshot) return [];
    return [...graphSnapshot.nodes].sort((a, b) => b.weight - a.weight).slice(0, 5);
  }, [graphSnapshot]);

  const handleChatSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!chatInput.trim()) return;
    setChatBusy(true);
    setChatError(null);
    try {
      const response = await postChatMessage(CHAT_USER_ID, chatInput);
      setChatHistory(response.history || []);
      setChatInput("");
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "Chat request failed");
    } finally {
      setChatBusy(false);
    }
  };

  const handleCopyTradeToggle = async () => {
    try {
      await toggleCopyTradeAgent(!copytradeEnabled);
      const status = await getAgentStatus();
      setAgentStatus(status);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to toggle copy trade agent");
    }
  };

  const handleLogSourceChange = (next: string) => {
    setLogSource(next);
  };

  const handleSettingsFieldChange = (key: keyof DashboardSettings, value: string | number) => {
    setSettingsDraft((prev) => (prev ? { ...prev, [key]: value } : prev));
    setSettingsMessage(null);
  };

  const handleSettingsSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!settingsDraft) return;
    setSettingsBusy(true);
    setSettingsMessage(null);
    try {
      const response = await updateDashboardSettings(settingsDraft);
      setSettingsState(response.settings);
      setSettingsDraft(response.settings);
      setReviewSnapshot(response.settings.latest_review_snapshot ?? reviewSnapshot ?? null);
      setSettingsMessage("Settings saved");
      void refresh();
    } catch (err) {
      setSettingsMessage(err instanceof Error ? err.message : "Unable to save settings");
    } finally {
      setSettingsBusy(false);
    }
  };

  const handleReviewTrigger = async () => {
    try {
      await triggerReviewRun();
      setSettingsMessage("Review run queued");
    } catch (err) {
      setSettingsMessage(err instanceof Error ? err.message : "Unable to queue review");
    }
  };

  const handlePipelineTest = async (pipeline: "trend" | "fact" | "fusion" | "prune") => {
    setPipelineBusy(true);
    setPipelineMessage(null);
    try {
      let response: { success: boolean; message?: string } = { success: false };
      let detail: string | undefined;
      if (pipeline === "trend") {
        if (!selectedTicker) throw new Error("Select a ticker first");
        const res = await runTrendPipeline(selectedTicker);
        response = { success: res.success, message: res.message };
        if (res.success && res.assessment) {
          detail = `${res.assessment.ticker} trend ${res.assessment.recommended_action} @ ${(res.assessment.confidence * 100).toFixed(1)}% (${res.assessment.agentic ? "agentic" : "fallback"})`;
        }
      } else if (pipeline === "fact") {
        if (!selectedTicker) throw new Error("Select a ticker first");
        const res = await runFactPipeline(selectedTicker);
        response = { success: res.success, message: res.message };
        if (res.success && res.insight) {
          detail = `${res.insight.ticker} fact sentiment ${res.insight.sentiment_score.toFixed(2)} (${res.insight.agentic ? "agentic" : "fallback"})`;
        }
      } else if (pipeline === "fusion") {
        if (!selectedTicker) throw new Error("Select a ticker first");
        const res = await runFusionPipeline(selectedTicker);
        response = { success: res.success, message: res.message };
        if (res.success && res.recommendation) {
          detail = `${res.recommendation.ticker} fusion ${res.recommendation.action} @ ${(res.recommendation.confidence * 100).toFixed(1)}% (${res.recommendation.agentic ? "agentic" : "fallback"})`;
        }
      } else {
        const res = await runPrunePipeline();
        response = { success: res.success };
        detail = res.success ? "Memory pruning pipeline executed" : undefined;
      }
      const message =
        detail ||
        response.message ||
        (response.success ? "Pipeline execution complete" : "Pipeline returned no result");
      setPipelineMessage(message);
      if (response.success) {
        void refresh();
      }
      try {
        const decisions = await getAiDecisions();
        setAiDecisions(decisions.decisions || []);
      } catch {
        // best-effort; keep previous decisions
      }
    } catch (err) {
      setPipelineMessage(err instanceof Error ? err.message : "Unable to execute pipeline");
    } finally {
      setPipelineBusy(false);
    }
  };

  const updatePipelineLiveField = (pipeline: string, updates: Partial<PipelineLiveEntry>) => {
    setPipelineLiveDraft((prev) => {
      if (!prev || !prev[pipeline]) return prev;
      return {
        ...prev,
        [pipeline]: {
          ...prev[pipeline],
          ...updates,
        },
      };
    });
    setPipelineLiveMessage(null);
  };

  const handlePipelineLiveToggle = (pipeline: string, enabled: boolean) => {
    updatePipelineLiveField(pipeline, { enabled });
  };

  const handlePipelineLiveIntervalChange = (pipeline: string, interval: string) => {
    updatePipelineLiveField(pipeline, { interval });
  };

  const handlePipelineLiveSave = async () => {
    if (!pipelineLiveDraft) return;
    setPipelineLiveBusy(true);
    setPipelineLiveMessage(null);
    try {
      const payload = Object.entries(pipelineLiveDraft).reduce(
        (acc, [key, value]) => ({
          ...acc,
          [key]: {
            enabled: value.enabled,
            interval: value.interval,
          },
        }),
        {} as Record<string, { enabled: boolean; interval: string }>,
      );
      const response = await updatePipelineLiveConfig(payload);
      setPipelineLiveConfig(response.pipelines);
      setPipelineLiveDraft(JSON.parse(JSON.stringify(response.pipelines)) as PipelineLiveConfig);
      setPipelineLiveMessage("Live-mode configuration saved");
    } catch (err) {
      setPipelineLiveMessage(err instanceof Error ? err.message : "Unable to update live-mode");
    } finally {
      setPipelineLiveBusy(false);
    }
  };

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

      <div className="grid">
        <section className="section chat-panel">
          <div className="section-header">
            <h2>Live Copilot Chat</h2>
            <span className="muted">Session: {CHAT_USER_ID}</span>
          </div>
          <div className="chat-history">
            {chatHistory.length === 0 ? (
              <p className="muted">Start the conversation to interrogate the agentic fusion stack.</p>
            ) : (
              chatHistory.map((entry, index) => (
                <div key={`${entry.timestamp}-${index}`} className={`chat-entry ${entry.role}`}>
                  <div className="chat-meta">
                    <span className="chat-role">{entry.role === "assistant" ? "Fusion Analyst" : "You"}</span>
                    <span className="chat-time">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                  </div>
                  <div className="chat-content">{entry.content}</div>
                </div>
              ))
            )}
          </div>
          {chatError && <div className="error-text">{chatError}</div>}
          <form className="chat-input" onSubmit={handleChatSubmit}>
            <input
              value={chatInput}
              onChange={(event) => setChatInput(event.target.value)}
              placeholder="Ask about today's calls or risk posture…"
            />
            <button type="submit" disabled={chatBusy || !chatInput.trim()}>
              {chatBusy ? "Thinking…" : "Send"}
            </button>
          </form>
        </section>

        <section className="section agent-monitor">
          <div className="section-header">
            <h2>Pipeline Monitor</h2>
            <button className="refresh-button" onClick={handleCopyTradeToggle}>
              {copytradeEnabled ? "Pause Copy Trade" : "Resume Copy Trade"}
            </button>
          </div>
          <p className="muted">
            Copy trade agent is currently <strong>{copytradeEnabled ? "enabled" : "disabled"}</strong>.
          </p>
          <div className="agent-controls">
            <label>
              Agent
              <select value={selectedAgent} onChange={(event) => setSelectedAgent(event.target.value as "fusion" | "trend" | "fact")}>
                <option value="fusion">Fusion</option>
                <option value="trend">Trend</option>
                <option value="fact">Fact</option>
              </select>
            </label>
            <label>
              Ticker
              <select value={selectedTicker} onChange={(event) => setSelectedTicker(event.target.value)}>
                {availableTickers.map((ticker: string) => (
                  <option key={ticker} value={ticker}>
                    {ticker}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="agent-details">
            {selectedAgent === "fusion" ? (
              selectedFusion ? (
                <div className="detail-card">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <h3>{selectedFusion.ticker}</h3>
                    {agenticBadge(selectedFusion.agentic)}
                  </div>
                  <p>
                    Action: <strong>{selectedFusion.action}</strong> &nbsp; • &nbsp; Confidence:{" "}
                    {(selectedFusion.confidence * 100).toFixed(1)}%
                  </p>
                  <p>
                    Allocation target: <strong>{formatPercent(selectedFusion.percent_allocation)}</strong> &nbsp; • &nbsp; Risk level:{" "}
                    {selectedFusion.risk_level}
                  </p>
                  {selectedFusion.ai_explanation && <p className="muted">{selectedFusion.ai_explanation}</p>}
                  {selectedFusion.rationale && <p className="muted">{selectedFusion.rationale}</p>}
                  {selectedFusion.components && (
                    <details>
                      <summary>Component inputs</summary>
                      <pre className="detail-json">{JSON.stringify(selectedFusion.components, null, 2)}</pre>
                    </details>
                  )}
                </div>
              ) : (
                <p className="muted">No fusion output yet for {selectedTicker || "selected ticker"}.</p>
              )
            ) : selectedAgent === "trend" ? (
              trendDetail ? (
                <div className="detail-card">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <h3>{trendDetail.ticker}</h3>
                    {agenticBadge(trendDetail.agentic)}
                  </div>
                  <p>
                    Composite trend score: <strong>{(trendDetail.trend_score * 100).toFixed(1)}%</strong> &nbsp; • &nbsp; Momentum:{" "}
                    {(trendDetail.momentum * 100).toFixed(1)}%
                  </p>
                  {trendDetail.volatility != null && (
                    <p>Volatility proxy: {(trendDetail.volatility * 100).toFixed(2)}%</p>
                  )}
                  {trendDetail.ai_explanation && <p className="muted">{trendDetail.ai_explanation}</p>}
                  <pre className="detail-json">{JSON.stringify(trendDetail.supporting_signals ?? {}, null, 2)}</pre>
                </div>
              ) : (
                <p className="muted">Trend agent has not published a signal for {selectedTicker || "this asset"} yet.</p>
              )
            ) : factDetail ? (
              <div className="detail-card">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <h3>{factDetail.ticker ?? "Market"}</h3>
                  {agenticBadge(factDetail.agentic)}
                </div>
                <p>
                  Sentiment: <strong>{factDetail.sentiment_score.toFixed(2)}</strong> &nbsp; • &nbsp; Confidence:{" "}
                  {(factDetail.confidence * 100).toFixed(1)}%
                </p>
                <p>{factDetail.thesis}</p>
                {factDetail.ai_explanation && <p className="muted">{factDetail.ai_explanation}</p>}
                {(factDetail.anomalies ?? []).length > 0 && (
                  <ul className="reference-list">
                    {(factDetail.anomalies ?? []).map((item, idx) => (
                      <li key={`anomaly-${idx}`}>{item}</li>
                    ))}
                  </ul>
                )}
                {(factDetail.references ?? []).length > 0 && (
                  <ul className="reference-list">
                    {(factDetail.references ?? []).slice(0, 3).map((ref, idx) => {
                      const reference = ref as { title?: string; source?: string };
                      return <li key={idx}>{reference.title ?? reference.source ?? "Referenced insight"}</li>;
                    })}
                  </ul>
                )}
                {factDetail.sentiment_breakdown && (
                  <details>
                    <summary>Sentiment breakdown</summary>
                    <pre className="detail-json">
                      {JSON.stringify(factDetail.sentiment_breakdown, null, 2)}
                    </pre>
                  </details>
                )}
                {factDetail.market_indicators && (
                  <details>
                    <summary>Market indicator snapshot</summary>
                    <pre className="detail-json">
                      {JSON.stringify(factDetail.market_indicators, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            ) : (
              <p className="muted">Fact agent has no recent insight for {selectedTicker || "this asset"}.</p>
            )}
          </div>
          <div className="pipeline-actions">
            <button
              type="button"
              className="refresh-button"
              onClick={() => handlePipelineTest("trend")}
              disabled={pipelineBusy}
            >
              Test Trend Pipeline
            </button>
            <button
              type="button"
              className="refresh-button"
              onClick={() => handlePipelineTest("fact")}
              disabled={pipelineBusy}
            >
              Test Fact Pipeline
            </button>
            <button
              type="button"
              className="refresh-button"
              onClick={() => handlePipelineTest("fusion")}
              disabled={pipelineBusy}
            >
              Test Fusion Pipeline
            </button>
            <button
              type="button"
              className="refresh-button"
              onClick={() => handlePipelineTest("prune")}
              disabled={pipelineBusy}
            >
              Run Memory Prune
            </button>
          </div>
          {pipelineLiveDraft ? (
            <div className="pipeline-live-controls">
              <h3 style={{ marginTop: "1.5rem" }}>Live Mode Scheduling</h3>
              <table className="allocation-table">
                <thead>
                  <tr>
                    <th>Pipeline</th>
                    <th>Enabled</th>
                    <th>Interval</th>
                    <th>Effective Cadence</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(pipelineLiveDraft).map(([key, config]) => (
                    <tr key={key}>
                      <td>{formatPipelineName(key)}</td>
                      <td>
                        <input
                          type="checkbox"
                          checked={config.enabled}
                          onChange={(event) => handlePipelineLiveToggle(key, event.target.checked)}
                        />
                      </td>
                      <td>
                        <select
                          value={config.interval}
                          onChange={(event) => handlePipelineLiveIntervalChange(key, event.target.value)}
                        >
                          {pipelineIntervals.map((option) => (
                            <option key={option} value={option}>
                              {formatPipelineName(option)}
                            </option>
                          ))}
                        </select>
                      </td>
                      <td>{approximateInterval(pipelineLiveConfig?.[key]?.seconds)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="settings-actions" style={{ marginTop: "0.75rem" }}>
                <button
                  type="button"
                  className="refresh-button"
                  onClick={handlePipelineLiveSave}
                  disabled={pipelineLiveBusy}
                >
                  {pipelineLiveBusy ? "Saving…" : "Save Live Mode"}
                </button>
                {pipelineLiveMessage && <span className="muted">{pipelineLiveMessage}</span>}
              </div>
            </div>
          ) : (
            <p className="muted" style={{ marginTop: "1rem" }}>
              Live-mode configuration unavailable.
            </p>
          )}
          <div className="decision-log" style={{ marginTop: "1.5rem" }}>
            <h3>Agent Decision Trace</h3>
            {aiDecisions.length === 0 ? (
              <p className="muted">
                No recent workforce decisions recorded. Trigger a pipeline or wait for the orchestrator to process new signals.
              </p>
            ) : (
              aiDecisions.slice(0, 8).map((decision, index) => {
                const statusLabel = (decision.status ?? "unknown").toLowerCase();
                const badgeClass =
                  statusLabel === "completed"
                    ? "badge low"
                    : statusLabel === "degraded"
                    ? "badge medium"
                    : "badge high";
                return (
                  <details
                    key={decision.decision_id ?? `${decision.ticker ?? "market"}-${index}`}
                    className="detail-card"
                    open={decision.status !== "completed"}
                  >
                    <summary>
                      <strong>{decision.ticker ?? "Market"}</strong> • {decision.action ?? "N/A"} •{" "}
                      <span className={badgeClass}>{decision.status ?? "unknown"}</span>{" "}
                      {agenticBadge(decision.agentic)}
                      {decision.completed_at && (
                        <span className="muted" style={{ marginLeft: "0.5rem" }}>
                          {new Date(decision.completed_at).toLocaleTimeString()}
                        </span>
                      )}
                    </summary>
                    {decision.task_description && <p className="muted">{decision.task_description}</p>}
                    {decision.steps && decision.steps.length > 0 ? (
                      <ol className="reference-list">
                        {decision.steps.map((step, stepIndex) => (
                          <li key={`${decision.decision_id}-step-${stepIndex}`}>
                            <strong>{step.step}</strong>
                            {step.timestamp && (
                              <span className="muted" style={{ marginLeft: "0.5rem" }}>
                                {new Date(step.timestamp).toLocaleTimeString()}
                              </span>
                            )}
                            {step.description && <div className="muted">{step.description}</div>}
                            {step.result && <div>{step.result}</div>}
                            {step.messages && step.messages.length > 0 && (
                              <pre className="detail-json">{step.messages.join("\n")}</pre>
                            )}
                          </li>
                        ))}
                      </ol>
                    ) : (
                      <p>No step-by-step transcript recorded for this decision.</p>
                    )}
                    {decision.result && (
                      <p>
                        <strong>Outcome:</strong> {decision.result}
                      </p>
                    )}
                    {decision.ai_explanation && (
                      <p className="muted">
                        <strong>AI Summary:</strong> {decision.ai_explanation}
                      </p>
                    )}
                    {decision.error && <p className="error-text">Error: {decision.error}</p>}
                  </details>
                );
              })
            )}
          </div>
          {pipelineMessage && <p className="muted">{pipelineMessage}</p>}
        </section>
      </div>

      <div className="grid">
        <section className="section">
          <div className="section-header">
            <h2>Latest Trades</h2>
            <span className="muted">
              {latestTrades.length === 0
                ? "Trades will appear once executions are evaluated."
                : `Showing ${latestTrades.length} of ${recentTrades.length} tracked trades.`}
            </span>
          </div>
          {latestTrades.length === 0 ? (
            <p className="muted">No trades have been executed during this session.</p>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table className="trades-table">
                <thead>
                  <tr>
                    <th>Evaluated</th>
                    <th>Ticker</th>
                    <th>Action</th>
                    <th>Qty</th>
                    <th>Entry</th>
                    <th>Eval</th>
                    <th>PnL</th>
                    <th>Reward</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {latestTrades.map((trade) => {
                    const evaluatedAt =
                      trade.reward_evaluated_at ?? trade.timestamp ?? "";
                    return (
                      <tr key={trade.trade_id}>
                        <td>{evaluatedAt ? new Date(evaluatedAt).toLocaleString() : "Pending"}</td>
                        <td>{trade.ticker}</td>
                        <td>{trade.action}</td>
                        <td>{trade.quantity?.toFixed(4)}</td>
                        <td>{trade.entry_price != null ? formatCurrency(trade.entry_price) : "—"}</td>
                        <td>
                          {trade.evaluation_price != null ? formatCurrency(trade.evaluation_price) : "—"}
                        </td>
                        <td
                          className={
                            trade.pnl != null
                              ? trade.pnl > 0
                                ? "positive"
                                : trade.pnl < 0
                                ? "negative"
                                : ""
                              : ""
                          }
                        >
                          {trade.pnl != null ? formatCurrency(trade.pnl) : "—"}
                        </td>
                        <td
                          className={
                            trade.reward != null
                              ? trade.reward > 0
                                ? "positive"
                                : trade.reward < 0
                                ? "negative"
                                : ""
                              : ""
                          }
                        >
                          {trade.reward != null ? formatPercent(trade.reward) : "—"}
                        </td>
                        <td>{trade.status ?? "PENDING"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="section">
          <h2>Agent Reward Summary</h2>
          {agentRewardEntries.length === 0 ? (
            <p className="muted">Reward attribution will populate after trades are evaluated.</p>
          ) : (
            <ul className="rewards-list">
              {agentRewardEntries.map(([agent, stats]) => (
                <li key={agent} className="reward-card">
                  <div className="reward-title">{agent}</div>
                  <div className="muted">
                    Trades: <strong>{stats.trades}</strong>
                  </div>
                  <div className="muted">
                    Total reward: <strong>{formatPercent(stats.total_reward)}</strong>
                  </div>
                  <div className="muted">
                    Avg reward: <strong>{formatPercent(stats.average_reward)}</strong>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      <div className="grid">
        <section className="section">
          <h2>Fusion Snapshot</h2>
          {topFusionRows.length === 0 ? (
            <p className="muted">Fusion agent has not produced any guidance yet.</p>
          ) : (
            <table className="allocation-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Action</th>
                  <th>Confidence</th>
                  <th>Allocation</th>
                  <th>Mode</th>
                  <th>Risk</th>
                </tr>
              </thead>
              <tbody>
                {topFusionRows.map((item) => (
                  <tr key={item.ticker}>
                    <td>{item.ticker}</td>
                    <td>{item.action}</td>
                    <td>{(item.confidence * 100).toFixed(1)}%</td>
                    <td>{formatPercent(item.percent_allocation)}</td>
                    <td>{agenticBadge(item.agentic)}</td>
                    <td>{item.risk_level}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        <section className="section">
          <h2>Graph Memory Snapshot</h2>
          <p className="muted">
            Nodes: <strong>{graphNodeCount}</strong> • Edges: <strong>{graphEdgeCount}</strong>
          </p>
          {topGraphNodes.length === 0 ? (
            <p className="muted">No graph entries yet.</p>
          ) : (
            <ul className="graph-list">
              {topGraphNodes.map((node) => (
                <li key={node.node_id}>
                  <strong>{node.label}</strong>
                  <span className="muted"> — weight {node.weight.toFixed(2)} • {node.node_type}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      <section className="section">
        <div className="section-header" style={{ alignItems: "baseline" }}>
          <h2>Agent Weight Review</h2>
          {reviewSnapshot ? (
            <div className="muted">
              Generated {new Date(reviewSnapshot.generated_at).toLocaleString()}{" "}
              {agenticBadge(reviewSnapshot.agentic)}
            </div>
          ) : (
            <span className="muted">No review snapshot captured yet.</span>
          )}
        </div>

        {reviewSnapshot ? (
          <>
            <p className="muted" style={{ marginBottom: "0.75rem" }}>
              {reviewSnapshot.ai_explanation || "Review agent did not provide an explanation."}
            </p>
            <div style={{ overflowX: "auto", marginBottom: "0.75rem" }}>
              <table className="allocation-table">
                <thead>
                  <tr>
                    <th>Agent</th>
                    <th>Weight</th>
                    <th>Δ vs previous</th>
                  </tr>
                </thead>
                <tbody>
                  {reviewWeights.map(([agent, weight]) => (
                    <tr key={agent}>
                      <td>{agentDisplayName(agent)}</td>
                      <td>{formatPercent(weight)}</td>
                      <td>{weightDelta(agent, weight)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {reviewMetrics.length > 0 && (
              <div style={{ marginBottom: "0.75rem" }}>
                <strong>Top metric signals</strong>
                <ul className="graph-list">
                  {reviewMetrics.map(([agent, score]) => (
                    <li key={agent}>
                      <strong>{agentDisplayName(agent)}</strong>
                      <span className="muted"> — score {score.toFixed(2)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {reviewSnapshot.failure_reason && (
              <p className="muted">Note: {reviewSnapshot.failure_reason}</p>
            )}
            {reviewSnapshot.raw_response && (
              <details>
                <summary className="muted">View raw assistant output</summary>
                <pre className="log-output">{reviewSnapshot.raw_response}</pre>
              </details>
            )}
          </>
        ) : (
          <p className="muted">
            Queue a weight review to populate this section and let the CAMEL reviewer rebalance agents.
          </p>
        )}
      </section>

      <section className="section">
        <div className="section-header">
          <h2>System Settings</h2>
          <button type="button" className="refresh-button" onClick={handleReviewTrigger}>
            Queue Weight Review
          </button>
        </div>
        {settingsDraft ? (
          <form className="settings-form" onSubmit={handleSettingsSubmit}>
            <div className="settings-grid">
              <label>
                Schedule profile
                <select
                  value={settingsDraft.schedule_profile}
                  onChange={(event) => handleSettingsFieldChange("schedule_profile", event.target.value)}
                >
                  <option value="minutes">Minutes</option>
                  <option value="hours">Hours</option>
                  <option value="days">Days</option>
                </select>
              </label>
              <label>
                Observation interval
                <select
                  value={settingsDraft.observation_interval}
                  onChange={(event) => handleSettingsFieldChange("observation_interval", event.target.value)}
                >
                  <option value="minutes">Minutes</option>
                  <option value="hours">Hours</option>
                  <option value="days">Days</option>
                </select>
              </label>
              <label>
                Decision interval
                <select
                  value={settingsDraft.decision_interval}
                  onChange={(event) => handleSettingsFieldChange("decision_interval", event.target.value)}
                >
                  <option value="minutes">Minutes</option>
                  <option value="hours">Hours</option>
                  <option value="days">Days</option>
                </select>
              </label>
              <label>
                Forecast interval
                <select
                  value={settingsDraft.forecast_interval}
                  onChange={(event) => handleSettingsFieldChange("forecast_interval", event.target.value)}
                >
                  <option value="minutes">Minutes</option>
                  <option value="hours">Hours</option>
                  <option value="days">Days</option>
                </select>
              </label>
              <label>
                Memory prune limit
                <input
                  type="number"
                  min={10}
                  value={settingsDraft.memory_prune_limit}
                  onChange={(event) => handleSettingsFieldChange("memory_prune_limit", Number(event.target.value))}
                />
              </label>
              <label>
                Similarity threshold
                <input
                  type="number"
                  step={0.01}
                  min={0}
                  max={1}
                  value={settingsDraft.memory_prune_similarity_threshold}
                  onChange={(event) =>
                    handleSettingsFieldChange("memory_prune_similarity_threshold", Number(event.target.value))
                  }
                />
              </label>
              <label>
                Review interval (hours)
                <input
                  type="number"
                  min={1}
                  value={settingsDraft.review_interval_hours}
                  onChange={(event) => handleSettingsFieldChange("review_interval_hours", Number(event.target.value))}
                />
              </label>
            </div>
            <label className="settings-textarea">
              Review prompt
              <textarea
                value={settingsDraft.review_prompt}
                onChange={(event) => handleSettingsFieldChange("review_prompt", event.target.value)}
                rows={4}
              />
            </label>
            <div className="settings-forecast">
              <h3>Available Forecast Tickers</h3>
              {forecastTickers.length === 0 ? (
                <p className="muted">No tickers returned by forecasting API yet.</p>
              ) : (
                <div className="ticker-cloud">
                  {forecastTickers.map((ticker) => (
                    <span key={ticker}>{ticker}</span>
                  ))}
                </div>
              )}
            </div>
            <div className="settings-actions">
              <button type="submit" className="refresh-button" disabled={settingsBusy}>
                {settingsBusy ? "Saving…" : "Save Settings"}
              </button>
              {settingsMessage && <span className="muted">{settingsMessage}</span>}
            </div>
          </form>
        ) : (
          <p className="muted">Settings unavailable. Refresh the dashboard to load configuration.</p>
        )}
      </section>

      <section className="section">
        <div className="section-header">
          <h2>Recent Logs</h2>
          <div className="log-controls">
            <label>
              Source
              <select value={logSource} onChange={(event) => handleLogSourceChange(event.target.value)}>
                <option value="trading">Trading</option>
                <option value="errors">Errors</option>
                <option value="decisions">Decisions</option>
                <option value="portfolio">Portfolio Plans</option>
              </select>
            </label>
          </div>
        </div>
        <pre className="log-output">{logs.length ? logs.join("\n") : "No log entries captured yet."}</pre>
        {logfireEnabled && <p className="muted">Logfire forwarding enabled for this deployment.</p>}
      </section>
    </div>
  );
};

export default App;

