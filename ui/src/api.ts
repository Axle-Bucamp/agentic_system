import {
  AllocationEntry,
  PlanSummary,
  WalletPlan,
  WeightedNewsEntry,
  ChatEntry,
  AgentStatusResponse,
  TrendAssessment,
  FactInsight,
  FusionRecommendation,
  DashboardSettings,
  TradeRecord,
  AgentRewardResponse,
  PortfolioTradesResponse,
  PipelineLiveConfig,
  AiDecisionResponse,
} from "./types";

async function fetchJSON<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

async function postJSON<T>(url: string, payload: unknown, method = "POST"): Promise<T> {
  const response = await fetch(url, {
    method,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export interface WalletPlanResponse {
  plan: WalletPlan | null;
  summary: PlanSummary | null;
}

export interface WalletPlanHistoryResponse {
  history: WalletPlan[];
}

export const getWalletPlan = () => fetchJSON<WalletPlanResponse>("/api/orchestrator/wallet-plan");

export const getWalletPlanHistory = (limit = 10) =>
  fetchJSON<WalletPlanHistoryResponse>(`/api/orchestrator/wallet-plan/history?limit=${limit}`);

export const getCopyTradeScores = () =>
  fetchJSON<WalletScoresResponse>("/api/copytrade/wallet-scores");

export interface WeightedNewsResponse {
  entries: WeightedNewsEntry[];
}

export const getWeightedNews = () => fetchJSON<WeightedNewsResponse>("/api/news/weighted");

export const getAgentScheduleProfile = () => fetchJSON<{ profile: string }>("/api/agent-schedule/profile");

export const getChatHistory = (userId: string, limit = 50) =>
  fetchJSON<{ user_id: string; history: import("./types").ChatEntry[] }>(
    `/api/chat/${encodeURIComponent(userId)}?limit=${limit}`,
  );

export const postChatMessage = (userId: string, message: string) =>
  postJSON<{ user_id: string; reply: string; history: import("./types").ChatEntry[] }>("/api/chat", {
    user_id: userId,
    message,
  });

export const getDashboardSettings = () =>
  fetchJSON<{ settings: import("./types").DashboardSettings }>("/api/settings");

export const updateDashboardSettings = (payload: Partial<import("./types").DashboardSettings>) =>
  postJSON<{ settings: import("./types").DashboardSettings }>("/api/settings", payload);

export const triggerReviewRun = () => postJSON<{ queued: boolean }>("/api/review/run", {});

export const getForecastTickers = () =>
  fetchJSON<{ tickers: Array<{ symbol: string; has_dqn?: boolean }>; count: number }>("/api/tickers/available");

export const toggleCopyTradeAgent = (enabled: boolean) =>
  postJSON<{ enabled: boolean }>("/api/agents/copytrade/toggle", { enabled });

export const runTrendPipeline = (ticker: string) =>
  postJSON<{ success: boolean; assessment?: TrendAssessment; message?: string }>(
    "/api/pipelines/trend/run",
    { ticker },
  );

export const runFactPipeline = (ticker: string) =>
  postJSON<{ success: boolean; insight?: FactInsight; message?: string }>("/api/pipelines/fact/run", { ticker });

export const runFusionPipeline = (ticker: string) =>
  postJSON<{ success: boolean; recommendation?: FusionRecommendation; message?: string }>(
    "/api/pipelines/fusion/run",
    { ticker },
  );

export const runPrunePipeline = () => postJSON<{ success: boolean }>("/api/pipelines/prune/run", {});

export const getAgentStatus = () =>
  fetchJSON<import("./types").AgentStatusResponse>("/api/agents/status");

export const getFusionRecommendations = () =>
  fetchJSON<import("./types").FusionRecommendationsResponse>("/api/pipelines/fusion");

export const getTrendAssessment = (ticker: string) =>
  fetchJSON<{ assessment: import("./types").TrendAssessment | null }>(
    `/api/pipelines/trend/${encodeURIComponent(ticker)}`,
  );

export const getFactInsight = (ticker: string) =>
  fetchJSON<{ insight: import("./types").FactInsight | null }>(
    `/api/pipelines/fact/${encodeURIComponent(ticker)}`,
  );

export const getGraphSnapshot = (nodes = 100, edges = 200) =>
  fetchJSON<import("./types").GraphSnapshotResponse>(
    `/api/memory/graph?nodes=${nodes}&edges=${edges}`,
  );

export const getRecentLogs = (source = "trading", limit = 120) =>
  fetchJSON<import("./types").LogResponse>(
    `/api/logs/recent?source=${encodeURIComponent(source)}&limit=${limit}`,
  );

export const getPortfolioTrades = (limit = 20, offset = 0) =>
  fetchJSON<PortfolioTradesResponse>(
    `/api/portfolios/main/trades?limit=${limit}&offset=${offset}`,
  );

export const getAgentRewards = () => fetchJSON<AgentRewardResponse>("/api/agents/rewards");

export const getPipelineLiveConfig = () =>
  fetchJSON<{ pipelines: PipelineLiveConfig; intervals: string[] }>("/api/pipelines/live-config");

export const updatePipelineLiveConfig = (payload: Record<string, { enabled?: boolean; interval?: string }>) =>
  postJSON<{ pipelines: PipelineLiveConfig }>("/api/pipelines/live-config", { pipelines: payload });

export const getAiDecisions = (limit = 25) =>
  fetchJSON<AiDecisionResponse>(`/api/ai/decisions?limit=${limit}`);

