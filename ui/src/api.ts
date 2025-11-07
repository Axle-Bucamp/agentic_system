import { AllocationEntry, PlanSummary, WalletPlan, WalletScoresResponse, WeightedNewsEntry } from "./types";

async function fetchJSON<T>(url: string): Promise<T> {
  const response = await fetch(url);
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

