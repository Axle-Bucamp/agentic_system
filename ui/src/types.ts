export interface AllocationEntry {
  allocation: number;
  allocation_pct: number;
  target_usdc: number;
  confidence: number;
  composite_score: number;
  risk_level: string;
  risk_score: number;
  gas_fee_estimate: number;
  stop_loss: [number, number];
  pnl_estimate: number;
  signals: Record<string, unknown>;
}

export interface WalletPlan {
  generated_at: string;
  profile?: string;
  total_value_usdc: number;
  allocations: Record<string, AllocationEntry>;
  context?: Array<Record<string, unknown>>;
}

export interface PlanSummary {
  generated_at?: string | null;
  summary?: string | null;
}

export interface WalletScoresResponse {
  wallets: Record<
    string,
    {
      success_rate: number;
      trade_count: number;
      signal_score?: number;
      last_signal_at?: string | null;
    }
  >;
}

export interface WeightedNewsEntry {
  ticker?: string | null;
  weighted_score?: number;
  total_weight?: number;
  last_updated?: string;
}

export interface ChatEntry {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

export interface FusionRecommendation {
  ticker: string;
  action: string;
  confidence: number;
  percent_allocation: number;
  risk_level: string;
  rationale?: string;
  components?: Record<string, unknown>;
  generated_at?: string;
  agentic?: boolean;
  ai_explanation?: string | null;
}

export interface FusionRecommendationsResponse {
  items: FusionRecommendation[];
}

export interface TrendAssessment {
  ticker: string;
  trend_score: number;
  momentum: number;
  volatility?: number | null;
  recommended_action: string;
  confidence: number;
  supporting_signals?: Record<string, unknown>;
  generated_at?: string;
  agentic?: boolean;
  ai_explanation?: string | null;
}

export interface FactInsight {
  ticker?: string | null;
  sentiment_score: number;
  confidence: number;
  thesis: string;
  references: Array<Record<string, unknown>>;
  anomalies?: string[];
  generated_at?: string;
  sentiment_breakdown?: Record<string, unknown>;
  market_indicators?: Record<string, unknown>;
  agentic?: boolean;
  ai_explanation?: string | null;
}

export interface AgentStatusResponse {
  copytrade: Record<string, unknown>;
  trend: Array<{ ticker: string; generated_at?: string }>;
  fact: Array<{ ticker: string; generated_at?: string }>;
  fusion: Array<{ ticker: string; generated_at?: string; action?: string; confidence?: number }>;
  logfire_enabled: boolean;
}

export interface GraphNode {
  node_id: string;
  label: string;
  node_type: string;
  weight: number;
  metadata?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

export interface GraphEdge {
  edge_id: string;
  source: string;
  target: string;
  relation: string;
  weight: number;
  metadata?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

export interface GraphSnapshotResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface LogResponse {
  source: string;
  path: string;
  lines: string[];
  logfire_enabled: boolean;
}

export interface DashboardSettings {
  schedule_profile: string;
  memory_prune_limit: number;
  memory_prune_similarity_threshold: number;
  review_interval_hours: number;
  review_prompt: string;
  observation_interval: string;
  decision_interval: string;
  forecast_interval: string;
}

export interface PipelineLiveEntry {
  enabled: boolean;
  interval: string;
  seconds?: number;
}

export type PipelineLiveConfig = Record<string, PipelineLiveEntry>;

export interface TradeRecord {
  trade_id: string;
  ticker: string;
  action: string;
  quantity: number;
  executed_price?: number | null;
  entry_price?: number | null;
  evaluation_price?: number | null;
  pnl?: number | null;
  reward?: number | null;
  status?: string | null;
  confidence?: number | null;
  timestamp?: string;
  reward_evaluated_at?: string;
}

export interface AgentRewardSummary {
  total_reward: number;
  trades: number;
  average_reward: number;
}

export interface AgentRewardResponse {
  rewards: Record<string, AgentRewardSummary>;
}

export interface PortfolioTradesResponse {
  trades: TradeRecord[];
  count: number;
  limit: number;
  offset: number;
  total: number;
}

export interface AiDecisionStep {
  step: string;
  description?: string;
  result?: string;
  messages?: string[];
  timestamp?: string;
}

export interface AiDecision {
  decision_id?: string;
  ticker?: string;
  action?: string;
  status?: string;
  task_description?: string;
  result?: string;
  result_text?: string;
  result_raw?: unknown;
  completed_at?: string;
  timestamp?: string;
  error?: string;
  steps?: AiDecisionStep[];
  agentic?: boolean;
  ai_explanation?: string | null;
}

export interface AiDecisionResponse {
  decisions: AiDecision[];
  count: number;
  total: number;
}

