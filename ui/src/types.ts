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

