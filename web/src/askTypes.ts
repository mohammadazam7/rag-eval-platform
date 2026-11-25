import type { Chunk } from "./engine";

export interface Interval {
  mean: number;
  low: number;
  high: number;
}

export interface BenchmarkRow {
  configuration: string;
  metrics: Record<string, Interval>;
}

/** Shape of web/public/ask.json, written by scripts/export_ask_data.py. */
export interface AskData {
  corpus: string;
  documents: number;
  k: number;
  chunks: Chunk[];
  suggested: string[];
  questions: { total: number; answerable: number };
  benchmark: BenchmarkRow[];
}
