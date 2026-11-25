import { useState } from "react";

import type { AskData, BenchmarkRow } from "./askTypes";

const METRICS = [
  ["recall@5", "Right passage in the top 5", "Share of answerable questions where a passage holding the answer is among the first five found."],
  ["mrr", "How high the right passage ranks", "Mean reciprocal rank: 1 when the right passage is always first, 0.5 when it is second."],
  ["abstention_correct", "Knows when to say nothing", "Share of all questions handled correctly: answered when the manual has the answer, declined when it does not."],
] as const;

const NAMES: Record<string, string> = {
  BM25: "Keyword search (BM25)",
  "Dense (hashing, 512)": "Vector search, hashing baseline",
  "Hybrid (RRF)": "Hybrid: keyword + hashing vectors",
  "Hybrid + re-rank": "Hybrid, re-ranked",
  "Dense (word vectors)": "Vector search, word vectors",
  "Hybrid (BM25 + word vectors)": "Hybrid: keyword + word vectors",
};

const W = 720;
const LEFT = 250;
const RIGHT = 56;
const ROW = 44;

function Chart({ rows, metric }: { rows: BenchmarkRow[]; metric: string }) {
  const sorted = [...rows].sort((a, b) => b.metrics[metric].mean - a.metrics[metric].mean);
  const x = (v: number) => LEFT + v * (W - LEFT - RIGHT);
  const height = sorted.length * ROW + 34;
  return (
    <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`Bar chart of ${metric} by search method`} className="chart">
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
        <g key={tick}>
          <line x1={x(tick)} x2={x(tick)} y1={0} y2={height - 28} className="grid" />
          <text x={x(tick)} y={height - 8} textAnchor="middle" className="tick">
            {tick * 100}%
          </text>
        </g>
      ))}
      {sorted.map((row, i) => {
        const m = row.metrics[metric];
        const y = i * ROW + 8;
        return (
          <g key={row.configuration} className={i === 0 ? "bar best" : "bar"}>
            <text x={LEFT - 12} y={y + 18} textAnchor="end" className="bar-label">
              {NAMES[row.configuration] ?? row.configuration}
            </text>
            <rect x={x(0)} y={y} width={x(m.mean) - x(0)} height={26} rx={3} />
            <line x1={x(m.low)} x2={x(m.high)} y1={y + 13} y2={y + 13} className="whisker" />
            <line x1={x(m.low)} x2={x(m.low)} y1={y + 7} y2={y + 19} className="whisker" />
            <line x1={x(m.high)} x2={x(m.high)} y1={y + 7} y2={y + 19} className="whisker" />
            <text x={x(m.high) + 8} y={y + 18} className="bar-value">
              {Math.round(m.mean * 100)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function ResultsView({ data }: { data: AskData }) {
  const [metric, setMetric] = useState<string>(METRICS[0][0]);
  const chosen = METRICS.find(([id]) => id === metric) ?? METRICS[0];
  const unanswerable = data.questions.total - data.questions.answerable;
  return (
    <main className="results">
      <h1>Six ways to search the manual, measured on the same {data.questions.total} questions</h1>
      <p className="standfirst">
        {data.questions.answerable} questions have an answer in the manual, each with the exact
        sentence that proves it. {unanswerable} have no answer there, to test whether the system
        declines instead of guessing.
      </p>

      <div className="metric-picker" role="group" aria-label="What to compare">
        {METRICS.map(([id, label]) => (
          <button key={id} type="button" aria-pressed={metric === id} onClick={() => setMetric(id)}>
            {label}
          </button>
        ))}
      </div>
      <p className="metric-note">{chosen[2]}</p>

      <figure>
        <Chart rows={data.benchmark} metric={metric} />
        <figcaption>
          Bars show the average. The thin line on each bar is the 95% range: with only{" "}
          {data.questions.total} questions the ranges are wide and mostly overlap.
        </figcaption>
      </figure>

      <section className="findings" aria-labelledby="findings-title">
        <h2 id="findings-title">What the numbers say</h2>
        <ul>
          <li>
            Plain keyword search is the strongest single method on this manual, where questions
            reuse the manual&rsquo;s own terms.
          </li>
          <li>
            Combining it with vector search did not help here. The ranges overlap, so the fair
            reading is &ldquo;no evidence that hybrid helps&rdquo;, not &ldquo;hybrid is worse&rdquo;.
          </li>
          <li>
            The vector methods tested are simple baselines. A modern sentence encoder has not
            been run on this benchmark yet, and that is the comparison that matters next.
          </li>
        </ul>
      </section>
    </main>
  );
}
