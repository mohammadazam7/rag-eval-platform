import { lazy, Suspense, useEffect, useMemo, useState } from "react";

import { AskView } from "./AskView";
import type { AskData } from "./askTypes";
import { BM25 } from "./engine";
import { ResultsView } from "./ResultsView";

const MapView = lazy(() => import("./MapView").then((m) => ({ default: m.MapView })));

const TABS = [
  ["ask", "Ask"],
  ["results", "Measured results"],
  ["map", "Map"],
] as const;
type Tab = (typeof TABS)[number][0];

const tabFromHash = (): Tab => {
  const hash = typeof window === "undefined" ? "" : window.location.hash.slice(1);
  return TABS.some(([id]) => id === hash) ? (hash as Tab) : "ask";
};

type LoadState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: AskData };

export function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash);
  const [state, setState] = useState<LoadState>({ status: "loading" });

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}ask.json`)
      .then((res) => {
        if (!res.ok) throw new Error(`ask.json: HTTP ${res.status}`);
        return res.json() as Promise<AskData>;
      })
      .then((data) => setState({ status: "ready", data }))
      .catch((err: unknown) =>
        setState({ status: "error", message: err instanceof Error ? err.message : String(err) }),
      );
  }, []);

  // Building the index tokenizes every passage once; about 10 ms for this corpus.
  const index = useMemo(
    () => (state.status === "ready" ? new BM25(state.data.chunks) : null),
    [state],
  );

  const select = (next: Tab) => {
    window.history.replaceState(null, "", `#${next}`);
    setTab(next);
  };

  return (
    <div className="shell">
      <header className="masthead">
        <a className="wordmark" href="#ask" onClick={() => select("ask")}>
          rag-eval-platform
        </a>
        <nav aria-label="Views">
          {TABS.map(([id, label]) => (
            <button
              key={id}
              type="button"
              className="tab"
              aria-current={tab === id ? "page" : undefined}
              onClick={() => select(id)}
            >
              {label}
            </button>
          ))}
        </nav>
        <a className="source" href="https://github.com/mohammadazam7/rag-eval-platform">
          Source on GitHub
        </a>
      </header>

      {state.status === "loading" && <p role="status" className="notice">Loading the manual…</p>}
      {state.status === "error" && (
        <p role="alert" className="notice">
          The manual could not be loaded ({state.message}). Reload the page to try again.
        </p>
      )}
      {state.status === "ready" && index && tab === "ask" && <AskView data={state.data} index={index} />}
      {state.status === "ready" && tab === "results" && <ResultsView data={state.data} />}
      {tab === "map" && (
        <Suspense fallback={<p role="status" className="notice">Loading the map…</p>}>
          <MapView />
        </Suspense>
      )}
    </div>
  );
}
