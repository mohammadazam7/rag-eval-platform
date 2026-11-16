import { lazy, Suspense, useEffect, useState } from "react";

import { ChunkTable } from "./ChunkTable";
import { canShowGalaxy } from "./environment";
import type { GalleryData, RetrievalMethod } from "./types";

// Three.js and the fiber renderer are only needed when the 3D view can actually show,
// so visitors without WebGL or with reduced motion never download them.
const Galaxy = lazy(() => import("./Galaxy").then((m) => ({ default: m.Galaxy })));

type LoadState = { status: "loading" } | { status: "error"; message: string } | { status: "ready"; data: GalleryData };

export function App() {
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const [method, setMethod] = useState<RetrievalMethod>("hybrid");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showGalaxy, setShowGalaxy] = useState(false);

  useEffect(() => {
    setShowGalaxy(canShowGalaxy());
  }, []);

  useEffect(() => {
    const base = import.meta.env.BASE_URL;
    fetch(`${base}data.json`)
      .then((res) => {
        if (!res.ok) throw new Error(`data.json: HTTP ${res.status}`);
        return res.json() as Promise<GalleryData>;
      })
      .then((data) => setState({ status: "ready", data }))
      .catch((err: unknown) =>
        setState({ status: "error", message: err instanceof Error ? err.message : String(err) }),
      );
  }, []);

  if (state.status === "loading") return <p role="status">Loading the gallery data…</p>;
  if (state.status === "error") {
    return <p role="alert">Could not load the gallery data: {state.message}</p>;
  }

  const { data } = state;
  return (
    <main>
      <h1>Retrieval Galaxy</h1>
      <p>
        {data.chunks.length} chunks from the <code>{data.corpus}</code> corpus, laid out by
        embedding similarity. Question: <strong>{data.question.text}</strong>
      </p>
      <fieldset>
        <legend>Retrieval method</legend>
        {(["bm25", "hybrid"] as const).map((m) => (
          <label key={m}>
            <input
              type="radio"
              name="method"
              value={m}
              checked={method === m}
              onChange={() => setMethod(m)}
            />
            {m}
          </label>
        ))}
      </fieldset>
      {showGalaxy && (
        <div style={{ height: "480px" }}>
          <Suspense fallback={<p role="status">Loading the 3D view…</p>}>
            <Galaxy
              data={data}
              method={method}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          </Suspense>
        </div>
      )}
      <ChunkTable data={data} method={method} selectedId={selectedId} onSelect={setSelectedId} />
    </main>
  );
}
