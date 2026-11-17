import { lazy, Suspense, useEffect, useState, type CSSProperties } from "react";

import { ChunkTable } from "./ChunkTable";
import { canShowGalaxy } from "./environment";
import { chunkStatus, findChunk, type ChunkStatus } from "./gallery";
import type { GalleryData, RetrievalMethod } from "./types";

// Three.js and the fiber renderer are only needed when the 3D view can actually show,
// so visitors without WebGL or with reduced motion never download them.
const Galaxy = lazy(() => import("./Galaxy").then((m) => ({ default: m.Galaxy })));

const METHOD_LABEL: Record<RetrievalMethod, string> = { bm25: "BM25", hybrid: "Hybrid (RRF)" };
const STATUS_TEXT: Record<ChunkStatus, string> = {
  "gold-and-retrieved": "gold evidence, retrieved",
  gold: "gold evidence, not retrieved",
  retrieved: "retrieved",
  none: "not retrieved",
};
const STATUS_KEY = [
  ["Gold and retrieved", "var(--both)"],
  ["Gold, missed", "var(--gold)"],
  ["Retrieved", "var(--retrieved)"],
  ["Other chunks", "var(--none)"],
] as const;

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
  const selected = selectedId ? findChunk(data, selectedId) : undefined;
  return (
    <main>
      <h1>Retrieval Galaxy</h1>
      <p className="lede">
        {data.chunks.length} chunks from the <code>{data.corpus}</code> corpus, laid out by
        embedding similarity. Question: <strong>{data.question.text}</strong>
      </p>
      <div className="controls">
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
              {METHOD_LABEL[m]}
            </label>
          ))}
        </fieldset>
        <ul className="key" aria-label="Colour key">
          {STATUS_KEY.map(([label, color]) => (
            <li key={label} style={{ "--dot": color } as CSSProperties}>
              {label}
            </li>
          ))}
        </ul>
      </div>
      <div className={showGalaxy ? "layout" : "layout table-only"}>
        {showGalaxy && (
          <div className="stage">
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
        <div className="table-wrap">
          <ChunkTable
            data={data}
            method={method}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        </div>
      </div>
      <p className="selected" aria-live="polite">
        {selected ? (
          <>
            <span className="doc">
              {selected.doc_id} · {STATUS_TEXT[chunkStatus(data, method, selected)]}
            </span>
            <br />
            {selected.preview}
          </>
        ) : (
          "Select a chunk in the table or the 3D view to read it."
        )}
      </p>
    </main>
  );
}
