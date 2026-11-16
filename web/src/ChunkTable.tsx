import { useCallback, useEffect, useRef, type KeyboardEvent } from "react";

import { chunkStatus, hitRank, orderedChunks } from "./gallery";
import type { GalleryData, RetrievalMethod } from "./types";

interface Props {
  data: GalleryData;
  method: RetrievalMethod;
  selectedId: string | null;
  onSelect: (chunkId: string) => void;
}

const STATUS_LABEL: Record<string, string> = {
  "gold-and-retrieved": "gold, retrieved",
  gold: "gold, not retrieved",
  retrieved: "retrieved",
  none: "",
};

/** The accessible, always-available view: every chunk the question could draw on, in
 * retrieval order, each one a focusable row. This is what shows when WebGL is missing
 * or the visitor asked for reduced motion, and it stays in sync with the 3D view the
 * rest of the time, so neither view can show something the other does not. */
export function ChunkTable({ data, method, selectedId, onSelect }: Props) {
  const rows = orderedChunks(data, method);
  const selectedRow = useRef<HTMLTableRowElement>(null);

  // A chunk picked in the 3D view may be far down the table: bring its row into view.
  useEffect(() => {
    selectedRow.current?.scrollIntoView?.({ block: "nearest" });
  }, [selectedId]);

  const onKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTableRowElement>, chunkId: string) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onSelect(chunkId);
      }
    },
    [onSelect],
  );

  return (
    <table aria-label={`Chunks for "${data.question.text}", ${method} retrieval`}>
      <caption>
        {rows.length} chunks · {data.retrieval[method].length} retrieved by {method}
      </caption>
      <thead>
        <tr>
          <th scope="col">Rank</th>
          <th scope="col">Document</th>
          <th scope="col">Preview</th>
          <th scope="col">Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((chunk) => {
          const rank = hitRank(data, method, chunk.id);
          const status = chunkStatus(data, method, chunk);
          const selected = chunk.id === selectedId;
          return (
            <tr
              key={chunk.id}
              ref={selected ? selectedRow : undefined}
              tabIndex={0}
              aria-selected={selected}
              data-status={status}
              onClick={() => onSelect(chunk.id)}
              onKeyDown={(event) => onKeyDown(event, chunk.id)}
            >
              <td>{rank ?? "-"}</td>
              <td>{chunk.doc_id}</td>
              <td>{chunk.preview}</td>
              <td>{STATUS_LABEL[status]}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
