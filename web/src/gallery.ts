/** Pure helpers over GalleryData, kept free of React and Three.js so they are easy to
 * unit test: what a chunk's status is for the selected retrieval method, and the order
 * the table and the galaxy agree to show chunks in. */

import type { GalleryChunk, GalleryData, RetrievalMethod } from "./types";

export type ChunkStatus = "gold-and-retrieved" | "gold" | "retrieved" | "none";

export function hitRank(
  data: GalleryData,
  method: RetrievalMethod,
  chunkId: string,
): number | null {
  const hit = data.retrieval[method].find((h) => h.chunk_id === chunkId);
  return hit ? hit.rank : null;
}

export function chunkStatus(
  data: GalleryData,
  method: RetrievalMethod,
  chunk: GalleryChunk,
): ChunkStatus {
  const retrieved = hitRank(data, method, chunk.id) !== null;
  if (chunk.gold && retrieved) return "gold-and-retrieved";
  if (chunk.gold) return "gold";
  if (retrieved) return "retrieved";
  return "none";
}

/** Retrieved chunks first (best rank first), then everything else by id, so the table
 * and the keyboard-navigable list always agree on an order. */
export function orderedChunks(data: GalleryData, method: RetrievalMethod): GalleryChunk[] {
  const byId = new Map(data.chunks.map((c) => [c.id, c]));
  const ranked = [...data.retrieval[method]]
    .sort((a, b) => a.rank - b.rank)
    .map((hit) => byId.get(hit.chunk_id))
    .filter((c): c is GalleryChunk => c !== undefined);
  const rankedIds = new Set(ranked.map((c) => c.id));
  const rest = data.chunks.filter((c) => !rankedIds.has(c.id)).sort((a, b) => a.id.localeCompare(b.id));
  return [...ranked, ...rest];
}

export function findChunk(data: GalleryData, chunkId: string): GalleryChunk | undefined {
  return data.chunks.find((c) => c.id === chunkId);
}
