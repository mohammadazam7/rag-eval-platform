import { describe, expect, it } from "vitest";

import { chunkStatus, findChunk, hitRank, orderedChunks } from "./gallery";
import type { GalleryData } from "./types";

function data(): GalleryData {
  return {
    corpus: "fastapi-tutorial",
    question: { id: "a01", text: "When do background tasks run?", gold_chunk_ids: ["c2"] },
    chunks: [
      { id: "c1", doc_id: "a.md", x: 0, y: 0, z: 0, preview: "one", gold: false },
      { id: "c2", doc_id: "a.md", x: 1, y: 0, z: 0, preview: "two", gold: true },
      { id: "c3", doc_id: "b.md", x: 0, y: 1, z: 0, preview: "three", gold: false },
    ],
    retrieval: {
      bm25: [
        { chunk_id: "c2", rank: 1, score: 5 },
        { chunk_id: "c1", rank: 2, score: 2 },
      ],
      hybrid: [{ chunk_id: "c2", rank: 1, score: 0.03 }],
    },
  };
}

describe("hitRank", () => {
  it("returns the rank a method retrieved a chunk at", () => {
    expect(hitRank(data(), "bm25", "c1")).toBe(2);
  });

  it("returns null for a chunk the method did not retrieve", () => {
    expect(hitRank(data(), "hybrid", "c1")).toBeNull();
  });
});

describe("chunkStatus", () => {
  it.each([
    ["c2", "bm25", "gold-and-retrieved"],
    ["c2", "hybrid", "gold-and-retrieved"],
    ["c1", "bm25", "retrieved"],
    ["c1", "hybrid", "none"],
    ["c3", "bm25", "none"],
  ] as const)("chunk %s under %s is %s", (id, method, expected) => {
    const d = data();
    const chunk = findChunk(d, id);
    if (!chunk) throw new Error("fixture chunk missing");
    expect(chunkStatus(d, method, chunk)).toBe(expected);
  });

  it("never marks a gold chunk as plain none or plain retrieved", () => {
    const d = data();
    const gold = findChunk(d, "c2");
    if (!gold) throw new Error("fixture chunk missing");
    for (const method of ["bm25", "hybrid"] as const) {
      expect(chunkStatus(d, method, gold)).not.toBe("none");
      expect(chunkStatus(d, method, gold)).not.toBe("retrieved");
    }
  });
});

describe("orderedChunks", () => {
  it("puts retrieved chunks first in rank order, then the rest by id", () => {
    const ids = orderedChunks(data(), "bm25").map((c) => c.id);
    expect(ids).toEqual(["c2", "c1", "c3"]);
  });

  it("falls back to id order entirely when nothing was retrieved", () => {
    const empty: GalleryData = { ...data(), retrieval: { bm25: [], hybrid: [] } };
    expect(orderedChunks(empty, "bm25").map((c) => c.id)).toEqual(["c1", "c2", "c3"]);
  });

  it("includes every chunk exactly once", () => {
    const ids = orderedChunks(data(), "hybrid").map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids.sort()).toEqual(["c1", "c2", "c3"]);
  });
});

describe("findChunk", () => {
  it("returns undefined for an id that is not in the data", () => {
    expect(findChunk(data(), "missing")).toBeUndefined();
  });
});
