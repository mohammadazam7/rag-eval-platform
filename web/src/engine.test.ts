import { describe, expect, it } from "vitest";

import data from "../public/ask.json";
import { answer, BM25, sentences, tokenize } from "./engine";
import fixture from "./parity.fixture.json";

const index = new BM25(data.chunks);

describe("parity with the Python pipeline", () => {
  it("covers every benchmark question", () => {
    expect(fixture).toHaveLength(44);
  });

  it.each(fixture.map((q) => [q.id, q] as const))("%s retrieves and answers the same", (_id, q) => {
    const result = answer(index, q.question, 4);
    expect(result.retrieved.map((h) => h.chunk.id)).toEqual(q.retrieved.map((r) => r.chunk_id));
    result.retrieved.forEach((hit, i) => expect(hit.score).toBeCloseTo(q.retrieved[i].score, 5));
    expect(result.text).toBe(q.answer);
    expect([...new Set(result.quotes.map((s) => s.chunkId))]).toEqual(q.cited);
  });
});

describe("text handling", () => {
  it("tokenizes to lowercase alphanumeric runs", () => {
    expect(tokenize("Run *after* the response, v0.95.0!")).toEqual(["run", "after", "the", "response", "v0", "95", "0"]);
  });

  it("drops table rows and markup, and keeps list items apart", () => {
    const text = "# Title\n\nFirst **bold** line\nwraps here. Second one!\n| a | b |\n- item one\n- item `two`";
    expect(sentences(text)).toEqual(["Title", "First bold line wraps here.", "Second one!", "item one", "item two"]);
  });

  it("abstains when no sentence shares enough of the question", () => {
    expect(answer(index, "Who won the World Cup in 2022?").abstained).toBe(true);
    expect(answer(index, "the of and").abstained).toBe(true);
  });
});
