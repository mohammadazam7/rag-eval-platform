/**
 * BM25 retrieval and the extractive answerer, in the browser.
 *
 * This is a line-for-line port of the Python pipeline (`rag_eval.retrieval`,
 * `rag_eval.prompting`, `rag_eval.generation`) so the page can answer questions without
 * a server. `engine.test.ts` runs it on every benchmark question and requires the same
 * retrieved chunks and the same answer text as the Python code produced
 * (`parity.fixture.json`, written by `scripts/export_ask_data.py`).
 *
 * Where Python and JavaScript differ (what counts as whitespace, how lines split), the
 * Python behaviour is spelled out here instead of trusting the built-ins to agree.
 */

export interface Chunk {
  id: string;
  doc: string;
  start: number;
  end: number;
  text: string;
}

export interface Hit {
  chunk: Chunk;
  score: number;
}

export interface Quote {
  label: string;
  chunkId: string;
  /** Index of the sentence within `sentences(chunk.text)`. */
  position: number;
  sentence: string;
  /** Share of the question's content terms the sentence contains. */
  overlap: number;
}

export interface Answer {
  question: string;
  retrieved: Hit[];
  /** Label (`P1`, `P2`, …) to chunk, for the passages the answerer was shown. */
  passages: Map<string, Chunk>;
  text: string;
  quotes: Quote[];
  abstained: boolean;
}

export const ABSTAIN = "I don't know.";

// Python's str.split(), str.strip() and \s on str treat exactly these as whitespace.
const WS = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const WS_RUN = new RegExp(`[${WS}]+`, "g");
const WS_EDGES = new RegExp(`^[${WS}]+|[${WS}]+$`, "g");
// str.splitlines() boundaries.
const LINE_BREAK = /\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]/;
const SENTENCE_END = new RegExp(`(?<=[.!?])[${WS}]+`);
const MARKUP = /\*\*|__|`|^#+\s*|^[-*]\s+/g;
const NUMBERED_ITEM = new RegExp(`^\\d+\\.[${WS}]`);
const WORD = /[a-z0-9]+/g;

const STOPWORDS = new Set(
  (
    "a an and are as at be by do does for from how in is it of on or that the this to was " +
    "what when where which who why with"
  ).split(" "),
);

const strip = (text: string): string => text.replace(WS_EDGES, "");
const words = (text: string): string[] => {
  const trimmed = strip(text);
  return trimmed ? trimmed.split(WS_RUN) : [];
};

export function tokenize(text: string): string[] {
  return text.toLowerCase().match(WORD) ?? [];
}

export function contentTerms(text: string): Set<string> {
  return new Set(tokenize(text).filter((t) => !STOPWORDS.has(t)));
}

/** Okapi BM25 with the Lucene IDF variant, which is never negative. */
export class BM25 {
  private readonly tf: Map<string, number>[];
  private readonly lengths: number[];
  private readonly avgLength: number;
  private readonly idf = new Map<string, number>();

  constructor(
    readonly chunks: readonly Chunk[],
    private readonly k1 = 1.2,
    private readonly b = 0.75,
  ) {
    this.tf = chunks.map((c) => {
      const counts = new Map<string, number>();
      for (const t of tokenize(c.text)) counts.set(t, (counts.get(t) ?? 0) + 1);
      return counts;
    });
    this.lengths = this.tf.map((counts) => {
      let n = 0;
      for (const v of counts.values()) n += v;
      return n;
    });
    const total = this.lengths.reduce((a, n) => a + n, 0);
    this.avgLength = this.lengths.length ? total / this.lengths.length : 0;
    const df = new Map<string, number>();
    for (const counts of this.tf) for (const t of counts.keys()) df.set(t, (df.get(t) ?? 0) + 1);
    const n = chunks.length;
    for (const [t, d] of df) this.idf.set(t, Math.log(1 + (n - d + 0.5) / (d + 0.5)));
  }

  score(queryTokens: readonly string[], index: number): number {
    const tf = this.tf[index];
    const norm = this.avgLength
      ? this.k1 * (1 - this.b + (this.b * this.lengths[index]) / this.avgLength)
      : 0;
    let total = 0;
    for (const term of queryTokens) {
      const f = tf.get(term) ?? 0;
      if (f) total += ((this.idf.get(term) ?? 0) * f * (this.k1 + 1)) / (f + norm);
    }
    return total;
  }

  search(query: string, k = 10): Hit[] {
    const tokens = tokenize(query);
    const hits: Hit[] = [];
    this.chunks.forEach((chunk, i) => {
      const score = this.score(tokens, i);
      if (score > 0) hits.push({ chunk, score });
    });
    hits.sort((a, b) => b.score - a.score || (a.chunk.id < b.chunk.id ? -1 : a.chunk.id > b.chunk.id ? 1 : 0));
    return hits.slice(0, k);
  }
}

/** Prose blocks of `text`: wrapped lines joined, table rows dropped. */
function paragraphs(text: string): string[] {
  const blocks: string[][] = [[]];
  for (const raw of text.split(LINE_BREAK)) {
    const line = strip(raw);
    if (!line || line.startsWith("|")) {
      blocks.push([]);
      continue;
    }
    if (line.startsWith("#") || line.startsWith("- ") || line.startsWith("* ") || NUMBERED_ITEM.test(line)) {
      blocks.push([]);
    }
    blocks[blocks.length - 1].push(line);
  }
  return blocks.filter((b) => b.length).map((b) => b.join(" "));
}

/** Prose sentences of `text`, with Markdown markup removed and table rows skipped. */
export function sentences(text: string): string[] {
  const out: string[] = [];
  for (const block of paragraphs(text)) {
    for (const part of block.replace(MARKUP, "").split(SENTENCE_END)) {
      const sentence = words(part).join(" ");
      if (sentence) out.push(sentence);
    }
  }
  return out;
}

/** Number chunks in order and keep as many as fit in the context budget. */
export function packPassages(chunks: readonly Chunk[], maxContextTokens = 1500): Map<string, Chunk> {
  const passages = new Map<string, Chunk>();
  const seen = new Set<string>();
  let used = 0;
  for (const chunk of chunks) {
    if (seen.has(chunk.id)) continue;
    const cost = words(chunk.text).length + 1;
    if (used + cost > maxContextTokens) continue;
    seen.add(chunk.id);
    passages.set(`P${passages.size + 1}`, chunk);
    used += cost;
  }
  return passages;
}

/**
 * Quote up to `maxSentences` sentences that contain at least `minOverlap` of the
 * question's content terms, best first; abstain when none qualifies.
 */
export function extract(
  question: string,
  passages: Map<string, Chunk>,
  maxSentences = 2,
  minOverlap = 0.5,
): { text: string; quotes: Quote[] } {
  const wanted = contentTerms(question);
  if (!wanted.size) return { text: ABSTAIN, quotes: [] };
  const candidates: (Quote & { rank: number })[] = [];
  let rank = 0;
  for (const [label, chunk] of passages) {
    sentences(chunk.text).forEach((sentence, position) => {
      const terms = contentTerms(sentence);
      let shared = 0;
      for (const t of wanted) if (terms.has(t)) shared += 1;
      const overlap = shared / wanted.size;
      if (overlap >= minOverlap) candidates.push({ label, chunkId: chunk.id, position, sentence, overlap, rank });
    });
    rank += 1;
  }
  if (!candidates.length) return { text: ABSTAIN, quotes: [] };
  candidates.sort((a, b) => b.overlap - a.overlap || a.rank - b.rank || a.position - b.position);
  const quotes = candidates.slice(0, maxSentences);
  const parts = quotes.map(({ label, sentence }) => {
    const s = ".!?".includes(sentence[sentence.length - 1]) ? sentence : `${sentence}.`;
    return `${s.slice(0, -1)} [${label}]${s[s.length - 1]}`;
  });
  return {
    text: parts.join(" "),
    quotes: quotes.map(({ rank: _rank, ...quote }) => quote),
  };
}

export function answer(index: BM25, question: string, k = 4): Answer {
  const asked = strip(question);
  const retrieved = index.search(asked, k);
  const passages = packPassages(retrieved.map((h) => h.chunk));
  const { text, quotes } = extract(asked, passages);
  return { question: asked, retrieved, passages, text, quotes, abstained: text === ABSTAIN };
}
