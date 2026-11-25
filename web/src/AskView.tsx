import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";

import type { AskData } from "./askTypes";
import { answer, contentTerms, sentences, tokenize, type Answer, type BM25, type Chunk } from "./engine";

interface Props {
  data: AskData;
  index: BM25;
}

interface Run {
  answer: Answer;
  ms: number;
}

/** Split an answer into its quoted sentences and their [Pn] labels. */
function answerParts(result: Answer): { text: string; label: string; end: string }[] {
  return result.quotes.map((q) => {
    const s = ".!?".includes(q.sentence[q.sentence.length - 1]) ? q.sentence : `${q.sentence}.`;
    return { text: tidy(s.slice(0, -1)), label: q.label, end: s[s.length - 1] };
  });
}

/**
 * Display-only clean-up of the manual's Markdown: code-include directives, heading
 * anchors, link targets and emphasis asterisks are noise to a reader. The engine still
 * works on the original text, so answers stay identical to the Python pipeline's.
 */
export function tidy(text: string): string {
  return text
    .replace(/\{\*[^}]*\*\}/g, "")
    .replace(/\{\s*#[^}]*\}/g, "")
    .replace(/\/\/\/(\s*(note|tip|info|warning|check|details)\b(\s*\|)?)?/g, "")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/\*([^*\s][^*]*)\*/g, "$1")
    .replace(/\s{2,}/g, " ")
    .replace(/\s+([.,:;!?])/g, "$1")
    .trim();
}

/** Wrap the words of `text` that are content terms of the question in <b>. */
function Emphasised({ text, terms }: { text: string; terms: Set<string> }) {
  const pieces = tidy(text).split(/([A-Za-z0-9]+)/);
  return (
    <>
      {pieces.map((piece, i) =>
        i % 2 === 1 && terms.has(piece.toLowerCase()) ? <b key={i}>{piece}</b> : piece,
      )}
    </>
  );
}

export function AskView({ data, index }: Props) {
  const [draft, setDraft] = useState(data.suggested[0] ?? "");
  const [run, setRun] = useState<Run | null>(null);
  const [openLabel, setOpenLabel] = useState<string | null>(null);
  const markRef = useRef<HTMLElement>(null);

  const ask = (question: string) => {
    if (!tokenize(question).length) return;
    const started = performance.now();
    const result = answer(index, question, data.k);
    setRun({ answer: result, ms: performance.now() - started });
    setOpenLabel(result.quotes[0]?.label ?? (result.passages.size ? "P1" : null));
  };

  // Answer the first suggested question on load, so the page opens on a real result.
  useEffect(() => {
    if (data.suggested[0]) ask(data.suggested[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    markRef.current?.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
  }, [openLabel, run]);

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    ask(draft);
  };

  const result = run?.answer;
  const terms = useMemo(() => (result ? contentTerms(result.question) : new Set<string>()), [result]);
  const open: Chunk | undefined = result && openLabel ? result.passages.get(openLabel) : undefined;
  const quotedHere = result?.quotes.filter((q) => q.label === openLabel).map((q) => q.position) ?? [];

  return (
    <main className="ask">
      <section className="desk" aria-labelledby="ask-title">
        <h1 id="ask-title">Ask the FastAPI manual. Check every answer.</h1>
        <p className="standfirst">
          Answers are quoted from the manual&rsquo;s {data.documents} pages, and each one shows the
          passage it came from. When the manual has no answer, it says so.
        </p>

        <form className="ask-form" onSubmit={onSubmit}>
          <label htmlFor="question">Your question</label>
          <div className="ask-row">
            <input
              id="question"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              autoComplete="off"
              spellCheck={false}
            />
            <button type="submit">Ask</button>
          </div>
        </form>

        <ul className="suggestions" aria-label="Questions to try">
          {[...data.suggested.slice(1), "Who won the World Cup in 2022?"].map((q) => (
            <li key={q}>
              <button
                type="button"
                onClick={() => {
                  setDraft(q);
                  ask(q);
                }}
              >
                {q}
              </button>
            </li>
          ))}
        </ul>

        {result && (
          <article className={result.abstained ? "answer abstained" : "answer"} aria-live="polite">
            {result.abstained ? (
              <>
                <p className="answer-text">The manual does not answer this.</p>
                <p className="answer-note">
                  No sentence in the closest passages covers at least half of the question&rsquo;s
                  key words, so nothing is quoted. The passages it looked at are listed below.
                </p>
              </>
            ) : (
              <p className="answer-text">
                {answerParts(result).map((part, i) => (
                  <span key={i}>
                    <mark
                      className={part.label === openLabel ? "quote on" : "quote"}
                      role="button"
                      tabIndex={0}
                      onClick={() => setOpenLabel(part.label)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          setOpenLabel(part.label);
                        }
                      }}
                    >
                      {part.text}
                    </mark>
                    <button type="button" className="cite" onClick={() => setOpenLabel(part.label)}>
                      {part.label.slice(1)}
                    </button>
                    {part.end}{" "}
                  </span>
                ))}
              </p>
            )}
            <p className="meta">
              Searched {data.chunks.length} passages and answered in {run.ms < 1 ? "under 1" : Math.round(run.ms)} ms,
              in your browser. Nothing was sent to a server.
            </p>
          </article>
        )}

        {result && result.retrieved.length > 0 && (
          <section aria-labelledby="found-title" className="found">
            <h2 id="found-title">Passages it read</h2>
            <ol>
              {[...result.passages].map(([label, chunk], i) => {
                const cited = result.quotes.some((q) => q.label === label);
                return (
                  <li key={chunk.id}>
                    <button
                      type="button"
                      className="passage"
                      aria-pressed={label === openLabel}
                      onClick={() => setOpenLabel(label)}
                    >
                      <span className="passage-doc">{chunk.doc}</span>
                      <span className="passage-score">
                        match {result.retrieved[i].score.toFixed(1)}
                        {cited ? ", quoted" : ""}
                      </span>
                      <span className="passage-preview">{tidy(sentences(chunk.text).slice(0, 2).join(" "))}</span>
                    </button>
                  </li>
                );
              })}
            </ol>
          </section>
        )}
      </section>

      <aside className="reader" aria-labelledby="reader-title">
        {open ? (
          <>
            <h2 id="reader-title">
              <span className="file">{open.doc}</span>
              <span className="span">
                characters {open.start.toLocaleString("en")} to {open.end.toLocaleString("en")}
              </span>
            </h2>
            <div className="sheet">
              {sentences(open.text).map((sentence, i) =>
                quotedHere.includes(i) ? (
                  <mark key={`${open.id}-${i}`} ref={i === quotedHere[0] ? markRef : undefined} className="source-mark">
                    <Emphasised text={sentence} terms={terms} />{" "}
                  </mark>
                ) : (
                  <span key={`${open.id}-${i}`}>
                    <Emphasised text={sentence} terms={terms} />{" "}
                  </span>
                ),
              )}
            </div>
            <p className="reader-note">
              {quotedHere.length
                ? "The highlighted sentence is the one quoted in the answer. Bold words are the words from your question."
                : "Nothing from this passage was quoted. Bold words are the words from your question."}
            </p>
          </>
        ) : (
          <p className="reader-empty">Ask a question to see the passage behind the answer.</p>
        )}
      </aside>
    </main>
  );
}
