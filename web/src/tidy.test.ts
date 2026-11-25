import { describe, expect, it } from "vitest";

import { tidy } from "./AskView";

describe("tidy", () => {
  it("removes code-include directives and heading anchors", () => {
    expect(tidy("run afterwards: {* ../../docs_src/x/tutorial002.py hl[13,15] *} In this example")).toBe(
      "run afterwards: In this example",
    );
    expect(tidy("Background Tasks { #background-tasks } You can")).toBe("Background Tasks You can");
  });

  it("keeps link text and drops emphasis asterisks", () => {
    expect(tidy("see the [Advanced User Guide](../advanced/x.md) to run *after* returning")).toBe(
      "see the Advanced User Guide to run after returning",
    );
  });

  it("removes admonition markers", () => {
    expect(tidy("from typing /// note FastAPI added support /// more text")).toBe(
      "from typing FastAPI added support more text",
    );
    expect(tidy("the response. /// note | Technical Details If you have")).toBe("the response. Technical Details If you have");
  });

  it("leaves ordinary sentences alone", () => {
    expect(tidy("In version 0.95.0, 2 * 3 is 6.")).toBe("In version 0.95.0, 2 * 3 is 6.");
  });
});
