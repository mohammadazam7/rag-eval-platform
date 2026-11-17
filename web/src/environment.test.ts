import { afterEach, describe, expect, it, vi } from "vitest";

import { canShowGalaxy, hasWebGL, prefersReducedMotion } from "./environment";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("prefersReducedMotion", () => {
  it("reflects the matchMedia query", () => {
    vi.spyOn(window, "matchMedia").mockReturnValue({ matches: true } as MediaQueryList);
    expect(prefersReducedMotion()).toBe(true);
  });

  it("is false when the query does not match", () => {
    vi.spyOn(window, "matchMedia").mockReturnValue({ matches: false } as MediaQueryList);
    expect(prefersReducedMotion()).toBe(false);
  });
});

describe("hasWebGL", () => {
  it("is false when no context can be created", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    expect(hasWebGL()).toBe(false);
  });

  it("is true when a context comes back", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      {} as RenderingContext,
    );
    expect(hasWebGL()).toBe(true);
  });
});

describe("canShowGalaxy", () => {
  it("is false when reduced motion is requested even with WebGL available", () => {
    vi.spyOn(window, "matchMedia").mockReturnValue({ matches: true } as MediaQueryList);
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      {} as RenderingContext,
    );
    expect(canShowGalaxy()).toBe(false);
  });
});
