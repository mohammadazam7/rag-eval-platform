/** Whether the 3D view should be offered at all: it needs WebGL, and it is pure motion
 * so a visitor who asked for reduced motion gets the table only. Both checks are read
 * once; neither matters enough to watch for changes mid-session. */

export function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

export function hasWebGL(): boolean {
  if (typeof document === "undefined") return false;
  try {
    const canvas = document.createElement("canvas");
    return !!(
      canvas.getContext("webgl2") ||
      canvas.getContext("webgl") ||
      canvas.getContext("experimental-webgl")
    );
  } catch {
    return false;
  }
}

export function canShowGalaxy(): boolean {
  return hasWebGL() && !prefersReducedMotion();
}
