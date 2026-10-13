/**
 * Canvas rendering utilities for the cross-attention heatmap.
 *
 * The decoder produces one cross-attention weight per image patch — a g×g
 * spatial grid (7×7 = 49 for EfficientNetV2-S at 224 px). This module
 * bilinearly interpolates them up to the displayed image size and colours them
 * with a "hot" colormap:
 *   t=0   → fully transparent   (image shows through)
 *   t≈0.4 → warm red overlay
 *   t≈0.7 → bright yellow overlay
 *   t=1   → near-white overlay  (peak attention)
 *
 * The grid size is derived from the number of weights rather than hard-coded,
 * so a different backbone / input size needs no frontend change.
 */

/** Side length g of the square grid holding `n` patches, or null if `n` is not g². */
export function gridSizeFor(n: number): number | null {
  const g = Math.round(Math.sqrt(n));
  return g > 0 && g * g === n ? g : null;
}

/**
 * Bilinearly sample a g×g grid at normalised image coordinates (u, v) ∈ [0, 1].
 *
 * Patch i covers [i/g, (i+1)/g), so its *centre* sits at (i + 0.5)/g. Sampling
 * relative to centres (not corners) keeps the heatmap aligned with the photo
 * instead of being shifted by half a patch.
 */
export function sampleGrid(
  weights: readonly number[],
  g: number,
  u: number,
  v: number,
): number {
  const gx = Math.min(g - 1, Math.max(0, u * g - 0.5));
  const gy = Math.min(g - 1, Math.max(0, v * g - 0.5));

  const x0 = Math.floor(gx);
  const y0 = Math.floor(gy);
  const x1 = Math.min(x0 + 1, g - 1);
  const y1 = Math.min(y0 + 1, g - 1);
  const dx = gx - x0;
  const dy = gy - y0;

  const at = (x: number, y: number): number => weights[y * g + x] ?? 0;
  return (
    at(x0, y0) * (1 - dx) * (1 - dy) +
    at(x1, y0) * dx * (1 - dy) +
    at(x0, y1) * (1 - dx) * dy +
    at(x1, y1) * dx * dy
  );
}

/** "Hot" colormap: t ∈ [0, 1] → [r, g, b, a], alpha growing with t. */
export function heatColor(t: number): [number, number, number, number] {
  return [
    clamp255(t * 3.0),
    clamp255(t * 3.0 - 1.0),
    clamp255(t * 3.0 - 2.0),
    Math.round(Math.min(1, Math.max(0, t)) * 210),
  ];
}

/**
 * Render a bilinearly-interpolated attention heatmap onto `canvas`.
 *
 * The canvas dimensions are used as-is — callers should set canvas.width
 * and canvas.height to match the displayed image before calling this.
 */
export function renderAttentionHeatmap(
  canvas: HTMLCanvasElement,
  weights: readonly number[],
): void {
  const g = gridSizeFor(weights.length);
  if (g === null) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const { width, height } = canvas;
  if (width === 0 || height === 0) return;

  const imageData = ctx.createImageData(width, height);
  const px = imageData.data;

  // Normalise to [0, 1] so the colormap always spans the full range even
  // when attention is concentrated in a small region.
  let lo = Infinity;
  let hi = -Infinity;
  for (const w of weights) {
    if (w < lo) lo = w;
    if (w > hi) hi = w;
  }
  const range = hi - lo > 1e-9 ? hi - lo : 1e-9;

  for (let py = 0; py < height; py++) {
    const v = (py + 0.5) / height;
    for (let pxX = 0; pxX < width; pxX++) {
      const t = (sampleGrid(weights, g, (pxX + 0.5) / width, v) - lo) / range;
      const [r, gr, b, a] = heatColor(t);

      const i = (py * width + pxX) * 4;
      px[i] = r;
      px[i + 1] = gr;
      px[i + 2] = b;
      px[i + 3] = a;
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

/** Clear the canvas completely — called when no token is selected. */
export function clearCanvas(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('2d');
  if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * Describe the dominant attended region in plain English.
 * Used for aria-labels and screen-reader announcements.
 *
 * Returns something like "upper left", "center", or "lower right".
 */
export function describeAttentionRegion(weights: readonly number[]): string {
  const g = gridSizeFor(weights.length);
  if (g === null) return 'unknown region';

  // Attention-weighted centroid in normalised image coordinates (0..1).
  let total = 0;
  let cx = 0;
  let cy = 0;
  for (let r = 0; r < g; r++) {
    for (let c = 0; c < g; c++) {
      const w = Math.max(0, weights[r * g + c] ?? 0);
      total += w;
      cx += (c + 0.5) * w;
      cy += (r + 0.5) * w;
    }
  }
  const x = total > 0 ? cx / total / g : 0.5;
  const y = total > 0 ? cy / total / g : 0.5;

  // thirds of the image
  const xLabel = x < 1 / 3 ? 'left' : x > 2 / 3 ? 'right' : 'center';
  const yLabel = y < 1 / 3 ? 'upper' : y > 2 / 3 ? 'lower' : 'middle';

  if (xLabel === 'center' && yLabel === 'middle') return 'center';
  if (xLabel === 'center') return yLabel;
  if (yLabel === 'middle') return xLabel;
  return `${yLabel} ${xLabel}`;
}

function clamp255(v: number): number {
  return Math.round(Math.min(1, Math.max(0, v)) * 255);
}
