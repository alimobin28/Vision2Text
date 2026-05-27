/**
 * Canvas rendering utilities for the 7×7 cross-attention heatmap.
 *
 * The decoder produces 49 cross-attention weights (one per image patch in a
 * 7×7 spatial grid). This module bilinearly interpolates them up to the
 * displayed image dimensions and composites them using the "hot" colormap:
 *   t=0  → fully transparent   (image shows through)
 *   t≈0.4 → warm red overlay
 *   t≈0.7 → bright yellow overlay
 *   t=1   → near-white overlay  (peak attention)
 */

export const GRID = 7;           // EfficientNetV2-S with image_size=224 → 7×7
export const N_PATCHES = GRID * GRID; // 49

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
  if (weights.length !== N_PATCHES) return;

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

  const maxX = Math.max(width - 1, 1);
  const maxY = Math.max(height - 1, 1);

  for (let py = 0; py < height; py++) {
    for (let px_ = 0; px_ < width; px_++) {
      // Map pixel coordinates to the 7×7 grid (fractional)
      const gx = (px_ / maxX) * (GRID - 1);
      const gy = (py / maxY) * (GRID - 1);

      // Bilinear interpolation
      const x0 = Math.floor(gx);
      const x1 = Math.min(x0 + 1, GRID - 1);
      const y0 = Math.floor(gy);
      const y1 = Math.min(y0 + 1, GRID - 1);
      const dx = gx - x0;
      const dy = gy - y0;

      const v =
        weights[y0 * GRID + x0] * (1 - dx) * (1 - dy) +
        weights[y0 * GRID + x1] *      dx  * (1 - dy) +
        weights[y1 * GRID + x0] * (1 - dx) *      dy  +
        weights[y1 * GRID + x1] *      dx  *      dy;

      const t = (v - lo) / range; // 0..1

      // "Hot" colormap — three linear ramps give black→red→yellow→white.
      // Alpha scales with t so low-attention areas stay transparent.
      const r = clamp255(t * 3.0);
      const g = clamp255(t * 3.0 - 1.0);
      const b = clamp255(t * 3.0 - 2.0);
      const a = Math.round(t * 210); // 0..210

      const i = (py * width + px_) * 4;
      px[i]     = r;
      px[i + 1] = g;
      px[i + 2] = b;
      px[i + 3] = a;
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

/** Clear the canvas completely — called when no token is hovered. */
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
  if (weights.length !== N_PATCHES) return 'unknown region';

  let total = 0;
  let cx = 0;
  let cy = 0;

  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      const w = Math.max(0, weights[r * GRID + c]);
      total += w;
      cx += c * w;
      cy += r * w;
    }
  }

  if (total > 0) {
    cx /= total;
    cy /= total;
  } else {
    cx = 3;
    cy = 3;
  }

  // thirds: [0, 2.33) = left/upper, [2.33, 4.67) = center/middle, [4.67, 7) = right/lower
  const xLabel = cx < 7 / 3 ? 'left' : cx > (7 * 2) / 3 ? 'right' : 'center';
  const yLabel = cy < 7 / 3 ? 'upper' : cy > (7 * 2) / 3 ? 'lower' : 'middle';

  if (xLabel === 'center' && yLabel === 'middle') return 'center';
  if (xLabel === 'center') return yLabel;
  if (yLabel === 'middle') return xLabel;
  return `${yLabel} ${xLabel}`;
}

function clamp255(v: number): number {
  return Math.round(Math.min(1, Math.max(0, v)) * 255);
}
