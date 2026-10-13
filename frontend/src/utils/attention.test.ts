import { describe, expect, it } from 'vitest';
import {
  describeAttentionRegion,
  gridSizeFor,
  heatColor,
  sampleGrid,
} from './attention';

/** A g×g grid that is 1 at (row, col) and 0 elsewhere. */
function spike(g: number, row: number, col: number): number[] {
  const w = new Array<number>(g * g).fill(0);
  w[row * g + col] = 1;
  return w;
}

describe('gridSizeFor', () => {
  it('recovers the side of a square grid', () => {
    expect(gridSizeFor(49)).toBe(7);
    expect(gridSizeFor(16)).toBe(4);
    expect(gridSizeFor(1)).toBe(1);
  });

  it('rejects non-square or empty inputs', () => {
    expect(gridSizeFor(0)).toBeNull();
    expect(gridSizeFor(48)).toBeNull();
    expect(gridSizeFor(50)).toBeNull();
  });
});

describe('sampleGrid', () => {
  it('returns the exact weight at a patch centre', () => {
    const g = 7;
    const w = spike(g, 2, 4);
    // centre of patch (row 2, col 4) in normalised image coordinates
    expect(sampleGrid(w, g, (4 + 0.5) / g, (2 + 0.5) / g)).toBeCloseTo(1, 10);
  });

  it('interpolates halfway between two neighbouring centres', () => {
    const g = 4;
    const w = spike(g, 1, 1);
    // midway between the centres of (1,1) and (1,2)
    expect(sampleGrid(w, g, (1 + 1) / g, (1 + 0.5) / g)).toBeCloseTo(0.5, 10);
  });

  it('clamps at the image border instead of reading outside the grid', () => {
    const g = 7;
    const w = spike(g, 0, 0);
    expect(sampleGrid(w, g, 0, 0)).toBeCloseTo(1, 10);
    expect(sampleGrid(w, g, 1, 1)).toBeCloseTo(0, 10);
  });

  it('is aligned with the photo: the top-left patch lights the top-left corner, not the middle', () => {
    const g = 7;
    const w = spike(g, 0, 0);
    expect(sampleGrid(w, g, 0.05, 0.05)).toBeGreaterThan(0.7);
    expect(sampleGrid(w, g, 0.5, 0.5)).toBe(0);
  });
});

describe('heatColor', () => {
  it('is fully transparent at 0 and opaque-ish white at 1', () => {
    expect(heatColor(0)).toEqual([0, 0, 0, 0]);
    expect(heatColor(1)).toEqual([255, 255, 255, 210]);
  });

  it('clamps out-of-range input', () => {
    expect(heatColor(-5)).toEqual(heatColor(0));
    expect(heatColor(5)).toEqual(heatColor(1));
  });

  it('goes red → yellow → white as t grows', () => {
    const [r1, g1] = heatColor(0.33);
    const [r2, g2, b2] = heatColor(0.66);
    expect(r1).toBeGreaterThan(200);
    expect(g1).toBeLessThan(10);
    expect(r2).toBe(255);
    expect(g2).toBeGreaterThan(200);
    expect(b2).toBeLessThan(10);
  });
});

describe('describeAttentionRegion', () => {
  it('names the corners and the centre', () => {
    expect(describeAttentionRegion(spike(7, 0, 0))).toBe('upper left');
    expect(describeAttentionRegion(spike(7, 0, 6))).toBe('upper right');
    expect(describeAttentionRegion(spike(7, 6, 0))).toBe('lower left');
    expect(describeAttentionRegion(spike(7, 6, 6))).toBe('lower right');
    expect(describeAttentionRegion(spike(7, 3, 3))).toBe('center');
  });

  it('names the edges', () => {
    expect(describeAttentionRegion(spike(7, 0, 3))).toBe('upper');
    expect(describeAttentionRegion(spike(7, 6, 3))).toBe('lower');
    expect(describeAttentionRegion(spike(7, 3, 0))).toBe('left');
    expect(describeAttentionRegion(spike(7, 3, 6))).toBe('right');
  });

  it('treats uniform attention as the centre', () => {
    expect(describeAttentionRegion(new Array<number>(49).fill(1 / 49))).toBe('center');
  });

  it('works for other grid sizes', () => {
    expect(describeAttentionRegion(spike(4, 0, 0))).toBe('upper left');
    expect(describeAttentionRegion(spike(4, 3, 3))).toBe('lower right');
  });

  it('handles empty / all-zero / malformed input', () => {
    expect(describeAttentionRegion([])).toBe('unknown region');
    expect(describeAttentionRegion(new Array<number>(48).fill(1))).toBe('unknown region');
    expect(describeAttentionRegion(new Array<number>(49).fill(0))).toBe('center');
  });
});
