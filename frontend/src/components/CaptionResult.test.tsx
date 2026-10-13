// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import CaptionResult from './CaptionResult';
import type { CaptionResponse } from '../types';

/** 7×7 attention that is concentrated in one cell. */
function spike(row: number, col: number): number[] {
  const w = new Array<number>(49).fill(0);
  w[row * 7 + col] = 1;
  return w;
}

const result: CaptionResponse = {
  caption: 'a dog runs',
  beam_size: 4,
  inference_ms: 123.4,
  tokens: ['a', 'dog', 'runs'],
  attention_weights: [spike(3, 3), spike(0, 0), spike(6, 6)],
};

beforeEach(() => {
  // jsdom has neither ResizeObserver nor a 2D canvas; the behaviour under test
  // is selection + accessibility, not pixels (those live in attention.test.ts).
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe(): void {}
      unobserve(): void {}
      disconnect(): void {}
    },
  );
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function renderResult(): void {
  render(<CaptionResult imageUrl="blob:test" result={result} />);
}

const word = (name: string): HTMLElement => screen.getByRole('button', { name: new RegExp(`^${name}`) });

describe('CaptionResult', () => {
  it('renders the caption, one button per word and the stats', () => {
    renderResult();
    expect(screen.getByText('a dog runs')).toBeTruthy();
    expect(screen.getAllByRole('button')).toHaveLength(3);
    expect(screen.getByText('123 ms')).toBeTruthy();
  });

  it('previews a word on hover and describes where the model looked', () => {
    renderResult();
    fireEvent.mouseEnter(word('dog'));
    expect(word('dog').getAttribute('aria-label')).toBe('dog — attended to upper left region');
    fireEvent.mouseLeave(word('dog'));
    expect(word('dog').getAttribute('aria-label')).toBe('dog');
  });

  it('previews a word on keyboard focus too', () => {
    renderResult();
    fireEvent.focus(word('runs'));
    expect(word('runs').getAttribute('aria-label')).toBe('runs — attended to lower right region');
    fireEvent.blur(word('runs'));
    expect(word('runs').getAttribute('aria-label')).toBe('runs');
  });

  it('keeps a clicked word selected after the pointer leaves (touch support)', () => {
    renderResult();
    fireEvent.mouseEnter(word('dog'));
    fireEvent.click(word('dog'));
    fireEvent.mouseLeave(word('dog'));

    expect(word('dog').getAttribute('aria-pressed')).toBe('true');
    expect(word('dog').getAttribute('aria-label')).toContain('upper left');
  });

  it('toggles the pin off on a second click and clears the selection', () => {
    renderResult();
    fireEvent.click(word('dog'));
    fireEvent.click(word('dog'));
    expect(word('dog').getAttribute('aria-pressed')).toBe('false');
    expect(word('dog').getAttribute('aria-label')).toBe('dog');
  });

  it('lets hover override the pinned word, then falls back to it', () => {
    renderResult();
    fireEvent.click(word('dog'));                    // pin "dog" (upper left)
    fireEvent.mouseEnter(word('runs'));              // hover "runs" (lower right)

    expect(word('runs').getAttribute('aria-label')).toContain('lower right');
    expect(word('dog').getAttribute('aria-pressed')).toBe('true');

    fireEvent.mouseLeave(word('runs'));
    expect(word('dog').getAttribute('aria-label')).toContain('upper left');
  });

  it('Escape clears both hover and pin', () => {
    renderResult();
    fireEvent.click(word('dog'));
    fireEvent.keyDown(word('dog'), { key: 'Escape' });
    expect(word('dog').getAttribute('aria-pressed')).toBe('false');
    expect(word('dog').getAttribute('aria-label')).toBe('dog');
  });
});
