import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from 'react';
import {
  clearCanvas,
  describeAttentionRegion,
  renderAttentionHeatmap,
} from '../utils/attention';
import type { CaptionResponse } from '../types';

interface Props {
  imageUrl: string;
  result: CaptionResponse;
}

export default function CaptionResult({ imageUrl, result }: Props): JSX.Element {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Two ways to select a word:
  //   • hover / keyboard focus  → temporary preview (mouse and keyboard users)
  //   • click / tap / Enter     → "pinned", stays until toggled off (touch users,
  //                               who have no hover)
  // A hovered word wins over the pinned one while the pointer is on it.
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [pinnedIdx, setPinnedIdx] = useState<number | null>(null);
  const activeIdx = hoverIdx ?? pinnedIdx;

  // Rendered dimensions of the <img> element (CSS pixels, not natural size).
  const [imgDims, setImgDims] = useState({ w: 0, h: 0 });

  // A new result starts with nothing selected.
  useEffect(() => {
    setHoverIdx(null);
    setPinnedIdx(null);
  }, [result]);

  // Track image dimensions via ResizeObserver so the canvas stays in sync
  // even when the viewport is resized.
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setImgDims({
          w: Math.round(entry.contentRect.width),
          h: Math.round(entry.contentRect.height),
        });
      }
    });

    observer.observe(img);
    // If the image is already rendered (e.g. from cache), grab dimensions now.
    if (img.complete && img.naturalWidth > 0) {
      setImgDims({ w: img.offsetWidth, h: img.offsetHeight });
    }

    return () => observer.disconnect();
  }, [imageUrl]);

  // Redraw whenever the active token or image dimensions change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || imgDims.w === 0 || imgDims.h === 0) return;

    // Resize the canvas to match the image (this also clears it).
    canvas.width = imgDims.w;
    canvas.height = imgDims.h;

    if (activeIdx === null) return;

    const weights = result.attention_weights[activeIdx];
    if (!weights) return;

    renderAttentionHeatmap(canvas, weights);
  }, [activeIdx, imgDims, result.attention_weights]);

  // Clear the canvas when nothing is selected any more.
  useEffect(() => {
    if (activeIdx === null && canvasRef.current) {
      clearCanvas(canvasRef.current);
    }
  }, [activeIdx]);

  const handleClick = useCallback((idx: number): void => {
    setPinnedIdx((prev) => {
      if (prev === idx) {
        // unpinning: also drop the hover so a tap on touch screens really clears it
        setHoverIdx(null);
        return null;
      }
      return idx;
    });
  }, []);

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLButtonElement>): void => {
    // Enter/Space already fire click (→ pin) natively; Escape dismisses everything.
    if (e.key === 'Escape') {
      setHoverIdx(null);
      setPinnedIdx(null);
    }
  }, []);

  // Compute aria description for the currently active token.
  const activeWeights =
    activeIdx !== null ? result.attention_weights[activeIdx] : null;
  const regionDescription =
    activeWeights !== null && activeWeights !== undefined
      ? describeAttentionRegion(activeWeights)
      : null;

  const canvasAriaLabel =
    activeIdx !== null && regionDescription !== null
      ? `Attention heatmap for "${result.tokens[activeIdx]}": model focused on the ${regionDescription} region`
      : 'No word selected — hover, focus or tap a word below to see the attention heatmap';

  return (
    <section className="result" aria-label="Caption result">

      {/* Heatmap canvas is absolutely placed over the image.
          It announces itself to screen readers via aria-label. */}
      <div className="result__image-wrap">
        <img
          ref={imgRef}
          src={imageUrl}
          alt="The uploaded image that was captioned"
          className="result__image"
          onLoad={() => {
            const img = imgRef.current;
            if (img) setImgDims({ w: img.offsetWidth, h: img.offsetHeight });
          }}
        />
        <canvas
          ref={canvasRef}
          className="result__canvas"
          role="img"
          aria-label={canvasAriaLabel}
          aria-live="polite"
          aria-atomic="true"
        />
      </div>

      {/* Screen reader live region: announces the active word on hover/focus */}
      <div aria-live="polite" aria-atomic="true" className="sr-only">
        {activeIdx !== null && regionDescription !== null
          ? `Showing attention for "${result.tokens[activeIdx]}" — focused on the ${regionDescription} region`
          : ''}
      </div>

      {/* Full caption as readable text */}
      <p className="result__caption" aria-label={`Caption: ${result.caption}`}>
        {result.caption}
      </p>

      {/* Word buttons — preview on hover/focus, pin on click/tap/Enter */}
      <div>
        <p className="result__tokens-label" id="tokens-hint">
          Hover or focus a word to preview which image regions drove it; click or
          tap to keep it selected.
        </p>
        <div
          className="result__tokens"
          role="group"
          aria-labelledby="tokens-hint"
        >
          {result.tokens.map((token, idx) => {
            const isActive = activeIdx === idx;
            const isPinned = pinnedIdx === idx;
            const localRegion =
              isActive && regionDescription !== null ? regionDescription : null;

            return (
              <button
                key={idx}
                type="button"
                className={['token-btn', isActive ? 'token-btn--active' : ''].join(' ')}
                aria-pressed={isPinned}
                aria-label={
                  localRegion !== null
                    ? `${token} — attended to ${localRegion} region`
                    : token
                }
                onMouseEnter={() => setHoverIdx(idx)}
                onMouseLeave={() => setHoverIdx(null)}
                onFocus={() => setHoverIdx(idx)}
                onBlur={() => setHoverIdx(null)}
                onClick={() => handleClick(idx)}
                onKeyDown={handleKeyDown}
              >
                {token}
              </button>
            );
          })}
        </div>
      </div>

      {/* Inference stats */}
      <dl className="result__stats" aria-label="Inference statistics">
        <span>
          <dt style={{ display: 'inline' }}>Inference</dt>{' '}
          <dd style={{ display: 'inline' }}>{result.inference_ms.toFixed(0)} ms</dd>
        </span>
        <span>
          <dt style={{ display: 'inline' }}>Beam size</dt>{' '}
          <dd style={{ display: 'inline' }}>{result.beam_size}</dd>
        </span>
        <span>
          <dt style={{ display: 'inline' }}>Words</dt>{' '}
          <dd style={{ display: 'inline' }}>{result.tokens.length}</dd>
        </span>
      </dl>
    </section>
  );
}
