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
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  // Rendered dimensions of the <img> element (CSS pixels, not natural size).
  const [imgDims, setImgDims] = useState({ w: 0, h: 0 });

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
    canvas.width  = imgDims.w;
    canvas.height = imgDims.h;

    if (activeIdx === null) return;

    const weights = result.attention_weights[activeIdx];
    if (!weights) return;

    renderAttentionHeatmap(canvas, weights);
  }, [activeIdx, imgDims, result.attention_weights]);

  // Clear the canvas when activeIdx goes to null (mouse left all tokens).
  useEffect(() => {
    if (activeIdx === null && canvasRef.current) {
      clearCanvas(canvasRef.current);
    }
  }, [activeIdx]);

  const activate = useCallback((idx: number): void => setActiveIdx(idx), []);
  const deactivate = useCallback((): void => setActiveIdx(null), []);

  const handleTokenKeyDown = useCallback(
    (e: KeyboardEvent<HTMLButtonElement>, idx: number): void => {
      // Enter/Space toggle the selection; Escape dismisses.
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        setActiveIdx((prev) => (prev === idx ? null : idx));
      } else if (e.key === 'Escape') {
        setActiveIdx(null);
      }
    },
    [],
  );

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
      : 'No token selected — hover or focus a word below to see the attention heatmap';

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

      {/* Token buttons — hover or focus to see that word's attention map */}
      <div>
        <p className="result__tokens-label" id="tokens-hint">
          Hover or focus a word to highlight which image regions drove it.
        </p>
        <div
          className="result__tokens"
          role="group"
          aria-labelledby="tokens-hint"
        >
          {result.tokens.map((token, idx) => {
            const isActive = activeIdx === idx;
            const localRegion =
              isActive && regionDescription !== null ? regionDescription : null;

            return (
              <button
                key={idx}
                className={[
                  'token-btn',
                  isActive ? 'token-btn--active' : '',
                ].join(' ')}
                aria-pressed={isActive}
                aria-label={
                  localRegion !== null
                    ? `${token} — attended to ${localRegion} region`
                    : token
                }
                onMouseEnter={() => activate(idx)}
                onMouseLeave={deactivate}
                onFocus={() => activate(idx)}
                onBlur={deactivate}
                onKeyDown={(e) => handleTokenKeyDown(e, idx)}
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
          <dt style={{ display: 'inline' }}>Tokens</dt>{' '}
          <dd style={{ display: 'inline' }}>{result.tokens.length}</dd>
        </span>
      </dl>
    </section>
  );
}
