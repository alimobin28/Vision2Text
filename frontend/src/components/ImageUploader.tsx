import {
  useCallback,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type KeyboardEvent,
} from 'react';
import { captionImage } from '../api/client';
import type { CaptionResponse } from '../types';

interface Props {
  onResult: (file: File, result: CaptionResponse) => void;
}

const ACCEPTED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);
const MAX_BYTES = 10 * 1024 * 1024;
// The API accepts 1–16; these are the useful range. Wider beams are slower.
const BEAM_OPTIONS = [1, 2, 3, 4, 5, 8] as const;
const DEFAULT_BEAM = 4;

export default function ImageUploader({ onResult }: Props): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  // 0..1 while bytes are uploading; null once upload is done (server processing)
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [beamSize, setBeamSize] = useState<number>(DEFAULT_BEAM);

  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const prevPreviewRef = useRef<string | null>(null);

  const applyFile = useCallback((f: File): void => {
    if (!ACCEPTED_TYPES.has(f.type)) {
      setError(`Unsupported format "${f.type}". Please use JPEG, PNG, or WEBP.`);
      return;
    }
    if (f.size > MAX_BYTES) {
      setError(`File is ${(f.size / 1_048_576).toFixed(1)} MB — maximum is 10 MB.`);
      return;
    }
    setError(null);
    setFile(f);

    // Revoke the previous preview URL before creating a new one.
    if (prevPreviewRef.current) URL.revokeObjectURL(prevPreviewRef.current);
    const url = URL.createObjectURL(f);
    prevPreviewRef.current = url;
    setPreview(url);
  }, []);

  const handleDrop = (e: DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) applyFile(dropped);
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>): void => {
    const picked = e.target.files?.[0];
    if (picked) applyFile(picked);
    // Reset input value so re-selecting the same file fires onChange again.
    e.target.value = '';
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>): void => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      inputRef.current?.click();
    }
  };

  const handleSubmit = async (): Promise<void> => {
    if (!file || uploading) return;

    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setUploading(true);
    setProgress(0);
    setError(null);

    try {
      const result = await captionImage(
        file,
        beamSize,
        (p) => {
          setProgress(p);
          // Once the bytes are fully uploaded, switch to the indeterminate state
          // (server is now running inference — we can't know how long it'll take).
          if (p >= 1) setProgress(null);
        },
        abortRef.current.signal,
      );
      onResult(file, result);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      setError(err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      setUploading(false);
      setProgress(null);
    }
  };

  const isReady = file !== null && !uploading;

  return (
    <section aria-label="Image upload">
      {/* Hidden file input — triggered by the drop zone or button */}
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        style={{ display: 'none' }}
        onChange={handleChange}
        aria-hidden="true"
        tabIndex={-1}
      />

      {/* Drop zone — keyboard-activatable via Enter/Space */}
      <div
        role="button"
        tabIndex={0}
        aria-label={
          preview
            ? 'Image selected. Press Enter to choose a different image.'
            : 'Select an image to caption. Press Enter to open file browser.'
        }
        className={[
          'drop-zone',
          isDragging ? 'drop-zone--dragging' : '',
        ].join(' ')}
        onClick={() => inputRef.current?.click()}
        onKeyDown={handleKeyDown}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragEnter={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        {preview !== null ? (
          <img
            src={preview}
            alt="Preview of the selected image"
            className="drop-zone__preview"
          />
        ) : (
          <>
            {/* Upload icon */}
            <svg
              className="drop-zone__icon"
              aria-hidden="true"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
              />
            </svg>
            <p className="drop-zone__hint">
              <strong>Click to browse</strong> or drag an image here
            </p>
            <p className="drop-zone__formats">JPEG · PNG · WEBP · max 10 MB</p>
          </>
        )}
      </div>

      {/* Progress bar — shown only while uploading */}
      {uploading && (
        <div
          className="progress-bar"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={progress !== null ? Math.round(progress * 100) : undefined}
          aria-label={
            progress !== null
              ? `Uploading: ${Math.round(progress * 100)}%`
              : 'Processing image on server…'
          }
        >
          <div
            className={[
              'progress-bar__fill',
              progress === null ? 'progress-bar__fill--indeterminate' : '',
            ].join(' ')}
            style={progress !== null ? { width: `${progress * 100}%` } : undefined}
          />
        </div>
      )}

      {/* Error display — role="alert" announces immediately to screen readers */}
      {error !== null && (
        <p className="error-banner" role="alert">
          {error}
        </p>
      )}

      {/* Footer: filename + submit button */}
      <div className="uploader-footer">
        <span className="uploader-footer__filename" aria-live="polite">
          {file !== null ? file.name : 'No file selected'}
        </span>
        <label
          className="beam-select"
          title="Higher values search more candidate captions: usually slightly better, but slower."
        >
          Beam size
          <select
            value={beamSize}
            onChange={(e) => setBeamSize(Number(e.target.value))}
            disabled={uploading}
          >
            {BEAM_OPTIONS.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <button
          className="btn btn--primary"
          disabled={!isReady}
          onClick={() => { void handleSubmit(); }}
          aria-busy={uploading}
        >
          {uploading ? 'Processing…' : 'Generate caption'}
        </button>
      </div>
    </section>
  );
}
