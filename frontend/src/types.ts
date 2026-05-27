/**
 * Mirrors src/api/schemas.py — if the Python model changes, update this too.
 */

export interface CaptionResponse {
  readonly caption: string;
  readonly beam_size: number;
  readonly inference_ms: number;
  /** BPE token strings, 1-to-1 with attention_weights rows. */
  readonly tokens: readonly string[];
  /**
   * Shape [T, 49]: cross-attention weights averaged over decoder layers.
   * attention_weights[i] is the 7×7 spatial weight map for tokens[i].
   */
  readonly attention_weights: ReadonlyArray<readonly number[]>;
}

/** FastAPI sends validation errors as { detail: string } or { detail: ValidationError[] }. */
export interface ValidationErrorItem {
  readonly msg: string;
  readonly loc: readonly string[];
  readonly type: string;
}

export type ApiErrorDetail = string | readonly ValidationErrorItem[];

export interface ApiErrorResponse {
  readonly detail: ApiErrorDetail;
}

export function extractErrorMessage(err: ApiErrorResponse): string {
  if (typeof err.detail === 'string') return err.detail;
  return err.detail.map((e) => e.msg).join('; ');
}

export function isApiErrorResponse(value: unknown): value is ApiErrorResponse {
  return (
    typeof value === 'object' &&
    value !== null &&
    'detail' in value
  );
}

export function isCaptionResponse(value: unknown): value is CaptionResponse {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v['caption'] === 'string' &&
    typeof v['beam_size'] === 'number' &&
    Array.isArray(v['tokens']) &&
    Array.isArray(v['attention_weights'])
  );
}
