import {
  type CaptionResponse,
  type ApiErrorResponse,
  extractErrorMessage,
  isApiErrorResponse,
  isCaptionResponse,
} from '../types';

/**
 * POST /caption with XHR so we can track upload progress.
 *
 * @param file       The image file to caption.
 * @param beamSize   Number of beams for the decoder (1–16).
 * @param onProgress Called with 0..1 as bytes upload. Reaches 1 when the
 *                   upload finishes; the server may still be processing.
 * @param signal     AbortSignal — cancels the request mid-flight.
 */
export function captionImage(
  file: File,
  beamSize: number,
  onProgress: (progress: number) => void,
  signal: AbortSignal,
): Promise<CaptionResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    // Upload progress: 0 → 1 as bytes leave the browser.
    xhr.upload.onprogress = (e: ProgressEvent): void => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };

    xhr.onload = (): void => {
      const body: unknown = parseJson(xhr.responseText);

      if (xhr.status >= 200 && xhr.status < 300) {
        if (isCaptionResponse(body)) {
          resolve(body);
        } else {
          reject(new Error('Server returned an unexpected response shape.'));
        }
        return;
      }

      if (isApiErrorResponse(body)) {
        reject(new Error(extractErrorMessage(body as ApiErrorResponse)));
      } else {
        reject(new Error(`Request failed with HTTP ${xhr.status}.`));
      }
    };

    xhr.onerror = (): void => reject(new Error('Network error — is the server running?'));
    xhr.ontimeout = (): void => reject(new Error('Request timed out.'));

    // Wire up AbortSignal so callers can cancel (e.g. component unmount).
    const onAbort = (): void => xhr.abort();
    signal.addEventListener('abort', onAbort, { once: true });
    xhr.onabort = (): void => {
      signal.removeEventListener('abort', onAbort);
      reject(new DOMException('Request was aborted.', 'AbortError'));
    };

    const form = new FormData();
    form.append('file', file);
    form.append('beam_size', String(beamSize));

    xhr.open('POST', '/caption');
    xhr.timeout = 120_000; // 2-minute hard cap — model can be slow on CPU
    xhr.send(form);
  });
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}
