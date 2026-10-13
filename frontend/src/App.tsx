import { useCallback, useEffect, useRef, useState } from 'react';
import { checkBackend, type BackendStatus } from './api/client';
import CaptionResult from './components/CaptionResult';
import ImageUploader from './components/ImageUploader';
import type { CaptionResponse } from './types';

interface ResultState {
  imageUrl: string;
  result: CaptionResponse;
}

const STATUS_MESSAGE: Record<Exclude<BackendStatus, 'checking' | 'ready'>, string> = {
  loading: 'The model is still loading — captions will work in a moment.',
  down: 'Cannot reach the caption service. Is the API running?',
};

export default function App(): JSX.Element {
  const [resultState, setResultState] = useState<ResultState | null>(null);
  const [status, setStatus] = useState<BackendStatus>('checking');
  const prevUrlRef = useRef<string | null>(null);

  const refreshStatus = useCallback((signal?: AbortSignal): void => {
    setStatus('checking');
    checkBackend(signal).then(setStatus, () => undefined); // AbortError: ignore
  }, []);

  // Check the backend once on mount so a stopped server is explained up front
  // instead of surfacing as a generic network error after an upload.
  useEffect(() => {
    const controller = new AbortController();
    refreshStatus(controller.signal);
    return () => controller.abort();
  }, [refreshStatus]);

  const handleResult = useCallback((file: File, result: CaptionResponse): void => {
    // Revoke the previous object URL to avoid memory leaks.
    if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current);
    const url = URL.createObjectURL(file);
    prevUrlRef.current = url;
    setResultState({ imageUrl: url, result });
    setStatus('ready'); // a successful caption proves the service is up
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Vision2Text</h1>
        <p>Upload a photo — hover each word to see where the model looked.</p>
      </header>

      <main>
        {(status === 'down' || status === 'loading') && (
          <p className="status-banner" role="status">
            {STATUS_MESSAGE[status]}{' '}
            <button type="button" className="status-banner__retry" onClick={() => refreshStatus()}>
              Retry
            </button>
          </p>
        )}

        <div className="card">
          <ImageUploader onResult={handleResult} />
        </div>

        {resultState !== null && (
          <div className="card" style={{ marginTop: '1.5rem' }}>
            <CaptionResult
              imageUrl={resultState.imageUrl}
              result={resultState.result}
            />
          </div>
        )}
      </main>
    </div>
  );
}
