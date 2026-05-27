import { useCallback, useRef, useState } from 'react';
import ImageUploader from './components/ImageUploader';
import CaptionResult from './components/CaptionResult';
import type { CaptionResponse } from './types';

interface ResultState {
  imageUrl: string;
  result: CaptionResponse;
}

export default function App(): JSX.Element {
  const [resultState, setResultState] = useState<ResultState | null>(null);
  const prevUrlRef = useRef<string | null>(null);

  const handleResult = useCallback((file: File, result: CaptionResponse): void => {
    // Revoke the previous object URL to avoid memory leaks.
    if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current);
    const url = URL.createObjectURL(file);
    prevUrlRef.current = url;
    setResultState({ imageUrl: url, result });
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Vision2Text</h1>
        <p>Upload a photo — hover each word to see where the model looked.</p>
      </header>

      <main>
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
