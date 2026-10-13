import { describe, expect, it } from 'vitest';
import { extractErrorMessage, isApiErrorResponse, isCaptionResponse } from './types';

const valid = {
  caption: 'a dog',
  beam_size: 4,
  inference_ms: 12.5,
  tokens: ['a', 'dog'],
  attention_weights: [new Array<number>(49).fill(0.02), new Array<number>(49).fill(0.02)],
};

describe('isCaptionResponse', () => {
  it('accepts a well-formed response', () => {
    expect(isCaptionResponse(valid)).toBe(true);
  });

  it.each([
    ['null', null],
    ['a string', 'caption'],
    ['missing tokens', { ...valid, tokens: undefined }],
    ['non-array attention', { ...valid, attention_weights: 'x' }],
    ['non-numeric beam', { ...valid, beam_size: '4' }],
  ])('rejects %s', (_label, value) => {
    expect(isCaptionResponse(value)).toBe(false);
  });
});

describe('API errors', () => {
  it('recognises FastAPI error bodies', () => {
    expect(isApiErrorResponse({ detail: 'nope' })).toBe(true);
    expect(isApiErrorResponse({ message: 'nope' })).toBe(false);
    expect(isApiErrorResponse(null)).toBe(false);
  });

  it('extracts a plain string detail', () => {
    expect(extractErrorMessage({ detail: 'Unsupported file type' })).toBe('Unsupported file type');
  });

  it('joins validation-error messages', () => {
    const detail = [
      { msg: 'beam_size must be <= 16', loc: ['body', 'beam_size'], type: 'less_than_equal' },
      { msg: 'field required', loc: ['body', 'file'], type: 'missing' },
    ];
    expect(extractErrorMessage({ detail })).toBe('beam_size must be <= 16; field required');
  });
});
