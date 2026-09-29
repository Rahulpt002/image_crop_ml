/** API client for the backend. */
import type {
  DatasetResponse,
  DatasetStats,
  TrainingExample,
  TrainingConfig,
  TrainingStatus,
  TrainingRunHistory,
  ModelStatus,
  PredictionResponse,
  HealthCheck,
} from './types';

const BASE = '';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, options);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

// ─── Health ──────────────────────────────────────────────────────────

export const api = {
  health: () => request<HealthCheck>('/api/health'),

  // ─── Dataset ───────────────────────────────────────────────────────

  dataset: {
    list: (skip = 0, limit = 50, split?: string, status?: string, batchId?: string) => {
      const params = new URLSearchParams({ skip: String(skip), limit: String(limit) });
      if (split) params.set('split', split);
      if (status) params.set('status', status);
      if (batchId) params.set('batch_id', batchId);
      return request<DatasetResponse>(`/api/dataset?${params}`);
    },

    stats: () => request<DatasetStats>('/api/dataset/stats'),

    get: (id: string) => request<TrainingExample>(`/api/dataset/${id}`),

    upload: (original: File, target?: File, batchId?: string, skipDuplicates = true) => {
      const form = new FormData();
      form.append('original', original);
      if (target) form.append('target', target);
      if (batchId) form.append('batch_id', batchId);
      form.append('skip_duplicates', String(skipDuplicates));
      return request<TrainingExample>('/api/dataset/upload', {
        method: 'POST',
        body: form,
      });
    },

    checkHashes: (hashes: string[]) =>
      request<{ duplicates: string[]; count: number }>('/api/dataset/check-hashes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(hashes),
      }),

    update: (id: string, data: Partial<TrainingExample>) =>
      request<TrainingExample>(`/api/dataset/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      }),

    setLabel: (id: string, crop: { crop_x: number; crop_y: number; crop_width: number; crop_height: number }) => {
      const form = new FormData();
      form.append('crop_x', String(crop.crop_x));
      form.append('crop_y', String(crop.crop_y));
      form.append('crop_width', String(crop.crop_width));
      form.append('crop_height', String(crop.crop_height));
      return request<TrainingExample>(`/api/dataset/${id}/label`, {
        method: 'POST',
        body: form,
      });
    },

    delete: (id: string) =>
      request<{ status: string }>(`/api/dataset/${id}`, { method: 'DELETE' }),

    split: (config?: { train_ratio: number; val_ratio: number; test_ratio: number; seed: number }) =>
      request<DatasetStats>('/api/dataset/split', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config ?? {}),
      }),
  },

  // ─── Training ──────────────────────────────────────────────────────

  training: {
    start: (config?: Partial<TrainingConfig>) =>
      request<{ status: string; model_version: string; config: TrainingConfig }>('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config ?? {}),
      }),

    stop: () =>
      request<{ status: string }>('/api/training/stop', { method: 'POST' }),

    status: () => request<TrainingStatus>('/api/training/status'),

    history: (skip = 0, limit = 20) =>
      request<TrainingRunHistory[]>(`/api/training/history?skip=${skip}&limit=${limit}`),
  },

  // ─── Model ────────────────────────────────────────────────────────

  model: {
    status: () => request<ModelStatus>('/api/model/status'),

    activate: (version: string) =>
      request<{ status: string }>(`/api/model/activate/${version}`, { method: 'POST' }),

    predict: (image: File) => {
      const form = new FormData();
      form.append('image', image);
      return request<PredictionResponse>('/api/model/predict', {
        method: 'POST',
        body: form,
      });
    },

    correct: (data: {
      prediction_id: string;
      corrected_x: number;
      corrected_y: number;
      corrected_width: number;
      corrected_height: number;
      add_to_dataset: boolean;
    }) =>
      request<{ status: string }>('/api/model/correct', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      }),

    export: (
      predictionId: string,
      cropCoords?: { x: number; y: number; width: number; height: number }
    ) => {
      const query = cropCoords
        ? `?crop_x=${cropCoords.x}&crop_y=${cropCoords.y}&crop_width=${cropCoords.width}&crop_height=${cropCoords.height}`
        : '';
      return request<{ status: string; path: string; filename: string; width: number; height: number; dpi: number }>(
        `/api/model/export/${predictionId}${query}`,
        { method: 'POST' }
      );
    },
  },
};
