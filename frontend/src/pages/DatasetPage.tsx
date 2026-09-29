import { useState, useEffect, useCallback } from 'react';
import {
  Database,
  Trash2,
  Eye,
  Loader2,
  RefreshCw,
  Shuffle,
  CheckCircle,
  AlertCircle,
} from 'lucide-react';
import { api } from '../api';
import type { TrainingExample, DatasetResponse } from '../types';

export default function DatasetPage() {
  const [data, setData] = useState<DatasetResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [splitLoading, setSplitLoading] = useState(false);
  const [selectedExample, setSelectedExample] = useState<TrainingExample | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [batchFilter, setBatchFilter] = useState<string>('all');

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const splitParam = filter !== 'all' ? filter : undefined;
      const statusParam = statusFilter !== 'all' ? statusFilter : undefined;
      const batchParam = batchFilter !== 'all' ? batchFilter : undefined;
      const resp = await api.dataset.list(0, 100, splitParam, statusParam, batchParam);
      setData(resp);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [filter, statusFilter, batchFilter]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this training example?')) return;
    try {
      await api.dataset.delete(id);
      loadData();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleSplit = async () => {
    setSplitLoading(true);
    try {
      await api.dataset.split();
      loadData();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setSplitLoading(false);
    }
  };

  const stats = data?.stats;

  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text-primary)]">Dataset</h1>
          <p className="text-sm text-[var(--color-text-secondary)] mt-1">
            Manage your training examples and dataset splits.
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={handleSplit}
            disabled={splitLoading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--color-surface-card)] border border-[var(--color-border-subtle)] text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-hover)] transition-colors disabled:opacity-50"
          >
            {splitLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Shuffle className="w-4 h-4" />}
            Assign Splits
          </button>
          <button
            onClick={loadData}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--color-surface-card)] border border-[var(--color-border-subtle)] text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-hover)] transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Stats cards */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 mb-8">
          {[
            { label: 'Total', value: stats.total, color: 'text-[var(--color-text-primary)]' },
            { label: 'Labeled', value: stats.labeled, color: 'text-[var(--color-success)]' },
            { label: 'Pending', value: stats.pending, color: 'text-[var(--color-warning)]' },
            { label: 'Train', value: stats.train, color: 'text-blue-400' },
            { label: 'Validation', value: stats.validation, color: 'text-purple-400' },
            { label: 'Test', value: stats.test, color: 'text-cyan-400' },
            { label: 'Unassigned', value: stats.unassigned, color: 'text-[var(--color-text-muted)]' },
          ].map(({ label, value, color }) => (
            <div
              key={label}
              className="p-4 rounded-xl bg-[var(--color-surface-card)] border border-[var(--color-border-subtle)]"
            >
              <p className="text-[10px] uppercase tracking-wider text-[var(--color-text-muted)] font-semibold mb-1">
                {label}
              </p>
              <p className={`text-2xl font-bold font-mono ${color}`}>{value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Filter tabs */}
      <div className="flex gap-1 mb-6 p-1 rounded-lg bg-[var(--color-surface-secondary)] w-fit">
        {['all', 'train', 'validation', 'test', 'unassigned'].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
              filter === f
                ? 'bg-[var(--color-accent-primary)] text-white'
                : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]'
            }`}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {/* Status + Batch filters */}
      <div className="flex gap-4 mb-6 flex-wrap">
        {/* Status filter */}
        <div className="flex gap-1 p-1 rounded-lg bg-[var(--color-surface-secondary)] w-fit">
          {['all', 'labeled', 'pending'].map((f) => (
            <button
              key={f}
              onClick={() => setStatusFilter(f)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                statusFilter === f
                  ? 'bg-[var(--color-accent-primary)] text-white'
                  : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]'
              }`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>

        {/* Batch filter */}
        {data?.batch_ids && data.batch_ids.length > 0 && (
          <select
            value={batchFilter}
            onChange={(e) => setBatchFilter(e.target.value)}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-[var(--color-surface-secondary)] border border-[var(--color-border-subtle)] text-[var(--color-text-secondary)] cursor-pointer"
          >
            <option value="all">All Batches</option>
            {data.batch_ids.map((b) => (
              <option key={b} value={b}>{b}</option>
            ))}
          </select>
        )}
      </div>

      {error && (
        <div className="mb-6 p-4 rounded-xl border border-[var(--color-danger)]/30 bg-[var(--color-danger)]/5 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-[var(--color-danger)]" />
          <p className="text-sm text-[var(--color-danger)]">{error}</p>
        </div>
      )}

      {/* Examples list */}
      {loading ? (
        <div className="flex items-center justify-center h-48">
          <Loader2 className="w-6 h-6 animate-spin text-[var(--color-text-muted)]" />
        </div>
      ) : data?.examples.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-48 text-center">
          <Database className="w-10 h-10 text-[var(--color-text-muted)] mb-3" />
          <p className="text-sm text-[var(--color-text-secondary)]">No training examples yet.</p>
          <p className="text-xs text-[var(--color-text-muted)] mt-1">
            Go to the Trainer page to add examples.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {data?.examples.map((ex) => (
            <div
              key={ex.id}
              className="flex items-center gap-4 p-4 rounded-xl bg-[var(--color-surface-card)] border border-[var(--color-border-subtle)] hover:border-[var(--color-border-active)]/30 transition-colors group"
            >
              {/* Original thumbnail */}
              <div className="w-16 h-16 rounded-lg overflow-hidden bg-black/30 shrink-0">
                <img
                  src={ex.original_path.replace(/.*\/uploads\//, '/uploads/')}
                  alt="Original"
                  className="w-full h-full object-cover"
                />
              </div>

              {/* Target thumbnail */}
              <div className="w-16 h-16 rounded-lg overflow-hidden bg-black/30 shrink-0">
                {ex.target_path ? (
                  <img
                    src={ex.target_path.replace(/.*\/uploads\//, '/uploads/')}
                    alt="Target"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-[var(--color-text-muted)]">
                    <span className="text-[10px]">No target</span>
                  </div>
                )}
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-[var(--color-text-primary)] truncate">
                  {ex.original_filename}
                </p>
                <div className="flex gap-4 mt-1 text-xs text-[var(--color-text-muted)]">
                  <span>{ex.original_width}×{ex.original_height}</span>
                  {ex.crop_width && (
                    <span className="font-mono">
                      Crop: {ex.crop_width}×{ex.crop_height}
                    </span>
                  )}
                  {ex.confidence !== null && (
                    <span>
                      Conf: {(ex.confidence * 100).toFixed(0)}%
                    </span>
                  )}
                  {ex.batch_id && (
                    <span className="text-[var(--color-accent-secondary)]">
                      {ex.batch_id}
                    </span>
                  )}
                </div>
              </div>

              {/* Status badge */}
              <div className="shrink-0">
                {ex.status === 'labeled' ? (
                  <span className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-semibold bg-[var(--color-success)]/10 text-[var(--color-success)]">
                    <CheckCircle className="w-3 h-3" />
                    Labeled
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-semibold bg-[var(--color-warning)]/10 text-[var(--color-warning)]">
                    <AlertCircle className="w-3 h-3" />
                    Pending
                  </span>
                )}
              </div>

              {/* Split badge */}
              <div className="shrink-0">
                <span className={`inline-block px-2 py-1 rounded-md text-[10px] font-semibold ${
                  ex.split === 'train' ? 'bg-blue-500/10 text-blue-400' :
                  ex.split === 'validation' ? 'bg-purple-500/10 text-purple-400' :
                  ex.split === 'test' ? 'bg-cyan-500/10 text-cyan-400' :
                  'bg-white/5 text-[var(--color-text-muted)]'
                }`}>
                  {ex.split}
                </span>
              </div>

              {/* Actions */}
              <div className="shrink-0 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={() => setSelectedExample(ex)}
                  className="p-2 rounded-lg hover:bg-[var(--color-surface-hover)] text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors"
                >
                  <Eye className="w-4 h-4" />
                </button>
                <button
                  onClick={() => handleDelete(ex.id)}
                  className="p-2 rounded-lg hover:bg-[var(--color-danger)]/10 text-[var(--color-text-muted)] hover:text-[var(--color-danger)] transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Preview modal */}
      {selectedExample && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
          onClick={() => setSelectedExample(null)}
        >
          <div
            className="bg-[var(--color-surface-secondary)] rounded-2xl border border-[var(--color-border-subtle)] p-6 max-w-3xl w-full mx-4 max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-semibold mb-4">{selectedExample.original_filename}</h3>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="text-xs text-[var(--color-text-muted)] mb-2 uppercase font-semibold">Original</p>
                <img
                  src={selectedExample.original_path.replace(/.*\/uploads\//, '/uploads/')}
                  alt="Original"
                  className="rounded-lg w-full object-contain max-h-64 bg-black/30"
                />
              </div>
              {selectedExample.target_path && (
                <div>
                  <p className="text-xs text-[var(--color-text-muted)] mb-2 uppercase font-semibold">Target Crop</p>
                  <img
                    src={selectedExample.target_path.replace(/.*\/uploads\//, '/uploads/')}
                    alt="Target"
                    className="rounded-lg w-full object-contain max-h-64 bg-black/30"
                  />
                </div>
              )}
            </div>
            {selectedExample.norm_x !== null && (
              <div className="grid grid-cols-4 gap-3">
                {['x', 'y', 'width', 'height'].map((key) => (
                  <div key={key} className="p-3 rounded-lg bg-[var(--color-surface-tertiary)]">
                    <p className="text-[10px] uppercase text-[var(--color-text-muted)] font-semibold">{key}</p>
                    <p className="text-sm font-mono font-bold text-[var(--color-accent-secondary)]">
                      {(selectedExample[`norm_${key}` as keyof TrainingExample] as number)?.toFixed(4)}
                    </p>
                  </div>
                ))}
              </div>
            )}
            <button
              onClick={() => setSelectedExample(null)}
              className="mt-4 px-4 py-2 rounded-lg border border-[var(--color-border-subtle)] text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-hover)]"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
