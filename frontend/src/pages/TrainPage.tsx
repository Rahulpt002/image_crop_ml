import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play,
  Square,
  Loader2,
  Cpu,
  Settings,
  TrendingDown,
  BarChart3,
  Clock,
  CheckCircle,
  XCircle,
} from 'lucide-react';
import { api } from '../api';
import type { TrainingConfig, TrainingStatus, DatasetStats, TrainingRunHistory } from '../types';

export default function TrainPage() {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [history, setHistory] = useState<TrainingRunHistory[]>([]);
  const [config, setConfig] = useState<TrainingConfig>({
    backbone: 'mobilenet_v3_small',
    image_size: 224,
    batch_size: 8,
    learning_rate: 0.0001,
    epochs: 30,
    optimizer: 'adamw',
    loss_function: 'smooth_l1',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollingRef = useRef<number | null>(null);

  // Load stats & status
  const loadAll = useCallback(async () => {
    try {
      const [s, st, h] = await Promise.all([
        api.dataset.stats(),
        api.training.status(),
        api.training.history(),
      ]);
      setStats(s);
      setStatus(st);
      setHistory(h);
    } catch (e: any) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  // Poll during training
  useEffect(() => {
    if (status?.status === 'training') {
      pollingRef.current = window.setInterval(async () => {
        try {
          const st = await api.training.status();
          setStatus(st);
          if (st.status !== 'training') {
            if (pollingRef.current) clearInterval(pollingRef.current);
            loadAll(); // Refresh everything when done
          }
        } catch {
          // Ignore polling errors
        }
      }, 2000);
      return () => {
        if (pollingRef.current) clearInterval(pollingRef.current);
      };
    }
  }, [status?.status, loadAll]);

  const handleStart = async () => {
    setLoading(true);
    setError(null);
    try {
      await api.training.start(config);
      const st = await api.training.status();
      setStatus(st);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    try {
      await api.training.stop();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const isTraining = status?.status === 'training';

  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text-primary)]">Train Model</h1>
          <p className="text-sm text-[var(--color-text-secondary)] mt-1">
            Configure and train your crop prediction model.
          </p>
        </div>
        <div className="flex gap-3">
          {isTraining ? (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[var(--color-danger)] text-white text-sm font-semibold hover:opacity-90 transition-opacity"
            >
              <Square className="w-4 h-4" />
              Stop Training
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={loading || !stats || stats.train === 0}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-[var(--color-accent-primary)] to-purple-500 text-white text-sm font-semibold hover:opacity-90 transition-opacity disabled:opacity-50"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              Train Model
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Configuration */}
        <div className="lg:col-span-1 space-y-6">
          {/* Dataset info */}
          <div className="p-5 rounded-xl bg-[var(--color-surface-card)] border border-[var(--color-border-subtle)]">
            <h3 className="text-xs uppercase tracking-wider text-[var(--color-text-muted)] font-semibold mb-3 flex items-center gap-2">
              <BarChart3 className="w-3.5 h-3.5" />
              Dataset
            </h3>
            {stats ? (
              <div className="space-y-2 text-sm">
                {[
                  { label: 'Total', value: stats.total },
                  { label: 'Train', value: stats.train, color: 'text-blue-400' },
                  { label: 'Validation', value: stats.validation, color: 'text-purple-400' },
                  { label: 'Test', value: stats.test, color: 'text-cyan-400' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="flex justify-between">
                    <span className="text-[var(--color-text-muted)]">{label}</span>
                    <span className={`font-mono font-bold ${color || 'text-[var(--color-text-primary)]'}`}>
                      {value}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <Loader2 className="w-4 h-4 animate-spin text-[var(--color-text-muted)]" />
            )}
          </div>

          {/* Config */}
          <div className="p-5 rounded-xl bg-[var(--color-surface-card)] border border-[var(--color-border-subtle)]">
            <h3 className="text-xs uppercase tracking-wider text-[var(--color-text-muted)] font-semibold mb-4 flex items-center gap-2">
              <Settings className="w-3.5 h-3.5" />
              Configuration
            </h3>
            <div className="space-y-4">
              <div>
                <label className="block text-xs text-[var(--color-text-muted)] mb-1">Backbone</label>
                <select
                  value={config.backbone}
                  onChange={(e) => setConfig({ ...config, backbone: e.target.value })}
                  disabled={isTraining}
                  className="w-full px-3 py-2 rounded-lg bg-[var(--color-surface-tertiary)] border border-[var(--color-border-subtle)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent-primary)]"
                >
                  <option value="mobilenet_v3_small">MobileNetV3 Small</option>
                  <option value="mobilenet_v3_large">MobileNetV3 Large</option>
                  <option value="efficientnet_b0">EfficientNet B0</option>
                  <option value="resnet18">ResNet18</option>
                  <option value="resnet34">ResNet34</option>
                </select>
              </div>

              {[
                { key: 'image_size', label: 'Image Size', type: 'number' },
                { key: 'batch_size', label: 'Batch Size', type: 'number' },
                { key: 'learning_rate', label: 'Learning Rate', type: 'number', step: '0.0001' },
                { key: 'epochs', label: 'Epochs', type: 'number' },
              ].map(({ key, label, type, step }) => (
                <div key={key}>
                  <label className="block text-xs text-[var(--color-text-muted)] mb-1">{label}</label>
                  <input
                    type={type}
                    value={(config as any)[key]}
                    step={step}
                    onChange={(e) =>
                      setConfig({
                        ...config,
                        [key]: type === 'number' ? Number(e.target.value) : e.target.value,
                      })
                    }
                    disabled={isTraining}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-surface-tertiary)] border border-[var(--color-border-subtle)] text-sm text-[var(--color-text-primary)] font-mono focus:outline-none focus:border-[var(--color-accent-primary)]"
                  />
                </div>
              ))}

              <div>
                <label className="block text-xs text-[var(--color-text-muted)] mb-1">Optimizer</label>
                <select
                  value={config.optimizer}
                  onChange={(e) => setConfig({ ...config, optimizer: e.target.value })}
                  disabled={isTraining}
                  className="w-full px-3 py-2 rounded-lg bg-[var(--color-surface-tertiary)] border border-[var(--color-border-subtle)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent-primary)]"
                >
                  <option value="adamw">AdamW</option>
                  <option value="adam">Adam</option>
                </select>
              </div>

              <div>
                <label className="block text-xs text-[var(--color-text-muted)] mb-1">Loss</label>
                <select
                  value={config.loss_function}
                  onChange={(e) => setConfig({ ...config, loss_function: e.target.value })}
                  disabled={isTraining}
                  className="w-full px-3 py-2 rounded-lg bg-[var(--color-surface-tertiary)] border border-[var(--color-border-subtle)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent-primary)]"
                >
                  <option value="smooth_l1">Smooth L1</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Training progress + history */}
        <div className="lg:col-span-2 space-y-6">
          {/* Training progress */}
          {status && status.status !== 'idle' && (
            <div className="p-6 rounded-xl bg-[var(--color-surface-card)] border border-[var(--color-border-subtle)]">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xs uppercase tracking-wider text-[var(--color-text-muted)] font-semibold flex items-center gap-2">
                  <TrendingDown className="w-3.5 h-3.5" />
                  Training Progress
                </h3>
                <span
                  className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-semibold ${
                    status.status === 'training'
                      ? 'bg-blue-500/10 text-blue-400'
                      : status.status === 'completed'
                      ? 'bg-[var(--color-success)]/10 text-[var(--color-success)]'
                      : status.status === 'failed'
                      ? 'bg-[var(--color-danger)]/10 text-[var(--color-danger)]'
                      : 'bg-white/5 text-[var(--color-text-muted)]'
                  }`}
                >
                  {status.status === 'training' && <Loader2 className="w-3 h-3 animate-spin" />}
                  {status.status.toUpperCase()}
                </span>
              </div>

              {/* Progress bar */}
              <div className="mb-6">
                <div className="flex justify-between text-xs text-[var(--color-text-muted)] mb-1.5">
                  <span>
                    Epoch {status.epoch} / {status.total_epochs}
                  </span>
                  <span>{status.progress.toFixed(1)}%</span>
                </div>
                <div className="h-2 rounded-full bg-[var(--color-surface-tertiary)] overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-[var(--color-accent-primary)] to-purple-500 transition-all duration-500"
                    style={{ width: `${status.progress}%` }}
                  />
                </div>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: 'Train Loss', value: status.train_loss?.toFixed(5), icon: TrendingDown },
                  { label: 'Val Loss', value: status.val_loss?.toFixed(5), icon: TrendingDown },
                  { label: 'Best Val Loss', value: status.best_val_loss?.toFixed(5), icon: CheckCircle },
                  { label: 'Device', value: status.device?.toUpperCase(), icon: Cpu },
                ].map(({ label, value, icon: Icon }) => (
                  <div key={label} className="p-3 rounded-lg bg-[var(--color-surface-tertiary)]">
                    <div className="flex items-center gap-1.5 mb-1">
                      <Icon className="w-3 h-3 text-[var(--color-text-muted)]" />
                      <p className="text-[10px] uppercase tracking-wider text-[var(--color-text-muted)] font-semibold">
                        {label}
                      </p>
                    </div>
                    <p className="text-lg font-bold font-mono text-[var(--color-text-primary)]">
                      {value ?? '—'}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="p-4 rounded-xl border border-[var(--color-danger)]/30 bg-[var(--color-danger)]/5 flex items-center gap-3">
              <XCircle className="w-5 h-5 text-[var(--color-danger)]" />
              <p className="text-sm text-[var(--color-danger)]">{error}</p>
            </div>
          )}

          {/* Training history */}
          <div className="p-5 rounded-xl bg-[var(--color-surface-card)] border border-[var(--color-border-subtle)]">
            <h3 className="text-xs uppercase tracking-wider text-[var(--color-text-muted)] font-semibold mb-4 flex items-center gap-2">
              <Clock className="w-3.5 h-3.5" />
              Training History
            </h3>
            {history.length === 0 ? (
              <p className="text-sm text-[var(--color-text-muted)]">No training runs yet.</p>
            ) : (
              <div className="space-y-2">
                {history.map((run) => (
                  <div
                    key={run.id}
                    className="flex items-center gap-4 p-3 rounded-lg bg-[var(--color-surface-tertiary)] text-sm"
                  >
                    <span className="font-mono font-bold text-[var(--color-accent-secondary)] w-12">
                      {run.model_version}
                    </span>
                    <span className="text-[var(--color-text-muted)] text-xs w-20">
                      {run.backbone.replace('mobilenet_v3_', 'MNv3-').replace('efficientnet_', 'EN-')}
                    </span>
                    <span className="text-[var(--color-text-muted)] text-xs">
                      {run.current_epoch}/{run.epochs} epochs
                    </span>
                    <span className="text-[var(--color-text-muted)] text-xs font-mono">
                      val: {run.best_val_loss?.toFixed(4) ?? '—'}
                    </span>
                    {run.mean_iou !== null && (
                      <span className="text-[var(--color-success)] text-xs font-mono">
                        IoU: {run.mean_iou.toFixed(3)}
                      </span>
                    )}
                    <span
                      className={`ml-auto px-2 py-0.5 rounded text-[10px] font-semibold ${
                        run.status === 'completed'
                          ? 'bg-[var(--color-success)]/10 text-[var(--color-success)]'
                          : run.status === 'failed'
                          ? 'bg-[var(--color-danger)]/10 text-[var(--color-danger)]'
                          : 'bg-white/5 text-[var(--color-text-muted)]'
                      }`}
                    >
                      {run.status}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
