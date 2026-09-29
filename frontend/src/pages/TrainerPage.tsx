import { useState, useCallback } from 'react';
import { Upload, CheckCircle, AlertTriangle, Loader2 } from 'lucide-react';
import ImageUploader from '../components/ImageUploader';
import CropEditor from '../components/CropEditor';
import { api } from '../api';
import type { TrainingExample } from '../types';

export default function TrainerPage() {
  const [originalFile, setOriginalFile] = useState<File | null>(null);
  const [originalPreview, setOriginalPreview] = useState<string | null>(null);
  const [originalDims, setOriginalDims] = useState<{ w: number; h: number } | null>(null);

  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [targetPreview, setTargetPreview] = useState<string | null>(null);
  const [targetDims, setTargetDims] = useState<{ w: number; h: number } | null>(null);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TrainingExample | null>(null);
  const [error, setError] = useState<string | null>(null);

  // For manual crop editing
  const [showEditor, setShowEditor] = useState(false);
  const [crop, setCrop] = useState({ x: 0, y: 0, width: 100, height: 100 });

  const loadImageDims = (file: File): Promise<{ w: number; h: number }> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        resolve({ w: img.naturalWidth, h: img.naturalHeight });
        URL.revokeObjectURL(img.src);
      };
      img.src = URL.createObjectURL(file);
    });
  };

  const handleOriginalFile = useCallback(async (file: File, url: string) => {
    setOriginalFile(file);
    setOriginalPreview(url);
    const dims = await loadImageDims(file);
    setOriginalDims(dims);
    setResult(null);
    setError(null);
    setShowEditor(false);
  }, []);

  const handleTargetFile = useCallback(async (file: File, url: string) => {
    setTargetFile(file);
    setTargetPreview(url);
    const dims = await loadImageDims(file);
    setTargetDims(dims);
    setResult(null);
    setError(null);
    setShowEditor(false);
  }, []);

  const handleCreateExample = async () => {
    if (!originalFile) return;
    setLoading(true);
    setError(null);

    try {
      const example = await api.dataset.upload(originalFile, targetFile ?? undefined);
      setResult(example);

      if (example.confidence !== null && example.confidence < 0.5 && originalDims) {
        // Low confidence — show crop editor
        setCrop({
          x: example.crop_x ?? Math.round(originalDims.w * 0.1),
          y: example.crop_y ?? Math.round(originalDims.h * 0.1),
          width: example.crop_width ?? Math.round(originalDims.w * 0.8),
          height: example.crop_height ?? Math.round(originalDims.h * 0.8),
        });
        setShowEditor(true);
      } else if (example.crop_x !== null) {
        setCrop({
          x: example.crop_x!,
          y: example.crop_y!,
          width: example.crop_width!,
          height: example.crop_height!,
        });
      }
    } catch (e: any) {
      setError(e.message || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveLabel = async () => {
    if (!result) return;
    setLoading(true);
    try {
      const updated = await api.dataset.setLabel(result.id, {
        crop_x: crop.x,
        crop_y: crop.y,
        crop_width: crop.width,
        crop_height: crop.height,
      });
      setResult(updated);
      setShowEditor(false);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setOriginalFile(null);
    setOriginalPreview(null);
    setOriginalDims(null);
    setTargetFile(null);
    setTargetPreview(null);
    setTargetDims(null);
    setResult(null);
    setError(null);
    setShowEditor(false);
  };

  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-[var(--color-text-primary)]">Training Studio</h1>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Upload an original image and your manually cropped version to create a training example.
        </p>
      </div>

      {/* Upload area */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <ImageUploader
          label="Original Image"
          onFile={handleOriginalFile}
          previewUrl={originalPreview}
          imageWidth={originalDims?.w}
          imageHeight={originalDims?.h}
          onClear={() => {
            setOriginalFile(null);
            setOriginalPreview(null);
            setOriginalDims(null);
            setResult(null);
          }}
        />
        <ImageUploader
          label="Target Crop"
          onFile={handleTargetFile}
          previewUrl={targetPreview}
          imageWidth={targetDims?.w}
          imageHeight={targetDims?.h}
          onClear={() => {
            setTargetFile(null);
            setTargetPreview(null);
            setTargetDims(null);
            setResult(null);
          }}
        />
      </div>

      {/* Dimensions info */}
      {originalDims && (
        <div className="flex gap-6 mb-6">
          <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
            <span className="text-[var(--color-text-muted)]">Original:</span>
            <span className="font-mono">{originalDims.w} × {originalDims.h}</span>
          </div>
          {targetDims && (
            <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
              <span className="text-[var(--color-text-muted)]">Target:</span>
              <span className="font-mono">{targetDims.w} × {targetDims.h}</span>
            </div>
          )}
        </div>
      )}

      {/* Action button */}
      {originalFile && !result && (
        <button
          onClick={handleCreateExample}
          disabled={loading}
          className="flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-[var(--color-accent-primary)] to-purple-500 text-white font-semibold text-sm hover:opacity-90 transition-opacity disabled:opacity-50"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Upload className="w-4 h-4" />
          )}
          Create Training Example
        </button>
      )}

      {/* Error */}
      {error && (
        <div className="mt-6 p-4 rounded-xl border border-[var(--color-danger)]/30 bg-[var(--color-danger)]/5 flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-[var(--color-danger)]" />
          <p className="text-sm text-[var(--color-danger)]">{error}</p>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="mt-8 space-y-6">
          {/* Status banner */}
          <div
            className={`p-4 rounded-xl border flex items-center gap-3 ${
              result.status === 'labeled'
                ? 'border-[var(--color-success)]/30 bg-[var(--color-success)]/5'
                : 'border-[var(--color-warning)]/30 bg-[var(--color-warning)]/5'
            }`}
          >
            {result.status === 'labeled' ? (
              <CheckCircle className="w-5 h-5 text-[var(--color-success)]" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-[var(--color-warning)]" />
            )}
            <div>
              <p className="text-sm font-medium text-[var(--color-text-primary)]">
                {result.status === 'labeled'
                  ? 'Training example saved!'
                  : 'Low confidence — please review or manually set the crop rectangle.'}
              </p>
              {result.confidence !== null && (
                <p className="text-xs text-[var(--color-text-muted)] mt-0.5">
                  Extraction confidence: {(result.confidence * 100).toFixed(1)}%
                  {result.extraction_method && ` (${result.extraction_method})`}
                </p>
              )}
            </div>
          </div>

          {/* Crop coordinates */}
          {result.crop_x !== null && !showEditor && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { label: 'Crop X', value: result.crop_x, norm: result.norm_x },
                { label: 'Crop Y', value: result.crop_y, norm: result.norm_y },
                { label: 'Crop Width', value: result.crop_width, norm: result.norm_width },
                { label: 'Crop Height', value: result.crop_height, norm: result.norm_height },
              ].map(({ label, value, norm }) => (
                <div
                  key={label}
                  className="p-4 rounded-xl bg-[var(--color-surface-card)] border border-[var(--color-border-subtle)]"
                >
                  <p className="text-[10px] uppercase tracking-wider text-[var(--color-text-muted)] mb-1 font-semibold">
                    {label}
                  </p>
                  <p className="text-lg font-bold font-mono text-[var(--color-text-primary)]">
                    {value}
                  </p>
                  <p className="text-xs font-mono text-[var(--color-accent-secondary)]">
                    {norm?.toFixed(4)}
                  </p>
                </div>
              ))}
            </div>
          )}

          {/* Crop editor for manual correction */}
          {(showEditor || result.status === 'pending') && originalPreview && originalDims && (
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
                Manual Crop Editor
              </h3>
              <CropEditor
                imageUrl={originalPreview}
                imageWidth={originalDims.w}
                imageHeight={originalDims.h}
                crop={crop}
                onChange={setCrop}
              />
              <div className="flex gap-3">
                <button
                  onClick={handleSaveLabel}
                  disabled={loading}
                  className="px-5 py-2.5 rounded-lg bg-[var(--color-accent-primary)] text-white text-sm font-semibold hover:opacity-90 transition-opacity disabled:opacity-50"
                >
                  {loading ? 'Saving...' : 'Save Label'}
                </button>
                <button
                  onClick={() => {
                    if (originalDims) {
                      setCrop({
                        x: Math.round(originalDims.w * 0.1),
                        y: Math.round(originalDims.h * 0.1),
                        width: Math.round(originalDims.w * 0.8),
                        height: Math.round(originalDims.h * 0.8),
                      });
                    }
                  }}
                  className="px-5 py-2.5 rounded-lg border border-[var(--color-border-subtle)] text-[var(--color-text-secondary)] text-sm font-medium hover:bg-[var(--color-surface-hover)] transition-colors"
                >
                  Reset
                </button>
                <button
                  onClick={() => setShowEditor(false)}
                  className="px-5 py-2.5 rounded-lg border border-[var(--color-border-subtle)] text-[var(--color-text-secondary)] text-sm font-medium hover:bg-[var(--color-surface-hover)] transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Actions */}
          {result.status === 'labeled' && (
            <div className="flex gap-3">
              <button
                onClick={() => setShowEditor(true)}
                className="px-5 py-2.5 rounded-lg border border-[var(--color-border-subtle)] text-[var(--color-text-secondary)] text-sm font-medium hover:bg-[var(--color-surface-hover)] transition-colors"
              >
                Edit Crop
              </button>
              <button
                onClick={handleReset}
                className="px-5 py-2.5 rounded-lg bg-gradient-to-r from-[var(--color-accent-primary)] to-purple-500 text-white text-sm font-semibold hover:opacity-90 transition-opacity"
              >
                Add Another Example
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
