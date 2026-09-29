import { useState, useRef, useEffect } from 'react';
import {
  Scissors,
  Sparkles,
  Download,
  RotateCcw,
  Loader2,
  AlertTriangle,
  Save,
} from 'lucide-react';
import ImageUploader from '../components/ImageUploader';
import CropEditor from '../components/CropEditor';
import { api } from '../api';
import type { PredictionResponse } from '../types';

export default function CropPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [crop, setCrop] = useState({ x: 0, y: 0, width: 100, height: 100 });
  const [originalCrop, setOriginalCrop] = useState({ x: 0, y: 0, width: 100, height: 100 });
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const loadDims = (f: File): Promise<{ w: number; h: number }> =>
    new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        resolve({ w: img.naturalWidth, h: img.naturalHeight });
        URL.revokeObjectURL(img.src);
      };
      img.src = URL.createObjectURL(f);
    });

  const handleFile = async (f: File, url: string) => {
    setFile(f);
    setPreview(url);
    const d = await loadDims(f);
    setDims(d);
    setPrediction(null);
    setError(null);

    // Auto-predict
    setLoading(true);
    try {
      const pred = await api.model.predict(f);
      setPrediction(pred);
      if (pred.pred_crop_x !== null) {
        const c = {
          x: pred.pred_crop_x!,
          y: pred.pred_crop_y!,
          width: pred.pred_crop_width!,
          height: pred.pred_crop_height!,
        };
        setCrop(c);
        setOriginalCrop(c);
      }
    } catch (e: any) {
      setError(e.message);
      // Set default crop
      setCrop({ x: 0, y: 0, width: d.w, height: d.h });
    } finally {
      setLoading(false);
    }
  };

  // Draw crop preview
  useEffect(() => {
    if (!canvasRef.current || !preview) return;
    const canvas = canvasRef.current;
    const img = new Image();
    img.onload = () => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const maxSize = 500;
      const scale = Math.min(maxSize / crop.width, maxSize / crop.height, 1);
      canvas.width = Math.max(1, crop.width * scale);
      canvas.height = Math.max(1, crop.height * scale);
      ctx.drawImage(
        img,
        crop.x, crop.y, crop.width, crop.height,
        0, 0, canvas.width, canvas.height
      );
    };
    img.src = preview;
  }, [preview, crop]);

  const [exporting, setExporting] = useState(false);

  const fallbackCanvasDownload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 150;
    canvas.height = 200;
    const ctx = canvas.getContext('2d');
    if (!ctx || !preview) return;
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, crop.x, crop.y, crop.width, crop.height, 0, 0, 150, 200);
      const link = document.createElement('a');
      link.download = `portrait_150x200_${file?.name || 'image'}.jpg`;
      link.href = canvas.toDataURL('image/jpeg', 0.95);
      link.click();
    };
    img.src = preview;
  };

  const handleDownload = async () => {
    if (prediction) {
      setExporting(true);
      try {
        const res = await api.model.export(prediction.id, crop);
        const link = document.createElement('a');
        link.download = res.filename || `portrait_150x200_${file?.name || 'crop'}.jpg`;
        link.href = `/exports/${res.filename}`;
        link.click();
      } catch (err) {
        console.warn('Backend export failed, falling back to canvas download:', err);
        fallbackCanvasDownload();
      } finally {
        setExporting(false);
      }
    } else {
      fallbackCanvasDownload();
    }
  };

  const handleSaveCorrection = async () => {
    if (!prediction || !dims) return;
    try {
      await api.model.correct({
        prediction_id: prediction.id,
        corrected_x: crop.x / dims.w,
        corrected_y: crop.y / dims.h,
        corrected_width: crop.width / dims.w,
        corrected_height: crop.height / dims.h,
        add_to_dataset: true,
      });
      alert('Correction saved to training dataset!');
    } catch (e: any) {
      setError(e.message);
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-[var(--color-text-primary)] flex items-center gap-3">
          <div className="p-2 rounded-xl bg-gradient-to-br from-[var(--color-accent-primary)] to-purple-500">
            <Scissors className="w-5 h-5 text-white" />
          </div>
          AI Portrait Crop
        </h1>
        <p className="text-sm text-[var(--color-text-secondary)] mt-2">
          Upload a portrait photo to detect face/head, frame in 3:4 aspect ratio, and export at 150 × 200 px (300 DPI).
        </p>
      </div>

      {!file ? (
        <div className="max-w-xl mx-auto">
          <ImageUploader
            label="Upload Image to Crop"
            onFile={handleFile}
            previewUrl={null}
          />
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left: Original with crop overlay */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
                Original — 3:4 Crop Framing
              </p>
              {prediction?.face_detected && (
                <span className="text-xs text-emerald-400 flex items-center gap-1 font-medium bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  Face Localized
                </span>
              )}
            </div>

            {loading ? (
              <div className="flex flex-col items-center justify-center h-80 rounded-xl bg-[var(--color-surface-card)] border border-[var(--color-border-subtle)]">
                <Loader2 className="w-8 h-8 animate-spin text-[var(--color-accent-primary)] mb-3" />
                <p className="text-sm text-[var(--color-text-muted)]">AI is analyzing face and learned portrait framing...</p>
              </div>
            ) : dims ? (
              <CropEditor
                imageUrl={preview!}
                imageWidth={dims.w}
                imageHeight={dims.h}
                crop={crop}
                onChange={setCrop}
                aspectRatio={0.75}
                faceBox={
                  prediction?.face_detected && prediction.face_box_x !== null
                    ? {
                        x: prediction.face_box_x!,
                        y: prediction.face_box_y!,
                        width: prediction.face_box_width!,
                        height: prediction.face_box_height!,
                      }
                    : null
                }
              />
            ) : null}

            {error && (
              <div className="p-3 rounded-lg border border-[var(--color-warning)]/30 bg-[var(--color-warning)]/5 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-[var(--color-warning)]" />
                <p className="text-xs text-[var(--color-warning)]">
                  {error.includes('No active model')
                    ? 'No trained model available. Showing manual 3:4 crop mode.'
                    : error}
                </p>
              </div>
            )}
          </div>

          {/* Right: Result */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
                {prediction ? 'AI Crop Result' : 'Crop Preview'}
              </p>
              <span className="text-xs font-mono font-medium text-[var(--color-accent-secondary)] bg-[var(--color-accent-primary)]/10 px-2 py-0.5 rounded border border-[var(--color-accent-primary)]/20">
                150 × 200 px · 300 DPI
              </span>
            </div>

            <div className="rounded-xl overflow-hidden border border-[var(--color-border-subtle)] bg-black/30 flex items-center justify-center min-h-[200px]">
              <canvas ref={canvasRef} className="max-w-full max-h-[500px]" />
            </div>

            {prediction && (
              <div className="flex items-center justify-between text-xs text-[var(--color-text-muted)]">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-3.5 h-3.5 text-[var(--color-accent-secondary)]" />
                  <span>Predicted by model {prediction.model_version}</span>
                </div>
                <span>Aspect Ratio: 3:4 (0.75)</span>
              </div>
            )}

            {/* Action buttons */}
            <div className="flex flex-wrap gap-3">
              {prediction && (
                <button
                  onClick={() => setCrop({ ...originalCrop })}
                  className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-[var(--color-accent-primary)]/10 text-[var(--color-accent-secondary)] text-sm font-medium hover:bg-[var(--color-accent-primary)]/20 transition-colors"
                >
                  <Sparkles className="w-4 h-4" />
                  Reset to AI
                </button>
              )}
              <button
                onClick={() => {
                  if (dims) {
                    // Reset to centered 3:4 crop
                    const targetAr = 0.75;
                    const w = Math.min(dims.w, dims.h * targetAr);
                    const h = w / targetAr;
                    setCrop({
                      x: Math.round((dims.w - w) / 2),
                      y: Math.round((dims.h - h) / 2),
                      width: Math.round(w),
                      height: Math.round(h),
                    });
                  }
                }}
                className="flex items-center gap-2 px-4 py-2.5 rounded-lg border border-[var(--color-border-subtle)] text-[var(--color-text-secondary)] text-sm font-medium hover:bg-[var(--color-surface-hover)] transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                Reset Frame
              </button>
              <button
                onClick={handleDownload}
                disabled={exporting}
                className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-gradient-to-r from-[var(--color-accent-primary)] to-purple-500 text-white text-sm font-semibold hover:opacity-90 transition-opacity disabled:opacity-50"
              >
                {exporting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Download className="w-4 h-4" />
                )}
                Export 150×200 (300 DPI)
              </button>
            </div>

            {/* Save correction */}
            {prediction && (
              <div className="pt-4 border-t border-[var(--color-border-subtle)]">
                <button
                  onClick={handleSaveCorrection}
                  className="flex items-center gap-2 px-4 py-2.5 rounded-lg border border-[var(--color-border-subtle)] text-[var(--color-text-secondary)] text-sm font-medium hover:bg-[var(--color-surface-hover)] transition-colors"
                >
                  <Save className="w-4 h-4" />
                  Save Correction to Dataset
                </button>
                <p className="text-[10px] text-[var(--color-text-muted)] mt-1.5">
                  Saves your final crop as a new training example for future model improvement.
                </p>
              </div>
            )}

            {/* New image */}
            <div className="pt-2">
              <button
                onClick={() => {
                  setFile(null);
                  setPreview(null);
                  setDims(null);
                  setPrediction(null);
                  setError(null);
                }}
                className="text-xs text-[var(--color-accent-secondary)] hover:underline"
              >
                ← Upload a different image
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
