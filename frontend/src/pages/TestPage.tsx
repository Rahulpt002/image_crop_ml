import { useState } from 'react';
import {
  FlaskConical,
  Loader2,
  AlertTriangle,
  Eye,
} from 'lucide-react';
import ImageUploader from '../components/ImageUploader';
import CropEditor from '../components/CropEditor';
import { api } from '../api';
import type { PredictionResponse } from '../types';

export default function TestPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [crop, setCrop] = useState({ x: 0, y: 0, width: 100, height: 100 });

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
  };

  const handlePredict = async () => {
    if (!file || !dims) return;
    setLoading(true);
    setError(null);
    try {
      const pred = await api.model.predict(file);
      setPrediction(pred);
      if (pred.pred_crop_x !== null) {
        setCrop({
          x: pred.pred_crop_x!,
          y: pred.pred_crop_y!,
          width: pred.pred_crop_width!,
          height: pred.pred_crop_height!,
        });
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCorrect = async (addToDataset: boolean) => {
    if (!prediction || !dims) return;
    try {
      await api.model.correct({
        prediction_id: prediction.id,
        corrected_x: crop.x / dims.w,
        corrected_y: crop.y / dims.h,
        corrected_width: crop.width / dims.w,
        corrected_height: crop.height / dims.h,
        add_to_dataset: addToDataset,
      });
      alert(addToDataset ? 'Correction saved and added to training dataset!' : 'Correction saved!');
    } catch (e: any) {
      setError(e.message);
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-[var(--color-text-primary)]">Test Model</h1>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Upload an image to test the model's crop prediction.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left: Upload & predict */}
        <div className="space-y-6">
          <ImageUploader
            label="Test Image"
            onFile={handleFile}
            previewUrl={preview}
            imageWidth={dims?.w}
            imageHeight={dims?.h}
            onClear={() => {
              setFile(null);
              setPreview(null);
              setDims(null);
              setPrediction(null);
            }}
          />

          {file && !prediction && (
            <button
              onClick={handlePredict}
              disabled={loading}
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-[var(--color-accent-primary)] to-purple-500 text-white font-semibold text-sm hover:opacity-90 transition-opacity disabled:opacity-50 w-full justify-center"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <FlaskConical className="w-4 h-4" />
              )}
              Predict Crop
            </button>
          )}

          {error && (
            <div className="p-4 rounded-xl border border-[var(--color-danger)]/30 bg-[var(--color-danger)]/5 flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-[var(--color-danger)]" />
              <p className="text-sm text-[var(--color-danger)]">{error}</p>
            </div>
          )}

          {/* Prediction values */}
          {prediction && prediction.pred_x !== null && (
            <div className="p-4 rounded-xl bg-[var(--color-surface-card)] border border-[var(--color-border-subtle)]">
              <h3 className="text-xs uppercase tracking-wider text-[var(--color-text-muted)] font-semibold mb-3 flex items-center gap-2">
                <Eye className="w-3.5 h-3.5" />
                Prediction
              </h3>
              <div className="grid grid-cols-4 gap-3">
                {[
                  { label: 'x', value: prediction.pred_x },
                  { label: 'y', value: prediction.pred_y },
                  { label: 'width', value: prediction.pred_width },
                  { label: 'height', value: prediction.pred_height },
                ].map(({ label, value }) => (
                  <div key={label} className="p-2 rounded-lg bg-[var(--color-surface-tertiary)]">
                    <p className="text-[10px] uppercase text-[var(--color-text-muted)] font-semibold">{label}</p>
                    <p className="text-sm font-mono font-bold text-[var(--color-accent-secondary)]">
                      {value?.toFixed(4)}
                    </p>
                  </div>
                ))}
              </div>
              <div className="flex items-center justify-between text-xs text-[var(--color-text-muted)] mt-2">
                <span>Model: {prediction.model_version}</span>
                <span className="font-mono text-[var(--color-accent-secondary)] font-medium">
                  Output: 150×200 px · 300 DPI
                </span>
              </div>
              {prediction.face_detected ? (
                <div className="mt-2 pt-2 border-t border-[var(--color-border-subtle)] text-xs text-emerald-400 flex items-center gap-1.5 font-medium">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block animate-pulse" />
                  Face localized {prediction.face_confidence ? `(${Math.round(prediction.face_confidence * 100)}% conf)` : ''} · 3:4 framing applied
                </div>
              ) : (
                <div className="mt-2 pt-2 border-t border-[var(--color-border-subtle)] text-xs text-[var(--color-text-muted)]">
                  No face detected · Full CropNet framing applied
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: Crop preview */}
        <div className="space-y-6">
          {prediction && preview && dims && (
            <>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-muted)] mb-2">
                  Predicted Crop — 3:4 Aspect Ratio Locked
                </p>
                <CropEditor
                  imageUrl={preview}
                  imageWidth={dims.w}
                  imageHeight={dims.h}
                  crop={crop}
                  onChange={setCrop}
                  aspectRatio={0.75}
                  faceBox={
                    prediction.face_detected && prediction.face_box_x !== null
                      ? {
                          x: prediction.face_box_x!,
                          y: prediction.face_box_y!,
                          width: prediction.face_box_width!,
                          height: prediction.face_box_height!,
                        }
                      : null
                  }
                />
              </div>

              {/* Live crop preview */}
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-muted)] mb-2">
                  Crop Preview
                </p>
                <div className="rounded-xl overflow-hidden border border-[var(--color-border-subtle)] bg-black/30">
                  <canvas
                    ref={(canvas) => {
                      if (!canvas || !preview) return;
                      const img = new Image();
                      img.onload = () => {
                        const ctx = canvas.getContext('2d');
                        if (!ctx) return;
                        const maxW = 400;
                        const scale = Math.min(maxW / crop.width, maxW / crop.height);
                        canvas.width = crop.width * scale;
                        canvas.height = crop.height * scale;
                        ctx.drawImage(
                          img,
                          crop.x, crop.y, crop.width, crop.height,
                          0, 0, canvas.width, canvas.height
                        );
                      };
                      img.src = preview;
                    }}
                    className="w-full"
                  />
                </div>
              </div>

              {/* Actions */}
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={() => handleCorrect(false)}
                  className="px-5 py-2.5 rounded-lg border border-[var(--color-border-subtle)] text-[var(--color-text-secondary)] text-sm font-medium hover:bg-[var(--color-surface-hover)] transition-colors"
                >
                  Save Correction
                </button>
                <button
                  onClick={() => handleCorrect(true)}
                  className="px-5 py-2.5 rounded-lg bg-[var(--color-accent-primary)] text-white text-sm font-semibold hover:opacity-90 transition-opacity"
                >
                  Add to Training Dataset
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
