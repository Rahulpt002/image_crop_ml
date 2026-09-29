import { useCallback, useRef, useState } from 'react';
import { Upload, X } from 'lucide-react';

interface Props {
  label: string;
  accept?: string;
  onFile: (file: File, previewUrl: string) => void;
  previewUrl?: string | null;
  imageWidth?: number | null;
  imageHeight?: number | null;
  onClear?: () => void;
}

export default function ImageUploader({
  label,
  accept = 'image/jpeg,image/png,image/webp',
  onFile,
  previewUrl,
  imageWidth,
  imageHeight,
  onClear,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      const url = URL.createObjectURL(file);
      onFile(file, url);
    },
    [onFile]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="flex flex-col gap-2">
      <label className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
        {label}
      </label>

      {previewUrl ? (
        <div className="relative group rounded-xl overflow-hidden border border-[var(--color-border-subtle)] bg-[var(--color-surface-tertiary)]">
          <img
            src={previewUrl}
            alt={label}
            className="w-full h-64 object-contain bg-black/20"
          />
          {onClear && (
            <button
              onClick={onClear}
              className="absolute top-2 right-2 p-1.5 rounded-lg bg-black/60 text-white opacity-0 group-hover:opacity-100 transition-opacity hover:bg-black/80"
            >
              <X className="w-4 h-4" />
            </button>
          )}
          {imageWidth && imageHeight && (
            <div className="absolute bottom-2 left-2 px-2 py-1 rounded-md bg-black/60 text-[10px] text-white font-mono">
              {imageWidth} × {imageHeight}
            </div>
          )}
        </div>
      ) : (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`flex flex-col items-center justify-center h-64 rounded-xl border-2 border-dashed cursor-pointer transition-all duration-200 ${
            dragOver
              ? 'border-[var(--color-accent-primary)] bg-[var(--color-accent-primary)]/5'
              : 'border-[var(--color-border-subtle)] bg-[var(--color-surface-tertiary)] hover:border-[var(--color-text-muted)] hover:bg-[var(--color-surface-hover)]'
          }`}
        >
          <Upload className="w-8 h-8 text-[var(--color-text-muted)] mb-3" />
          <p className="text-sm text-[var(--color-text-secondary)]">
            Drop image or <span className="text-[var(--color-accent-secondary)] font-medium">browse</span>
          </p>
          <p className="text-xs text-[var(--color-text-muted)] mt-1">
            JPG, PNG, WebP
          </p>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
          e.target.value = '';
        }}
      />
    </div>
  );
}
