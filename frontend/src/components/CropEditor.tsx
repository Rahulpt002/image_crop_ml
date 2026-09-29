import { useRef, useState, useEffect, useCallback } from 'react';

interface CropBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Props {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  crop: CropBox;
  onChange: (crop: CropBox) => void;
  readOnly?: boolean;
  aspectRatio?: number;
  faceBox?: { x: number; y: number; width: number; height: number } | null;
}

export default function CropEditor({
  imageUrl,
  imageWidth,
  imageHeight,
  crop,
  onChange,
  readOnly = false,
  aspectRatio,
  faceBox,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [displaySize, setDisplaySize] = useState({ w: 0, h: 0 });
  const [dragging, setDragging] = useState<null | 'move' | 'nw' | 'ne' | 'sw' | 'se' | 'n' | 's' | 'e' | 'w'>(null);
  const [dragStart, setDragStart] = useState({ mx: 0, my: 0, crop: crop });

  // Calculate display dimensions preserving aspect ratio
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => {
      const containerW = el.clientWidth;
      const maxH = 500;
      const scale = Math.min(containerW / imageWidth, maxH / imageHeight);
      setDisplaySize({
        w: Math.round(imageWidth * scale),
        h: Math.round(imageHeight * scale),
      });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [imageWidth, imageHeight]);

  const scaleX = displaySize.w / imageWidth;
  const scaleY = displaySize.h / imageHeight;

  // Display crop rectangle
  const displayCrop = {
    x: crop.x * scaleX,
    y: crop.y * scaleY,
    w: crop.width * scaleX,
    h: crop.height * scaleY,
  };

  // Normalized values
  const normX = (crop.x / imageWidth).toFixed(4);
  const normY = (crop.y / imageHeight).toFixed(4);
  const normW = (crop.width / imageWidth).toFixed(4);
  const normH = (crop.height / imageHeight).toFixed(4);

  const clamp = (val: number, min: number, max: number) => Math.max(min, Math.min(max, val));

  const handleMouseDown = useCallback(
    (e: React.MouseEvent, handle: typeof dragging) => {
      if (readOnly) return;
      e.preventDefault();
      e.stopPropagation();
      setDragging(handle);
      setDragStart({ mx: e.clientX, my: e.clientY, crop: { ...crop } });
    },
    [crop, readOnly]
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!dragging) return;

      const dx = (e.clientX - dragStart.mx) / scaleX;
      const dy = (e.clientY - dragStart.my) / scaleY;
      const c = { ...dragStart.crop };

      if (dragging === 'move') {
        c.x = clamp(c.x + dx, 0, imageWidth - c.width);
        c.y = clamp(c.y + dy, 0, imageHeight - c.height);
      } else if (aspectRatio) {
        const base = dragStart.crop;
        if (dragging === 'se') {
          let w = Math.max(30, base.width + dx);
          let h = w / aspectRatio;
          if (base.x + w > imageWidth) {
            w = imageWidth - base.x;
            h = w / aspectRatio;
          }
          if (base.y + h > imageHeight) {
            h = imageHeight - base.y;
            w = h * aspectRatio;
          }
          c.width = w;
          c.height = h;
        } else if (dragging === 'nw') {
          let w = Math.max(30, base.width - dx);
          let h = w / aspectRatio;
          if (base.x + base.width - w < 0) {
            w = base.x + base.width;
            h = w / aspectRatio;
          }
          if (base.y + base.height - h < 0) {
            h = base.y + base.height;
            w = h * aspectRatio;
          }
          c.x = base.x + base.width - w;
          c.y = base.y + base.height - h;
          c.width = w;
          c.height = h;
        } else if (dragging === 'ne') {
          let w = Math.max(30, base.width + dx);
          let h = w / aspectRatio;
          if (base.x + w > imageWidth) {
            w = imageWidth - base.x;
            h = w / aspectRatio;
          }
          if (base.y + base.height - h < 0) {
            h = base.y + base.height;
            w = h * aspectRatio;
          }
          c.y = base.y + base.height - h;
          c.width = w;
          c.height = h;
        } else if (dragging === 'sw') {
          let w = Math.max(30, base.width - dx);
          let h = w / aspectRatio;
          if (base.x + base.width - w < 0) {
            w = base.x + base.width;
            h = w / aspectRatio;
          }
          if (base.y + h > imageHeight) {
            h = imageHeight - base.y;
            w = h * aspectRatio;
          }
          c.x = base.x + base.width - w;
          c.width = w;
          c.height = h;
        } else if (dragging === 'e' || dragging === 'w') {
          const delta = dragging === 'e' ? dx : -dx;
          let w = clamp(base.width + delta, 30, dragging === 'e' ? imageWidth - base.x : base.x + base.width);
          let h = w / aspectRatio;
          if (h > imageHeight) {
            h = imageHeight;
            w = h * aspectRatio;
          }
          let newY = clamp(base.y + (base.height - h) / 2, 0, imageHeight - h);
          c.y = newY;
          c.height = h;
          c.width = w;
          if (dragging === 'w') {
            c.x = base.x + base.width - w;
          }
        } else if (dragging === 'n' || dragging === 's') {
          const delta = dragging === 's' ? dy : -dy;
          let h = clamp(base.height + delta, 40, dragging === 's' ? imageHeight - base.y : base.y + base.height);
          let w = h * aspectRatio;
          if (w > imageWidth) {
            w = imageWidth;
            h = w / aspectRatio;
          }
          let newX = clamp(base.x + (base.width - w) / 2, 0, imageWidth - w);
          c.x = newX;
          c.width = w;
          c.height = h;
          if (dragging === 'n') {
            c.y = base.y + base.height - h;
          }
        }
      } else {
        // Freeform resize
        if (dragging.includes('w')) {
          const newX = clamp(c.x + dx, 0, c.x + c.width - 20);
          c.width = c.width - (newX - c.x);
          c.x = newX;
        }
        if (dragging.includes('e')) {
          c.width = clamp(c.width + dx, 20, imageWidth - c.x);
        }
        if (dragging.includes('n')) {
          const newY = clamp(c.y + dy, 0, c.y + c.height - 20);
          c.height = c.height - (newY - c.y);
          c.y = newY;
        }
        if (dragging.includes('s')) {
          c.height = clamp(c.height + dy, 20, imageHeight - c.y);
        }
      }

      onChange({
        x: Math.round(c.x),
        y: Math.round(c.y),
        width: Math.round(c.width),
        height: Math.round(c.height),
      });
    },
    [dragging, dragStart, scaleX, scaleY, imageWidth, imageHeight, onChange, aspectRatio]
  );

  const handleMouseUp = useCallback(() => {
    setDragging(null);
  }, []);

  useEffect(() => {
    if (dragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [dragging, handleMouseMove, handleMouseUp]);

  const handleSize = 8;

  return (
    <div className="flex flex-col gap-4">
      {/* Canvas */}
      <div
        ref={containerRef}
        className="relative select-none overflow-hidden rounded-xl border border-[var(--color-border-subtle)] bg-black/30"
        style={{ width: '100%' }}
      >
        <div
          className="relative mx-auto"
          style={{ width: displaySize.w, height: displaySize.h }}
        >
          {/* Image */}
          <img
            src={imageUrl}
            alt="Original"
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            draggable={false}
          />

          {/* Dimmed overlay outside crop */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 ${displaySize.w} ${displaySize.h}`}
          >
            <defs>
              <mask id="cropMask">
                <rect width="100%" height="100%" fill="white" />
                <rect
                  x={displayCrop.x}
                  y={displayCrop.y}
                  width={displayCrop.w}
                  height={displayCrop.h}
                  fill="black"
                />
              </mask>
            </defs>
            <rect
              width="100%"
              height="100%"
              fill="rgba(0,0,0,0.55)"
              mask="url(#cropMask)"
            />
          </svg>

          {/* Face Detection Box (if present) */}
          {faceBox && (
            <div
              className="absolute border border-dashed border-emerald-400 pointer-events-none z-10 rounded-sm"
              style={{
                left: faceBox.x * scaleX,
                top: faceBox.y * scaleY,
                width: faceBox.width * scaleX,
                height: faceBox.height * scaleY,
                boxShadow: '0 0 10px rgba(52, 211, 153, 0.35)',
              }}
            >
              <span className="absolute -top-4 left-0 bg-emerald-500 text-black text-[9px] font-bold px-1 py-0.5 rounded-sm tracking-wide">
                FACE
              </span>
            </div>
          )}

          {/* Crop rectangle */}
          <div
            className={`absolute border-2 border-[var(--color-accent-primary)] ${
              readOnly ? '' : 'cursor-move'
            }`}
            style={{
              left: displayCrop.x,
              top: displayCrop.y,
              width: displayCrop.w,
              height: displayCrop.h,
              boxShadow: '0 0 0 1px rgba(99,102,241,0.3), 0 0 20px rgba(99,102,241,0.15)',
            }}
            onMouseDown={(e) => handleMouseDown(e, 'move')}
          >
            {/* Rule of thirds guides */}
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute left-1/3 top-0 bottom-0 w-px bg-white/15" />
              <div className="absolute left-2/3 top-0 bottom-0 w-px bg-white/15" />
              <div className="absolute top-1/3 left-0 right-0 h-px bg-white/15" />
              <div className="absolute top-2/3 left-0 right-0 h-px bg-white/15" />
            </div>

            {/* Resize handles */}
            {!readOnly && (
              <>
                {(['nw', 'ne', 'sw', 'se', 'n', 's', 'e', 'w'] as const).map((pos) => {
                  const style: React.CSSProperties = { position: 'absolute', width: handleSize, height: handleSize };
                  const cursor = {
                    nw: 'nw-resize', ne: 'ne-resize', sw: 'sw-resize', se: 'se-resize',
                    n: 'n-resize', s: 's-resize', e: 'e-resize', w: 'w-resize',
                  }[pos];

                  if (pos.includes('n')) style.top = -handleSize / 2;
                  if (pos.includes('s')) style.bottom = -handleSize / 2;
                  if (pos.includes('w')) style.left = -handleSize / 2;
                  if (pos.includes('e')) style.right = -handleSize / 2;
                  if (pos === 'n' || pos === 's') { style.left = '50%'; style.marginLeft = -handleSize / 2; }
                  if (pos === 'e' || pos === 'w') { style.top = '50%'; style.marginTop = -handleSize / 2; }

                  return (
                    <div
                      key={pos}
                      className="bg-white border border-[var(--color-accent-primary)] rounded-sm"
                      style={{ ...style, cursor }}
                      onMouseDown={(e) => handleMouseDown(e, pos)}
                    />
                  );
                })}
              </>
            )}
          </div>
        </div>
      </div>

      {/* Coordinate info */}
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded-lg bg-[var(--color-surface-tertiary)] border border-[var(--color-border-subtle)]">
          <div className="flex items-center justify-between mb-2">
            <p className="text-[10px] uppercase tracking-wider text-[var(--color-text-muted)] font-semibold">
              Pixel Coordinates
            </p>
            {aspectRatio && (
              <span className="text-[10px] bg-[var(--color-accent-primary)]/15 text-[var(--color-accent-secondary)] px-1.5 py-0.5 rounded font-mono font-medium">
                3:4 Portrait Lock
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs font-mono">
            <span className="text-[var(--color-text-muted)]">X:</span>
            <span className="text-[var(--color-text-primary)]">{crop.x}</span>
            <span className="text-[var(--color-text-muted)]">Y:</span>
            <span className="text-[var(--color-text-primary)]">{crop.y}</span>
            <span className="text-[var(--color-text-muted)]">W:</span>
            <span className="text-[var(--color-text-primary)]">{crop.width}</span>
            <span className="text-[var(--color-text-muted)]">H:</span>
            <span className="text-[var(--color-text-primary)]">{crop.height}</span>
          </div>
        </div>
        <div className="p-3 rounded-lg bg-[var(--color-surface-tertiary)] border border-[var(--color-border-subtle)]">
          <p className="text-[10px] uppercase tracking-wider text-[var(--color-text-muted)] mb-2 font-semibold">
            Normalized (0–1)
          </p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs font-mono">
            <span className="text-[var(--color-text-muted)]">x:</span>
            <span className="text-[var(--color-accent-secondary)]">{normX}</span>
            <span className="text-[var(--color-text-muted)]">y:</span>
            <span className="text-[var(--color-accent-secondary)]">{normY}</span>
            <span className="text-[var(--color-text-muted)]">w:</span>
            <span className="text-[var(--color-accent-secondary)]">{normW}</span>
            <span className="text-[var(--color-text-muted)]">h:</span>
            <span className="text-[var(--color-accent-secondary)]">{normH}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
