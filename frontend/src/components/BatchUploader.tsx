/**
 * BatchUploader — Batch dataset import component.
 *
 * Handles:
 * 1. Drag & drop / file selection for originals and targets
 * 2. Automatic filename-stem pairing
 * 3. Duplicate detection via file hashing
 * 4. Controlled concurrent upload (4 at a time)
 * 5. Progress tracking with per-pair status
 * 6. Post-upload summary with review flow
 */

import { useState, useCallback, useRef, useMemo } from 'react';
import { api } from '../api';
import type { FilePair } from '../types';

const ALLOWED_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp']);
const MAX_CONCURRENT = 4;

function getFileStem(filename: string): string {
  const lastDot = filename.lastIndexOf('.');
  return lastDot > 0 ? filename.substring(0, lastDot).toLowerCase() : filename.toLowerCase();
}

function getFileExt(filename: string): string {
  const lastDot = filename.lastIndexOf('.');
  return lastDot > 0 ? filename.substring(lastDot).toLowerCase() : '';
}

function isValidImageFile(file: File): boolean {
  if (file.size === 0) return false;
  const ext = getFileExt(file.name);
  return ALLOWED_EXTENSIONS.has(ext);
}

function generateBatchId(): string {
  const now = new Date();
  const pad = (n: number, w = 2) => String(n).padStart(w, '0');
  return `batch_${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}



type Phase = 'select' | 'pairing' | 'uploading' | 'complete';

interface BatchStats {
  matched: number;
  missingTarget: number;
  missingOriginal: number;
  duplicates: number;
  invalid: number;
}

export default function BatchUploader() {
  const [phase, setPhase] = useState<Phase>('select');
  const [originals, setOriginals] = useState<File[]>([]);
  const [targets, setTargets] = useState<File[]>([]);
  const [pairs, setPairs] = useState<FilePair[]>([]);
  const [batchId] = useState(generateBatchId);
  const [showPairingTable, setShowPairingTable] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({ done: 0, total: 0, success: 0, failed: 0, review: 0, skipped: 0 });
  const [isDraggingOrig, setIsDraggingOrig] = useState(false);
  const [isDraggingTarget, setIsDraggingTarget] = useState(false);
  const [isHashing, setIsHashing] = useState(false);
  const stopRef = useRef(false);

  const origInputRef = useRef<HTMLInputElement>(null);
  const targetInputRef = useRef<HTMLInputElement>(null);
  const origFolderRef = useRef<HTMLInputElement>(null);
  const targetFolderRef = useRef<HTMLInputElement>(null);

  // ─── File selection handlers ──────────────────────────────────

  const handleFiles = useCallback((files: FileList | File[], type: 'original' | 'target') => {
    const fileArray = Array.from(files).filter(f => {
      if (!isValidImageFile(f)) return false;
      return true;
    });
    if (type === 'original') {
      setOriginals(prev => [...prev, ...fileArray]);
    } else {
      setTargets(prev => [...prev, ...fileArray]);
    }
    setPhase('select');
    setPairs([]);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent, type: 'original' | 'target') => {
    e.preventDefault();
    setIsDraggingOrig(false);
    setIsDraggingTarget(false);

    // Handle folder drops
    const items = e.dataTransfer.items;
    const entries: FileSystemEntry[] = [];
    if (items) {
      for (let i = 0; i < items.length; i++) {
        const entry = items[i].webkitGetAsEntry?.();
        if (entry) entries.push(entry);
      }
    }

    if (entries.length > 0 && entries.some(e => e.isDirectory)) {
      // Read directory contents
      const readDir = (dirEntry: FileSystemDirectoryEntry): Promise<File[]> => {
        return new Promise((resolve) => {
          const reader = dirEntry.createReader();
          const allFiles: File[] = [];
          const readEntries = () => {
            reader.readEntries((entries) => {
              if (entries.length === 0) {
                resolve(allFiles);
                return;
              }
              const promises = entries.map(entry => {
                if (entry.isFile) {
                  return new Promise<File | null>((res) => {
                    (entry as FileSystemFileEntry).file(f => res(f), () => res(null));
                  });
                }
                return Promise.resolve(null);
              });
              Promise.all(promises).then(results => {
                allFiles.push(...results.filter((f): f is File => f !== null));
                readEntries();
              });
            });
          };
          readEntries();
        });
      };

      Promise.all(entries.map(entry => {
        if (entry.isDirectory) return readDir(entry as FileSystemDirectoryEntry);
        return new Promise<File[]>((resolve) => {
          (entry as FileSystemFileEntry).file(f => resolve([f]), () => resolve([]));
        });
      })).then(results => {
        handleFiles(results.flat(), type);
      });
      return;
    }

    // Regular file drops
    if (e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files, type);
    }
  }, [handleFiles]);

  // ─── Pairing logic ────────────────────────────────────────────

  const runPairing = useCallback(async () => {
    setIsHashing(true);
    setPhase('pairing');

    const origMap = new Map<string, File>();
    const targetMap = new Map<string, File>();
    const invalidFiles: string[] = [];

    originals.forEach(f => {
      if (!isValidImageFile(f)) { invalidFiles.push(f.name); return; }
      origMap.set(getFileStem(f.name), f);
    });
    targets.forEach(f => {
      if (!isValidImageFile(f)) { invalidFiles.push(f.name); return; }
      targetMap.set(getFileStem(f.name), f);
    });

    // Get all unique stems
    const allStems = new Set([...origMap.keys(), ...targetMap.keys()]);

    // Build pairs
    const newPairs: FilePair[] = [];
    for (const stem of allStems) {
      const orig = origMap.get(stem) || null;
      const tgt = targetMap.get(stem) || null;

      let status: FilePair['status'] = 'matched';
      if (!orig) status = 'missing_original';
      else if (!tgt) status = 'missing_target';

      newPairs.push({
        stem,
        original: orig,
        target: tgt,
        status,
        uploadStatus: 'pending',
      });
    }

    // Sort: matched first, then unmatched
    newPairs.sort((a, b) => {
      if (a.status === 'matched' && b.status !== 'matched') return -1;
      if (a.status !== 'matched' && b.status === 'matched') return 1;
      return a.stem.localeCompare(b.stem);
    });

    setPairs(newPairs);
    setIsHashing(false);
    setShowPairingTable(true);
  }, [originals, targets]);

  // ─── Upload logic ─────────────────────────────────────────────

  const matchedPairs = useMemo(() => pairs.filter(p => p.status === 'matched'), [pairs]);

  const startUpload = useCallback(async () => {
    stopRef.current = false;
    setPhase('uploading');
    const toUpload = matchedPairs.filter(p => p.uploadStatus !== 'skipped');
    setUploadProgress({ done: 0, total: toUpload.length, success: 0, failed: 0, review: 0, skipped: 0 });

    const uploadedResults: unknown[] = [];
    let successCount = 0;
    let failedCount = 0;
    let reviewCount = 0;
    let skippedCount = 0;

    // Controlled concurrency pool
    let index = 0;
    const uploadNext = async (): Promise<void> => {
      while (index < toUpload.length && !stopRef.current) {
        const i = index++;
        const pair = toUpload[i];

        // Update pair status
        setPairs(prev => prev.map(p => p.stem === pair.stem ? { ...p, uploadStatus: 'uploading' as const } : p));

        try {
          const result = await api.dataset.upload(pair.original!, pair.target!, batchId);
          uploadedResults.push(result);

          // Check if it was a duplicate (same file_hash returned for existing)
          if (result.file_hash && pair.original) {
            // Successfully uploaded
          }

          if (result.status === 'labeled' && result.confidence && result.confidence >= 0.7) {
            successCount++;
          } else if (result.status === 'pending' || (result.confidence && result.confidence < 0.7)) {
            reviewCount++;
          } else {
            successCount++;
          }

          setPairs(prev => prev.map(p => p.stem === pair.stem ? { ...p, uploadStatus: 'success' as const, result } : p));
        } catch (err) {
          failedCount++;
          setPairs(prev => prev.map(p => p.stem === pair.stem ? { ...p, uploadStatus: 'failed' as const, error: String(err) } : p));
        }

        setUploadProgress({
          done: successCount + failedCount + reviewCount + skippedCount,
          total: toUpload.length,
          success: successCount,
          failed: failedCount,
          review: reviewCount,
          skipped: skippedCount,
        });
      }
    };

    // Launch concurrent workers
    const workers = Array.from({ length: Math.min(MAX_CONCURRENT, toUpload.length) }, () => uploadNext());
    await Promise.all(workers);


    setPhase('complete');
  }, [matchedPairs, batchId]);

  // ─── Stats ────────────────────────────────────────────────────

  const stats: BatchStats = useMemo(() => ({
    matched: pairs.filter(p => p.status === 'matched').length,
    missingTarget: pairs.filter(p => p.status === 'missing_target').length,
    missingOriginal: pairs.filter(p => p.status === 'missing_original').length,
    duplicates: pairs.filter(p => p.status === 'duplicate').length,
    invalid: pairs.filter(p => p.status === 'invalid').length,
  }), [pairs]);

  const origInvalid = originals.filter(f => !isValidImageFile(f));
  const targetInvalid = targets.filter(f => !isValidImageFile(f));

  // ─── Render ───────────────────────────────────────────────────

  return (
    <div className="batch-uploader">
      {/* Selection Phase */}
      {(phase === 'select' || phase === 'pairing') && (
        <>
          <div className="batch-dropzones">
            {/* Original Images */}
            <div className="batch-dropzone-section">
              <h3>Original Images</h3>
              <div
                className={`batch-dropzone ${isDraggingOrig ? 'dragging' : ''}`}
                onDragOver={e => { e.preventDefault(); setIsDraggingOrig(true); }}
                onDragLeave={() => setIsDraggingOrig(false)}
                onDrop={e => handleDrop(e, 'original')}
                onClick={() => origInputRef.current?.click()}
              >
                <div className="dropzone-icon">📁</div>
                <p className="dropzone-text">
                  {originals.length > 0
                    ? <><strong>{originals.length}</strong> original images selected</>
                    : <>Drop original images here</>
                  }
                </p>
                <div className="dropzone-actions">
                  <button className="btn-sm" onClick={e => { e.stopPropagation(); origInputRef.current?.click(); }}>
                    Select Files
                  </button>
                  <button className="btn-sm" onClick={e => { e.stopPropagation(); origFolderRef.current?.click(); }}>
                    Select Folder
                  </button>
                </div>
              </div>
              <input ref={origInputRef} type="file" multiple accept="image/*" hidden
                onChange={e => e.target.files && handleFiles(e.target.files, 'original')} />
              <input ref={origFolderRef} type="file" multiple accept="image/*" hidden
                {...{ webkitdirectory: '', directory: '' } as any}
                onChange={e => e.target.files && handleFiles(e.target.files, 'original')} />
            </div>

            {/* Target Images */}
            <div className="batch-dropzone-section">
              <h3>Target Crop Images</h3>
              <div
                className={`batch-dropzone ${isDraggingTarget ? 'dragging' : ''}`}
                onDragOver={e => { e.preventDefault(); setIsDraggingTarget(true); }}
                onDragLeave={() => setIsDraggingTarget(false)}
                onDrop={e => handleDrop(e, 'target')}
                onClick={() => targetInputRef.current?.click()}
              >
                <div className="dropzone-icon">🎯</div>
                <p className="dropzone-text">
                  {targets.length > 0
                    ? <><strong>{targets.length}</strong> target crop images selected</>
                    : <>Drop target images here</>
                  }
                </p>
                <div className="dropzone-actions">
                  <button className="btn-sm" onClick={e => { e.stopPropagation(); targetInputRef.current?.click(); }}>
                    Select Files
                  </button>
                  <button className="btn-sm" onClick={e => { e.stopPropagation(); targetFolderRef.current?.click(); }}>
                    Select Folder
                  </button>
                </div>
              </div>
              <input ref={targetInputRef} type="file" multiple accept="image/*" hidden
                onChange={e => e.target.files && handleFiles(e.target.files, 'target')} />
              <input ref={targetFolderRef} type="file" multiple accept="image/*" hidden
                {...{ webkitdirectory: '', directory: '' } as any}
                onChange={e => e.target.files && handleFiles(e.target.files, 'target')} />
            </div>
          </div>

          {/* File count + clear buttons */}
          {(originals.length > 0 || targets.length > 0) && (
            <div className="batch-file-summary">
              <span>{originals.length} originals, {targets.length} targets</span>
              <div className="batch-file-actions">
                <button className="btn-sm btn-ghost" onClick={() => { setOriginals([]); setTargets([]); setPairs([]); setPhase('select'); }}>
                  Clear All
                </button>
                <button className="btn-primary" onClick={runPairing} disabled={originals.length === 0 || targets.length === 0 || isHashing}>
                  {isHashing ? 'Analyzing...' : 'Pair Images'}
                </button>
              </div>
            </div>
          )}

          {/* Validation warnings */}
          {(origInvalid.length > 0 || targetInvalid.length > 0) && (
            <div className="batch-warning">
              ⚠ {origInvalid.length + targetInvalid.length} invalid files skipped (unsupported format or zero-byte)
            </div>
          )}
        </>
      )}

      {/* Pairing Results */}
      {phase === 'pairing' && pairs.length > 0 && (
        <div className="batch-pairing-results">
          <h3>Pairing Results</h3>
          <div className="batch-stats-grid">
            <div className="batch-stat matched">
              <div className="batch-stat-value">{stats.matched}</div>
              <div className="batch-stat-label">✓ Matched</div>
            </div>
            <div className="batch-stat unmatched">
              <div className="batch-stat-value">{stats.missingTarget + stats.missingOriginal}</div>
              <div className="batch-stat-label">⚠ Unmatched</div>
            </div>
            <div className="batch-stat duplicate">
              <div className="batch-stat-value">{stats.duplicates}</div>
              <div className="batch-stat-label">↻ Duplicates</div>
            </div>
          </div>

          {/* Toggle pairing table */}
          <button className="btn-sm btn-ghost" onClick={() => setShowPairingTable(!showPairingTable)}>
            {showPairingTable ? 'Hide' : 'Show'} Pairing Details
          </button>

          {showPairingTable && (
            <div className="batch-pairing-table-wrapper">
              <table className="batch-pairing-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Stem</th>
                    <th>Original</th>
                    <th>Target</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {pairs.slice(0, 200).map((pair, i) => (
                    <tr key={pair.stem} className={`pair-row pair-${pair.status}`}>
                      <td>{i + 1}</td>
                      <td className="stem-cell">{pair.stem}</td>
                      <td>{pair.original?.name ?? '—'}</td>
                      <td>{pair.target?.name ?? '—'}</td>
                      <td>
                        <span className={`pair-badge badge-${pair.status}`}>
                          {pair.status === 'matched' && '✓ Matched'}
                          {pair.status === 'missing_target' && '⚠ Missing target'}
                          {pair.status === 'missing_original' && '⚠ Missing original'}
                          {pair.status === 'duplicate' && '↻ Duplicate'}
                          {pair.status === 'invalid' && '✕ Invalid'}
                        </span>
                      </td>
                    </tr>
                  ))}
                  {pairs.length > 200 && (
                    <tr><td colSpan={5} style={{ textAlign: 'center', opacity: 0.6 }}>
                      ... and {pairs.length - 200} more
                    </td></tr>
                  )}
                </tbody>
              </table>
            </div>
          )}

          {/* Upload button */}
          {stats.matched > 0 && (
            <div className="batch-upload-action">
              <button className="btn-primary btn-lg" onClick={startUpload}>
                Import {stats.matched} Training Example{stats.matched !== 1 ? 's' : ''}
              </button>
              <button className="btn-sm btn-ghost" onClick={() => { setPairs([]); setPhase('select'); }}>
                Cancel
              </button>
            </div>
          )}
        </div>
      )}

      {/* Upload Progress */}
      {phase === 'uploading' && (
        <div className="batch-progress">
          <h3>Importing Training Examples...</h3>
          <div className="progress-header">
            <span className="progress-count">{uploadProgress.done} / {uploadProgress.total}</span>
            <span className="progress-pct">{uploadProgress.total > 0 ? ((uploadProgress.done / uploadProgress.total) * 100).toFixed(1) : 0}%</span>
          </div>
          <div className="progress-bar-container">
            <div className="progress-bar-fill" style={{ width: `${uploadProgress.total > 0 ? (uploadProgress.done / uploadProgress.total) * 100 : 0}%` }} />
          </div>
          <div className="progress-stats">
            <span className="stat-success">✓ Successful: {uploadProgress.success}</span>
            <span className="stat-review">⚠ Review: {uploadProgress.review}</span>
            <span className="stat-failed">✕ Failed: {uploadProgress.failed}</span>
          </div>

          {/* Live pair status (virtualized: show last 10) */}
          <div className="batch-live-status">
            {pairs.filter(p => p.uploadStatus !== 'pending').slice(-10).map(pair => (
              <div key={pair.stem} className={`live-pair live-${pair.uploadStatus}`}>
                <span className="live-icon">
                  {pair.uploadStatus === 'uploading' && '⏳'}
                  {pair.uploadStatus === 'success' && '✓'}
                  {pair.uploadStatus === 'failed' && '✕'}
                  {pair.uploadStatus === 'skipped' && '↻'}
                </span>
                <span className="live-name">{pair.stem}</span>
              </div>
            ))}
          </div>

          <button className="btn-sm btn-ghost" onClick={() => { stopRef.current = true; }}>
            Stop Upload
          </button>
        </div>
      )}

      {/* Complete Phase */}
      {phase === 'complete' && (
        <div className="batch-complete">
          <h3>Batch Import Complete</h3>
          <div className="batch-complete-stats">
            <div className="complete-stat">
              <div className="complete-value">{uploadProgress.total}</div>
              <div className="complete-label">Total</div>
            </div>
            <div className="complete-stat success">
              <div className="complete-value">{uploadProgress.success}</div>
              <div className="complete-label">Verified</div>
            </div>
            <div className="complete-stat review">
              <div className="complete-value">{uploadProgress.review}</div>
              <div className="complete-label">Needs Review</div>
            </div>
            <div className="complete-stat failed">
              <div className="complete-value">{uploadProgress.failed}</div>
              <div className="complete-label">Failed</div>
            </div>
          </div>

          <div className="batch-complete-actions">
            {uploadProgress.review > 0 && (
              <a href={`/dataset?status=pending&batch_id=${batchId}`} className="btn-primary">
                Review Low Confidence ({uploadProgress.review})
              </a>
            )}
            <a href="/dataset" className="btn-secondary">
              View Dataset
            </a>
            {uploadProgress.failed > 0 && (
              <button className="btn-sm btn-ghost" onClick={() => {
                setPairs(prev => prev.map(p => p.uploadStatus === 'failed' ? { ...p, uploadStatus: 'pending' as const } : p));
                setPhase('uploading');
                startUpload();
              }}>
                Retry Failed ({uploadProgress.failed})
              </button>
            )}
            <button className="btn-sm btn-ghost" onClick={() => {
              setPhase('select');
              setOriginals([]);
              setTargets([]);
              setPairs([]);

            }}>
              New Batch
            </button>
          </div>

          {/* Batch ID */}
          <div className="batch-id-display">
            <span className="batch-id-label">Batch ID:</span>
            <code>{batchId}</code>
          </div>
        </div>
      )}
    </div>
  );
}
