/** API types matching backend Pydantic schemas. */

export interface CropRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface NormalizedCropRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface TrainingExample {
  id: string;
  created_at: string;
  updated_at: string;
  original_filename: string;
  original_path: string;
  original_width: number;
  original_height: number;
  target_filename: string | null;
  target_path: string | null;
  target_width: number | null;
  target_height: number | null;
  crop_x: number | null;
  crop_y: number | null;
  crop_width: number | null;
  crop_height: number | null;
  norm_x: number | null;
  norm_y: number | null;
  norm_width: number | null;
  norm_height: number | null;
  extraction_method: string;
  confidence: number | null;
  split: string;
  status: string;
  batch_id: string | null;
  file_hash: string | null;
  notes: string | null;
}

export interface DatasetStats {
  total: number;
  labeled: number;
  pending: number;
  train: number;
  validation: number;
  test: number;
  unassigned: number;
}

export interface DatasetResponse {
  total: number;
  examples: TrainingExample[];
  stats: DatasetStats;
  batch_ids: string[];
}

export interface TrainingConfig {
  backbone: string;
  image_size: number;
  batch_size: number;
  learning_rate: number;
  epochs: number;
  optimizer: string;
  loss_function: string;
}

export interface TrainingStatus {
  status: string;
  run_id: string | null;
  epoch: number;
  total_epochs: number;
  train_loss: number | null;
  val_loss: number | null;
  best_val_loss: number | null;
  device: string | null;
  progress: number;
}

export interface TrainingRunHistory {
  id: string;
  created_at: string;
  completed_at: string | null;
  model_version: string;
  backbone: string;
  image_size: number;
  batch_size: number;
  learning_rate: number;
  epochs: number;
  train_samples: number;
  val_samples: number;
  test_samples: number;
  status: string;
  current_epoch: number;
  train_loss: number | null;
  val_loss: number | null;
  best_val_loss: number | null;
  device: string | null;
  mean_iou: number | null;
}

export interface ModelInfo {
  id: string;
  version: string;
  created_at: string;
  backbone: string;
  image_size: number;
  val_loss: number | null;
  mean_iou: number | null;
  is_active: boolean;
  dataset_size: number;
}

export interface ModelStatus {
  has_model: boolean;
  active_version: string | null;
  versions: ModelInfo[];
  device: string | null;
}

export interface PredictionResponse {
  id: string;
  image_filename: string;
  image_path: string;
  image_width: number;
  image_height: number;
  model_version: string | null;
  pred_x: number | null;
  pred_y: number | null;
  pred_width: number | null;
  pred_height: number | null;
  pred_crop_x: number | null;
  pred_crop_y: number | null;
  pred_crop_width: number | null;
  pred_crop_height: number | null;
}

export interface HealthCheck {
  status: string;
  device: string;
  version: string;
}

/** Batch upload types */
export type PairStatus = 'matched' | 'missing_target' | 'missing_original' | 'duplicate' | 'invalid';

export interface FilePair {
  stem: string;
  original: File | null;
  target: File | null;
  status: PairStatus;
  uploadStatus: 'pending' | 'uploading' | 'success' | 'failed' | 'skipped';
  error?: string;
  result?: TrainingExample;
}
