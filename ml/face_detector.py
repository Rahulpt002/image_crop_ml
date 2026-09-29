"""
Face detection module with replaceable detector interface.

Provides:
    - FaceDetection: dataclass for detection results
    - FaceDetector: abstract base class (swap implementations freely)
    - MediaPipeFaceDetector: lightweight detector suitable for M2 Mac
    - get_face_detector(): factory that returns the configured detector

The face detector is used during inference to provide face/head location
as context for the learned cropping model. It is NOT the primary cropping
mechanism — CropNet handles that.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """A single detected face in an image."""
    x: int              # top-left x in absolute pixels
    y: int              # top-left y in absolute pixels
    width: int          # box width in pixels
    height: int         # box height in pixels
    confidence: float   # 0–1
    landmarks: dict = field(default_factory=dict)  # optional: eyes, nose, etc.

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
        }


class FaceDetector(ABC):
    """
    Abstract face detector interface.

    Swap implementations by subclassing:
        - MediaPipeFaceDetector (default, lightweight)
        - OpenCVFaceDetector (Haar/DNN-based)
        - RetinaFaceDetector (higher accuracy, heavier)

    All implementations must return a list of FaceDetection objects
    sorted by confidence descending.
    """

    @abstractmethod
    def detect(self, image: Image.Image) -> List[FaceDetection]:
        """
        Detect faces in a PIL Image.

        Args:
            image: RGB PIL Image (EXIF already applied)

        Returns:
            List of FaceDetection sorted by confidence (highest first).
            Empty list if no faces detected.
        """
        ...

    def detect_primary(self, image: Image.Image) -> Optional[FaceDetection]:
        """Convenience: return the highest-confidence face, or None."""
        detections = self.detect(image)
        return detections[0] if detections else None


class MediaPipeFaceDetector(FaceDetector):
    """
    Face detector using Google MediaPipe.

    Lightweight, runs on CPU, optimized for M-series Macs.
    Uses the short-range model (for faces within ~2m of camera).
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._detector = None

    def _get_detector(self):
        """Lazy-init the MediaPipe detector (avoids import cost at startup)."""
        if self._detector is None:
            import mediapipe as mp

            base_options = mp.tasks.BaseOptions(
                model_asset_path=self._get_model_path(),
            )
            options = mp.tasks.vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=self.min_confidence,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
            )
            self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)

        return self._detector

    def _get_model_path(self) -> str:
        """Get the MediaPipe face detection model path, downloading if needed."""
        import os
        from pathlib import Path

        # Store model in the project's models directory
        models_dir = Path(__file__).parent.parent / "models" / "mediapipe"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "blaze_face_short_range.tflite"

        if not model_path.exists():
            logger.info("Downloading MediaPipe face detection model...")
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(url, str(model_path))
            logger.info(f"Model saved to {model_path}")

        return str(model_path)

    def detect(self, image: Image.Image) -> List[FaceDetection]:
        """Detect faces using MediaPipe."""
        import mediapipe as mp

        detector = self._get_detector()

        # Convert PIL Image to MediaPipe Image
        img_array = np.array(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

        # Run detection
        result = detector.detect(mp_image)

        img_w, img_h = image.size
        detections = []

        for det in result.detections:
            bbox = det.bounding_box
            x = max(0, bbox.origin_x)
            y = max(0, bbox.origin_y)
            w = min(bbox.width, img_w - x)
            h = min(bbox.height, img_h - y)

            confidence = det.categories[0].score if det.categories else 0.0

            # Extract landmarks if available
            landmarks = {}
            if det.keypoints:
                landmark_names = [
                    "right_eye", "left_eye", "nose_tip",
                    "mouth_center", "right_ear_tragion", "left_ear_tragion",
                ]
                for i, kp in enumerate(det.keypoints):
                    if i < len(landmark_names):
                        landmarks[landmark_names[i]] = {
                            "x": int(kp.x * img_w),
                            "y": int(kp.y * img_h),
                        }

            detections.append(FaceDetection(
                x=x, y=y, width=w, height=h,
                confidence=confidence,
                landmarks=landmarks,
            ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.info(f"MediaPipe detected {len(detections)} face(s)")
        return detections


class OpenCVHaarFaceDetector(FaceDetector):
    """
    Fallback face detector using OpenCV Haar cascades.

    Less accurate than MediaPipe but requires no extra dependencies
    beyond opencv-python (already installed).
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    def detect(self, image: Image.Image) -> List[FaceDetection]:
        import cv2

        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return []

        detections = []
        for (x, y, w, h) in faces:
            # Haar doesn't provide a true confidence, so use a fixed value
            detections.append(FaceDetection(
                x=int(x), y=int(y), width=int(w), height=int(h),
                confidence=0.8,
            ))

        detections.sort(key=lambda d: d.width * d.height, reverse=True)
        return detections


# ── Factory ─────────────────────────────────────────────────────────

_detector_instance: Optional[FaceDetector] = None


def get_face_detector(backend: str = "mediapipe") -> FaceDetector:
    """
    Get a face detector instance (cached singleton).

    Args:
        backend: "mediapipe" (default) or "opencv"

    Returns:
        A FaceDetector implementation.
    """
    global _detector_instance

    if _detector_instance is not None:
        return _detector_instance

    if backend == "mediapipe":
        _detector_instance = MediaPipeFaceDetector()
    elif backend == "opencv":
        _detector_instance = OpenCVHaarFaceDetector()
    else:
        raise ValueError(f"Unknown face detector backend: {backend}")

    logger.info(f"Initialized face detector: {backend}")
    return _detector_instance
