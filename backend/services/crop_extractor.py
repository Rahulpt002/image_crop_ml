"""
Crop label extraction service.

Determines the crop rectangle of a target image relative to its original
using feature matching. Designed as a replaceable service.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from backend.config.settings import CROP_CONFIDENCE_THRESHOLD
from backend.services.image_service import load_image_as_array

logger = logging.getLogger(__name__)


@dataclass
class CropExtractionResult:
    """Result of crop label extraction."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    method: str

    def to_normalized(self, orig_width: int, orig_height: int) -> dict:
        """Convert to normalized coordinates."""
        return {
            "x": self.x / orig_width,
            "y": self.y / orig_height,
            "width": self.width / orig_width,
            "height": self.height / orig_height,
        }


class CropLabelExtractor:
    """
    Determines where a cropped image sits within its original.

    Uses a multi-strategy approach:
    1. Template matching (fast, works for exact crops)
    2. Feature matching with SIFT/ORB (robust to scale changes)

    Designed as a replaceable service — swap the implementation
    without changing the interface.
    """

    def __init__(self, confidence_threshold: float = CROP_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def extract_crop(
        self,
        original_path: Path,
        target_path: Path,
    ) -> CropExtractionResult:
        """
        Determine the crop rectangle of target_image within original_image.

        The target may have been resized after cropping, so we try
        multiple strategies.
        """
        original = load_image_as_array(original_path)
        target = load_image_as_array(target_path)

        orig_h, orig_w = original.shape[:2]
        target_h, target_w = target.shape[:2]

        # Strategy 1: Try multi-scale template matching
        result = self._template_match_multiscale(original, target, orig_w, orig_h)
        if result and result.confidence >= self.confidence_threshold:
            logger.info(f"Template matching succeeded: confidence={result.confidence:.3f}")
            return result

        # Strategy 2: Feature-based matching (ORB)
        result = self._feature_match(original, target, orig_w, orig_h)
        if result and result.confidence >= self.confidence_threshold:
            logger.info(f"Feature matching succeeded: confidence={result.confidence:.3f}")
            return result

        # Strategy 3: Try SIFT if available
        result = self._sift_match(original, target, orig_w, orig_h)
        if result and result.confidence >= self.confidence_threshold:
            logger.info(f"SIFT matching succeeded: confidence={result.confidence:.3f}")
            return result

        # Return best attempt with low confidence
        logger.warning("Crop extraction confidence is low — manual correction recommended.")
        if result:
            return result

        # Fallback: center crop estimate based on aspect ratios
        return self._aspect_ratio_fallback(orig_w, orig_h, target_w, target_h)

    def _template_match_multiscale(
        self,
        original: np.ndarray,
        target: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> Optional[CropExtractionResult]:
        """Multi-scale template matching."""
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

        best_val = -1.0
        best_loc = None
        best_scale = 1.0
        target_h, target_w = target_gray.shape[:2]

        # Try scales where target would fit within original
        for scale in np.linspace(0.2, 1.0, 20):
            scaled_w = int(target_w * scale)
            scaled_h = int(target_h * scale)

            if scaled_w >= orig_w or scaled_h >= orig_h:
                continue
            if scaled_w < 10 or scaled_h < 10:
                continue

            resized_target = cv2.resize(target_gray, (scaled_w, scaled_h))

            # Also try matching with the target at its original scale
            # but we need the crop region to be smaller than the original
            result = cv2.matchTemplate(original_gray, resized_target, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale = scale

        if best_loc is None:
            return None

        crop_w = int(target_w * best_scale)
        crop_h = int(target_h * best_scale)
        x, y = best_loc

        # Clamp to image boundaries
        x = max(0, min(x, orig_w - crop_w))
        y = max(0, min(y, orig_h - crop_h))
        crop_w = min(crop_w, orig_w - x)
        crop_h = min(crop_h, orig_h - y)

        # Template matching confidence is the correlation value
        confidence = float(max(0.0, min(1.0, best_val)))

        return CropExtractionResult(
            x=x, y=y, width=crop_w, height=crop_h,
            confidence=confidence, method="template_matching",
        )

    def _feature_match(
        self,
        original: np.ndarray,
        target: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> Optional[CropExtractionResult]:
        """ORB feature-based matching."""
        try:
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

            orb = cv2.ORB_create(nfeatures=2000)
            kp1, des1 = orb.detectAndCompute(original_gray, None)
            kp2, des2 = orb.detectAndCompute(target_gray, None)

            if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
                return None

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des2, des1, k=2)

            # Lowe's ratio test
            good_matches = []
            for m_pair in matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 4:
                return None

            return self._matches_to_crop(
                good_matches, kp2, kp1, orig_w, orig_h, "orb_feature_matching"
            )

        except Exception as e:
            logger.warning(f"Feature matching failed: {e}")
            return None

    def _sift_match(
        self,
        original: np.ndarray,
        target: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> Optional[CropExtractionResult]:
        """SIFT feature-based matching (if available)."""
        try:
            sift = cv2.SIFT_create(nfeatures=2000)
        except cv2.error:
            return None

        try:
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

            kp1, des1 = sift.detectAndCompute(original_gray, None)
            kp2, des2 = sift.detectAndCompute(target_gray, None)

            if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
                return None

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(des2, des1, k=2)

            good_matches = []
            for m_pair in matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 4:
                return None

            return self._matches_to_crop(
                good_matches, kp2, kp1, orig_w, orig_h, "sift_feature_matching"
            )

        except Exception as e:
            logger.warning(f"SIFT matching failed: {e}")
            return None

    def _matches_to_crop(
        self,
        matches,
        kp_target,
        kp_original,
        orig_w: int,
        orig_h: int,
        method: str,
    ) -> Optional[CropExtractionResult]:
        """Convert feature matches to a crop rectangle."""
        src_pts = np.float32([kp_target[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_original[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        inliers = mask.ravel().sum()
        confidence = float(inliers / len(matches))

        # Map target corners to original image space
        target_h, target_w = int(max(p[0][1] for p in src_pts)), int(max(p[0][0] for p in src_pts))
        corners = np.float32([
            [0, 0], [target_w, 0], [target_w, target_h], [0, target_h]
        ]).reshape(-1, 1, 2)

        mapped_corners = cv2.perspectiveTransform(corners, H)
        mapped = mapped_corners.reshape(-1, 2)

        x_min = max(0, int(np.min(mapped[:, 0])))
        y_min = max(0, int(np.min(mapped[:, 1])))
        x_max = min(orig_w, int(np.max(mapped[:, 0])))
        y_max = min(orig_h, int(np.max(mapped[:, 1])))

        crop_w = x_max - x_min
        crop_h = y_max - y_min

        if crop_w <= 0 or crop_h <= 0:
            return None

        return CropExtractionResult(
            x=x_min, y=y_min, width=crop_w, height=crop_h,
            confidence=confidence, method=method,
        )

    def _aspect_ratio_fallback(
        self,
        orig_w: int,
        orig_h: int,
        target_w: int,
        target_h: int,
    ) -> CropExtractionResult:
        """Fallback: estimate a center crop matching the target aspect ratio."""
        target_ratio = target_w / target_h

        if orig_w / orig_h > target_ratio:
            # Original is wider — constrain by height
            crop_h = orig_h
            crop_w = int(crop_h * target_ratio)
        else:
            # Original is taller — constrain by width
            crop_w = orig_w
            crop_h = int(crop_w / target_ratio)

        x = (orig_w - crop_w) // 2
        y = (orig_h - crop_h) // 2

        return CropExtractionResult(
            x=x, y=y, width=crop_w, height=crop_h,
            confidence=0.1,  # Low confidence — manual review needed
            method="aspect_ratio_fallback",
        )
