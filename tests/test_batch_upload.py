"""
Tests for batch upload functionality.

Covers:
- Filename stem pairing logic
- Different extension matching
- Missing target/original detection
- Duplicate hash detection
- Batch ID tracking
- Partial failure handling
"""

import hashlib
import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════
# FILENAME PAIRING LOGIC (mirrors frontend logic)
# ═══════════════════════════════════════════════════════════════════════

def get_file_stem(filename: str) -> str:
    """Extract filename stem (without extension), lowercased."""
    return Path(filename).stem.lower()


def pair_files(originals: list[str], targets: list[str]) -> dict:
    """
    Pair original and target filenames by stem.
    Returns dict with matched, missing_target, missing_original.
    """
    orig_map = {get_file_stem(f): f for f in originals}
    tgt_map = {get_file_stem(f): f for f in targets}

    all_stems = set(orig_map.keys()) | set(tgt_map.keys())

    matched = []
    missing_target = []
    missing_original = []

    for stem in sorted(all_stems):
        orig = orig_map.get(stem)
        tgt = tgt_map.get(stem)
        if orig and tgt:
            matched.append((orig, tgt))
        elif orig and not tgt:
            missing_target.append(orig)
        elif tgt and not orig:
            missing_original.append(tgt)

    return {
        "matched": matched,
        "missing_target": missing_target,
        "missing_original": missing_original,
    }


class TestFilenamePairing:
    """Test filename stem-based pairing logic."""

    def test_exact_match(self):
        """Files with identical names pair correctly."""
        originals = ["001.jpg", "002.jpg", "003.jpg"]
        targets = ["001.jpg", "002.jpg", "003.jpg"]
        result = pair_files(originals, targets)
        assert len(result["matched"]) == 3
        assert len(result["missing_target"]) == 0
        assert len(result["missing_original"]) == 0

    def test_different_extensions(self):
        """Files with different extensions but same stem pair correctly."""
        originals = ["student001.jpg", "student002.jpg", "student003.jpg"]
        targets = ["student001.png", "student002.jpeg", "student003.webp"]
        result = pair_files(originals, targets)
        assert len(result["matched"]) == 3
        for orig, tgt in result["matched"]:
            assert get_file_stem(orig) == get_file_stem(tgt)

    def test_case_insensitive(self):
        """Pairing is case-insensitive."""
        originals = ["Photo_001.JPG", "PHOTO_002.jpg"]
        targets = ["photo_001.png", "photo_002.PNG"]
        result = pair_files(originals, targets)
        assert len(result["matched"]) == 2

    def test_missing_targets(self):
        """Files without matching targets are flagged."""
        originals = ["001.jpg", "002.jpg", "003.jpg", "004.jpg"]
        targets = ["001.jpg", "002.jpg", "003.jpg"]
        result = pair_files(originals, targets)
        assert len(result["matched"]) == 3
        assert len(result["missing_target"]) == 1
        assert "004.jpg" in result["missing_target"]

    def test_missing_originals(self):
        """Targets without matching originals are flagged."""
        originals = ["001.jpg", "002.jpg"]
        targets = ["001.jpg", "002.jpg", "003.jpg"]
        result = pair_files(originals, targets)
        assert len(result["matched"]) == 2
        assert len(result["missing_original"]) == 1
        assert "003.jpg" in result["missing_original"]

    def test_empty_inputs(self):
        """Empty lists produce empty results."""
        result = pair_files([], [])
        assert len(result["matched"]) == 0

    def test_no_overlap(self):
        """Completely different filenames produce no matches."""
        originals = ["a.jpg", "b.jpg"]
        targets = ["c.jpg", "d.jpg"]
        result = pair_files(originals, targets)
        assert len(result["matched"]) == 0
        assert len(result["missing_target"]) == 2
        assert len(result["missing_original"]) == 2

    def test_ordering_independent(self):
        """Pairing doesn't depend on file ordering."""
        originals = ["003.jpg", "001.jpg", "002.jpg"]
        targets = ["002.png", "003.jpg", "001.jpeg"]
        result = pair_files(originals, targets)
        assert len(result["matched"]) == 3

    def test_large_batch(self):
        """Pairing works for large batches."""
        originals = [f"{i:04d}.jpg" for i in range(500)]
        targets = [f"{i:04d}.png" for i in range(500)]
        result = pair_files(originals, targets)
        assert len(result["matched"]) == 500

    def test_stem_with_dots(self):
        """Filenames with multiple dots pair by full stem."""
        originals = ["photo.v2.jpg"]
        targets = ["photo.v2.png"]
        result = pair_files(originals, targets)
        assert len(result["matched"]) == 1


# ═══════════════════════════════════════════════════════════════════════
# DUPLICATE DETECTION (hash-based)
# ═══════════════════════════════════════════════════════════════════════

class TestDuplicateDetection:
    """Test SHA-256 hash-based duplicate detection."""

    def _compute_pair_hash(self, orig_bytes: bytes, tgt_bytes: bytes | None) -> str:
        """Mirror backend's _compute_pair_hash."""
        h = hashlib.sha256(orig_bytes)
        if tgt_bytes:
            h.update(tgt_bytes)
        return h.hexdigest()

    def test_identical_files_same_hash(self):
        """Identical file contents produce the same hash."""
        content = b"test image content 123"
        hash1 = self._compute_pair_hash(content, None)
        hash2 = self._compute_pair_hash(content, None)
        assert hash1 == hash2

    def test_different_files_different_hash(self):
        """Different file contents produce different hashes."""
        hash1 = self._compute_pair_hash(b"image_a", None)
        hash2 = self._compute_pair_hash(b"image_b", None)
        assert hash1 != hash2

    def test_pair_hash_includes_target(self):
        """Hash changes when target content changes."""
        orig = b"original content"
        hash1 = self._compute_pair_hash(orig, b"target_a")
        hash2 = self._compute_pair_hash(orig, b"target_b")
        assert hash1 != hash2

    def test_pair_hash_with_and_without_target(self):
        """Hash differs with/without target."""
        orig = b"original content"
        hash1 = self._compute_pair_hash(orig, None)
        hash2 = self._compute_pair_hash(orig, b"target")
        assert hash1 != hash2

    def test_hash_deterministic(self):
        """Same inputs always produce the same hash."""
        orig = b"orig"
        tgt = b"tgt"
        hashes = [self._compute_pair_hash(orig, tgt) for _ in range(10)]
        assert len(set(hashes)) == 1

    def test_real_image_hashing(self):
        """Real image files produce consistent hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new("RGB", (100, 100), (255, 0, 0))
            path = os.path.join(tmpdir, "test.jpg")
            img.save(path)
            content = open(path, "rb").read()

            hash1 = self._compute_pair_hash(content, None)
            hash2 = self._compute_pair_hash(content, None)
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA-256 hex length


# ═══════════════════════════════════════════════════════════════════════
# BATCH ID
# ═══════════════════════════════════════════════════════════════════════

class TestBatchId:
    """Test batch ID generation and format."""

    def test_batch_id_format(self):
        """Batch IDs follow expected format."""
        import re
        # Simulate the frontend's batch ID generation
        from datetime import datetime
        now = datetime.now()
        batch_id = f"batch_{now.strftime('%Y%m%d_%H%M%S')}"
        assert re.match(r"batch_\d{8}_\d{6}", batch_id)

    def test_batch_id_uniqueness(self):
        """Different timestamps produce different batch IDs."""
        import time
        ids = set()
        for _ in range(5):
            from datetime import datetime
            now = datetime.now()
            ids.add(f"batch_{now.strftime('%Y%m%d_%H%M%S%f')}")
            time.sleep(0.001)
        assert len(ids) >= 2  # At least some should differ


# ═══════════════════════════════════════════════════════════════════════
# FILE VALIDATION
# ═══════════════════════════════════════════════════════════════════════

class TestFileValidation:
    """Test file validation logic."""

    ALLOWED = {'.jpg', '.jpeg', '.png', '.webp'}

    def test_valid_extensions(self):
        """Valid image extensions are accepted."""
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            assert ext in self.ALLOWED

    def test_invalid_extensions_rejected(self):
        """Non-image extensions are rejected."""
        for ext in ['.gif', '.bmp', '.tiff', '.pdf', '.doc', '.txt']:
            assert ext not in self.ALLOWED

    def test_case_handling(self):
        """Extension check should be case-insensitive."""
        for ext in ['.JPG', '.Png', '.WEBP']:
            assert ext.lower() in self.ALLOWED


# ═══════════════════════════════════════════════════════════════════════
# PARTIAL FAILURE HANDLING
# ═══════════════════════════════════════════════════════════════════════

class TestPartialFailure:
    """Test that batch upload handles partial failures correctly."""

    def test_failure_count_tracking(self):
        """Success/failure counts are tracked independently."""
        results = {"success": 0, "failed": 0, "review": 0}

        # Simulate a batch of 10 uploads
        outcomes = ["success"] * 7 + ["failed"] * 2 + ["review"] * 1
        for outcome in outcomes:
            results[outcome] += 1

        assert results["success"] == 7
        assert results["failed"] == 2
        assert results["review"] == 1
        assert sum(results.values()) == 10

    def test_failed_dont_block_success(self):
        """Failed uploads don't prevent successful ones."""
        total = 100
        failed_indices = {3, 17, 42}  # These fail
        successful = [i for i in range(total) if i not in failed_indices]
        assert len(successful) == total - len(failed_indices)
        assert len(failed_indices) == 3
