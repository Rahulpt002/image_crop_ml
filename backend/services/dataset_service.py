"""Dataset management service."""

import logging
import random
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from backend.config.settings import DEFAULT_TRAIN_SPLIT, DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT, SPLIT_RANDOM_SEED
from backend.models.db_models import TrainingExample
from backend.schemas.api_schemas import DatasetStats, DatasetSplitConfig

logger = logging.getLogger(__name__)


def get_dataset_stats(db: Session) -> DatasetStats:
    """Calculate dataset statistics."""
    total = db.query(TrainingExample).count()
    labeled = db.query(TrainingExample).filter(TrainingExample.status == "labeled").count()
    pending = db.query(TrainingExample).filter(TrainingExample.status == "pending").count()
    train = db.query(TrainingExample).filter(TrainingExample.split == "train").count()
    validation = db.query(TrainingExample).filter(TrainingExample.split == "validation").count()
    test = db.query(TrainingExample).filter(TrainingExample.split == "test").count()
    unassigned = db.query(TrainingExample).filter(TrainingExample.split == "unassigned").count()

    return DatasetStats(
        total=total,
        labeled=labeled,
        pending=pending,
        train=train,
        validation=validation,
        test=test,
        unassigned=unassigned,
    )


def assign_splits(
    db: Session,
    config: Optional[DatasetSplitConfig] = None,
) -> DatasetStats:
    """
    Assign train/val/test splits to all labeled examples.
    Uses deterministic random seed to avoid data leakage on re-runs.
    """
    if config is None:
        config = DatasetSplitConfig()

    # Get all labeled examples
    examples = (
        db.query(TrainingExample)
        .filter(TrainingExample.status == "labeled")
        .all()
    )

    if not examples:
        return get_dataset_stats(db)

    # Deterministic shuffle
    rng = random.Random(config.seed)
    ids = [e.id for e in examples]
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)
    # Remaining go to test
    n_test = n - n_train - n_val

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    for example in examples:
        if example.id in train_ids:
            example.split = "train"
        elif example.id in val_ids:
            example.split = "validation"
        else:
            example.split = "test"

    db.commit()
    logger.info(f"Assigned splits: train={n_train}, val={n_val}, test={n_test}")

    return get_dataset_stats(db)


def get_examples_by_split(
    db: Session,
    split: str,
) -> List[TrainingExample]:
    """Get all labeled examples for a given split."""
    return (
        db.query(TrainingExample)
        .filter(
            TrainingExample.status == "labeled",
            TrainingExample.split == split,
        )
        .all()
    )
