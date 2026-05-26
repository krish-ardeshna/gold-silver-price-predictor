from dataclasses import dataclass

DEFAULT_TRAIN_RATIO = 0.75
DEFAULT_VAL_RATIO = 0.15


@dataclass(frozen=True)
class TimeSplit:
    train_end: int
    val_end: int
    total_rows: int
    train_ratio: float
    val_ratio: float

    @property
    def test_start(self) -> int:
        return self.val_end

    @property
    def test_rows(self) -> int:
        return self.total_rows - self.val_end

    @property
    def train_rows(self) -> int:
        return self.train_end

    @property
    def val_rows(self) -> int:
        return self.val_end - self.train_end


def compute_time_split(
    total_rows: int,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> TimeSplit:
    """Create deterministic time-based split boundaries."""
    if total_rows < 3:
        raise ValueError("Need at least 3 rows for train/validation/test splits.")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must leave room for a test split.")

    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    if train_end <= 0:
        raise ValueError("Training split is empty.")

    if val_end <= train_end:
        raise ValueError("Validation split is empty.")

    if val_end >= total_rows:
        raise ValueError("Test split is empty.")

    return TimeSplit(
        train_end=train_end,
        val_end=val_end,
        total_rows=total_rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
