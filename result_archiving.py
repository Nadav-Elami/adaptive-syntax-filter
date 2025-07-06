from __future__ import annotations

"""Helper utilities for storing and retrieving pipeline artifacts.

Directory layout
----------------
Artifacts are stored under a root *archive* directory (by default `results/`) in
folders organised as::

    {date_run}/{experiment_id}/{stage}/{artifact_files}

Where
    date_run      ``YYYYMMDD`` local date when the pipeline was executed
    experiment_id Free-form identifier set by the caller (e.g. hyperparam grid key)
    stage         Processing stage (e.g. raw, processed, figures, metrics)

The module exposes two high-level helper functions:

* :pyfunc:`stash_artifact` – copy a file (or directory) into the archive and
  return its new path.
* :pyfunc:`retrieve_artifacts` – yield paths matching query parameters.
"""

from pathlib import Path
from datetime import datetime
import logging
import shutil
from typing import Iterable, Iterator, Optional, List

logger = logging.getLogger(__name__)

ARCHIVE_ROOT = Path("results")

# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------

def _build_target_path(
    date_run: str | None,
    experiment_id: str,
    stage: str,
    artifact_name: str,
    create_dirs: bool = True,
) -> Path:
    date_str = date_run or datetime.now().strftime("%Y%m%d")
    target_dir = ARCHIVE_ROOT / date_str / experiment_id / stage
    if create_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / artifact_name


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def stash_artifact(
    src: str | Path,
    *,
    experiment_id: str,
    stage: str = "raw",
    date_run: str | None = None,
    tag: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Copy *src* into the archive tree and return its destination path.

    Args:
        src: Path to file or directory to be stored.
        experiment_id: Identifier for the experiment/run.
        stage: Processing stage (default ``raw``).
        date_run: Override date folder (``YYYYMMDD``). If ``None`` uses today.
        tag: Optional tag appended to filename stem before extension.
        overwrite: Overwrite target if exists (default ``False``).

    Returns:
        Destination path inside the archive.
    """
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(src)

    name = src_path.name
    if tag:
        name = f"{src_path.stem}_{tag}{src_path.suffix}"

    dest = _build_target_path(date_run, experiment_id, stage, name, create_dirs=True)

    if dest.exists():
        if overwrite:
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        else:
            raise FileExistsError(dest)

    if src_path.is_dir():
        shutil.copytree(src_path, dest)
    else:
        shutil.copy2(src_path, dest)

    logger.info("Artifact stashed: %s -> %s", src_path, dest)
    return dest


def retrieve_artifacts(
    *,
    experiment_id: Optional[str] = None,
    stage: Optional[str] = None,
    date_run: Optional[str] = None,
) -> Iterator[Path]:
    """Yield artifact paths matching *experiment_id*, *stage* and *date_run*.

    Args:
        experiment_id: If provided, filter by experiment directory.
        stage: Filter by processing stage.
        date_run: Filter by date folder ``YYYYMMDD``.

    Yields:
        Paths to matching artifact files.
    """
    root = ARCHIVE_ROOT
    if date_run:
        root = root / date_run
    if experiment_id:
        root = root / experiment_id
    if stage:
        root = root / stage

    if not root.exists():
        logger.warning("Archive path not found: %s", root)
        return iter([])  # empty iterator

    return (p for p in root.rglob("*") if p.is_file())


# Convenience wrapper ----------------------------------------------------------

def list_artifacts(**kwargs) -> List[Path]:
    """Return list wrapper around :pyfunc:`retrieve_artifacts`."""
    return list(retrieve_artifacts(**kwargs)) 