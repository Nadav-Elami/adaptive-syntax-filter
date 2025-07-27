from __future__ import annotations

"""Batch runner for Adaptive Syntax Filter experiments.

This utility executes the analysis pipeline over *multiple* configuration
files, either collected from a directory or specified in a manifest CSV. It is
intended for large-scale experiments and high-throughput parameter sweeps.

Key characteristics
-------------------
• Uses *process* pool (`concurrent.futures.ProcessPoolExecutor`) by default for
  isolation and full CPU utilisation. Switch to serial execution with
  ``--serial``.
• Graceful handling of *Ctrl+C*: incomplete futures are cancelled and the pool
  is shut down cleanly.
• Per-config logging routed through the pipeline itself. Batch-level statistics
  summarised once all jobs finish.

The script can be invoked directly (``python batch_processing.py ...``) or via
`cli.py` in a future release.
"""

from csv import DictReader
from pathlib import Path
import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, CancelledError
from typing import List, Sequence

import research_pipeline  # local import – provides run_pipeline

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _discover_configs(input_path: Path) -> List[Path]:
    """Return all config paths under *input_path* (recursive)."""
    exts = {".yml", ".yaml", ".json"}
    return [p for p in input_path.rglob("*") if p.suffix.lower() in exts]


def _parse_manifest(csv_path: Path) -> List[Path]:
    """Extract config paths from a manifest CSV (column ``config_path``)."""
    with csv_path.open(newline="", encoding="utf-8") as fp:
        reader = DictReader(fp)
        if "config_path" not in reader.fieldnames:
            raise ValueError("Manifest must contain a 'config_path' column")
        return [Path(row["config_path"]).expanduser() for row in reader]


def _run_single(config_path: Path) -> bool:
    """Wrapper so that it can be pickled for *process* pool."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return research_pipeline.run_pipeline(config_path)


# -----------------------------------------------------------------------------
# Main execution logic
# -----------------------------------------------------------------------------

def _execute_batch(configs: Sequence[Path], max_workers: int, serial: bool) -> None:
    start = time.time()
    successes: List[Path] = []
    failures: List[Path] = []

    if serial or max_workers == 1:
        logger.info("Running %d jobs serially", len(configs))
        for cfg in configs:
            ok = _run_single(cfg)
            (successes if ok else failures).append(cfg)
    else:
        logger.info("Running %d jobs in parallel (%d workers)", len(configs), max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_cfg = {executor.submit(_run_single, c): c for c in configs}
            try:
                for future in as_completed(future_to_cfg):
                    cfg = future_to_cfg[future]
                    try:
                        ok = future.result()
                        (successes if ok else failures).append(cfg)
                        status = "✅" if ok else "❌"
                        logger.info("%s %s", status, cfg)
                    except Exception as exc:  # pylint: disable=broad-except
                        failures.append(cfg)
                        logger.error("❌ %s failed – %s", cfg, exc)
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt received – cancelling remaining jobs …")
                for fut in future_to_cfg:
                    fut.cancel()
                # Give the executor a moment to cancel futures
                executor.shutdown(cancel_futures=True)
                raise

    duration = time.time() - start
    logger.info("Batch finished in %.1f s (%d success / %d failure)", duration, len(successes), len(failures))

    if failures:
        logger.error("Some jobs failed. See log for details.")
        sys.exit(2)


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-mode execution of Adaptive Syntax Filter pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-folder", type=Path, help="Directory containing config files (recursive search)")
    src.add_argument("--manifest", type=Path, help="CSV listing config_path column")

    p.add_argument("--max-workers", type=int, default=4, help="Maximum parallel processes")
    p.add_argument("--serial", action="store_true", help="Force serial execution (overrides --max-workers)")
    return p


def main(argv: List[str] | None = None) -> None:  # noqa: D401
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args = _build_parser().parse_args(argv)

    if args.input_folder:
        configs = _discover_configs(args.input_folder)
        if not configs:
            logger.error("No config files found in %s", args.input_folder)
            sys.exit(1)
    else:  # manifest path is provided
        configs = _parse_manifest(args.manifest)
        if not configs:
            logger.error("Manifest contains no rows")
            sys.exit(1)

    try:
        _execute_batch(configs, max_workers=args.max_workers, serial=args.serial)
    except KeyboardInterrupt:
        logger.info("Batch interrupted by user")
        sys.exit(130)  # 128 + SIGINT


if __name__ == "__main__":
    main() 