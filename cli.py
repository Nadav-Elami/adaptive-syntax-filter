import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def _setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> None:
    """Configure root logger to log to stdout and a log file.

    Args:
        verbose: If ``True`` set console log level to ``DEBUG`` else ``INFO``.
        log_file: Optional path to a log file. If ``None`` a timestamped file is
            created under ``logs/``.
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"cli_{timestamp}.log"

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    # File handler (always DEBUG for maximum detail)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(file_handler)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers)
    logging.debug("Logging initialised. Log file: %s", log_file)


def _load_config(config_path: Path | str) -> Dict[str, Any]:
    """Load JSON or YAML configuration.

    Args:
        config_path: Path to a ``.json``, ``.yml`` or ``.yaml`` file.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() in {".json"}:
        return json.loads(path.read_text())
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyYAML is required for YAML config files. Install with 'pip install PyYAML'"
            ) from exc
        config = yaml.safe_load(path.read_text())
        
        # Ensure numeric values are properly typed (YAML sometimes loads them as strings)
        def _convert_numeric_values(obj):
            if isinstance(obj, dict):
                return {k: _convert_numeric_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert_numeric_values(item) for item in obj]
            elif isinstance(obj, str):
                # Try to convert string to float/int if it looks like a number
                try:
                    if '.' in obj or 'e' in obj.lower():
                        return float(obj)
                    else:
                        return int(obj)
                except ValueError:
                    return obj
            else:
                return obj
        
        return _convert_numeric_values(config)

    raise ValueError(f"Unsupported config type: {path.suffix}")


# -----------------------------------------------------------------------------
# Sub-command implementations
# -----------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> int:
    """Entry point for the ``run`` sub-command."""
    logger = logging.getLogger(__name__)
    logger.info("Starting pipeline run – config: %s", args.config)

    try:
        config = _load_config(args.config)
    except Exception as exc:
        logger.error("Failed to load configuration: %s", exc)
        return 1

    try:
        from research_pipeline import run_pipeline  # lazy import

        success: bool = run_pipeline(config)  # type: ignore[arg-type]
        logger.info("Pipeline finished – success=%s", success)
        return 0 if success else 2
    except ImportError as exc:
        logger.error(
            "research_pipeline.py not available or failed to import (%s). "
            "Stub exit.",
            exc,
        )
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Pipeline execution failed: %s", exc)
        return 1


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Entry point for the ``evaluate`` sub-command."""
    logger = logging.getLogger(__name__)
    logger.info("Running evaluation …")

    try:
        from research_pipeline import evaluate_results  # type: ignore

        success: bool = evaluate_results(args.results_dir)
        logger.info("Evaluation complete – success=%s", success)
        return 0 if success else 2
    except ImportError as exc:
        logger.error("research_pipeline.evaluate_results missing: %s", exc)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Evaluation failed: %s", exc)
        return 1


def _cmd_export(args: argparse.Namespace) -> int:
    """Entry point for the ``export`` sub-command."""
    logger = logging.getLogger(__name__)
    logger.info("Exporting figures: output_dir=%s", args.output_dir)

    try:
        from export_cli import export_figures  # type: ignore

        count: int = export_figures(args.output_dir, fmt=args.format)
        logger.info("Exported %d figure(s)", count)
        return 0
    except ImportError as exc:
        logger.error("export_cli.export_figures missing: %s", exc)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Figure export failed: %s", exc)
        return 1


def _cmd_clean(args: argparse.Namespace) -> int:
    """Entry point for the ``clean`` sub-command."""
    logger = logging.getLogger(__name__)
    logger.info("Cleaning temporary files and caches")

    patterns = [
        "**/__pycache__",
        "**/*.py[cod]",
        "**/*.log",
        "output",
        "logs",
    ]

    for pattern in patterns:
        for path in Path(".").glob(pattern):
            try:
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    for sub in path.rglob("*"):
                        if sub.is_file():
                            sub.unlink()
                    path.rmdir()
                logger.debug("Removed %s", path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Could not remove %s: %s", path, exc)

    logger.info("Clean complete")
    return 0


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adaptive Syntax Filter command-line interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging output.",
    )

    sub_parsers = parser.add_subparsers(dest="command", required=True)

    # run ---------------------------------------------------------------------
    run_parser = sub_parsers.add_parser("run", help="Execute full analysis pipeline")
    run_parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML/JSON pipeline configuration file.",
    )
    run_parser.set_defaults(func=_cmd_run)

    # evaluate ----------------------------------------------------------------
    eval_parser = sub_parsers.add_parser("evaluate", help="Evaluate existing results")
    eval_parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing pipeline outputs to evaluate.",
    )
    eval_parser.set_defaults(func=_cmd_evaluate)

    # export ------------------------------------------------------------------
    export_parser = sub_parsers.add_parser("export", help="Export publication-ready figures")
    export_parser.add_argument(
        "output_dir",
        type=str,
        help="Destination directory for exported figures.",
    )
    export_parser.add_argument(
        "--format",
        choices=["png", "svg", "pdf"],
        default="png",
        help="Output format.",
    )
    export_parser.set_defaults(func=_cmd_export)

    # clean -------------------------------------------------------------------
    clean_parser = sub_parsers.add_parser("clean", help="Remove temporary files and caches")
    clean_parser.set_defaults(func=_cmd_clean)

    return parser


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:  # noqa: D401, D401 is about docstring style
    """CLI entry point. Parse ``argv`` and dispatch to sub-command implementation."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Logging must be set up *after* parsing to respect --verbose flag.
    _setup_logging(verbose=args.verbose)

    exit_code = args.func(args)  # type: ignore[attr-defined]
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 