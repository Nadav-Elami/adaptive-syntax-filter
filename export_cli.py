from __future__ import annotations

"""Figure export utility for Adaptive Syntax Filter.

This module auto-discovers figure-generating callables in ``adaptive_syntax_filter.viz``
and serialises them to disk in the requested format. It is used by :pymod:`cli`
but can also be executed standalone::

    python -m export_cli output/figs --format pdf

Discovery strategy
------------------
A *figure-generating callable* is defined as a function with a name starting
with ``create_`` that returns a ``matplotlib.figure.Figure`` and has **no
required positional parameters**.

Anything matching this signature in the ``viz`` sub-package will be invoked.
Failures are logged and skipped so one problematic figure does not abort the
whole export process.
"""

from pathlib import Path
from typing import List, Callable
import logging
import inspect
import argparse
import sys

import matplotlib.pyplot as plt

from adaptive_syntax_filter import viz as _viz_pkg  # noqa: WPS436 – internal import

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Core implementation
# -----------------------------------------------------------------------------


def _is_zero_arg_creator(obj: Callable) -> bool:
    """Return ``True`` if *obj* is a zero-arg figure factory."""
    if not inspect.isfunction(obj):
        return False
    if not obj.__name__.startswith("create_"):
        return False
    sig = inspect.signature(obj)
    return all(p.default is not inspect._empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
               for p in sig.parameters.values())


def _discover_creators() -> List[Callable[[], plt.Figure]]:
    """Collect zero-argument figure factories from the ``viz`` package."""
    creators: List[Callable[[], plt.Figure]] = []
    for name in getattr(_viz_pkg, "__all__", []):
        obj = getattr(_viz_pkg, name)
        if _is_zero_arg_creator(obj):
            creators.append(obj)  # type: ignore[arg-type]
    # Also iterate through sub-modules to catch internal creators not re-exported
    for _, mod in vars(_viz_pkg).items():
        if inspect.ismodule(mod):
            for obj in vars(mod).values():
                if _is_zero_arg_creator(obj) and obj not in creators:
                    creators.append(obj)
    return creators


def export_figures(output_dir: str | Path, fmt: str = "png") -> int:  # noqa: D401
    """Export all discoverable figures to *output_dir*.

    Args:
        output_dir: Destination directory path.
        fmt: File format accepted by :py:meth:`matplotlib.figure.Figure.savefig`.

    Returns:
        Number of figures successfully exported.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    creators = _discover_creators()
    if not creators:
        logger.warning("No figure creators found – nothing to export")
        return 0

    exported = 0
    for fn in creators:
        try:
            fig = fn()  # type: ignore[misc]
            if not isinstance(fig, plt.Figure):
                logger.debug("%s did not return a Figure – skipped", fn.__name__)
                continue
            file_path = out / f"{fn.__name__}.{fmt}"
            fig.savefig(file_path, format=fmt)
            plt.close(fig)
            exported += 1
            logger.info("Saved %s", file_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to generate %s – %s", fn.__name__, exc)
    logger.info("Exported %d figure(s) to %s", exported, out)
    return exported


# -----------------------------------------------------------------------------
# CLI entry point (optional convenience)
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export publication-ready figures discovered in the viz package",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("output_dir", type=Path, help="Destination directory")
    p.add_argument("--format", choices=["png", "svg", "pdf"], default="png", help="Output format")
    return p


def main(argv: List[str] | None = None) -> None:  # noqa: D401
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    count = export_figures(args.output_dir, fmt=args.format)
    sys.exit(0 if count else 1)


if __name__ == "__main__":
    main() 