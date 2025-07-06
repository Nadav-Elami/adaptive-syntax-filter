import argparse
import json
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Helpers for (de)serialization
# -----------------------------------------------------------------------------

SUPPORTED_SUFFIXES = {".json", ".yml", ".yaml"}


def _read_config(path: Path) -> Dict[str, Any]:
    """Read a JSON / YAML configuration file.

    Args:
        path: Path to configuration file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If *path* has an unsupported suffix.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())

    # YAML branch -------------------------------------------------------------
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover – handled by tests
        raise ModuleNotFoundError(
            "PyYAML dependency missing. Install with 'pip install PyYAML'."
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


def _write_config(data: Dict[str, Any], path: Path) -> None:
    """Write *data* into *path* preserving the original format."""
    if path.suffix.lower() == ".json":
        text = json.dumps(data, indent=2)
        path.write_text(text)
        return

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML dependency missing. Install with 'pip install PyYAML'."
        ) from exc

    path.write_text(yaml.safe_dump(data, sort_keys=False))


# -----------------------------------------------------------------------------
# Key-path utilities
# -----------------------------------------------------------------------------

def _get_nested(data: Dict[str, Any], path: List[str]) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, dict):
            raise KeyError("->".join(path))
        node = node[key]
    return node


def _set_nested(data: Dict[str, Any], path: List[str], value: Any) -> None:
    node: Dict[str, Any] = data
    *parents, last = path
    for key in parents:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]  # type: ignore[assignment]
    node[last] = value


# -----------------------------------------------------------------------------
# Dataclasses for parsed args
# -----------------------------------------------------------------------------

@dataclass
class ShowArgs:
    config: Path
    key: Optional[str]


@dataclass
class DiffArgs:
    base: Path
    other: Path


@dataclass
class SetArgs:
    config: Path
    key: str
    value: str


# -----------------------------------------------------------------------------
# Command implementations
# -----------------------------------------------------------------------------

def _cmd_show(a: ShowArgs) -> int:  # noqa: D401
    cfg = _read_config(a.config)
    if a.key is None:
        json.dump(cfg, sys.stdout, indent=2, ensure_ascii=False)
        print()
    else:
        path = a.key.split(".")
        try:
            val = _get_nested(cfg, path)
            print(json.dumps(val, indent=2, ensure_ascii=False))
        except KeyError:
            logging.error("Key '%s' not found in config", a.key)
            return 1
    return 0


def _cmd_diff(a: DiffArgs) -> int:  # noqa: D401
    import difflib

    base_text = json.dumps(_read_config(a.base), indent=2, sort_keys=True)
    other_text = json.dumps(_read_config(a.other), indent=2, sort_keys=True)

    diff = difflib.unified_diff(
        base_text.splitlines(),
        other_text.splitlines(),
        fromfile=str(a.base),
        tofile=str(a.other),
        lineterm="",
    )
    for line in diff:
        print(line)
    return 0


def _cmd_set(a: SetArgs) -> int:  # noqa: D401
    cfg = _read_config(a.config)

    # Attempt to JSON-decode value to preserve types where possible.
    try:
        value: Any = json.loads(a.value)
    except json.JSONDecodeError:
        value = a.value  # treat as plain string

    _set_nested(cfg, a.key.split("."), value)

    logging.info("Updating key '%s' in %s", a.key, a.config)

    # Transactional write -----------------------------------------------------
    with tempfile.NamedTemporaryFile("w", delete=False, dir=a.config.parent, suffix=a.config.suffix) as tmp:
        tmp_path = Path(tmp.name)
        logging.debug("Writing temp file %s", tmp_path)
        _write_config(cfg, tmp_path)

    shutil.move(tmp_path, a.config)
    logging.info("Config updated successfully")
    return 0


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utility for inspecting and editing pipeline configuration files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # show --------------------------------------------------------------------
    p_show = sub.add_parser("show", help="Display configuration (or a specific key)")
    p_show.add_argument("config", type=Path, help="Path to config file")
    p_show.add_argument("key", nargs="?", help="Dot-separated key path (optional)")
    p_show.set_defaults(func=lambda ns: _cmd_show(ShowArgs(ns.config, ns.key)))

    # diff --------------------------------------------------------------------
    p_diff = sub.add_parser("diff", help="Show diff between two config files")
    p_diff.add_argument("base", type=Path, help="Baseline config file")
    p_diff.add_argument("other", type=Path, help="Other config file to compare against baseline")
    p_diff.set_defaults(func=lambda ns: _cmd_diff(DiffArgs(ns.base, ns.other)))

    # set ---------------------------------------------------------------------
    p_set = sub.add_parser("set", help="Set a configuration value transactionally")
    p_set.add_argument("config", type=Path, help="Path to config file")
    p_set.add_argument("key", help="Dot-separated key path to modify")
    p_set.add_argument("value", help="New value (JSON encoded if complex)")
    p_set.set_defaults(func=lambda ns: _cmd_set(SetArgs(ns.config, ns.key, ns.value)))

    return parser


# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] | None = None) -> None:  # noqa: D401
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = _build_parser()
    ns = parser.parse_args(argv)
    exit_code: int = ns.func()  # type: ignore[attr-defined]
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 