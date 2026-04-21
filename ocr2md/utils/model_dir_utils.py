from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

GLM_MODEL_DIR_HINTS: tuple[str, ...] = ("glm-ocr", "glm", "ocr")
GLM_DEFAULT_SUBDIR = "glm-ocr"

LAYOUT_MODEL_DIR_HINTS: tuple[str, ...] = (
    "pp-doclayout-v3",
    "doclayout",
    "layout",
)
LAYOUT_DEFAULT_SUBDIR = "pp-doclayout-v3"
LAYOUT_REQUIRED_FILES: frozenset[str] = frozenset(
    {"config.json", "preprocessor_config.json", "model.safetensors"}
)


def _normalize_name(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def _matches_hints(path: Path, preferred_dir_hints: Sequence[str]) -> bool:
    path_name = _normalize_name(path.name)
    return any(_normalize_name(hint) in path_name for hint in preferred_dir_hints)


def _is_valid_model_dir(path: Path, required_files: Iterable[str]) -> bool:
    if not path.is_dir():
        return False

    expected = {str(item).strip() for item in required_files if str(item).strip()}
    if not expected:
        return False

    file_names = {item.name for item in path.iterdir() if item.is_file()}
    return expected.issubset(file_names)


def _sort_candidate(path: Path, preferred_dir_hints: Sequence[str]) -> tuple[int, str]:
    return (0 if _matches_hints(path, preferred_dir_hints) else 1, path.name.lower())


def _base_candidates(
    raw_model_dir: str, search_roots: Optional[Sequence[Path]]
) -> list[Path]:
    raw = Path(str(raw_model_dir).strip()).expanduser()
    if raw.is_absolute():
        return [raw.resolve()]

    roots = list(search_roots or [Path.cwd()])
    resolved: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        candidate = (root / raw).resolve()
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            resolved.append(candidate)
    return resolved


def build_model_dir_candidates(
    raw_model_dir: str,
    *,
    required_files: Iterable[str],
    preferred_dir_hints: Sequence[str],
    default_subdir_name: Optional[str] = None,
    search_roots: Optional[Sequence[Path]] = None,
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def _append(path: Path) -> None:
        resolved = path.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            candidates.append(resolved)

    bases = _base_candidates(raw_model_dir, search_roots)
    for base in bases:
        _append(base)
        if not base.is_dir():
            continue

        valid_subdirs = [
            sub.resolve()
            for sub in base.iterdir()
            if sub.is_dir() and _is_valid_model_dir(sub, required_files)
        ]
        valid_subdirs.sort(key=lambda item: _sort_candidate(item, preferred_dir_hints))
        for subdir in valid_subdirs:
            _append(subdir)

    if default_subdir_name:
        for base in bases:
            if base.is_dir() and not _matches_hints(base, preferred_dir_hints):
                _append(base / default_subdir_name)

    return candidates


def resolve_model_dir_path(
    raw_model_dir: str,
    *,
    required_files: Iterable[str],
    preferred_dir_hints: Sequence[str],
    default_subdir_name: Optional[str] = None,
    search_roots: Optional[Sequence[Path]] = None,
    must_exist: bool,
) -> Path:
    candidates = build_model_dir_candidates(
        raw_model_dir,
        required_files=required_files,
        preferred_dir_hints=preferred_dir_hints,
        default_subdir_name=default_subdir_name,
        search_roots=search_roots,
    )

    for candidate in candidates:
        if _is_valid_model_dir(candidate, required_files):
            return candidate

    if must_exist:
        searched = ", ".join(str(item) for item in candidates) or str(raw_model_dir)
        raise FileNotFoundError(
            f"Cannot find a valid model directory. Checked: {searched}"
        )

    if default_subdir_name:
        normalized_default = _normalize_name(default_subdir_name)
        for candidate in reversed(candidates):
            if _normalize_name(candidate.name) == normalized_default:
                return candidate

    if candidates:
        return candidates[0]

    return Path(str(raw_model_dir).strip()).expanduser().resolve()
