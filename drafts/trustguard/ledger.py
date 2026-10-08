"""Create a local, inspectable evidence chain for one experiment run."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def dataframe_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    """Hash values, column order, dtypes and index without exporting raw data."""
    hasher = hashlib.sha256()
    hasher.update("|".join(map(str, X.columns)).encode())
    hasher.update("|".join(map(str, X.dtypes)).encode())
    # ``index=True`` prevents a row reordering from being mistaken for the same experiment.
    hasher.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    hasher.update(pd.util.hash_pandas_object(y, index=True).values.tobytes())
    return hasher.hexdigest()


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


class ReproducibilityLedger:
    """Write data identity, split evidence, configuration and artifact hashes.

    Raw data are intentionally never copied to the report directory.  A reviewer
    can compare fingerprints and split indices while the source data remains in
    its original controlled location.
    """

    package_names = ("tabularx", "numpy", "pandas", "scikit-learn", "optuna", "xgboost", "lightgbm", "catboost")

    def capture(
        self,
        output_dir: str | Path,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        split: Any | None = None,
        config: Mapping[str, Any] | None = None,
        command: str | None = None,
    ) -> dict[str, Path]:
        root = Path(output_dir)
        destination = root / "reproducibility_ledger"
        destination.mkdir(parents=True, exist_ok=True)
        split_path: Path | None = None
        split_summary: dict[str, Any] | None = None
        if split is not None:
            arrays = {
                key: np.asarray(getattr(split, key), dtype=np.int64)
                for key in ("train_indices", "valid_indices", "test_indices")
                if hasattr(split, key)
            }
            if arrays:
                split_path = destination / "split_indices.npz"
                # Index evidence is sufficient to reproduce the split; raw rows remain outside reports.
                np.savez_compressed(split_path, **arrays)
                split_summary = {key: {"count": int(len(arr)), "sha256": hashlib.sha256(arr.tobytes()).hexdigest()} for key, arr in arrays.items()}

        packages: dict[str, str | None] = {}
        for name in self.package_names:
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None

        payload = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data": {"rows": int(len(X)), "columns": int(X.shape[1]), "fingerprint_sha256": dataframe_fingerprint(X, y)},
            "split": split_summary,
            "config": _json_safe(dict(config or {})),
            "command": command,
            "environment": {"python": sys.version, "platform": platform.platform(), "packages": packages},
            "privacy_note": "Raw rows and target values are not stored in this evidence package.",
        }
        ledger_path = destination / "reproducibility_ledger.json"
        ledger_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path = self.write_artifact_manifest(root)
        result = {"ledger": ledger_path, "manifest": manifest_path}
        if split_path:
            result["split_indices"] = split_path
        return result

    @staticmethod
    def write_artifact_manifest(root: str | Path) -> Path:
        root = Path(root)
        manifest = root / "artifact_manifest.json"
        entries = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != manifest:
                # Excluding the manifest avoids an impossible self-referential hash.
                entries.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)})
        manifest.write_text(json.dumps({"algorithm": "sha256", "files": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


def verify_artifact_manifest(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_path = root / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    listed = set()
    for entry in payload["files"]:
        listed.add(entry["path"])
        path = root / entry["path"]
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            mismatches.append(entry["path"])
    current = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    unexpected = sorted(current - listed)
    missing = sorted(listed - current)
    return {
        "valid": not mismatches and not unexpected,
        "checked_files": len(payload["files"]),
        "mismatches": sorted(set(mismatches)),
        "missing": missing,
        "unexpected": unexpected,
    }
