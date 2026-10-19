"""Create or verify the frozen security-model release manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import joblib


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(model_path: Path, metrics_path: Path) -> dict:
    artifact = joblib.load(model_path)
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    evaluation_name = "official_test" if "official_test" in report else "held_out_test"
    return {
        "model_file": model_path.name,
        "sha256": sha256(model_path),
        "bytes": model_path.stat().st_size,
        "schema_version": artifact.get("schema_version"),
        "selected_model": artifact.get("selected_model"),
        "threshold": artifact["threshold"],
        "feature_count": len(artifact["feature_columns"]),
        "feature_columns": artifact["feature_columns"],
        "categorical_columns": artifact.get("categorical_columns", []),
        "training_created_at_utc": report.get("created_at_utc"),
        "versions": report.get("versions", {}),
        evaluation_name: report[evaluation_name],
    }


def _verified_artifact(model_path: Path, manifest_path: Path) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_bytes = model_path.read_bytes()
    if len(model_bytes) != manifest["bytes"]:
        raise ValueError("Model size does not match the release manifest")
    if hashlib.sha256(model_bytes).hexdigest() != manifest["sha256"]:
        raise ValueError("Model SHA-256 does not match the release manifest")
    artifact = joblib.load(io.BytesIO(model_bytes))
    checks = {
        "schema_version": artifact.get("schema_version"),
        "selected_model": artifact.get("selected_model"),
        "threshold": artifact["threshold"],
        "feature_count": len(artifact["feature_columns"]),
        "feature_columns": artifact["feature_columns"],
        "categorical_columns": artifact.get("categorical_columns", []),
    }
    for field, value in checks.items():
        if value != manifest[field]:
            raise ValueError(f"Model {field} does not match the release manifest")
    return manifest, artifact


def verify_manifest(model_path: Path, manifest_path: Path) -> dict:
    return _verified_artifact(model_path, manifest_path)[0]


def load_verified_model(model_path: str | Path, manifest_path: str | Path | None = None) -> dict:
    model_path = Path(model_path)
    manifest_path = Path(manifest_path) if manifest_path else model_path.with_name("model_manifest.json")
    return _verified_artifact(model_path, manifest_path)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/security/security_model.joblib")
    parser.add_argument("--metrics", default="artifacts/security/metrics.json")
    parser.add_argument("--manifest", default="artifacts/security/model_manifest.json")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    model_path, manifest_path = Path(args.model), Path(args.manifest)
    if args.write:
        manifest = build_manifest(model_path, Path(args.metrics))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        manifest = verify_manifest(model_path, manifest_path)
    print(json.dumps({"verified": True, "sha256": manifest["sha256"],
                      "selected_model": manifest["selected_model"],
                      "feature_count": manifest["feature_count"]}, indent=2))


if __name__ == "__main__":
    main()
