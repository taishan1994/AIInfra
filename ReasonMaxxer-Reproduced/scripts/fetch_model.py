#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def _set_hf_endpoint(endpoint: str) -> None:
    endpoint = endpoint.strip()
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"[info] HF_ENDPOINT={endpoint}")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    print(f"[info] HF_HUB_DISABLE_XET={os.environ['HF_HUB_DISABLE_XET']}")


def _download_huggingface(*, repo_id: str, revision: str | None, out_dir: Path) -> str:
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(out_dir),
    )
    return str(path)


def _download_modelscope(*, repo_id: str, revision: str | None, out_dir: Path) -> str:
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "modelscope is not installed. Run `pip install modelscope` or use --source huggingface."
        ) from exc

    path = ms_snapshot_download(
        model_id=repo_id,
        revision=revision,
        cache_dir=str(out_dir.parent),
        local_dir=str(out_dir),
        local_files_only=False,
    )
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a model checkpoint from Hugging Face or ModelScope for ReasonMaxxer."
    )
    parser.add_argument("--repo_id", required=True, help="Remote model repo id, e.g. Qwen/Qwen2.5-1.5B")
    parser.add_argument("--out_dir", required=True, help="Local directory for the downloaded model")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--source",
        choices=["auto", "huggingface", "modelscope"],
        default="auto",
        help="Use ModelScope first only when explicitly requested.",
    )
    parser.add_argument(
        "--hf_endpoint",
        default="https://hf-mirror.com",
        help="Set HF_ENDPOINT before Hugging Face downloads. Use empty string to disable.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    source = args.source
    if source == "auto":
        source = "huggingface"

    if source == "modelscope":
        path = _download_modelscope(repo_id=args.repo_id, revision=args.revision, out_dir=out_dir)
    else:
        _set_hf_endpoint(args.hf_endpoint)
        path = _download_huggingface(
            repo_id=args.repo_id,
            revision=args.revision,
            out_dir=out_dir,
        )
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
