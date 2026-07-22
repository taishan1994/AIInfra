#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
import requests


def _set_hf_endpoint(endpoint: str) -> None:
    endpoint = endpoint.strip()
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"[info] HF_ENDPOINT={endpoint}")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    print(f"[info] HF_HUB_DISABLE_XET={os.environ['HF_HUB_DISABLE_XET']}")


def _direct_download(*, endpoint: str, repo_id: str, repo_type: str, filename: str, local_path: Path) -> None:
    if not endpoint.strip():
        raise RuntimeError("Direct download fallback requires a non-empty --hf_endpoint")
    repo_prefix = "datasets" if repo_type == "dataset" else "models"
    url = f"{endpoint.rstrip('/')}/{repo_prefix}/{repo_id}/resolve/main/{filename}"
    print(f"[fallback] direct download {url}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp_path.replace(local_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Download public SimpleRL-Zoo parquet files from Hugging Face.")
    p.add_argument("--repo_id", default="hkust-nlp/SimpleRL-Zoo-Data")
    p.add_argument("--repo_type", default="dataset")
    p.add_argument("--style", choices=["abel", "qwen"], default="abel")
    p.add_argument(
        "--subset",
        choices=["gsm8k_level1", "level1to4", "level3to5"],
        default="level3to5",
        help="Dataset slice under the Hugging Face dataset repo.",
    )
    p.add_argument(
        "--splits",
        default="train,test",
        help="Comma-separated splits to download, e.g. train or train,test.",
    )
    p.add_argument("--out_dir", default="data/external/simpleRL")
    p.add_argument(
        "--hf_endpoint",
        default="https://hf-mirror.com",
        help="Set HF_ENDPOINT before download. Use empty string to disable mirror routing.",
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    _set_hf_endpoint(args.hf_endpoint)

    remote_dir = f"simplelr_{args.style}_{args.subset}"
    out_dir = Path(args.out_dir) / remote_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in [x.strip() for x in args.splits.split(",") if x.strip()]:
        filename = f"{remote_dir}/{split}.parquet"
        local_path = out_dir / f"{split}.parquet"
        if local_path.exists() and not args.force:
            print(f"[skip] {local_path}")
            continue
        try:
            cached_path = hf_hub_download(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                filename=filename,
                force_download=bool(args.force),
            )
            local_path.write_bytes(Path(cached_path).read_bytes())
        except Exception as exc:
            print(f"[warn] hf_hub_download failed for {filename}: {type(exc).__name__}: {exc}")
            _direct_download(
                endpoint=args.hf_endpoint,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                filename=filename,
                local_path=local_path,
            )
        print(f"[saved] {local_path}")


if __name__ == "__main__":
    main()
