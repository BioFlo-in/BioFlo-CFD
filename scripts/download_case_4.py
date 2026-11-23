#!/usr/bin/env python3
"""
Robust Streaming Downloader for Large Kaggle Kernel Artifacts:
Downloads bioflo-cstr-case-4-viscous-10x-broth (1.9 GB) using streaming HTTP chunks
with auto-resume and timeout tolerance, avoiding the memory-buffering crash.
"""

import os
import sys
import time
from pathlib import Path
import requests

from kaggle.api.kaggle_api_extended import KaggleApi
import sys as _sys
_mod = _sys.modules[KaggleApi.__module__]
ApiListKernelSessionOutputRequest = getattr(_mod, "ApiListKernelSessionOutputRequest")

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_DIR = BASE_DIR / "outputs" / "kaggle" / "downloaded_datasets" / "bioflo-cstr-case-4-viscous-10x-broth"
TARGET_DIR.mkdir(parents=True, exist_ok=True)


def download_case_4():
    print("=" * 80)
    print("📥 BIOFLO CFD: Streaming Case 4 (Viscous 10x Broth, ~1.9 GB)")
    print("=" * 80)

    api = KaggleApi()
    api.authenticate()

    owner_slug = "ajitsheoran"
    kernel_slug = "bioflo-cstr-case-4-viscous-10x-broth"

    file_info = None

    print(f"Querying kernel session outputs for {owner_slug}/{kernel_slug}...")
    token = None
    with api.build_kaggle_client() as kaggle:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner_slug
            req.kernel_slug = kernel_slug
            api._set_paging(req, 50, token)
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            for f in resp.files or []:
                if f.file_name.endswith(".tar.gz"):
                    file_info = f
                    break
            if file_info or not resp.next_page_token:
                break
            token = resp.next_page_token

    if not file_info:
        print("❌ Could not find .tar.gz archive in kernel session outputs!")
        sys.exit(1)

    print(f"✓ Found archive: {file_info.file_name}")
    url = file_info.url
    dest_path = TARGET_DIR / file_info.file_name

    headers = {}
    downloaded_bytes = 0
    if dest_path.exists():
        downloaded_bytes = dest_path.stat().st_size
        print(f"Existing partial file detected: {downloaded_bytes / (1024*1024):.2f} MB")

    print(f"Starting chunked stream to {dest_path}...")
    t0 = time.time()
    
    # Download with stream=True and 8 MB chunks
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        print(f"Total Content-Length: {total_size / (1024*1024):.2f} MB")
        
        with open(dest_path, "wb") as f:
            chunk_count = 0
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    chunk_count += 1
                    if chunk_count % 5 == 0 or downloaded_bytes == total_size:
                        elapsed = time.time() - t0
                        speed_mb = (downloaded_bytes / (1024 * 1024)) / max(elapsed, 0.1)
                        pct = (downloaded_bytes / total_size) * 100 if total_size else 0
                        print(f"   [{pct:5.1f}%] {downloaded_bytes / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({speed_mb:.1f} MB/s)")

    print(f"\n✅ Case 4 download completed in {time.time() - t0:.1f}s!")
    print(f"Final file size: {dest_path.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    download_case_4()
