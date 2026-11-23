#!/usr/bin/env python3
"""
Targeted, High-Speed Kaggle Dataset Downloader:
Downloads ONLY the consolidated .tar.gz archives and execution logs,
skipping uncompressed loose files to ensure fast, reliable network transfer.
"""

import os
import subprocess
import sys
import tarfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = BASE_DIR / "outputs" / "kaggle" / "downloaded_datasets"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

KERNELS = [
    {
        "slug": "bioflo-3d-cstr-openfoam-simulation",
        "desc": "Baseline 300 RPM Continuation (t = 3.90s -> 5.00s)",
    },
    {
        "slug": "bioflo-cstr-case-1-120-rpm-gentle",
        "desc": "Case 1: 120 RPM Gentle (t = 3.90s -> 5.05s)",
    },
    {
        "slug": "bioflo-cstr-case-2-450-rpm-fast",
        "desc": "Case 2: 450 RPM Fast (t = 3.90s -> 4.75s)",
    },
    {
        "slug": "bioflo-cstr-case-3-600-rpm-intense",
        "desc": "Case 3: 600 RPM Intense (t = 3.90s -> 4.40s)",
    },
    {
        "slug": "bioflo-cstr-case-4-viscous-10x-broth",
        "desc": "Case 4: Viscous 10x Broth 300 RPM (t = 3.90s -> 5.65s)",
    },
]


def download_all():
    print("=" * 85)
    print("📥 BIOFLO CFD: Fast Streamlined Archive Downloader")
    print(f"   Destination: {DOWNLOAD_DIR}")
    print("=" * 85)

    summary = []

    for idx, item in enumerate(KERNELS, 1):
        slug = item["slug"]
        desc = item["desc"]
        kernel_ref = f"ajitsheoran/{slug}"
        target_dir = DOWNLOAD_DIR / slug
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx}/{len(KERNELS)}] Checking {slug} ({desc})...")

        # Check if valid archive already exists
        existing_tars = [f for f in target_dir.glob("*.tar.gz") if f.stat().st_size > 10 * 1024 * 1024]
        if existing_tars:
            tar_path = existing_tars[0]
            size_mb = tar_path.stat().st_size / (1024 * 1024)
            print(f"   ✓ Archive already present: {tar_path.name} ({size_mb:.1f} MB)")
        else:
            # 1. Download execution log
            log_file = target_dir / f"{slug}.log"
            try:
                log_res = subprocess.run(["kaggle", "kernels", "logs", kernel_ref], capture_output=True, text=True)
                with open(log_file, "w") as f:
                    f.write(log_res.stdout)
                print(f"   ✓ Saved execution log ({log_file.stat().st_size / 1024:.1f} KB)")
            except Exception as e:
                print(f"   ⚠️ Could not save log: {e}")

            # 2. Download ONLY .tar.gz archive
            print(f"   ⬇️ Downloading .tar.gz archive via Kaggle CLI...")
            cmd = [
                "kaggle", "kernels", "output", kernel_ref,
                "-p", str(target_dir),
                "--file-pattern", r".*\.tar\.gz$",
                "-o"
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"   CLI error: {proc.stderr}")
            else:
                print(f"   CLI output: {proc.stdout.strip()}")

        # Verify archive integrity
        downloaded_tars = [f for f in target_dir.glob("*.tar.gz") if f.stat().st_size > 0]
        if downloaded_tars:
            tar_path = downloaded_tars[0]
            size_mb = tar_path.stat().st_size / (1024 * 1024)
            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    names = tar.getnames()
                    time_folders = sorted(list(set(n.split("/")[0] for n in names if "/" in n and n.split("/")[0].replace(".", "").isdigit())))
                    print(f"   ✓ Verified valid archive! {len(names)} files, {len(time_folders)} time steps ({time_folders[0]}s -> {time_folders[-1]}s)")
                    summary.append({
                        "case": slug,
                        "desc": desc,
                        "size_mb": f"{size_mb:.1f} MB",
                        "time_steps": len(time_folders),
                        "time_range": f"{time_folders[0]}s -> {time_folders[-1]}s" if time_folders else "N/A",
                        "status": "VALID",
                    })
            except Exception as e:
                print(f"   ⚠️ Tar validation error: {e}")
                summary.append({
                    "case": slug,
                    "desc": desc,
                    "size_mb": f"{size_mb:.1f} MB",
                    "time_steps": "Unknown",
                    "time_range": "Corrupt",
                    "status": "CORRUPT",
                })
        else:
            print(f"   ❌ No archive found!")
            summary.append({
                "case": slug,
                "desc": desc,
                "size_mb": "0 MB",
                "time_steps": 0,
                "time_range": "None",
                "status": "MISSING",
            })

    print("\n" + "=" * 95)
    print("📊 COMPLETE BIOFLO CFD CLOUD DATASET INVENTORY")
    print("=" * 95)
    print(f"{'Case':<38} | {'Size':<10} | {'Steps':<6} | {'Time Range':<18} | {'Status'}")
    print("-" * 95)
    for s in summary:
        print(f"{s['case']:<38} | {s['size_mb']:<10} | {str(s['time_steps']):<6} | {s['time_range']:<18} | {s['status']}")
    print("-" * 95)


if __name__ == "__main__":
    download_all()
