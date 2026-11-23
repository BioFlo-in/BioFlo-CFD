#!/usr/bin/env python3
"""
Comprehensive Ingestion Pipeline for All Cloud & Local 3D CFD Runs:
Extracts and maps all OpenFOAM multiphase volumetric fields across all regimes:
  1. Baseline 300 RPM CSTR (t = 0.00s -> 5.00s, 121 snapshots)
  2. Case 1 (120 RPM Gentle, t = 3.90s -> 5.05s, 24 snapshots)
  3. Case 2 (450 RPM Fast, t = 3.90s -> 4.75s, 18 snapshots)
  4. Case 3 (600 RPM Intense, t = 3.90s -> 4.40s, 11 snapshots)
  5. Case 4 (Viscous 10x Broth, t = 3.90s -> 5.65s, 36 snapshots)
  6. Pipe Flow & Channel Manifold (Domain 2, 50 snapshots)

Constructs:
  - data/cstr_massive_multiphase_dataset.pt (All 210 CSTR snapshots with physical metadata)
  - data/multi_domain_3d_cfd_dataset.pt (Unified 260-sample 8-channel foundation dataset)
"""

import os
import re
import sys
import time
import tarfile
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import scipy.ndimage as ndi
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from nlss.io.foam_reader import read_foam_field_file

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs" / "kaggle" / "downloaded_datasets"

GRID_DIM = 32
N_CELLS = 894950


def compute_sdf_from_mask(mask: np.ndarray, dx: float = 1.0) -> np.ndarray:
    dist_outside = ndi.distance_transform_edt(~mask, sampling=dx)
    dist_inside = ndi.distance_transform_edt(mask, sampling=dx)
    return (dist_outside - dist_inside).astype(np.float32)


def parse_internal_field_fast(text: str, is_vector: bool, n_cells: int = N_CELLS) -> np.ndarray:
    if_pos = text.find("internalField")
    if if_pos == -1:
        return np.zeros((n_cells, 3) if is_vector else (n_cells,), dtype=np.float32)

    bf_pos = text.find("boundaryField", if_pos)
    chunk = text[if_pos:bf_pos] if bf_pos != -1 else text[if_pos:]

    if "nonuniform" in chunk:
        start = chunk.find("(")
        end = chunk.rfind(")")
        if start != -1 and end != -1:
            raw_body = chunk[start + 1 : end]
            if is_vector:
                raw_body = raw_body.replace("(", " ").replace(")", " ")
                arr = np.fromstring(raw_body, sep=" ", dtype=np.float32).reshape(-1, 3)
                if arr.shape[0] == n_cells:
                    return arr
            else:
                arr = np.fromstring(raw_body, sep=" ", dtype=np.float32)
                if arr.shape[0] == n_cells:
                    return arr

    if "uniform" in chunk:
        val_str = chunk.split("uniform")[-1].split(";")[0].strip()
        if is_vector:
            val_clean = val_str.replace("(", "").replace(")", "").strip()
            parts = [float(x) for x in val_clean.split()]
            if len(parts) == 3:
                return np.tile(np.array(parts, dtype=np.float32), (n_cells, 1))
        else:
            try:
                v = float(val_str)
                return np.full((n_cells,), v, dtype=np.float32)
            except ValueError:
                pass

    return np.zeros((n_cells, 3) if is_vector else (n_cells,), dtype=np.float32)


def parse_field_from_tar(tar: tarfile.TarFile, member_name: str, is_vector: bool) -> np.ndarray:
    member = tar.getmember(member_name)
    f = tar.extractfile(member)
    text = f.read().decode("utf-8", errors="ignore")
    return parse_internal_field_fast(text, is_vector)


def parse_field_from_file(filepath: Path, is_vector: bool) -> np.ndarray:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return parse_internal_field_fast(text, is_vector)


def main():
    print("=" * 85)
    print("🚀 BioFlo-CFD: Massive Multi-Regime CFD Dataset Ingestion Engine")
    print(f"   Grid Resolution: {GRID_DIM}³ ({GRID_DIM**3:,} voxels)")
    print("=" * 85)

    # 1. Load cell coordinates and construct spatial cKDTree
    c_file = BASE_DIR / "openfoam_cases" / "professional_cstr_3d_gpu_mrf" / "1.95" / "C"
    if not c_file.exists():
        c_file = BASE_DIR / "openfoam_cases" / "professional_cstr_3d" / "0.5" / "C"
    assert c_file.exists(), f"Mesh cell coordinates file not found at {c_file}!"

    print("Loading 3D mesh cell coordinates (894,950 cells)...")
    t0 = time.time()
    c_field = read_foam_field_file(c_file)
    cell_coords = c_field.internal_field
    print(f"✓ Loaded {cell_coords.shape[0]:,} coordinates in {time.time() - t0:.2f}s")

    x_min, y_min, z_min = cell_coords.min(axis=0)
    x_max, y_max, z_max = cell_coords.max(axis=0)
    print(f"Domain Bounds: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}], Z=[{z_min:.3f}, {z_max:.3f}]")

    gx = np.linspace(x_min, x_max, GRID_DIM, dtype=np.float32)
    gy = np.linspace(y_min, y_max, GRID_DIM, dtype=np.float32)
    gz = np.linspace(z_min, z_max, GRID_DIM, dtype=np.float32)
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
    grid_pts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=-1)

    print("Building spatial cKDTree index...")
    t0 = time.time()
    tree = cKDTree(cell_coords)
    _, grid_indices = tree.query(grid_pts, k=1)
    print(f"✓ KDTree index constructed in {time.time() - t0:.2f}s")

    # Compute CSTR Geometry Mask & Signed Distance Function (SDF)
    Z_mesh, Y_mesh, X_mesh = np.meshgrid(gz, gy, gx, indexing="ij")
    r_mesh = np.sqrt(X_mesh**2 + Y_mesh**2)
    fluid_mask_cstr = (r_mesh <= 0.25)
    dx_cstr = (x_max - x_min) / (GRID_DIM - 1)
    sdf_cstr = compute_sdf_from_mask(fluid_mask_cstr, dx=dx_cstr)

    # 2. Define All Ingestion Cases
    cases_manifest = [
        {
            "id": "baseline_300rpm_0_1s",
            "name": "Baseline 300 RPM (0 - 1s)",
            "rpm": 300,
            "omega": -31.416,
            "nu": 1e-6,
            "type": "dir",
            "path": BASE_DIR / "openfoam_cases" / "professional_cstr_3d",
            "max_time": 1.0,
        },
        {
            "id": "baseline_300rpm_1_5s",
            "name": "Baseline 300 RPM (1 - 5s)",
            "rpm": 300,
            "omega": -31.416,
            "nu": 1e-6,
            "type": "dir",
            "path": BASE_DIR / "openfoam_cases" / "professional_cstr_3d_gpu_mrf",
            "min_time": 1.025,
        },
        {
            "id": "case_1_120rpm",
            "name": "Case 1: 120 RPM Gentle",
            "rpm": 120,
            "omega": -12.566,
            "nu": 1e-6,
            "type": "tar",
            "path": OUTPUTS_DIR / "bioflo-cstr-case-1-120-rpm-gentle" / "bioflo_3d_cstr_bioflo-cstr-case-1-120-rpm-gentle_reconstructed.tar.gz",
        },
        {
            "id": "case_2_450rpm",
            "name": "Case 2: 450 RPM Fast",
            "rpm": 450,
            "omega": -47.124,
            "nu": 1e-6,
            "type": "tar",
            "path": OUTPUTS_DIR / "bioflo-cstr-case-2-450-rpm-fast" / "bioflo_3d_cstr_bioflo-cstr-case-2-450-rpm-fast_reconstructed.tar.gz",
        },
        {
            "id": "case_3_600rpm",
            "name": "Case 3: 600 RPM Intense",
            "rpm": 600,
            "omega": -62.832,
            "nu": 1e-6,
            "type": "tar",
            "path": OUTPUTS_DIR / "bioflo-cstr-case-3-600-rpm-intense" / "bioflo_3d_cstr_bioflo-cstr-case-3-600-rpm-intense_reconstructed.tar.gz",
        },
        {
            "id": "case_4_viscous10x",
            "name": "Case 4: Viscous 10x Broth",
            "rpm": 300,
            "omega": -31.416,
            "nu": 1e-5,
            "type": "tar",
            "path": OUTPUTS_DIR / "bioflo-cstr-case-4-viscous-10x-broth" / "bioflo_3d_cstr_bioflo-cstr-case-4-viscous-10x-broth_reconstructed.tar.gz",
        },
    ]

    all_cstr_records = []
    total_snapshots = 0

    for cidx, cinfo in enumerate(cases_manifest, 1):
        print(f"\n[{cidx}/{len(cases_manifest)}] Processing {cinfo['name']}...")
        ptype = cinfo["type"]
        path = cinfo["path"]

        if not path.exists():
            print(f"   ⚠️ Path not found: {path}, skipping.")
            continue

        case_states = []
        case_times = []
        t_case_start = time.time()

        if ptype == "dir":
            t_dirs = [d for d in os.listdir(path) if re.match(r"^[0-9]+(\.[0-9]+)?$", d)]
            t_dirs = sorted(t_dirs, key=lambda x: float(x))
            for td in t_dirs:
                tval = float(td)
                if "min_time" in cinfo and tval < cinfo["min_time"]:
                    continue
                if "max_time" in cinfo and tval > cinfo["max_time"]:
                    continue

                u_f = path / td / "U"
                p_f = path / td / "p"
                a_f = path / td / "alpha.water"

                if not (u_f.exists() and (p_f.exists() or (path / td / "p_rgh").exists()) and a_f.exists()):
                    continue

                p_f_use = p_f if p_f.exists() else (path / td / "p_rgh")

                try:
                    u_raw = parse_field_from_file(u_f, is_vector=True)
                    p_raw = parse_field_from_file(p_f_use, is_vector=False)
                    a_raw = parse_field_from_file(a_f, is_vector=False)

                    u_g = u_raw[grid_indices].reshape(GRID_DIM, GRID_DIM, GRID_DIM, 3)
                    p_g = p_raw[grid_indices].reshape(GRID_DIM, GRID_DIM, GRID_DIM, 1)
                    a_g = a_raw[grid_indices].reshape(GRID_DIM, GRID_DIM, GRID_DIM, 1)

                    comb = np.concatenate([u_g, p_g, a_g], axis=-1)  # (D, H, W, 5)
                    comb = np.transpose(comb, (3, 0, 1, 2))  # (5, D, H, W)
                    case_states.append(comb)
                    case_times.append(tval)
                except Exception as e:
                    print(f"   Error parsing {td}: {e}")

        elif ptype == "tar":
            with tarfile.open(path, "r:gz") as tar:
                time_map = {}
                for m in tar.getmembers():
                    parts = [p for p in m.name.split("/") if p and p != "."]
                    if len(parts) >= 2 and parts[0].replace(".", "").isdigit():
                        t_str = parts[0]
                        fname = parts[1]
                        if fname in ("U", "p", "p_rgh", "alpha.water"):
                            time_map.setdefault(t_str, {})[fname] = m.name

                sorted_t_keys = sorted(time_map.keys(), key=lambda x: float(x))
                for t_str in sorted_t_keys:
                    fields = time_map[t_str]
                    if "U" not in fields or "alpha.water" not in fields:
                        continue
                    p_key = "p" if "p" in fields else ("p_rgh" if "p_rgh" in fields else None)
                    if not p_key:
                        continue

                    try:
                        u_raw = parse_field_from_tar(tar, fields["U"], is_vector=True)
                        p_raw = parse_field_from_tar(tar, fields[p_key], is_vector=False)
                        a_raw = parse_field_from_tar(tar, fields["alpha.water"], is_vector=False)

                        u_g = u_raw[grid_indices].reshape(GRID_DIM, GRID_DIM, GRID_DIM, 3)
                        p_g = p_raw[grid_indices].reshape(GRID_DIM, GRID_DIM, GRID_DIM, 1)
                        a_g = a_raw[grid_indices].reshape(GRID_DIM, GRID_DIM, GRID_DIM, 1)

                        comb = np.concatenate([u_g, p_g, a_g], axis=-1)
                        comb = np.transpose(comb, (3, 0, 1, 2))
                        case_states.append(comb)
                        case_times.append(float(t_str))
                    except Exception as e:
                        print(f"   Error parsing tar time {t_str}: {e}")

        if case_states:
            states_np = np.stack(case_states, axis=0)  # (T, 5, D, H, W)
            elapsed = time.time() - t_case_start
            print(f"   ✓ Ingested {states_np.shape[0]} snapshots in {elapsed:.1f}s ({elapsed/len(case_states):.2f}s/step) (t = {case_times[0]:.2f}s -> {case_times[-1]:.2f}s)")
            all_cstr_records.append({
                "case_id": cinfo["id"],
                "name": cinfo["name"],
                "rpm": cinfo["rpm"],
                "omega": cinfo["omega"],
                "nu": cinfo["nu"],
                "states": torch.from_numpy(states_np).float(),
                "times": case_times,
            })
            total_snapshots += states_np.shape[0]

    print(f"\n✓ Successfully ingested all CSTR cases: Total {total_snapshots} snapshots across {len(all_cstr_records)} cases!")

    # Save Massive CSTR Multiphase Dataset
    cstr_out_path = DATA_DIR / "cstr_massive_multiphase_dataset.pt"
    torch.save({
        "dataset_name": "BioFlo-CSTR-Massive-Multiphase-CFD",
        "cases": all_cstr_records,
        "grid_dim": GRID_DIM,
        "grid_bounds": {"x": (x_min, x_max), "y": (y_min, y_max), "z": (z_min, z_max)},
        "channels": ["ux", "uy", "uz", "p", "alpha.water"],
        "total_snapshots": total_snapshots,
        "fluid_mask": torch.from_numpy(fluid_mask_cstr),
        "sdf": torch.from_numpy(sdf_cstr),
    }, cstr_out_path)
    print(f"💾 Saved CSTR Multiphase Dataset to: {cstr_out_path} ({os.path.getsize(cstr_out_path)/(1024*1024):.1f} MB)")

    # 3. Assemble Unified Multi-Domain Foundation Dataset
    print("\n" + "=" * 85)
    print("🌐 Assembling Unified Multi-Domain 3D CFD Foundation Dataset (8 Channels)")
    print("   Channels: [ux, uy, uz, p, alpha, SDF, fluid_mask, domain_id]")
    print("=" * 85)

    multi_domain_samples = []

    # Domain 1: All CSTR Runs (domain_id = 1.0)
    sdf_tensor_cstr = torch.from_numpy(sdf_cstr).unsqueeze(0)  # (1, D, H, W)
    mask_tensor_cstr = torch.from_numpy(fluid_mask_cstr.astype(np.float32)).unsqueeze(0)  # (1, D, H, W)
    domain_id_cstr = torch.ones((1, GRID_DIM, GRID_DIM, GRID_DIM), dtype=torch.float32)

    for rec in all_cstr_records:
        states = rec["states"]  # (T, 5, D, H, W)
        for s in states:
            sample_8ch = torch.cat([s, sdf_tensor_cstr, mask_tensor_cstr, domain_id_cstr], dim=0)  # (8, D, H, W)
            multi_domain_samples.append(sample_8ch)

    print(f"✓ Added {len(multi_domain_samples)} CSTR samples across {len(all_cstr_records)} cases")

    # Domain 2: 3D Pipe Flow & Channel Manifold (domain_id = 2.0)
    pipe_path = DATA_DIR / "cfd_3d_dataset.pt"
    if pipe_path.exists():
        data_pipe = torch.load(pipe_path, weights_only=False)
        pipe_tensor = data_pipe["tensor"].float()  # (50, 4, D, H, W) -> [ux, uy, uz, p]
        T_pipe = pipe_tensor.shape[0]

        # Cylindrical pipe along Z-axis (R <= 0.85)
        x_p = np.linspace(-1, 1, GRID_DIM)
        y_p = np.linspace(-1, 1, GRID_DIM)
        z_p = np.linspace(-1, 1, GRID_DIM)
        Z_p, Y_p, X_p = np.meshgrid(z_p, y_p, x_p, indexing="ij")
        r_p = np.sqrt(X_p**2 + Y_p**2)
        fluid_mask_pipe = (r_p <= 0.85)
        sdf_pipe = compute_sdf_from_mask(fluid_mask_pipe, dx=2.0 / (GRID_DIM - 1))

        sdf_tensor_pipe = torch.from_numpy(sdf_pipe).unsqueeze(0)
        mask_tensor_pipe = torch.from_numpy(fluid_mask_pipe.astype(np.float32)).unsqueeze(0)
        alpha_pipe = torch.ones((1, GRID_DIM, GRID_DIM, GRID_DIM), dtype=torch.float32)
        domain_id_pipe = torch.full((1, GRID_DIM, GRID_DIM, GRID_DIM), 2.0, dtype=torch.float32)

        for s in pipe_tensor:
            sample_8ch = torch.cat([s, alpha_pipe, sdf_tensor_pipe, mask_tensor_pipe, domain_id_pipe], dim=0)
            multi_domain_samples.append(sample_8ch)

        print(f"✓ Added {T_pipe} Pipe Flow samples (Domain 2)")

    foundation_tensor = torch.stack(multi_domain_samples, dim=0)  # (N_total, 8, D, H, W)
    multi_domain_out_path = DATA_DIR / "multi_domain_3d_cfd_dataset.pt"

    torch.save({
        "dataset_name": "BioFlo-MultiDomain-3D-CFD-Foundation",
        "tensor": foundation_tensor,
        "num_samples": foundation_tensor.shape[0],
        "grid_dim": GRID_DIM,
        "channels": ["ux", "uy", "uz", "p", "alpha", "SDF", "fluid_mask", "domain_id"],
        "domains": ["CSTR_Bioreactor", "Pipe_Channel"],
        "cases_included": [r["name"] for r in all_cstr_records] + ["3D Cylindrical Pipe Flow"],
        "description": "Massive Multi-Domain 3D CFD Foundation Dataset spanning 5 CSTR operational regimes (120, 300, 450, 600 RPM + Viscous Broth) and Wall-Bounded Pipe Flow.",
    }, multi_domain_out_path)

    print("\n" + "=" * 85)
    print("✅ MULTI-DOMAIN FOUNDATION DATASET INGESTION COMPLETE!")
    print(f"   Final Tensor Shape: {foundation_tensor.shape} (N, C, D, H, W)")
    print(f"   Total Multi-Regime Snapshots: {foundation_tensor.shape[0]}")
    print(f"   Output File: {multi_domain_out_path} ({os.path.getsize(multi_domain_out_path)/(1024*1024):.1f} MB)")
    print("=" * 85)


if __name__ == "__main__":
    main()
