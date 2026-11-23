#!/usr/bin/env python3
"""
Diverse 3D CSTR Bioreactor Configurations Generator:
Establishes a comprehensive multi-configuration suite across:
  - Agitation speeds (RPM / tip speeds)
  - Fluid rheology (kinematic viscosity nu)
  - Vessel liquid fill levels (alpha phase distribution)

Uses relative symlinks to existing 894,950-cell polyMesh to incur ZERO disk space duplication!
"""

import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CASES_DIR = BASE_DIR / "openfoam_cases"
SUITE_DIR = CASES_DIR / "diverse_configurations_suite"
SRC_CASE = CASES_DIR / "professional_cstr_3d_gpu_mrf"
SRC_0_ORIG = CASES_DIR / "professional_cstr_3d" / "0.orig"


CONFIGURATIONS = [
    {
        "name": "case_1_gentle_rpm_120",
        "description": "Low-shear mammalian/stem-cell agitation regime",
        "rpm": 120,
        "omega": -12.566,
        "nu": 1e-6,
        "fill_z": 0.664,
        "regime": "Low Shear / Gentle Agitation",
    },
    {
        "name": "case_2_fast_rpm_450",
        "description": "High-intensity bacterial/yeast fermentation",
        "rpm": 450,
        "omega": -47.124,
        "nu": 1e-6,
        "fill_z": 0.664,
        "regime": "High Shear / Rapid Gas Dispersion",
    },
    {
        "name": "case_3_intense_rpm_600",
        "description": "Extreme turbulence & deep vortex core drawdown",
        "rpm": 600,
        "omega": -62.832,
        "nu": 1e-6,
        "fill_z": 0.664,
        "regime": "Severe Vortex Drawdown / Cavitation Risk",
    },
    {
        "name": "case_4_viscous_broth_10x",
        "description": "Viscous fungal/mycelial biopolymer media (10x water)",
        "rpm": 300,
        "omega": -31.416,
        "nu": 1e-5,
        "fill_z": 0.664,
        "regime": "Viscous Non-Water Biopolymer Broth",
    },
    {
        "name": "case_5_fill_90percent",
        "description": "High liquid volume fill (90% capacity, Z = 0.90m)",
        "rpm": 300,
        "omega": -31.416,
        "nu": 1e-6,
        "fill_z": 0.900,
        "regime": "Expanded Upper Recirculation Loop",
    },
    {
        "name": "case_6_fill_50percent",
        "description": "Low liquid volume fill (50% capacity, Z = 0.45m)",
        "rpm": 300,
        "omega": -31.416,
        "nu": 1e-6,
        "fill_z": 0.450,
        "regime": "Surface Impeller Air Ingestion",
    },
]


def setup_cases():
    print("=" * 85)
    print("🔬 Setting Up Diverse 3D CSTR Multi-Configuration Suite")
    print(f"   Destination: {SUITE_DIR}")
    print("=" * 85)

    os.makedirs(SUITE_DIR, exist_ok=True)

    summary_rows = []

    for cfg in CONFIGURATIONS:
        c_dir = SUITE_DIR / cfg["name"]
        os.makedirs(c_dir, exist_ok=True)

        # 1. Setup 'constant' directory with relative symlink to polyMesh
        const_dir = c_dir / "constant"
        os.makedirs(const_dir, exist_ok=True)

        poly_link = const_dir / "polyMesh"
        if not poly_link.exists():
            # Relative path to professional_cstr_3d_gpu_mrf/constant/polyMesh
            rel_target = Path("../../../professional_cstr_3d_gpu_mrf/constant/polyMesh")
            poly_link.symlink_to(rel_target)

        # Copy and modify MRFProperties
        mrf_src = SRC_CASE / "constant" / "MRFProperties"
        with open(mrf_src, "r") as f:
            mrf_text = f.read()

        # Replace omega
        import re
        mrf_text_mod = re.sub(r"omega\s+[^;]+;", f"omega       {cfg['omega']:.4f};", mrf_text)
        with open(const_dir / "MRFProperties", "w") as f:
            f.write(mrf_text_mod)

        # Copy and modify transportProperties
        tp_src = SRC_CASE / "constant" / "transportProperties"
        with open(tp_src, "r") as f:
            tp_text = f.read()

        tp_text_mod = re.sub(r"nu\s+1e-06;", f"nu              {cfg['nu']:g};", tp_text)
        with open(const_dir / "transportProperties", "w") as f:
            f.write(tp_text_mod)

        # Copy g and turbulenceProperties
        for extra in ["g", "turbulenceProperties"]:
            src_f = SRC_CASE / "constant" / extra
            if src_f.exists():
                shutil.copy(src_f, const_dir / extra)

        # 2. Setup 'system' directory
        sys_dir = c_dir / "system"
        os.makedirs(sys_dir, exist_ok=True)

        for s_file in ["controlDict", "fvSchemes", "decomposeParDict"]:
            src_f = SRC_CASE / "system" / s_file
            if src_f.exists():
                shutil.copy(src_f, sys_dir / s_file)

        # Copy fvSolution (using standard OpenFOAM solvers)
        shutil.copy(SRC_CASE / "system" / "fvSolution.foam", sys_dir / "fvSolution")

        # Modify setFieldsDict for liquid fill height
        setfields_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      setFieldsDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defaultFieldValues
(
    volScalarFieldValue alpha.water 1
);

regions
(
    boxToCell
    {{
        box (-4 -4 {cfg['fill_z']:.3f}) (4 4 4.0);
        fieldValues
        (
            volScalarFieldValue alpha.water 0
        );
    }}
);
// ************************************************************************* //
"""
        with open(sys_dir / "setFieldsDict", "w") as f:
            f.write(setfields_content)

        # 3. Setup initial '0' from 0.orig
        zero_dir = c_dir / "0"
        if zero_dir.exists():
            shutil.rmtree(zero_dir)
        shutil.copytree(SRC_0_ORIG, zero_dir)

        # Create touch case.foam for easy ParaView visualization
        (c_dir / "case.foam").touch()

        summary_rows.append({
            "Case Directory": cfg["name"],
            "RPM": cfg["rpm"],
            "Omega (rad/s)": f"{cfg['omega']:.2f}",
            "Viscosity (m²/s)": f"{cfg['nu']:g}",
            "Fill Height (m)": f"{cfg['fill_z']:.2f}",
            "Physical Regime": cfg["regime"],
        })

    # 4. Generate master launcher script
    launcher_sh = SUITE_DIR / "run_diverse_cases.sh"
    with open(launcher_sh, "w") as f:
        f.write("""#!/usr/bin/env bash
# Master Execution Script for Diverse 3D CSTR Suite
set -e

. /usr/lib/openfoam/openfoam2406/etc/bashrc 2>/dev/null || true

for case in case_*; do
    if [ -d "$case" ]; then
        echo "================================================================="
        echo "🚀 Initializing and Running OpenFOAM Case: $case"
        echo "================================================================="
        cd "$case"
        setFields > log.setFields 2>&1
        decomposePar -force > log.decomposePar 2>&1
        mpirun -np 4 interFoam -parallel > log.interFoam 2>&1 &
        echo "Case $case submitted to 4-core MPI!"
        cd ..
    fi
done
""")
    os.chmod(launcher_sh, 0o755)

    print("\n✅ Successfully generated 6 diverse configurations in:")
    print(f"   {SUITE_DIR}\n")
    print(f"{'Case Name':<28} | {'RPM':<5} | {'Omega':<8} | {'Viscosity':<10} | {'Fill (m)':<8} | {'Regime'}")
    print("-" * 88)
    for r in summary_rows:
        print(f"{r['Case Directory']:<28} | {r['RPM']:<5} | {r['Omega (rad/s)']:<8} | {r['Viscosity (m²/s)']:<10} | {r['Fill Height (m)']:<8} | {r['Physical Regime']}")
    print("-" * 88)


if __name__ == "__main__":
    setup_cases()
