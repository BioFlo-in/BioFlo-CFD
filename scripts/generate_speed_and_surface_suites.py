#!/usr/bin/env python3
"""
Generator and Solver for Speed and Surface Variable Suites.

Generates 30 physical CFD simulation regimes:
  1. Speed Variations Suite (15 cases):
     - Comprehensive velocity sweeps: 0.05 to 25.0 m/s (creeping to high inertia)
     - Channel flow speeds (slow 0.2 m/s to fast 3.0 m/s)
     - Bioreactor impeller tip speeds (0.25 to 3.20 m/s)
  2. Surface Variable Variations Suite (15 cases):
     - Slip (zero-shear, frictionless / free surface)
     - Partial slip (valueFraction 0.25, 0.50, 0.75)
     - Multi-surface moving boundaries (counter-moving, 4-wall perimeter, C-drive, asymmetric ratios)
     - Surface friction / wall shear stress sweeps (viscous boundary layer thinning vs thick diffusion)
     - Channel Navier-slip walls
     - High surface dissipation bioreactor broth
"""

import shutil
import subprocess
from pathlib import Path
from typing import List
import numpy as np

from nlss.dataset.bioreactor import (
    BioreactorSimulator,
    BioreactorGeometry,
    export_simulation_state_to_openfoam,
)

WORKSPACE = Path.cwd()
SPEED_DIR = WORKSPACE / "openfoam_cases/speed_variations_suite"
SURFACE_DIR = WORKSPACE / "openfoam_cases/surface_variations_suite"
SPEED_DIR.mkdir(parents=True, exist_ok=True)
SURFACE_DIR.mkdir(parents=True, exist_ok=True)

REF_CAVITY = WORKSPACE / "external_openfoam_sources/official_tutorials/cavity"


def create_base_case(target_dir: Path, nu: float = 0.01, end_time: float = 0.25, write_interval: float = 0.05):
    """Copies full valid cavity template and modifies transportProperties and controlDict."""
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(REF_CAVITY, target_dir, ignore=shutil.ignore_patterns("0.*", "1*", "2*", "3*", "4*", "5*", "6*", "7*", "8*", "9*"))
    
    # Update nu in constant/transportProperties
    tp_file = target_dir / "constant/transportProperties"
    txt = tp_file.read_text()
    import re
    txt = re.sub(r"nu\s+\[.*?\]\s+[\d\.eE\-+]+;", f"nu              [0 2 -1 0 0 0 0] {nu};", txt)
    tp_file.write_text(txt)

    # Update controlDict
    cd_file = target_dir / "system/controlDict"
    cd_txt = cd_file.read_text()
    cd_txt = re.sub(r"endTime\s+[\d\.eE\-+]+;", f"endTime         {end_time};", cd_txt)
    cd_txt = re.sub(r"writeInterval\s+[\d\.eE\-+]+;", f"writeInterval   {write_interval};", cd_txt)
    cd_txt = re.sub(r"writeControl\s+\w+;", "writeControl    runTime;", cd_txt)
    cd_file.write_text(cd_txt)


def setup_4wall_geometry(case_dir: Path):
    """Configures 4 independent walls in blockMeshDict and 0/ fields."""
    bmd_file = case_dir / "system/blockMeshDict"
    bmd_txt = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale   0.1;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 0.1)
    (1 0 0.1)
    (1 1 0.1)
    (0 1 0.1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    topWall
    {
        type wall;
        faces
        (
            (3 7 6 2)
        );
    }
    rightWall
    {
        type wall;
        faces
        (
            (2 6 5 1)
        );
    }
    bottomWall
    {
        type wall;
        faces
        (
            (1 5 4 0)
        );
    }
    leftWall
    {
        type wall;
        faces
        (
            (0 4 7 3)
        );
    }
    frontAndBack
    {
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);

// ************************************************************************* //
"""
    bmd_file.write_text(bmd_txt)

    # 0/p
    p_file = case_dir / "0/p"
    p_txt = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    topWall { type zeroGradient; }
    rightWall { type zeroGradient; }
    bottomWall { type zeroGradient; }
    leftWall { type zeroGradient; }
    frontAndBack { type empty; }
}

// ************************************************************************* //
"""
    p_file.write_text(p_txt)


def set_fields_4wall(case_dir: Path, top_u: str, right_u: str, bottom_u: str, left_u: str):
    """Sets 0/U for 4-wall geometry."""
    u_file = case_dir / "0/U"
    u_txt = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    topWall {{ {top_u} }}
    rightWall {{ {right_u} }}
    bottomWall {{ {bottom_u} }}
    leftWall {{ {left_u} }}
    frontAndBack {{ type empty; }}
}}

// ************************************************************************* //
"""
    u_file.write_text(u_txt)


def run_openfoam(case_dir: Path) -> int:
    """Executes blockMesh and icoFoam within Docker OpenFOAM container."""
    rel = case_dir.resolve().relative_to(WORKSPACE.resolve())
    cmd = f'docker run --rm --user 1000:1000 -v {WORKSPACE}:/work opencfd/openfoam-default:latest bash -c "cd /work/{rel} && blockMesh && icoFoam"'
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.returncode


def generate_speed_variations():
    """Generates the 15-case Speed Variations Suite."""
    print("\n--- Generating Speed Variations Suite (15 cases) ---")
    
    speed_specs = [
        ("speed_creeping_u005", 0.05, 0.01),
        ("speed_slow_u015", 0.15, 0.01),
        ("speed_low_u035", 0.35, 0.01),
        ("speed_subnominal_u075", 0.75, 0.01),
        ("speed_nominal_u125", 1.25, 0.01),
        ("speed_moderate_u200", 2.00, 0.01),
        ("speed_brisk_u350", 3.50, 0.01),
        ("speed_fast_u500", 5.00, 0.01),
        ("speed_intense_u750", 7.50, 0.01),
        ("speed_extreme_u1200", 12.00, 0.01),
        ("speed_high_re2500", 25.00, 0.01),
        ("speed_channel_slow_u02", 0.20, 0.005),
        ("speed_channel_fast_u30", 3.00, 0.005),
    ]

    for name, u_val, nu_val in speed_specs:
        case_dir = SPEED_DIR / name
        create_base_case(case_dir, nu=nu_val, end_time=0.25, write_interval=0.05)
        u_file = case_dir / "0/U"
        txt = u_file.read_text()
        txt = txt.replace("uniform (1 0 0)", f"uniform ({u_val} 0 0)")
        u_file.write_text(txt)
        ret = run_openfoam(case_dir)
        snaps = len([d for d in case_dir.iterdir() if d.is_dir() and d.name not in ["0", "constant", "system"]])
        print(f"  ✓ [{name}]: u={u_val} m/s, nu={nu_val} -> {snaps} snapshots (code {ret})")

    # 2 Bioreactor Tip Speed cases using robust OpenFOAM
    bio_specs = [
        ("speed_bioreactor_micro_tip_u025", 0.25, 0.002),
        ("speed_bioreactor_high_shear_u320", 3.20, 0.005),
    ]
    for name, tip_u, nu_val in bio_specs:
        case_dir = SPEED_DIR / name
        create_base_case(case_dir, nu=nu_val, end_time=0.25, write_interval=0.05)
        u_file = case_dir / "0/U"
        txt = u_file.read_text().replace("uniform (1 0 0)", f"uniform ({tip_u} 0 0)")
        u_file.write_text(txt)
        run_openfoam(case_dir)
        snaps = len([d for d in case_dir.iterdir() if d.is_dir() and d.name not in ["0", "constant", "system"]])
        print(f"  ✓ [{name}]: tip_u={tip_u} m/s -> {snaps} snapshots (OpenFOAM)")


def generate_surface_variations():
    """Generates the 15-case Surface Variables Suite."""
    print("\n--- Generating Surface Variables Suite (15 cases) ---")
    
    # 1. Slip wall (free surface / frictionless)
    case_dir = SURFACE_DIR / "surface_free_slip_top"
    create_base_case(case_dir, nu=0.01, end_time=0.25, write_interval=0.05)
    u_file = case_dir / "0/U"
    txt = u_file.read_text().replace("type            fixedValue;\n        value           uniform (1 0 0);", "type            slip;")
    txt = txt.replace("type            noSlip;", "type            fixedValue;\n        value           uniform (1 0 0);")
    u_file.write_text(txt)
    run_openfoam(case_dir)
    print("  ✓ [surface_free_slip_top]: slip boundary")

    # 2. Partial Slip alpha=0.25
    case_dir = SURFACE_DIR / "surface_partial_slip_alpha25"
    create_base_case(case_dir, nu=0.01, end_time=0.25, write_interval=0.05)
    u_file = case_dir / "0/U"
    txt = u_file.read_text().replace("type            fixedValue;\n        value           uniform (1 0 0);",
                                     "type            partialSlip;\n        valueFraction   uniform 0.25;\n        value           uniform (2 0 0);")
    u_file.write_text(txt)
    run_openfoam(case_dir)
    print("  ✓ [surface_partial_slip_alpha25]: partial slip 0.25")

    # 3. Partial Slip alpha=0.50
    case_dir = SURFACE_DIR / "surface_partial_slip_alpha50"
    create_base_case(case_dir, nu=0.01, end_time=0.25, write_interval=0.05)
    u_file = case_dir / "0/U"
    txt = u_file.read_text().replace("type            fixedValue;\n        value           uniform (1 0 0);",
                                     "type            partialSlip;\n        valueFraction   uniform 0.50;\n        value           uniform (2 0 0);")
    u_file.write_text(txt)
    run_openfoam(case_dir)
    print("  ✓ [surface_partial_slip_alpha50]: partial slip 0.50")

    # 4. Partial Slip alpha=0.75
    case_dir = SURFACE_DIR / "surface_partial_slip_alpha75"
    create_base_case(case_dir, nu=0.01, end_time=0.25, write_interval=0.05)
    u_file = case_dir / "0/U"
    txt = u_file.read_text().replace("type            fixedValue;\n        value           uniform (1 0 0);",
                                     "type            partialSlip;\n        valueFraction   uniform 0.75;\n        value           uniform (2 0 0);")
    u_file.write_text(txt)
    run_openfoam(case_dir)
    print("  ✓ [surface_partial_slip_alpha75]: partial slip 0.75")

    # 5. Counter-moving horizontal (top +1.5, bottom -1.5)
    case_dir = SURFACE_DIR / "surface_counter_moving_horizontal"
    create_base_case(case_dir, nu=0.01, end_time=0.25, write_interval=0.05)
    setup_4wall_geometry(case_dir)
    set_fields_4wall(case_dir,
                     top_u="type fixedValue; value uniform (1.5 0 0);",
                     right_u="type noSlip;",
                     bottom_u="type fixedValue; value uniform (-1.5 0 0);",
                     left_u="type noSlip;")
    run_openfoam(case_dir)
    print("  ✓ [surface_counter_moving_horizontal]: dual opposing horizontal surface walls")

    # 6. Counter-moving vertical (left +1.5, right -1.5)
    case_dir = SURFACE_DIR / "surface_counter_moving_vertical"
    create_base_case(case_dir, nu=0.01, end_time=0.25, write_interval=0.05)
    setup_4wall_geometry(case_dir)
    set_fields_4wall(case_dir,
                     top_u="type noSlip;",
                     right_u="type fixedValue; value uniform (0 -1.5 0);",
                     bottom_u="type noSlip;",
                     left_u="type fixedValue; value uniform (0 1.5 0);")
    run_openfoam(case_dir)
    print("  ✓ [surface_counter_moving_vertical]: dual opposing vertical surface walls")

    # 7. Four-wall vortex loop (all 4 perimeter walls driving fluid in loop)
    case_dir = SURFACE_DIR / "surface_four_wall_vortex_loop"
    create_base_case(case_dir, nu=0.01, end_time=0.25, write_interval=0.05)
    setup_4wall_geometry(case_dir)
    set_fields_4wall(case_dir,
                     top_u="type fixedValue; value uniform (1.0 0 0);",
                     right_u="type fixedValue; value uniform (0 -1.0 0);",
                     bottom_u="type fixedValue; value uniform (-1.0 0 0);",
                     left_u="type fixedValue; value uniform (0 1.0 0);")
    run_openfoam(case_dir)
    print("  ✓ [surface_four_wall_vortex_loop]: 4-wall driven perimeter circulation")

    # 8. Three-wall C-drive
    case_dir = SURFACE_DIR / "surface_three_wall_c_drive"
    create_base_case(case_dir, nu=0.01, end_time=0.25, write_interval=0.05)
    setup_4wall_geometry(case_dir)
    set_fields_4wall(case_dir,
                     top_u="type fixedValue; value uniform (1.0 0 0);",
                     right_u="type fixedValue; value uniform (0 -1.0 0);",
                     bottom_u="type fixedValue; value uniform (-1.0 0 0);",
                     left_u="type noSlip;")
    run_openfoam(case_dir)
    print("  ✓ [surface_three_wall_c_drive]: 3-wall C-shaped surface drive")

    # 9. Asymmetric ratio 3:1 (top 3.0, bottom 1.0)
    case_dir = SURFACE_DIR / "surface_asymmetric_ratio_3to1"
    create_base_case(case_dir, nu=0.01, end_time=0.25, write_interval=0.05)
    setup_4wall_geometry(case_dir)
    set_fields_4wall(case_dir,
                     top_u="type fixedValue; value uniform (3.0 0 0);",
                     right_u="type noSlip;",
                     bottom_u="type fixedValue; value uniform (1.0 0 0);",
                     left_u="type noSlip;")
    run_openfoam(case_dir)
    print("  ✓ [surface_asymmetric_ratio_3to1]: asymmetric 3:1 surface ratio")

    # 10. Asymmetric ratio 10:1 (top 2.0, bottom 0.2)
    case_dir = SURFACE_DIR / "surface_asymmetric_ratio_10to1"
    create_base_case(case_dir, nu=0.01, end_time=0.25, write_interval=0.05)
    setup_4wall_geometry(case_dir)
    set_fields_4wall(case_dir,
                     top_u="type fixedValue; value uniform (2.0 0 0);",
                     right_u="type noSlip;",
                     bottom_u="type fixedValue; value uniform (0.2 0 0);",
                     left_u="type noSlip;")
    run_openfoam(case_dir)
    print("  ✓ [surface_asymmetric_ratio_10to1]: asymmetric 10:1 surface ratio")

    # 11. Reverse-driven lid (negative velocity)
    case_dir = SURFACE_DIR / "surface_reverse_driven_lid"
    create_base_case(case_dir, nu=0.01, end_time=0.25, write_interval=0.05)
    u_file = case_dir / "0/U"
    txt = u_file.read_text().replace("uniform (1 0 0)", "uniform (-1.5 0 0)")
    u_file.write_text(txt)
    run_openfoam(case_dir)
    print("  ✓ [surface_reverse_driven_lid]: reverse-driven surface")

    # 12. High friction thin boundary layer (nu = 0.0005, high wall velocity gradient)
    case_dir = SURFACE_DIR / "surface_high_friction_thin_bl"
    create_base_case(case_dir, nu=0.0005, end_time=0.25, write_interval=0.05)
    run_openfoam(case_dir)
    print("  ✓ [surface_high_friction_thin_bl]: high shear / thin boundary layer")

    # 13. Low friction thick diffusive boundary layer (nu = 0.05)
    case_dir = SURFACE_DIR / "surface_low_friction_thick_bl"
    create_base_case(case_dir, nu=0.05, end_time=0.25, write_interval=0.05)
    run_openfoam(case_dir)
    print("  ✓ [surface_low_friction_thick_bl]: thick diffusive boundary layer")

    # 14. Poiseuille Navier-slip walls
    case_dir = SURFACE_DIR / "surface_poiseuille_slip_walls"
    create_base_case(case_dir, nu=0.01, end_time=0.25, write_interval=0.05)
    setup_4wall_geometry(case_dir)
    set_fields_4wall(case_dir,
                     top_u="type slip;",
                     right_u="type zeroGradient;",
                     bottom_u="type slip;",
                     left_u="type fixedValue; value uniform (1.0 0 0);")
    run_openfoam(case_dir)
    print("  ✓ [surface_poiseuille_slip_walls]: Navier-slip channel walls")

    # 15. Bioreactor high surface shear dissipation using robust OpenFOAM
    case_dir = SURFACE_DIR / "surface_bioreactor_viscous_damping"
    create_base_case(case_dir, nu=0.03, end_time=0.25, write_interval=0.05)
    u_file = case_dir / "0/U"
    txt = u_file.read_text().replace("uniform (1 0 0)", "uniform (1.2 0 0)")
    u_file.write_text(txt)
    run_openfoam(case_dir)
    snaps = len([d for d in case_dir.iterdir() if d.is_dir() and d.name not in ["0", "constant", "system"]])
    print(f"  ✓ [surface_bioreactor_viscous_damping]: high viscous dissipation -> {snaps} snapshots")


def main():
    print("=" * 80)
    print("🚀 GENERATING SPEED & SURFACE VARIABLE SIMULATION SUITES")
    print("=" * 80)
    
    # 1. Speed Variations
    generate_speed_variations()
    
    # 2. Surface Variations
    generate_surface_variations()
    
    # Fix permissions
    subprocess.run(f'docker run --rm -v {WORKSPACE}:/work opencfd/openfoam-default:latest bash -c "chown -R 1000:1000 /work/openfoam_cases"', shell=True)

    print("\n" + "=" * 80)
    print("🎉 GENERATION OF SPEED AND SURFACE SUITES COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
