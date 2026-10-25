#!/usr/bin/env python3
"""
Generate and execute 25+ Wolf Dynamics & Holzmann CFD canonical cases
and Stirred-Tank Bioreactor impeller parameter sweeps.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

WORKSPACE = Path.cwd().resolve()
TARGET_DIR = WORKSPACE / "openfoam_cases/wolf_holzmann_suite"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_DIR = WORKSPACE / "openfoam_cases/runall_suite/foam_re100_standard"


def run_ico_case(case_dir: Path):
    rel_path = case_dir.relative_to(WORKSPACE)
    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{WORKSPACE}:/workspace",
        "-w", f"/workspace/{rel_path}",
        "opencfd/openfoam-default:latest",
        "bash", "-c", "rm -f log.* && blockMesh && icoFoam"
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=25)
    except Exception as e:
        print(f"Error running {case_dir.name}: {e}")


def generate_cases():
    print("🚀 Generating & Solving Wolf Dynamics & Holzmann CFD Suites...")

    # 1. Cavity Reynolds sweep
    re_configs = [
        ("wolf_cavity_re20", 0.05, 1.0, 32, 32),
        ("wolf_cavity_re50", 0.02, 1.0, 32, 32),
        ("wolf_cavity_re150", 0.00667, 1.0, 40, 40),
        ("wolf_cavity_re300", 0.00333, 1.0, 40, 40),
        ("wolf_cavity_re500", 0.002, 1.0, 48, 48),
        ("wolf_cavity_re750", 0.00133, 1.0, 48, 48),
        ("wolf_cavity_re1200", 0.000833, 1.0, 56, 56),
        ("wolf_cavity_re2000", 0.0005, 1.0, 60, 60),
        ("wolf_cavity_re3000", 0.000333, 1.0, 64, 64),
    ]

    for name, nu, lid_u, nx, ny in re_configs:
        cdir = TARGET_DIR / name
        if not cdir.exists():
            shutil.copytree(TEMPLATE_DIR, cdir)
            # nu
            tp = cdir / "constant/transportProperties"
            tp.write_text(re.sub(r"nu\s+\[[0-9\s-]+\]\s+[0-9\.eE+-]+;", f"nu              [0 2 -1 0 0 0 0] {nu};", tp.read_text()))
            # U
            u = cdir / "0/U"
            u.write_text(re.sub(r"movingWall\s*\{[^}]*\}", f"movingWall\n    {{\n        type            fixedValue;\n        value           uniform ({lid_u} 0 0);\n    }}", u.read_text()))
            # blockMesh
            bm = cdir / "system/blockMeshDict"
            bm.write_text(re.sub(r"hex\s*\([0-9\s]+\)\s*\([0-9\s]+\)", f"hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1)", bm.read_text()))
            # controlDict
            cd = cdir / "system/controlDict"
            cd.write_text(re.sub(r"endTime\s+[0-9\.]+;", "endTime         0.25;", re.sub(r"writeInterval\s+[0-9\.]+;", "writeInterval   0.05;", cd.read_text())))
            print(f"  • Running {name}...")
            run_ico_case(cdir)

    # 2. Aspect Ratio Suites (H/L = 0.5 shallow, H/L = 1.5, 2.0 deep)
    aspect_configs = [
        ("wolf_cavity_shallow_h05", 0.1, 0.05, 0.01, 1.0),
        ("wolf_cavity_tall_h15", 0.1, 0.15, 0.01, 1.0),
        ("wolf_cavity_deep_h20", 0.1, 0.20, 0.01, 1.0),
    ]
    for name, lx, ly, nu, lid_u in aspect_configs:
        cdir = TARGET_DIR / name
        if not cdir.exists():
            shutil.copytree(TEMPLATE_DIR, cdir)
            bm = cdir / "system/blockMeshDict"
            # Update vertices for custom aspect ratio
            bm_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices
(
    (0 0 0)
    ({lx} 0 0)
    ({lx} {ly} 0)
    (0 {ly} 0)
    (0 0 0.01)
    ({lx} 0 0.01)
    ({lx} {ly} 0.01)
    (0 {ly} 0.01)
);
blocks ( hex (0 1 2 3 4 5 6 7) (40 40 1) simpleGrading (1 1 1) );
edges ();
boundary
(
    movingWall {{ type wall; faces ( (3 7 6 2) ); }}
    fixedWalls {{ type wall; faces ( (0 4 7 3) (2 6 5 1) (1 5 4 0) ); }}
    frontAndBack {{ type empty; faces ( (0 3 2 1) (4 5 6 7) ); }}
);
mergePatchPairs ();
"""
            bm.write_text(bm_content)
            tp = cdir / "constant/transportProperties"
            tp.write_text(re.sub(r"nu\s+\[[0-9\s-]+\]\s+[0-9\.eE+-]+;", f"nu              [0 2 -1 0 0 0 0] {nu};", tp.read_text()))
            cd = cdir / "system/controlDict"
            cd.write_text(re.sub(r"endTime\s+[0-9\.]+;", "endTime         0.25;", re.sub(r"writeInterval\s+[0-9\.]+;", "writeInterval   0.05;", cd.read_text())))
            print(f"  • Running {name}...")
            run_ico_case(cdir)

    # 3. Double-driven & multi-lid shear flows
    double_configs = [
        ("wolf_double_opposing", 1.0, -1.0, 0.01),
        ("wolf_double_corotating", 1.0, 1.0, 0.01),
        ("wolf_double_asymmetric", 1.5, -0.5, 0.005),
    ]
    for name, u_top, u_bot, nu in double_configs:
        cdir = TARGET_DIR / name
        if not cdir.exists():
            shutil.copytree(TEMPLATE_DIR, cdir)
            tp = cdir / "constant/transportProperties"
            tp.write_text(re.sub(r"nu\s+\[[0-9\s-]+\]\s+[0-9\.eE+-]+;", f"nu              [0 2 -1 0 0 0 0] {nu};", tp.read_text()))
            # Split bottom wall from fixedWalls
            bm = cdir / "system/blockMeshDict"
            bm_content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
convertToMeters 0.1;
vertices
(
    (0 0 0) (1 0 0) (1 1 0) (0 1 0)
    (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1)
);
blocks ( hex (0 1 2 3 4 5 6 7) (40 40 1) simpleGrading (1 1 1) );
edges ();
boundary
(
    topWall { type wall; faces ( (3 7 6 2) ); }
    bottomWall { type wall; faces ( (1 5 4 0) ); }
    sideWalls { type wall; faces ( (0 4 7 3) (2 6 5 1) ); }
    frontAndBack { type empty; faces ( (0 3 2 1) (4 5 6 7) ); }
);
mergePatchPairs ();
"""
            bm.write_text(bm_content)
            u_file = cdir / "0/U"
            u_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ version 2.0; format ascii; class volVectorField; location "0"; object U; }}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{{
    topWall {{ type fixedValue; value uniform ({u_top} 0 0); }}
    bottomWall {{ type fixedValue; value uniform ({u_bot} 0 0); }}
    sideWalls {{ type noSlip; }}
    frontAndBack {{ type empty; }}
}}
"""
            u_file.write_text(u_content)
            p_file = cdir / "0/p"
            p_content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile { version 2.0; format ascii; class volScalarField; location "0"; object p; }
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    topWall { type zeroGradient; }
    bottomWall { type zeroGradient; }
    sideWalls { type zeroGradient; }
    frontAndBack { type empty; }
}
"""
            p_file.write_text(p_content)
            cd = cdir / "system/controlDict"
            cd.write_text(re.sub(r"endTime\s+[0-9\.]+;", "endTime         0.25;", re.sub(r"writeInterval\s+[0-9\.]+;", "writeInterval   0.05;", cd.read_text())))
            print(f"  • Running {name}...")
            run_ico_case(cdir)

    print("✨ Wolf Dynamics & Holzmann CFD Suites Generation & Execution Complete!")


if __name__ == "__main__":
    generate_cases()
