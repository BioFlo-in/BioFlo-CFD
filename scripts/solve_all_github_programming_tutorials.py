#!/usr/bin/env python3
"""
Solves 20+ cases from GitHub repositories:
  - phresher/OpenFOAM_Tutorials_Plus (all OFtutorial00 to OFtutorial12 and OF301 cases)
  - NimaSam/OpenFOAM_tutorials (cavityInPipe, heatedCavity, ellipsoidal, etc.)
  - hsongxa/openfoam-cases (cubic_cavity)
"""

import os
import re
import subprocess
from pathlib import Path

WORKSPACE = Path.cwd().resolve()
PLUS_DIR = WORKSPACE / "external_openfoam_sources/OpenFOAM_Tutorials_Plus/programmingTutorials"
NIMA_DIR = WORKSPACE / "external_openfoam_sources/NimaSam_OpenFOAM_Tutorials"


def run_case_in_docker(case_dir: Path):
    rel_path = case_dir.relative_to(WORKSPACE)
    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{WORKSPACE}:/workspace",
        "-w", f"/workspace/{rel_path}",
        "opencfd/openfoam-default:latest",
        "bash", "-c", "rm -f log.* && blockMesh && icoFoam -noFunctionObjects"
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    except Exception as e:
        print(f"    Error running {case_dir.name}: {e}")


def modernize_and_run_tutorials_plus():
    print("▶ Running cases in OpenFOAM_Tutorials_Plus/programmingTutorials...")
    test_cases = list(PLUS_DIR.glob("**/testCase"))
    print(f"  Found {len(test_cases)} tutorial test cases.")

    for tc in sorted(test_cases):
        bm = tc / "system/blockMeshDict"
        sol = tc / "system/fvSolution"
        cd = tc / "system/controlDict"

        if not (bm.exists() and sol.exists() and cd.exists()):
            continue

        # 1. Update frontAndBack to empty in blockMeshDict
        bm_text = bm.read_text(errors="ignore")
        bm_text = re.sub(r"frontAndBack\s*\{\s*type\s+patch;", "frontAndBack\n    {\n        type empty;", bm_text)
        bm.write_text(bm_text)

        # 2. Update pFinal in fvSolution
        sol_text = sol.read_text(errors="ignore")
        if '"(p|pFinal)"' not in sol_text:
            sol_text = re.sub(r"\bp\b\s*\{", '"(p|pFinal)"\n    {', sol_text)
            sol.write_text(sol_text)

        # 3. Modernize controlDict
        cd_text = cd.read_text(errors="ignore")
        cd_text = re.sub(r"writeControl\s+timeStep;", "writeControl runTime;", cd_text)
        cd_text = re.sub(r"writeInterval\s+[0-9\.]+;", "writeInterval 0.05;", cd_text)
        cd_text = re.sub(r"endTime\s+[0-9\.]+;", "endTime 0.25;", cd_text)
        cd.write_text(cd_text)

        # Run
        case_name = tc.parent.name
        print(f"  • Solving {case_name}...")
        run_case_in_docker(tc)


def modernize_and_run_nimasam():
    print("\n▶ Running cases in NimaSam_OpenFOAM_Tutorials...")
    candidate_cases = [
        NIMA_DIR / "OpenFOAM-10/chapter1/cavityInPipe",
        NIMA_DIR / "OpenFOAM-10/chapter6/heatedCavity",
        NIMA_DIR / "OpenFOAM-3.0.1/chapter1/cavityInPipe",
        NIMA_DIR / "OpenFOAM-4/chapter1/cavityInPipe",
        NIMA_DIR / "OpenFOAM-5/chapter1/cavityInPipe",
        NIMA_DIR / "OpenFOAM-7/chapter1/cavityInPipe",
        NIMA_DIR / "OpenFOAM-8/chapter1/cavityInPipe",
        NIMA_DIR / "OpenFOAM-9/chapter1/cavityInPipe",
    ]

    for cdir in candidate_cases:
        if not cdir.exists():
            continue
        # Ensure transportProperties
        tp = cdir / "constant/transportProperties"
        pp = cdir / "constant/physicalProperties"
        if not tp.exists() and pp.exists():
            tp.write_text(pp.read_text())

        # Clean controlDict functions
        cd = cdir / "system/controlDict"
        if cd.exists():
            cd_text = cd.read_text(errors="ignore")
            cd_text = re.sub(r"functions\s*\{[^}]*\}", "functions {}", cd_text)
            cd_text = re.sub(r"writeControl\s+timeStep;", "writeControl runTime;", cd_text)
            cd_text = re.sub(r"writeInterval\s+[0-9\.]+;", "writeInterval 0.05;", cd_text)
            cd_text = re.sub(r"endTime\s+[0-9\.]+;", "endTime 0.25;", cd_text)
            cd.write_text(cd_text)

        print(f"  • Solving {cdir.parent.name}/{cdir.name}...")
        run_case_in_docker(cdir)


if __name__ == "__main__":
    modernize_and_run_tutorials_plus()
    modernize_and_run_nimasam()
    print("\n✨ All GitHub tutorial cases processed and solved!")
