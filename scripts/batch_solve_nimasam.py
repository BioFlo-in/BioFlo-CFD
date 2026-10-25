import os
import shutil
import subprocess
from pathlib import Path

repo = Path('external_openfoam_sources/NimaSam_OpenFOAM_Tutorials')
versions = ['OpenFOAM-10', 'OpenFOAM-9', 'OpenFOAM-8', 'OpenFOAM-7', 'OpenFOAM-5', 'OpenFOAM-4', 'OpenFOAM-3.0.1']

solved_count = 0

def run_docker(cmd):
    full_cmd = f'docker run --rm -v {Path.cwd()}:/work opencfd/openfoam-default:latest bash -c "{cmd}"'
    return subprocess.run(full_cmd, shell=True, capture_output=True, text=True)

# First ensure permissions
subprocess.run(f'docker run --rm -v {Path.cwd()}:/work opencfd/openfoam-default:latest bash -c "chown -R 1000:1000 /work/external_openfoam_sources/NimaSam_OpenFOAM_Tutorials"', shell=True)

for v in versions:
    for ch in ['chapter1/cavityInPipe', 'chapter6/heatedCavity']:
        case = repo / v / ch
        if not case.exists():
            continue
        
        # Check if already has >= 2 solved snapshots
        times = [p.name for p in case.iterdir() if p.is_dir() and p.name not in ['0', 'constant', 'system', 'dynamicCode', 'uniform', 'VTK', 'postProcessing'] and not p.name.startswith('.')]
        if len(times) >= 2:
            print(f'[{v} {ch}] Already solved with {len(times)} snapshots')
            solved_count += 1
            continue

        # Adjust controlDict
        cd = case / 'system/controlDict'
        if cd.exists():
            t = cd.read_text(errors='ignore')
            t = t.replace('writeControl    timeStep;', 'writeControl    runTime;')
            t = t.replace('writeInterval   20;', 'writeInterval   0.05;')
            t = t.replace('writeInterval   50;', 'writeInterval   0.05;')
            cd.write_text(t)

        # Ensure transportProperties
        if not (case / 'constant/transportProperties').exists() and (case / 'constant/physicalProperties').exists():
            shutil.copy(case / 'constant/physicalProperties', case / 'constant/transportProperties')

        # Fix blockMeshDict frontAndBack
        bmd = case / 'system/blockMeshDict'
        if not bmd.exists():
            bmd = case / 'constant/polyMesh/blockMeshDict'
        if bmd.exists():
            t = bmd.read_text(errors='ignore')
            t = t.replace('type patch;', 'type empty;') if 'frontAndBack' in t else t
            t = t.replace('frontAndBack\n    {\n        type            patch;', 'frontAndBack\n    {\n        type            empty;')
            bmd.write_text(t)

        res = run_docker(f'cd /work/{case} && blockMesh && icoFoam')
        new_times = [p.name for p in case.iterdir() if p.is_dir() and p.name not in ['0', 'constant', 'system', 'dynamicCode', 'uniform', 'VTK', 'postProcessing'] and not p.name.startswith('.')]
        print(f'[{v} {ch}] status={res.returncode}, snapshots={len(new_times)}')
        if len(new_times) >= 2:
            solved_count += 1

print(f"Total solved cases in NimaSam: {solved_count}")
