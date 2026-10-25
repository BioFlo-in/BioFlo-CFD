import subprocess
import shutil
from pathlib import Path

repo = Path('external_openfoam_sources/BasicOpenFOAMProgrammingTutorials')
valid_fvsol = Path('external_openfoam_sources/OpenFOAM_Tutorials_Plus/programmingTutorials/OFtutorial00_helloWorld/testCase/system/fvSolution').read_text()

# Fix ownership first
subprocess.run(f'docker run --rm -v {Path.cwd()}:/work opencfd/openfoam-default:latest bash -c "chown -R 1000:1000 /work/external_openfoam_sources/BasicOpenFOAMProgrammingTutorials"', shell=True)

solved = []

for c in sorted(list(repo.glob('**/testCase'))):
    bmd = c / 'system/blockMeshDict'
    if not bmd.exists():
        bmd = c / 'constant/polyMesh/blockMeshDict'
    if not bmd.exists():
        continue
    
    cd = c / 'system/controlDict'
    if not cd.exists():
        continue

    cd_txt = cd.read_text(errors='ignore')
    if 'icoFoam' not in cd_txt and 'application     unknown;' not in cd_txt:
        continue

    # Setup standard fvSolution
    (c / 'system/fvSolution').write_text(valid_fvsol)

    # Setup controlDict
    cd_txt = cd_txt.replace('application     unknown;', 'application     icoFoam;')
    cd_txt = cd_txt.replace('writeControl    timeStep;', 'writeControl    runTime;')
    cd_txt = cd_txt.replace('writeInterval   10;', 'writeInterval   0.05;')
    cd_txt = cd_txt.replace('writeInterval   20;', 'writeInterval   0.05;')
    cd_txt = cd_txt.replace('writeInterval   50;', 'writeInterval   0.05;')
    cd_txt = cd_txt.replace('endTime         0.5;', 'endTime         0.25;')
    cd.write_text(cd_txt)

    # Fix blockMeshDict frontAndBack
    b_txt = bmd.read_text(errors='ignore')
    if 'frontAndBack' in b_txt and 'type patch;' in b_txt:
        b_txt = b_txt.replace('type patch;', 'type empty;')
        bmd.write_text(b_txt)

    # Ensure 0 directory has valid p and U if missing
    if not (c / '0/p').exists():
        ref_0 = Path('external_openfoam_sources/OpenFOAM_Tutorials_Plus/programmingTutorials/OFtutorial00_helloWorld/testCase/0')
        if (c / '0').exists():
            shutil.rmtree(c / '0')
        shutil.copytree(ref_0, c / '0')

    # Ensure transportProperties
    if not (c / 'constant/transportProperties').exists() and (c / 'constant/physicalProperties').exists():
        shutil.copy(c / 'constant/physicalProperties', c / 'constant/transportProperties')

    # Run
    cmd = f'docker run --rm -v {Path.cwd()}:/work opencfd/openfoam-default:latest bash -c "cd /work/{c} && blockMesh && icoFoam"'
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    times = [p.name for p in c.iterdir() if p.is_dir() and p.name not in ['0', 'constant', 'system', 'dynamicCode', 'uniform', 'VTK', 'postProcessing'] and not p.name.startswith('.')]
    
    if len(times) >= 2:
        solved.append(c.relative_to(repo))
        print(f"[{len(solved)}] Solved: {c.relative_to(repo)} ({len(times)} snapshots)")
    else:
        print(f"Failed {c.relative_to(repo)} (exit {res.returncode}): {res.stderr[:200] if res.stderr else res.stdout[-200:]}")
    
    if len(solved) >= 25:
        break

print(f"\nSuccessfully solved {len(solved)} cases in BasicOpenFOAMProgrammingTutorials!")
