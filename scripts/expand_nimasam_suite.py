import shutil
import subprocess
from pathlib import Path

repo = Path('external_openfoam_sources/NimaSam_OpenFOAM_Tutorials')

# Variations for cavityInPipe
pipe_variations = [
    ('cavityInPipe_re200', 0.2, 0.001),
    ('cavityInPipe_re500', 0.5, 0.001),
    ('cavityInPipe_re2000', 2.0, 0.001),
    ('cavityInPipe_viscous', 0.1, 0.01),
]

for name, u_inlet, nu in pipe_variations:
    target = repo / 'OpenFOAM-10/chapter1' / name
    if (target / '0.25').exists():
        print(f"{name} already solved")
        continue
    
    src = repo / 'OpenFOAM-10/chapter1/cavityInPipe'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src, target, ignore=shutil.ignore_patterns('0.*', '1*', '2*', '3*', '4*', '5*', '6*', '7*', '8*', '9*'))
    
    # modify U inlet
    u_file = target / '0/U'
    if u_file.exists():
        txt = u_file.read_text()
        txt = txt.replace('uniform (1 0 0)', f'uniform ({u_inlet} 0 0)')
        u_file.write_text(txt)
        
    # modify nu
    tp_file = target / 'constant/transportProperties'
    if tp_file.exists():
        txt = tp_file.read_text()
        txt = txt.replace('0.01;', f'{nu};')
        tp_file.write_text(txt)
        
    # run
    cmd = f'docker run --rm -v {Path.cwd()}:/work opencfd/openfoam-default:latest bash -c "cd /work/{target} && blockMesh && icoFoam"'
    subprocess.run(cmd, shell=True)
    times = [p.name for p in target.iterdir() if p.is_dir() and p.name not in ['0', 'constant', 'system', 'dynamicCode', 'uniform', 'VTK', 'postProcessing'] and not p.name.startswith('.')]
    print(f"Solved {name}: {len(times)} snapshots")

# Variations for heatedCavity
cavity_variations = [
    ('heatedCavity_fast_u2', 2.0, 0.01),
    ('heatedCavity_slow_u02', 0.2, 0.01),
    ('heatedCavity_re500', 1.0, 0.002),
    ('heatedCavity_re2000', 1.0, 0.0005),
]

for name, u_wall, nu in cavity_variations:
    target = repo / 'OpenFOAM-10/chapter6' / name
    if (target / '0.5').exists():
        print(f"{name} already solved")
        continue
    
    src = repo / 'OpenFOAM-10/chapter6/heatedCavity'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src, target, ignore=shutil.ignore_patterns('0.*', '1*', '2*', '3*', '4*', '5*', '6*', '7*', '8*', '9*'))
    
    # modify movingWall U
    u_file = target / '0/U'
    if u_file.exists():
        txt = u_file.read_text()
        txt = txt.replace('uniform (1 0 0)', f'uniform ({u_wall} 0 0)')
        u_file.write_text(txt)
        
    # modify nu
    tp_file = target / 'constant/transportProperties'
    if tp_file.exists():
        txt = tp_file.read_text()
        txt = txt.replace('0.01;', f'{nu};')
        tp_file.write_text(txt)
        
    # run
    cmd = f'docker run --rm -v {Path.cwd()}:/work opencfd/openfoam-default:latest bash -c "cd /work/{target} && blockMesh && icoFoam"'
    subprocess.run(cmd, shell=True)
    times = [p.name for p in target.iterdir() if p.is_dir() and p.name not in ['0', 'constant', 'system', 'dynamicCode', 'uniform', 'VTK', 'postProcessing'] and not p.name.startswith('.')]
    print(f"Solved {name}: {len(times)} snapshots")

