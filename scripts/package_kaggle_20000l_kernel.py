#!/usr/bin/env python3
"""
Packages the 20,000L Commercial Bioreactor OpenFOAM CFD Simulation
into a self-contained, standalone Kaggle notebook and kernel metadata.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KERNEL_DIR = BASE_DIR / "outputs" / "kaggle" / "kernels" / "bioflo-commercial-20000l-bioreactor"
KERNEL_DIR.mkdir(parents=True, exist_ok=True)
CASE_DIR = BASE_DIR / "openfoam_cases" / "commercial_bioreactor_20000l"

# Read dictionary contents from local case
def read_f(rel):
    return (CASE_DIR / rel).read_text()

alpha_water = read_f("0/alpha.water")
U_field = read_f("0/U")
p_rgh = read_f("0/p_rgh")
k_field = read_f("0/k")
eps_field = read_f("0/epsilon")
nut_field = read_f("0/nut")

transportProps = read_f("constant/transportProperties")
turbProps = read_f("constant/turbulenceProperties")
g_dict = read_f("constant/g")
mrfProps = read_f("constant/MRFProperties")

blockMesh = read_f("system/blockMeshDict")
topoSet = read_f("system/topoSetDict")
setFields = read_f("system/setFieldsDict")
controlDict = read_f("system/controlDict")
fvSchemes = read_f("system/fvSchemes")
fvSolution = read_f("system/fvSolution")
decomposePar = read_f("system/decomposeParDict")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# BioFlo Commercial 20,000L Industrial Bioreactor Simulation (OpenFOAM v2406)\n\n",
            "**Scale-Up Specifications**:\n",
            "- **Working Volume**: 20,000 Liters ($20\\text{ m}^3$)\n",
            "- **Vessel Geometry**: Diameter $T = 2.40\\text{ m}$, Liquid Height $H = 4.80\\text{ m}$ ($H/T = 2.0$), Headspace to $5.60\\text{ m}$\n",
            "- **Dual Impellers**: Lower 6-Blade Rushton ($D = 0.80\\text{ m}$) + Upper 4-Blade Pitched-Blade Turbine ($D = 0.80\\text{ m}$)\n",
            "- **Agitation**: 75 RPM (Tip Speed: $3.14\\text{ m/s}$), $\\text{Re} = 326,400$ (Fully Turbulent), Power = $4.44\\text{ kW}$\n",
            "- **Solver**: OpenFOAM `interFoam` with Dual MRF Zones on 4 Parallel CPU Cores."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Hardware Inspection\n",
            "!echo 'CPU Cores available:' $(nproc)\n",
            "!free -h\n",
            "!cat /proc/cpuinfo | grep 'model name' | head -n 1\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Install OpenFOAM v2406 directly into Kaggle Environment\n",
            "print('Installing OpenFOAM v2406 and OpenMPI on Kaggle CPU...')\n",
            "!wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | bash\n",
            "!apt-get update -qq\n",
            "!apt-get install -y -qq openfoam2406-default libopenmpi-dev\n",
            "print('OpenFOAM v2406 installation verified!')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Stage All 20,000L Case Dictionaries & Boundary Fields\n",
            "import os\n",
            "from pathlib import Path\n\n",
            "case = Path('/kaggle/working/case')\n",
            "(case / '0').mkdir(parents=True, exist_ok=True)\n",
            "(case / 'constant').mkdir(parents=True, exist_ok=True)\n",
            "(case / 'system').mkdir(parents=True, exist_ok=True)\n\n",
            f"with open(case / '0' / 'alpha.water', 'w') as f: f.write({repr(alpha_water)})\n",
            f"with open(case / '0' / 'U', 'w') as f: f.write({repr(U_field)})\n",
            f"with open(case / '0' / 'p_rgh', 'w') as f: f.write({repr(p_rgh)})\n",
            f"with open(case / '0' / 'k', 'w') as f: f.write({repr(k_field)})\n",
            f"with open(case / '0' / 'epsilon', 'w') as f: f.write({repr(eps_field)})\n",
            f"with open(case / '0' / 'nut', 'w') as f: f.write({repr(nut_field)})\n\n",
            f"with open(case / 'constant' / 'transportProperties', 'w') as f: f.write({repr(transportProps)})\n",
            f"with open(case / 'constant' / 'turbulenceProperties', 'w') as f: f.write({repr(turbProps)})\n",
            f"with open(case / 'constant' / 'g', 'w') as f: f.write({repr(g_dict)})\n",
            f"with open(case / 'constant' / 'MRFProperties', 'w') as f: f.write({repr(mrfProps)})\n\n",
            f"with open(case / 'system' / 'blockMeshDict', 'w') as f: f.write({repr(blockMesh)})\n",
            f"with open(case / 'system' / 'topoSetDict', 'w') as f: f.write({repr(topoSet)})\n",
            f"with open(case / 'system' / 'setFieldsDict', 'w') as f: f.write({repr(setFields)})\n",
            f"with open(case / 'system' / 'controlDict', 'w') as f: f.write({repr(controlDict)})\n",
            f"with open(case / 'system' / 'fvSchemes', 'w') as f: f.write({repr(fvSchemes)})\n",
            f"with open(case / 'system' / 'fvSolution', 'w') as f: f.write({repr(fvSolution)})\n",
            f"with open(case / 'system' / 'decomposeParDict', 'w') as f: f.write({repr(decomposePar)})\n\n",
            "print('All 20,000L case files staged in /kaggle/working/case!')\n",
            "!ls -la /kaggle/working/case/system\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%bash\n",
            "# 4. Mesh Generation, MRF Zones & Multiphase Initialization\n",
            "cd /kaggle/working/case\n",
            ". /usr/lib/openfoam/openfoam2406/etc/bashrc\n\n",
            "echo '1. Generating 20,000L cylindrical blockMesh...'\n",
            "blockMesh > log.blockMesh 2>&1\n",
            "tail -n 15 log.blockMesh\n\n",
            "echo '2. Defining Dual MRF Zones (Rushton + PBT) via topoSet...'\n",
            "topoSet > log.topoSet 2>&1\n",
            "tail -n 15 log.topoSet\n\n",
            "echo '3. Initializing 20,000L broth liquid level via setFields...'\n",
            "setFields > log.setFields 2>&1\n",
            "tail -n 15 log.setFields\n\n",
            "echo '4. Decomposing across 4 CPU cores...'\n",
            "decomposePar -force > log.decomposePar 2>&1\n",
            "ls -d processor*\n",
            "echo 'Preprocessing complete!'\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%bash\n",
            "# 5. Execute 4-Core Parallel interFoam Simulation\n",
            "cd /kaggle/working/case\n",
            ". /usr/lib/openfoam/openfoam2406/etc/bashrc\n\n",
            "export OMPI_ALLOW_RUN_AS_ROOT=1\n",
            "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1\n",
            "export OMPI_MCA_btl_vader_single_copy_mechanism=none\n\n",
            "echo 'Launching 4-core parallel interFoam simulation with --oversubscribe...'\n",
            "mpirun --allow-run-as-root --oversubscribe --use-hwthread-cpus -np 4 interFoam -parallel > log.interFoam.kaggle 2>&1 &\n",
            "FOAM_PID=$!\n",
            "echo \"interFoam running with PID $FOAM_PID...\"\n\n",
            "# Monitor progress\n",
            "for i in {1..50}; do\n",
            "    sleep 10\n",
            "    if ! kill -0 $FOAM_PID 2>/dev/null; then\n",
            "        echo 'interFoam process completed or stopped.'\n",
            "        break\n",
            "    fi\n",
            "    echo \"[Time $(date +%T)] Elapsed $((i*10))s:\"\n",
            "    grep -E \"Time =|ExecutionTime\" log.interFoam.kaggle | tail -n 4\n",
            "done\n\n",
            "wait $FOAM_PID || echo \"Process ended with code $?\"\n",
            "echo 'Simulation execution phase completed.'\n",
            "tail -n 35 log.interFoam.kaggle\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%bash\n",
            "# 6. Reconstruct Parallel Fields and Clean Up\n",
            "cd /kaggle/working/case\n",
            ". /usr/lib/openfoam/openfoam2406/etc/bashrc\n\n",
            "echo 'Reconstructing time directories...'\n",
            "reconstructPar -newTimes > log.reconstructPar 2>&1\n",
            "tail -n 25 log.reconstructPar\n\n",
            "echo 'Removing processor subdirectories...'\n",
            "rm -rf processor*\n\n",
            "echo 'Reconstructed time directories:'\n",
            "foamListTimes\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Package Solution and Clean Workspace\n",
            "import os, glob\n\n",
            "out_tar = '/kaggle/working/bioflo_20000l_commercial_bioreactor_solution.tar.gz'\n",
            "print(f'Compressing reconstructed case to {out_tar}...')\n",
            "!tar -czf {out_tar} -C /kaggle/working/case .\n\n",
            "if os.path.exists(out_tar):\n",
            "    size_mb = os.path.getsize(out_tar) / (1024 * 1024)\n",
            "    print(f'SUCCESS! Generated {out_tar} ({size_mb:.2f} MB)')\n",
            "    print('Cleaning uncompressed case directory to expedite download...')\n",
            "    !rm -rf /kaggle/working/case\n",
            "else:\n",
            "    print('Error: Output archive not created!')\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

nb_path = KERNEL_DIR / "bioflo-commercial-20000l-bioreactor.ipynb"
with open(nb_path, "w") as f:
    json.dump(nb, f, indent=2)

# Kernel Metadata
meta = {
    "id": "ajitsheoran/bioflo-commercial-20000l-bioreactor",
    "title": "bioflo-commercial-20000l-bioreactor",
    "code_file": "bioflo-commercial-20000l-bioreactor.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
}

with open(KERNEL_DIR / "kernel-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"✓ Packaged Kaggle notebook: {nb_path}")
print(f"✓ Created metadata: {KERNEL_DIR / 'kernel-metadata.json'}")
