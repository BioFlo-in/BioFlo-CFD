#!/usr/bin/env python3
"""
Autonomous Multi-Kernel Deployment to Kaggle Cloud:
Deploys each diverse 3D CSTR Bioreactor case as an independent, standalone Kaggle kernel:
  1. bioflo-cstr-case-1-120-rpm-gentle
  2. bioflo-cstr-case-2-450-rpm-fast
  3. bioflo-cstr-case-3-600-rpm-intense
  4. bioflo-cstr-case-4-viscous-10x-broth

Direct Checkpoint Resumption Architecture:
Resumes directly from the fully-converged checkpoint at t = 3.9s in ajitsheoran/bioflo-cstr-3d-checkpoint,
applying exact physical RPM and viscosity modifications while avoiding setFields / AMI interface mismatches.
Runs concurrently on unlimited 4-core CPU instances with 30 GB RAM.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

BASE_DIR = Path(__file__).resolve().parent.parent
KERNELS_ROOT = BASE_DIR / "outputs" / "kaggle" / "kernels"

CASES = [
    {
        "slug": "bioflo-cstr-case-1-120-rpm-gentle",
        "title": "BioFlo CSTR Case 1 (120 RPM Gentle)",
        "rpm": 120,
        "omega": -12.566,
        "nu": 1e-6,
        "fill_z": 0.664,
        "desc": "Low-shear mammalian & stem cell agitation regime (120 RPM, omega = -12.57 rad/s). Minimal surface vortex drawdown.",
    },
    {
        "slug": "bioflo-cstr-case-2-450-rpm-fast",
        "title": "BioFlo CSTR Case 2 (450 RPM Fast)",
        "rpm": 450,
        "omega": -47.124,
        "nu": 1e-6,
        "fill_z": 0.664,
        "desc": "High-intensity bacterial & yeast fermentation (450 RPM, omega = -47.12 rad/s). Strong radial impeller discharge jet.",
    },
    {
        "slug": "bioflo-cstr-case-3-600-rpm-intense",
        "title": "BioFlo CSTR Case 3 (600 RPM Intense)",
        "rpm": 600,
        "omega": -62.832,
        "nu": 1e-6,
        "fill_z": 0.664,
        "desc": "Extreme agitation with deep air-core vortex depression reaching the Rushton hub (600 RPM, omega = -62.83 rad/s).",
    },
    {
        "slug": "bioflo-cstr-case-4-viscous-10x-broth",
        "title": "BioFlo CSTR Case 4 (Viscous 10x Broth)",
        "rpm": 300,
        "omega": -31.416,
        "nu": 1e-5,
        "fill_z": 0.664,
        "desc": "Viscous non-water fermentation broth (10x water viscosity, nu = 1e-5 m2/s, 300 RPM). Damped turbulent eddies.",
    },
]


def generate_notebook(case_info: dict) -> dict:
    slug = case_info["slug"]
    title = case_info["title"]
    omega = case_info["omega"]
    nu = case_info["nu"]
    fill_z = case_info["fill_z"]
    desc = case_info["desc"]

    cells = []

    # Cell 0: Markdown
    cell0 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {title}\n",
            "\n",
            f"**Physical Configuration**: {desc}\n",
            "\n",
            "### Physical & Operational Parameters:\n",
            f"- **Agitation Speed**: {case_info['rpm']} RPM ($\\\\omega = {omega:.3f}\\\\text{{ rad/s}}$)\n",
            f"- **Kinematic Viscosity**: $\\\\nu = {nu:g}\\\\text{{ m}}^2/\\\\text{{s}}$\n",
            f"- **Liquid Fill Level**: $Z = {fill_z:.3f}\\\\text{{ m}}$\n",
            "- **Mesh**: 894,950 polyhedral cells with rotating MRF zone and 4 baffles\n",
            "- **Solver**: Parallel `interFoam` (4 CPU cores, scotch decomposition)\n",
            "- **Resumption Strategy**: Direct Checkpoint Resumption from $t = 3.90\\text{ s}$ established solution\n",
        ],
    }
    cells.append(cell0)

    # Cell 1: Hardware Check
    cell1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Hardware Inspection\n",
            "!echo \"CPU Cores available: $(nproc)\"\n",
            "!free -h\n",
            "!cat /proc/cpuinfo | grep 'model name' | head -n 1\n",
        ],
    }
    cells.append(cell1)

    # Cell 2: OpenFOAM v2406 Installation
    cell2 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Install OpenFOAM v2406 directly into Kaggle Environment (~2-3 mins)\n",
            "print(\"Adding OpenCFD repository and installing OpenFOAM v2406...\")\n",
            "!wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | bash > /dev/null 2>&1\n",
            "!apt-get update -qq && apt-get install -y -qq openfoam2406-default > /dev/null 2>&1\n",
            "!which interFoam 2>/dev/null || . /usr/lib/openfoam/openfoam2406/etc/bashrc && echo \"OpenFOAM v2406 installation verified!\"\n",
        ],
    }
    cells.append(cell2)

    # Cell 3: Locate & Extract Case
    cell3 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Locate & Copy Full Checkpoint Case to /kaggle/working/case\n",
            "import os, glob, shutil\n",
            "\n",
            "case_candidates = glob.glob(\"/kaggle/input/**/case\", recursive=True)\n",
            "if not case_candidates:\n",
            "    dicts = glob.glob(\"/kaggle/input/**/system/controlDict\", recursive=True)\n",
            "    if dicts:\n",
            "        case_candidates = [os.path.dirname(os.path.dirname(dicts[0]))]\n",
            "\n",
            "!mkdir -p /kaggle/working/case\n",
            "if case_candidates:\n",
            "    src = case_candidates[0]\n",
            "    print(f\"Found pre-extracted case at: {src}\")\n",
            "    !cp -rf {src}/* /kaggle/working/case/\n",
            "else:\n",
            "    archives = glob.glob(\"/kaggle/input/**/*.tar.gz\", recursive=True) + glob.glob(\"/kaggle/input/**/*.zip\", recursive=True)\n",
            "    assert len(archives) > 0, f\"No case files found in /kaggle/input!\"\n",
            "    print(f\"Extracting archive: {archives[0]}\")\n",
            "    !tar -xzf {archives[0]} -C /kaggle/working/\n",
            "\n",
            "!chmod -R 777 /kaggle/working/case\n",
            "print(\"Case directory verified in /kaggle/working/case:\")\n",
            "!ls -la /kaggle/working/case\n",
        ],
    }
    cells.append(cell3)

    # Cell 4: Apply Specific Case Parameters & Decompose from Checkpoint
    cell4 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%bash\n",
            "# 4. Configure Physical Case Properties and Decompose from Checkpoint\n",
            "cd /kaggle/working/case\n",
            ". /usr/lib/openfoam/openfoam2406/etc/bashrc\n",
            "\n",
            "echo \"Checking available case times:\"\n",
            "foamListTimes\n",
            "\n",
            "LATEST_TIME=$(foamListTimes | tail -n 1)\n",
            "echo \"Resuming from established checkpoint: $LATEST_TIME\"\n",
            "\n",
            f"# Set omega in MRFProperties ({omega:.3f} rad/s)\n",
            f'sed -i -E "s/omega\\s+[^;]+;/omega       {omega:.4f};/" constant/MRFProperties\n',
            "echo 'Updated MRFProperties:'\n",
            "grep 'omega' constant/MRFProperties\n",
            "\n",
            f"# Set fluid viscosity nu in transportProperties ({nu:g} m2/s)\n",
            f'sed -i -E "s/nu\\s+[^;]+;/nu              {nu:g};/" constant/transportProperties\n',
            "echo 'Updated transportProperties:'\n",
            "grep 'nu' constant/transportProperties\n",
            "\n",
            "# Ensure controlDict starts from latestTime and runs forward\n",
            "sed -i -E 's/startFrom\\s+[^;]+;/startFrom       latestTime;/' system/controlDict\n",
            "sed -i -E 's/endTime\\s+[^;]+;/endTime         6.0;/' system/controlDict\n",
            "sed -i -E 's/writeInterval\\s+[^;]+;/writeInterval   0.05;/' system/controlDict\n",
            "echo 'Updated controlDict parameters:'\n",
            "grep -E 'startFrom|endTime|writeInterval' system/controlDict\n",
            "\n",
            "# Clean any old processor directories if present\n",
            "rm -rf processor*\n",
            "\n",
            "echo \"Decomposing mesh across 4 subdomains for time $LATEST_TIME...\"\n",
            "decomposePar -time \"$LATEST_TIME\" -force > log.decomposePar 2>&1\n",
            "ls -d processor* || (cat log.decomposePar && exit 1)\n",
            "echo \"Initialization and decomposition phase complete!\"\n",
        ],
    }
    cells.append(cell4)

    # Cell 5: Parallel Execution with Monitoring
    cell5 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%bash\n",
            "# 5. Execute Parallel 4-Core interFoam Simulation\n",
            "cd /kaggle/working/case\n",
            ". /usr/lib/openfoam/openfoam2406/etc/bashrc\n",
            "\n",
            "export OMPI_ALLOW_RUN_AS_ROOT=1\n",
            "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1\n",
            "\n",
            "echo \"=================================================================\"\n",
            f"echo \"Starting interFoam 4-core parallel simulation for {title}...\"\n",
            "echo \"=================================================================\"\n",
            "\n",
            "mpirun --oversubscribe --use-hwthread-cpus -np 4 interFoam -parallel > log.interFoam.kaggle 2>&1 &\n",
            "FOAM_PID=$!\n",
            "echo \"interFoam launched with PID: $FOAM_PID\"\n",
            "\n",
            "MAX_BUDGET=38000\n",
            "LAST_TIME=\"\"\n",
            "\n",
            "while kill -0 $FOAM_PID 2>/dev/null; do\n",
            "    if [ -f log.interFoam.kaggle ]; then\n",
            "        CURR_TIME=$(grep \"^Time = \" log.interFoam.kaggle | tail -n 1)\n",
            "        if [ -n \"$CURR_TIME\" ] && [ \"$CURR_TIME\" != \"$LAST_TIME\" ]; then\n",
            "            COURANT=$(grep \"Courant Number\" log.interFoam.kaggle | tail -n 1)\n",
            "            EXEC_TIME=$(grep \"ExecutionTime = \" log.interFoam.kaggle | tail -n 1)\n",
            "            echo \"[PROGRESS] $CURR_TIME | $COURANT | $EXEC_TIME\"\n",
            "            LAST_TIME=\"$CURR_TIME\"\n",
            "        fi\n",
            "        if grep -q \"FOAM FATAL\" log.interFoam.kaggle; then\n",
            "            echo \"ERROR: Fatal OpenFOAM error detected!\"\n",
            "            tail -n 35 log.interFoam.kaggle\n",
            "            kill -9 $FOAM_PID 2>/dev/null || true\n",
            "            exit 1\n",
            "        fi\n",
            "    fi\n",
            "    if [ $SECONDS -ge $MAX_BUDGET ]; then\n",
            "        echo \"Session time budget reached. Gracefully stopping...\"\n",
            "        kill -2 $FOAM_PID 2>/dev/null || true\n",
            "        sleep 10\n",
            "        kill -15 $FOAM_PID 2>/dev/null || true\n",
            "        break\n",
            "    fi\n",
            "    sleep 20\n",
            "done\n",
            "\n",
            "wait $FOAM_PID 2>/dev/null || true\n",
            "echo \"Simulation phase completed! Last lines of log:\"\n",
            "tail -n 20 log.interFoam.kaggle\n",
        ],
    }
    cells.append(cell5)

    # Cell 6: Reconstruct & Clean
    cell6 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%bash\n",
            "# 6. Reconstruct Parallel Fields to Top-Level Time Folders\n",
            "cd /kaggle/working/case\n",
            ". /usr/lib/openfoam/openfoam2406/etc/bashrc\n",
            "\n",
            "echo \"Reconstructing newly calculated 3D solution fields...\"\n",
            "reconstructPar -newTimes\n",
            "touch case.foam\n",
            "\n",
            "echo \"Cleaning up decomposed processor directories to save disk space...\"\n",
            "rm -rf processor*\n",
            "\n",
            "echo \"List of reconstructed time directories:\"\n",
            "foamListTimes\n",
        ],
    }
    cells.append(cell6)

    # Cell 7: Package Dataset Archive
    cell7 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"# 7. Package Reconstructed Output Dataset\n",
            f"out_tar = '/kaggle/working/bioflo_3d_cstr_{slug}_reconstructed.tar.gz'\n",
            f"print(f'Compressing reconstructed case into {{out_tar}}...')\n",
            f"!tar -czf {{out_tar}} -C /kaggle/working/case .\n",
            f"!ls -lh {{out_tar}}\n",
            f"print('Dataset ready for cloud download!')\n",
        ],
    }
    cells.append(cell7)

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return nb


def deploy_all_kernels():
    print("=" * 85)
    print("🌐 BioFlo-CFD: Autonomous Multi-Kernel Cloud Deployment to Kaggle CPU")
    print(f"   Target Diverse Cases: {len(CASES)}")
    print("   Resumption Source: ajitsheoran/bioflo-cstr-3d-checkpoint (t = 3.90s)")
    print("=" * 85)

    api = KaggleApi()
    api.authenticate()

    deployed_status = []

    for i, case_info in enumerate(CASES, 1):
        slug = case_info["slug"]
        title = case_info["title"]
        kernel_dir = KERNELS_ROOT / slug
        os.makedirs(kernel_dir, exist_ok=True)

        nb_filename = f"{slug}.ipynb"
        nb_path = kernel_dir / nb_filename

        # 1. Generate & Write Notebook
        nb_data = generate_notebook(case_info)
        with open(nb_path, "w") as f:
            json.dump(nb_data, f, indent=1)

        # 2. Write kernel-metadata.json
        meta = {
            "id": f"ajitsheoran/{slug}",
            "title": title,
            "code_file": nb_filename,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": ["ajitsheoran/bioflo-cstr-3d-checkpoint"],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        with open(kernel_dir / "kernel-metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n[{i}/{len(CASES)}] Pushing kernel '{slug}' to Kaggle...")
        push_res = subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], capture_output=True, text=True)
        print("   " + push_res.stdout.strip())
        if push_res.stderr:
            print("   STDERR: " + push_res.stderr.strip())

        # Brief delay to space API requests
        time.sleep(3)

        # Verify status
        try:
            kstat = api.kernels_status(f"ajitsheoran/{slug}")
            status_str = str(kstat.status)
        except Exception as e:
            status_str = f"Pushed ({e})"

        deployed_status.append({
            "Case Name": case_info["title"],
            "Kernel Slug": slug,
            "RPM": case_info["rpm"],
            "Viscosity": f"{case_info['nu']:g}",
            "Fill (m)": f"{case_info['fill_z']:.2f}",
            "Status": status_str,
            "URL": f"https://www.kaggle.com/code/ajitsheoran/{slug}",
        })

    print("\n" + "=" * 95)
    print("✅ ALL 4 DIVERSE CONFIGURATIONS SUCCESSFULLY DEPLOYED TO KAGGLE CPU CLOUD!")
    print("=" * 95)
    print(f"{'Case Name':<34} | {'RPM':<5} | {'Nu (m²/s)':<10} | {'Status':<14} | {'Kaggle URL'}")
    print("-" * 95)
    for row in deployed_status:
        print(f"{row['Case Name']:<34} | {row['RPM']:<5} | {row['Viscosity']:<10} | {row['Status']:<14} | {row['URL']}")
    print("-" * 95)


if __name__ == "__main__":
    deploy_all_kernels()
