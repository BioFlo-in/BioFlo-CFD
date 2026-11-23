#!/usr/bin/env bash
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
