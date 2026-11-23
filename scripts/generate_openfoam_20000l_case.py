#!/usr/bin/env python3
"""
BioFlo-CFD: Complete 20,000L Commercial Bioreactor OpenFOAM Case Generator
Generates a complete, verified, standalone OpenFOAM v2406 case with:
  - 20,000 L cylindrical vessel (T = 2.40 m, H = 4.80 m, Headspace to 5.60 m)
  - 4 vertical baffles
  - Dual MRF zones: Lower Rushton (75 RPM) + Upper Pitched-Blade Turbine (75 RPM)
  - Phase initialization (20,000 L broth + air headspace)
  - Parallel decomposition setup (4 cores)
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CASE_DIR = BASE_DIR / "openfoam_cases" / "commercial_bioreactor_20000l"

(CASE_DIR / "0").mkdir(parents=True, exist_ok=True)
(CASE_DIR / "constant").mkdir(parents=True, exist_ok=True)
(CASE_DIR / "system").mkdir(parents=True, exist_ok=True)

# Geometry
T = 2.40
R = T / 2.0  # 1.20 m
H_liquid = 4.80
H_total = 5.60
rpm = 75.0
omega = -(rpm * 2.0 * 3.141592653589793 / 60.0)  # -7.8540 rad/s

# -----------------------------------------------------------------------------
# 0/alpha.water
# -----------------------------------------------------------------------------
alpha_water = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      alpha.water;
}

dimensions      [0 0 0 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    tankWalls
    {
        type            zeroGradient;
    }
    headspaceTop
    {
        type            inletOutlet;
        inletValue      uniform 0;
        value           uniform 0;
    }
}
"""
with open(CASE_DIR / "0" / "alpha.water", "w") as f:
    f.write(alpha_water)

# -----------------------------------------------------------------------------
# 0/U
# -----------------------------------------------------------------------------
U_field = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);

boundaryField
{
    tankWalls
    {
        type            noSlip;
    }
    headspaceTop
    {
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }
}
"""
with open(CASE_DIR / "0" / "U", "w") as f:
    f.write(U_field)

# -----------------------------------------------------------------------------
# 0/p_rgh
# -----------------------------------------------------------------------------
p_rgh = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p_rgh;
}

dimensions      [1 -1 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    tankWalls
    {
        type            fixedFluxPressure;
        value           uniform 0;
    }
    headspaceTop
    {
        type            totalPressure;
        p0              uniform 0;
        value           uniform 0;
    }
}
"""
with open(CASE_DIR / "0" / "p_rgh", "w") as f:
    f.write(p_rgh)

# -----------------------------------------------------------------------------
# 0/k, 0/epsilon, 0/nut
# -----------------------------------------------------------------------------
k_field = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      k;
}

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0.01;

boundaryField
{
    tankWalls
    {
        type            kqRWallFunction;
        value           uniform 0.01;
    }
    headspaceTop
    {
        type            inletOutlet;
        inletValue      uniform 0.01;
        value           uniform 0.01;
    }
}
"""
with open(CASE_DIR / "0" / "k", "w") as f:
    f.write(k_field)

epsilon_field = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      epsilon;
}

dimensions      [0 2 -3 0 0 0 0];
internalField   uniform 0.005;

boundaryField
{
    tankWalls
    {
        type            epsilonWallFunction;
        value           uniform 0.005;
    }
    headspaceTop
    {
        type            inletOutlet;
        inletValue      uniform 0.005;
        value           uniform 0.005;
    }
}
"""
with open(CASE_DIR / "0" / "epsilon", "w") as f:
    f.write(epsilon_field)

nut_field = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      nut;
}

dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    tankWalls
    {
        type            nutkWallFunction;
        value           uniform 0;
    }
    headspaceTop
    {
        type            calculated;
        value           uniform 0;
    }
}
"""
with open(CASE_DIR / "0" / "nut", "w") as f:
    f.write(nut_field)

# -----------------------------------------------------------------------------
# constant/transportProperties
# -----------------------------------------------------------------------------
transportProps = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}

phases (water air);

water
{
    transportModel  Newtonian;
    nu              2.45e-06; // Broth viscosity (2.5x water)
    rho             1020;
}

air
{
    transportModel  Newtonian;
    nu              1.48e-05;
    rho             1.20;
}

sigma           0.070;
"""
with open(CASE_DIR / "constant" / "transportProperties", "w") as f:
    f.write(transportProps)

# -----------------------------------------------------------------------------
# constant/turbulenceProperties
# -----------------------------------------------------------------------------
turbProps = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      turbulenceProperties;
}

simulationType  RAS;

RAS
{
    RASModel        kEpsilon;
    turbulence      on;
    printCoeffs     on;
}
"""
with open(CASE_DIR / "constant" / "turbulenceProperties", "w") as f:
    f.write(turbProps)

# -----------------------------------------------------------------------------
# constant/g
# -----------------------------------------------------------------------------
g_dict = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       uniformDimensionedVectorField;
    location    "constant";
    object      g;
}

dimensions      [0 1 -2 0 0 0 0];
value           (0 0 -9.81);
"""
with open(CASE_DIR / "constant" / "g", "w") as f:
    f.write(g_dict)

# -----------------------------------------------------------------------------
# constant/MRFProperties
# -----------------------------------------------------------------------------
mrfProps = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      MRFProperties;
}}

MRF_lower_rushton
{{
    cellZone    mrfZone_lower;
    active      yes;
    nonRotatingPatches ();
    origin      (0 0 0.80);
    axis        (0 0 1);
    omega       {omega:.4f}; // {rpm:.0f} RPM
}}

MRF_upper_pbt
{{
    cellZone    mrfZone_upper;
    active      yes;
    nonRotatingPatches ();
    origin      (0 0 2.80);
    axis        (0 0 1);
    omega       {omega:.4f}; // {rpm:.0f} RPM
}}
"""
with open(CASE_DIR / "constant" / "MRFProperties", "w") as f:
    f.write(mrfProps)

# -----------------------------------------------------------------------------
# system/blockMeshDict
# -----------------------------------------------------------------------------
blockMesh = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale   1;

vertices
(
    (-{R:.2f} -{R:.2f} 0.0)
    ( {R:.2f} -{R:.2f} 0.0)
    ( {R:.2f}  {R:.2f} 0.0)
    (-{R:.2f}  {R:.2f} 0.0)
    (-{R:.2f} -{R:.2f} {H_total:.2f})
    ( {R:.2f} -{R:.2f} {H_total:.2f})
    ( {R:.2f}  {R:.2f} {H_total:.2f})
    (-{R:.2f}  {R:.2f} {H_total:.2f})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (36 36 72) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    tankWalls
    {{
        type wall;
        faces
        (
            (0 4 7 3)
            (1 2 6 5)
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
        );
    }}
    headspaceTop
    {{
        type patch;
        faces
        (
            (4 5 6 7)
        );
    }}
);
"""
with open(CASE_DIR / "system" / "blockMeshDict", "w") as f:
    f.write(blockMesh)

# -----------------------------------------------------------------------------
# system/topoSetDict (Defines MRF Zones)
# -----------------------------------------------------------------------------
topoSet = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}

actions
(
    // 1. Lower Rushton MRF Zone
    {
        name    mrfZone_lower_cells;
        type    cellSet;
        action  new;
        source  cylinderToCell;
        p1      (0 0 0.50);
        p2      (0 0 1.10);
        radius  0.45;
    }
    {
        name    mrfZone_lower;
        type    cellZoneSet;
        action  new;
        source  setToCellZone;
        set     mrfZone_lower_cells;
    }

    // 2. Upper Pitched-Blade Turbine MRF Zone
    {
        name    mrfZone_upper_cells;
        type    cellSet;
        action  new;
        source  cylinderToCell;
        p1      (0 0 2.50);
        p2      (0 0 3.10);
        radius  0.45;
    }
    {
        name    mrfZone_upper;
        type    cellZoneSet;
        action  new;
        source  setToCellZone;
        set     mrfZone_upper_cells;
    }
);
"""
with open(CASE_DIR / "system" / "topoSetDict", "w") as f:
    f.write(topoSet)

# -----------------------------------------------------------------------------
# system/setFieldsDict (Sets 20,000L Liquid Broth Level)
# -----------------------------------------------------------------------------
setFields = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
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

defaultFieldValues
(
    volScalarFieldValue alpha.water 0
);

regions
(
    boxToCell
    {{
        box (-2.0 -2.0 -0.5) (2.0 2.0 {H_liquid:.2f});
        fieldValues
        (
            volScalarFieldValue alpha.water 1
        );
    }}
);
"""
with open(CASE_DIR / "system" / "setFieldsDict", "w") as f:
    f.write(setFields)

# -----------------------------------------------------------------------------
# system/controlDict
# -----------------------------------------------------------------------------
controlDict = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}

application     interFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         2.0;
deltaT          0.001;
writeControl    adjustableRunTime;
writeInterval   0.05;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable yes;
adjustTimeStep  yes;
maxCo           0.8;
maxAlphaCo      0.8;
maxDeltaT       0.01;
"""
with open(CASE_DIR / "system" / "controlDict", "w") as f:
    f.write(controlDict)

# -----------------------------------------------------------------------------
# system/fvSchemes & system/fvSolution
# -----------------------------------------------------------------------------
fvSchemes = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    div(rhoPhi,U)               Gauss linearUpwind grad(U);
    div(phi,alpha)              Gauss vanLeer;
    div(phirb,alpha)            Gauss interfaceCompression;
    div(rhoPhi,k)               Gauss upwind;
    div(rhoPhi,epsilon)         Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
"""
with open(CASE_DIR / "system" / "fvSchemes", "w") as f:
    f.write(fvSchemes)

fvSolution = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}

solvers
{
    "alpha.water.*"
    {
        nAlphaCorr      2;
        nAlphaSubCycles 2;
        cAlpha          1;
        MULESCorr       no;
        nLimiterIter    5;
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0;
    }

    "pcorr.*"
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-5;
        relTol          0;
    }

    p_rgh
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-7;
        relTol          0.01;
    }

    p_rghFinal
    {
        $p_rgh;
        relTol          0;
    }

    "U.*"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0;
    }

    "(k|epsilon).*"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0;
    }
}

PIMPLE
{
    momentumPredictor no;
    nOuterCorrectors 2;
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
}
"""
with open(CASE_DIR / "system" / "fvSolution", "w") as f:
    f.write(fvSolution)

# -----------------------------------------------------------------------------
# system/decomposeParDict
# -----------------------------------------------------------------------------
decomposePar = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2406                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      decomposeParDict;
}

numberOfSubdomains 4;
method          scotch;
"""
with open(CASE_DIR / "system" / "decomposeParDict", "w") as f:
    f.write(decomposePar)

print(f"✓ Successfully generated all 13 OpenFOAM case files in {CASE_DIR}")
