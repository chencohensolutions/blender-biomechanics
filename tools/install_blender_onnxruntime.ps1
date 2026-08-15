<#
.SYNOPSIS
    Install a CPU-only onnxruntime build into Blender's per-user module directory.

.DESCRIPTION
    Blender's embedded Python is not a normal virtual environment, so packages
    must be installed with `--target` into the directory Blender's `bpy.utils`
    module search path (see biomechanics_addon/onnx_compat.py:user_modules_directory).
    This keeps native wheels out of the add-on ZIP and out of source control.

.PARAMETER BlenderPythonExe
    Path to Blender's embedded python.exe, e.g.
    "C:\Program Files\Blender Foundation\Blender 4.1\4.1\python\bin\python.exe".
    If omitted, this script searches common Blender install locations.

.PARAMETER TargetDirectory
    Destination module directory. If omitted, defaults to Blender's per-user
    scripts/modules folder under %APPDATA%\Blender Foundation\Blender\<version>\scripts\modules.
#>
param(
    [string]$BlenderPythonExe,
    [string]$TargetDirectory
)

$ErrorActionPreference = "Stop"

if (-not $BlenderPythonExe) {
    $searchPatterns = @(
        "$env:ProgramFiles\Blender Foundation\Blender *\*\python\bin\python.exe",
        "$env:LOCALAPPDATA\Programs\Blender Foundation\Blender *\*\python\bin\python.exe"
    )
    $BlenderPythonExe = $searchPatterns |
        ForEach-Object { Get-Item $_ -ErrorAction SilentlyContinue } |
        Sort-Object FullName -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}

if (-not $BlenderPythonExe -or -not (Test-Path $BlenderPythonExe -PathType Leaf)) {
    throw "Blender's embedded python.exe was not found. Pass -BlenderPythonExe explicitly."
}

Write-Host "Using Blender Python: $BlenderPythonExe"

if (-not $TargetDirectory) {
    $blenderVersionOutput = & $BlenderPythonExe -c "import sys; print('%d.%d' % sys.version_info[:2])"
    Write-Host "Blender embedded Python version: $blenderVersionOutput"
    $candidateRoots = Get-ChildItem "$env:APPDATA\Blender Foundation\Blender" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    if (-not $candidateRoots) {
        throw "Could not locate a Blender user config folder. Pass -TargetDirectory explicitly."
    }
    $TargetDirectory = Join-Path $candidateRoots[0].FullName "scripts\modules"
}

New-Item -ItemType Directory -Force -Path $TargetDirectory | Out-Null
Write-Host "Installing CPU onnxruntime into: $TargetDirectory"

& $BlenderPythonExe -m pip install --upgrade --target $TargetDirectory "onnxruntime==1.20.1"

if ($LASTEXITCODE -ne 0) {
    throw "pip install failed with exit code $LASTEXITCODE"
}

Write-Host "Done. In Blender, run 'Check Neural Model Compatibility' to verify the import."
