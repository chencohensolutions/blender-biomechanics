param(
    [Parameter(Mandatory = $true)]
    [string]$PackagePath,
    [string]$BlenderPath = $env:BLENDER_EXE
)

$ErrorActionPreference = "Stop"

if (-not $BlenderPath) {
    foreach ($commandName in @("blender", "blender-launcher.exe")) {
        $pathCommand = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($pathCommand) {
            $BlenderPath = $pathCommand.Source
            break
        }
    }
}

if (-not $BlenderPath) {
    $searchPatterns = @(
        "$env:ProgramFiles\Blender Foundation\Blender *\blender.exe",
        "${env:ProgramFiles(x86)}\Steam\steamapps\common\Blender\blender.exe",
        "$env:LOCALAPPDATA\Programs\Blender Foundation\Blender *\blender.exe",
        "$env:USERPROFILE\scoop\apps\blender\current\blender.exe"
    )
    $BlenderPath = $searchPatterns |
        ForEach-Object { Get-Item $_ -ErrorAction SilentlyContinue } |
        Sort-Object FullName -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}

if (-not $BlenderPath) {
    $storePackage = Get-AppxPackage "BlenderFoundation.Blender" -ErrorAction SilentlyContinue |
        Sort-Object Version -Descending |
        Select-Object -First 1
    if ($storePackage) {
        $storeAlias = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\blender-launcher.exe"
        if (Test-Path $storeAlias -PathType Leaf) {
            $BlenderPath = $storeAlias
        }
    }
}

if (-not $BlenderPath -or -not (Test-Path $BlenderPath -PathType Leaf)) {
    throw "Blender was not found. Add blender.exe to PATH, set BLENDER_EXE, or pass -BlenderPath."
}

$resolvedPackage = (Resolve-Path $PackagePath).Path
$configPath = Join-Path $resolvedPackage "training_config.json"
$inputPath = Join-Path $resolvedPackage "training_input.blend"
if (-not (Test-Path $configPath -PathType Leaf)) {
    throw "Training configuration not found: $configPath"
}
if (-not (Test-Path $inputPath -PathType Leaf)) {
    throw "Training input not found: $inputPath"
}

$workerPath = Join-Path $PSScriptRoot "headless_train.py"
Write-Host "Starting background training with $BlenderPath"
& $BlenderPath --background --factory-startup --python $workerPath -- --config $configPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}