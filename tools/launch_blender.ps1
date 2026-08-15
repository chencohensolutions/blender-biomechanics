param(
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
    throw "Blender was not found. Install Blender, add blender.exe to PATH, or set BLENDER_EXE to its full path. Microsoft Store installs require an enabled BlenderFoundation.Blender package."
}

$loaderPath = Join-Path $PSScriptRoot "dev_load.py"
Write-Host "Starting $BlenderPath"
& $BlenderPath --python $loaderPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}