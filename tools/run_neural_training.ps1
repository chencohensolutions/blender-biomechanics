[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PackagePath,
    [string]$BlenderPath = $env:BLENDER_EXE,
    [string]$PythonPath,
    [string]$DatasetPath,
    [string]$ModelPath,
    [int]$SampleBudget = 2000,
    [double]$MinimumClearanceMm = 0.5,
    [int]$MaxBalanceRounds = 2,
    [int]$BalanceRoundSize = 250,
    [int]$MaxRefinementPairs = 150,
    [int]$RefinementRounds = 3,
    [int]$RefinementSearchSubset = 2000,
    [int]$ChunkSize = 500,
    [int]$Epochs = 300,
    [int]$BatchSize = 32,
    [int]$HiddenSize = 128,
    [double]$LearningRate = 0.001,
    [double]$WeightDecay = 0.0001,
    [int]$Seed = -1,
    [double]$LeakageRadiusMm = 2.0,
    [double]$FullMeshAuthorityWeight = 20.0,
    [double]$TargetFalseSafeRate = 0.005,
    [double]$BoundaryWeightScale = 4.0,
    [double]$BoundaryBandwidthMm = 1.5,
    [double]$NeutralAnchorWeight = 5000.0,
    [double]$RotationAnchorWeight = 100.0,
    [double]$TranslationAnchorWeight = 100.0,
    [int]$Opset = 18,
    [double]$ParityTolerance = 0.005,
    [Nullable[int]]$CpuThreads,
    [switch]$ActiveLearning,
    [int]$ActiveRounds = 5,
    [int]$CandidatesPerRound = 4000,
    [int]$MaxNewSamplesPerRound = 60,
    [int]$MaxTotalNewSamples = 300,
    [double]$UncertaintyBand = 0.15,
    [Nullable[double]]$MaxRuntimeSeconds,
    [switch]$InstallRequirements,
    [string[]]$TrainArgument = @()
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot ".." -Resolve)).Path
$datasetScript = Join-Path $PSScriptRoot "generate_neural_dataset.py"
$trainerScript = Join-Path $PSScriptRoot "train_constraint_model.py"
$activeLearningScript = Join-Path $PSScriptRoot "active_learning_loop.py"
$requirementsPath = Join-Path $repositoryRoot "requirements-neural.txt"

function Format-CommandArgument {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value)
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Format-CommandLine {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $false)][string[]]$Arguments = @()
    )
    $parts = @((Format-CommandArgument $Executable))
    foreach ($argument in $Arguments) {
        $parts += Format-CommandArgument ([string]$argument)
    }
    return ($parts -join " ")
}

function Find-Executable {
    param(
        [string]$ExplicitPath,
        [string[]]$CommandNames,
        [string[]]$SearchPatterns,
        [string]$NotFoundMessage
    )
    if ($ExplicitPath) {
        $resolved = Resolve-Path $ExplicitPath -ErrorAction SilentlyContinue
        if ($resolved -and (Test-Path $resolved.Path -PathType Leaf)) {
            return $resolved.Path
        }
        throw "Executable not found: $ExplicitPath"
    }

    foreach ($commandName in $CommandNames) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    $matches = @(
        $SearchPatterns |
            ForEach-Object { Get-Item $_ -ErrorAction SilentlyContinue } |
            Where-Object { $_.PSIsContainer -eq $false }
    )
    if ($matches.Count -gt 0) {
        return ($matches | Sort-Object FullName -Descending | Select-Object -First 1).FullName
    }
    throw $NotFoundMessage
}

$resolvedPackage = (Resolve-Path $PackagePath -ErrorAction Stop).Path
$datasetPath = if ($DatasetPath) { [IO.Path]::GetFullPath($DatasetPath) } else { Join-Path $resolvedPackage "dataset" }
$modelPath = if ($ModelPath) { [IO.Path]::GetFullPath($ModelPath) } else { Join-Path $resolvedPackage "model" }
$statusPath = Join-Path $resolvedPackage "neural_training_status.json"
$logPath = Join-Path $resolvedPackage "neural_training.log"
$runId = [guid]::NewGuid().ToString()

$status = [ordered]@{
    schema_version = 1
    run_id = $runId
    state = "running"
    current_stage = "validation"
    command = $null
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    ended_at = $null
    exit_code = $null
    package_dir = $resolvedPackage
    dataset_dir = $datasetPath
    model_dir = $modelPath
    log_path = $logPath
    status_path = $statusPath
    error = $null
    stages = @()
    artifacts = [ordered]@{
        package = Join-Path $resolvedPackage "package.json"
        dataset_meta = Join-Path $datasetPath "dataset_meta.json"
        model_manifest = Join-Path $modelPath "model_manifest.json"
        metrics = Join-Path $modelPath "metrics.json"
        onnx = Join-Path $modelPath "constraint_model.onnx"
    }
    metrics = $null
}

function Write-RunStatus {
    $temporaryPath = "$statusPath.$PID.tmp"
    $json = $status | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText($temporaryPath, $json, [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporaryPath -Destination $statusPath -Force
}

function Write-RunLog {
    param([string]$Message)
    $line = "[$((Get-Date).ToUniversalTime().ToString('o'))] $Message"
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
    Write-Host $Message
}

function Invoke-NeuralStage {
    param(
        [Parameter(Mandatory = $true)][string]$StageName,
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $false)][string[]]$Arguments = @()
    )

    $commandLine = Format-CommandLine $Executable $Arguments
    $stage = [ordered]@{
        name = $StageName
        command = $commandLine
        started_at = (Get-Date).ToUniversalTime().ToString("o")
        ended_at = $null
        exit_code = $null
        result = "running"
    }
    $status.current_stage = $StageName
    $status.command = $commandLine
    $status.error = $null
    Write-RunStatus
    Write-RunLog "[$StageName] command: $commandLine"

    $previousErrorActionPreference = $ErrorActionPreference
    $previousPythonUtf8 = $env:PYTHONUTF8
    $previousPythonIoEncoding = $env:PYTHONIOENCODING
    try {
        # Native tools write useful warnings to stderr; do not let PowerShell's
        # Stop preference turn those warnings into a failed stage.
        $ErrorActionPreference = "Continue"
        $env:PYTHONUTF8 = "1"
        $env:PYTHONIOENCODING = "utf-8"
        & $Executable @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
    }
    catch {
        $exitCode = 1
        $stage.error = $_.Exception.Message
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        if ($null -eq $previousPythonUtf8) {
            Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
        }
        else {
            $env:PYTHONUTF8 = $previousPythonUtf8
        }
        if ($null -eq $previousPythonIoEncoding) {
            Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue
        }
        else {
            $env:PYTHONIOENCODING = $previousPythonIoEncoding
        }
    }

    $stage.ended_at = (Get-Date).ToUniversalTime().ToString("o")
    $stage.exit_code = $exitCode
    if ($exitCode -ne 0) {
        $stage.result = "failed"
        $status.stages += ,$stage
        $status.state = "failed"
        $status.exit_code = $exitCode
        $status.error = if ($stage.error) { $stage.error } else { "$StageName failed with exit code $exitCode" }
        Write-RunStatus
        throw $status.error
    }

    $stage.result = "completed"
    $status.stages += ,$stage
    $status.current_stage = $null
    $status.command = $null
    Write-RunStatus
}

function Assert-File {
    param([string]$Path, [string]$Description)
    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "$Description not found: $Path"
    }
}

function Update-ArtifactStatus {
    foreach ($key in @("package", "dataset_meta", "model_manifest", "metrics", "onnx")) {
        $artifactPath = $status.artifacts[$key]
        $status.artifacts[$key] = [ordered]@{
            path = $artifactPath
            exists = Test-Path $artifactPath -PathType Leaf
        }
    }
    $metricsPath = $status.artifacts.metrics.path
    if (Test-Path $metricsPath -PathType Leaf) {
        try {
            $metrics = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
            $status.metrics = [ordered]@{
                holdout_full_mesh = $metrics.holdout_full_mesh
                holdout_all_sources = $metrics.holdout_all_sources
                best_validation_loss = $metrics.best_validation_loss
            }
        }
        catch {
            $status.metrics = [ordered]@{ error = "Could not read metrics.json: $($_.Exception.Message)" }
        }
    }
}

try {
    New-Item -ItemType Directory -Path $resolvedPackage -Force | Out-Null
    New-Item -ItemType Directory -Path $datasetPath -Force | Out-Null
    New-Item -ItemType Directory -Path $modelPath -Force | Out-Null
    Write-RunStatus
    Write-RunLog "Run $runId started. Status: $statusPath"

    Assert-File (Join-Path $resolvedPackage "package.json") "Neural package"
    Assert-File (Join-Path $resolvedPackage "training_input.blend") "Background Blender input"
    Assert-File $datasetScript "Dataset generator"
    Assert-File $trainerScript "Constraint trainer"
    if ($ActiveLearning) {
        Assert-File $activeLearningScript "Active-learning loop"
    }

    $blenderSearchPatterns = @(
        "$env:ProgramFiles\Blender Foundation\Blender *\blender.exe",
        "${env:ProgramFiles(x86)}\Steam\steamapps\common\Blender\blender.exe",
        "$env:LOCALAPPDATA\Programs\Blender Foundation\Blender *\blender.exe",
        "$env:USERPROFILE\scoop\apps\blender\current\blender.exe",
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\blender-launcher.exe"
    )
    $BlenderPath = Find-Executable $BlenderPath @("blender", "blender-launcher.exe") $blenderSearchPatterns `
        "Blender was not found. Add blender.exe to PATH, set BLENDER_EXE, or pass -BlenderPath."

    $venvPython = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
    if (-not $PythonPath) {
        if (Test-Path $venvPython -PathType Leaf) {
            $PythonPath = $venvPython
        }
        else {
            $systemPython = Find-Executable $null @("python", "python3") @() `
                "Python was not found. Install Python or pass -PythonPath."
            Invoke-NeuralStage "environment" $systemPython @("-m", "venv", (Join-Path $repositoryRoot ".venv"))
            $PythonPath = $venvPython
        }
    }
    Assert-File $PythonPath "External Python interpreter"

    if ($InstallRequirements) {
        Assert-File $requirementsPath "Neural requirements"
        Invoke-NeuralStage "environment-install" $PythonPath @("-m", "pip", "install", "-r", $requirementsPath)
    }
    $dependencyCheck = "import numpy, torch, onnx, onnxruntime, onnxscript"
    try {
        & $PythonPath "-c" $dependencyCheck 2>&1 | Tee-Object -FilePath $logPath -Append
        $dependencyExitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
    }
    catch {
        $dependencyExitCode = 1
    }
    if ($dependencyExitCode -ne 0) {
        throw "External neural dependencies are unavailable. Run with -InstallRequirements or install requirements-neural.txt into $PythonPath."
    }

    $datasetArguments = @(
        "--package-dir", $resolvedPackage,
        "--output-dir", $datasetPath,
        "--sample-budget", $SampleBudget,
        "--minimum-clearance-mm", $MinimumClearanceMm,
        "--max-balance-rounds", $MaxBalanceRounds,
        "--balance-round-size", $BalanceRoundSize,
        "--max-refinement-pairs", $MaxRefinementPairs,
        "--refinement-rounds", $RefinementRounds,
        "--refinement-search-subset", $RefinementSearchSubset,
        "--chunk-size", $ChunkSize
    )
    Invoke-NeuralStage "dataset" $BlenderPath (@("--background", "--factory-startup", "--python", $datasetScript, "--") + $datasetArguments)

    $trainerArguments = @(
        "--package-dir", $resolvedPackage,
        "--dataset-dir", $datasetPath,
        "--output-dir", $modelPath,
        "--epochs", $Epochs,
        "--batch-size", $BatchSize,
        "--hidden-size", $HiddenSize,
        "--learning-rate", $LearningRate,
        "--weight-decay", $WeightDecay,
        "--leakage-radius-mm", $LeakageRadiusMm,
        "--full-mesh-authority-weight", $FullMeshAuthorityWeight,
        "--target-false-safe-rate", $TargetFalseSafeRate,
        "--boundary-weight-scale", $BoundaryWeightScale,
        "--boundary-bandwidth-mm", $BoundaryBandwidthMm,
        "--neutral-anchor-weight", $NeutralAnchorWeight,
        "--rotation-anchor-weight", $RotationAnchorWeight,
        "--translation-anchor-weight", $TranslationAnchorWeight,
        "--opset", $Opset,
        "--parity-tolerance", $ParityTolerance
    )
    if ($CpuThreads.HasValue) {
        $trainerArguments += @("--cpu-threads", $CpuThreads.Value)
    }
    if ($Seed -ge 0) {
        $trainerArguments += @("--seed", $Seed)
    }
    $trainerArguments += $TrainArgument
    $trainingCommand = @($trainerArguments)
    Invoke-NeuralStage "training" $PythonPath (@($trainerScript) + $trainingCommand)

    if ($ActiveLearning) {
        $activeArguments = @(
            $activeLearningScript,
            "--package-dir", $resolvedPackage,
            "--dataset-dir", $datasetPath,
            "--model-dir", $modelPath,
            "--rounds", $ActiveRounds,
            "--candidates-per-round", $CandidatesPerRound,
            "--max-new-samples-per-round", $MaxNewSamplesPerRound,
            "--max-total-new-samples", $MaxTotalNewSamples,
            "--uncertainty-band", $UncertaintyBand,
            "--minimum-clearance-mm", $MinimumClearanceMm,
            "--blender-exe", $BlenderPath
        )
        if ($MaxRuntimeSeconds.HasValue) {
            $activeArguments += @("--max-runtime-seconds", $MaxRuntimeSeconds.Value)
        }
        if ($Seed -ge 0) {
            $activeArguments += @("--seed", $Seed)
        }
        foreach ($argument in @(
            "--epochs=$Epochs",
            "--batch-size=$BatchSize",
            "--hidden-size=$HiddenSize",
            "--learning-rate=$LearningRate",
            "--weight-decay=$WeightDecay",
            "--leakage-radius-mm=$LeakageRadiusMm",
            "--full-mesh-authority-weight=$FullMeshAuthorityWeight",
            "--target-false-safe-rate=$TargetFalseSafeRate",
            "--neutral-anchor-weight=$NeutralAnchorWeight",
            "--rotation-anchor-weight=$RotationAnchorWeight",
            "--translation-anchor-weight=$TranslationAnchorWeight",
            "--opset=$Opset",
            "--parity-tolerance=$ParityTolerance"
        )) {
            $activeArguments += "--train-arg=$argument"
        }
        foreach ($argument in $TrainArgument) {
            $activeArguments += "--train-arg=$argument"
        }
        Invoke-NeuralStage "active-learning" $PythonPath $activeArguments
    }

    Assert-File (Join-Path $modelPath "model_manifest.json") "Complete model manifest"
    Assert-File (Join-Path $modelPath "constraint_model.onnx") "Published ONNX model"
    Update-ArtifactStatus
    Write-RunLog "Dataset artifacts: $datasetPath"
    Write-RunLog "Model artifacts: $modelPath"
    if ($status.metrics) {
        Write-RunLog ("Key metrics: " + ($status.metrics | ConvertTo-Json -Compress -Depth 8))
    }
    $status.state = if ($ActiveLearning -and (Get-Content -LiteralPath $logPath -Raw -ErrorAction SilentlyContinue) -match "converged") { "converged" } else { "completed" }
    $status.current_stage = $null
    $status.command = $null
    $status.exit_code = 0
    $status.ended_at = (Get-Date).ToUniversalTime().ToString("o")
    Write-RunStatus
    Write-RunLog "Run finished with state '$($status.state)'. Dataset: $datasetPath. Model: $modelPath."
    exit 0
}
catch {
    $status.state = "failed"
    $status.current_stage = if ($status.current_stage) { $status.current_stage } else { "validation" }
    $status.exit_code = if ($status.exit_code) { $status.exit_code } else { 1 }
    $status.error = $_.Exception.Message
    $status.ended_at = (Get-Date).ToUniversalTime().ToString("o")
    try {
        Update-ArtifactStatus
        Write-RunStatus
        Write-RunLog "Run failed in '$($status.current_stage)': $($status.error)"
    }
    catch {
        Write-Error "Run failed: $($status.error)"
    }
    exit $status.exit_code
}
