[CmdletBinding()]
param(
    [switch]$SkipTraining,
    [switch]$SkipFrontend,
    [string]$InputPath
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$virtualEnvironment = Join-Path $repositoryRoot ".venv"
$pythonExecutable = Join-Path $virtualEnvironment "Scripts\python.exe"

Push-Location $repositoryRoot
try {
    if (-not (Test-Path -LiteralPath $pythonExecutable)) {
        python -m venv $virtualEnvironment
    }
    & $pythonExecutable -m pip install --upgrade pip
    & $pythonExecutable -m pip install -e ".[dev,postgres,redis]"

    if (-not $SkipFrontend) {
        Push-Location (Join-Path $repositoryRoot "frontend")
        try {
            npm.cmd ci
        }
        finally {
            Pop-Location
        }
    }

    if (-not $SkipTraining) {
        $trainingScript = Join-Path $PSScriptRoot "train.ps1"
        if ($InputPath) {
            & $trainingScript -Python $pythonExecutable -InputPath $InputPath
        }
        else {
            & $trainingScript -Python $pythonExecutable
        }
    }

    Write-Host "Bootstrap complete. Activate with: .\.venv\Scripts\Activate.ps1"
}
finally {
    Pop-Location
}
