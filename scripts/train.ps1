[CmdletBinding()]
param(
    [string]$Python = "python",
    [int]$Seed = 20260819,
    [string]$InputPath
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot

Push-Location $repositoryRoot
try {
    $prepareArguments = @(
        "-m", "training.prepare_data",
        "--seed", "$Seed",
        "--output-dir", "data/processed"
    )
    if ($InputPath) {
        $resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
        $prepareArguments += @("--input", $resolvedInput)
    }
    & $Python @prepareArguments
    & $Python -m training.train_classifier --seed $Seed --data-dir data/processed `
        --output-dir artifacts/supervised-v1 --model-version supervised-v1 --overwrite
    & $Python -m training.train_anomaly --seed $Seed --data-dir data/processed `
        --output artifacts/anomaly-v1.joblib
    & $Python -m training.evaluate --data-dir data/processed `
        --bundle-dir artifacts/supervised-v1 --output reports/model_evaluation.md
    & $Python -m training.plot_evaluation --data-dir data/processed `
        --bundle-dir artifacts/supervised-v1 --output-dir reports/plots
    & $Python -m training.hybrid_eval --data-dir data/processed `
        --classifier-bundle artifacts/supervised-v1 `
        --anomaly-artifact artifacts/anomaly-v1.joblib
    & $Python -m training.responsible_ai --data-dir data/processed `
        --classifier-bundle artifacts/supervised-v1 `
        --anomaly-artifact artifacts/anomaly-v1.joblib
    Write-Host "Deterministic training and evaluation pipeline completed with seed $Seed."
}
finally {
    Pop-Location
}
