[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [switch]$SkipDemo
)

$ErrorActionPreference = "Stop"
$apiRoot = $BaseUrl.TrimEnd("/")

$health = Invoke-RestMethod -Uri "$apiRoot/health" -Method Get
if ($health.readiness.status -ne "ready") {
    throw "Backend is live but not ready. Inspect GET /health for failed dependencies."
}

$modelInfo = Invoke-RestMethod -Uri "$apiRoot/model-info" -Method Get
if (-not $modelInfo.model_version) {
    throw "Model metadata did not contain a model version."
}

$applicationId = "APP-SMOKE-001"
$payload = @{
    application_id = $applicationId
    user_id = "SMOKE-USER-001"
    event_timestamp = "2026-08-20T10:00:00Z"
    requested_loan_amount = 5000.0
    income = 80000.0
    account_age_days = 900
    device_id = "SMOKE-DEVICE-001"
    ip_address = "203.0.113.8"
} | ConvertTo-Json

$assessment = Invoke-RestMethod -Uri "$apiRoot/predict" -Method Post `
    -ContentType "application/json" -Body $payload
if ($assessment.application_id -ne $applicationId) {
    throw "Prediction response did not match the submitted public application ID."
}

$detail = Invoke-RestMethod -Uri "$apiRoot/applications/$applicationId" -Method Get
$analytics = Invoke-RestMethod -Uri "$apiRoot/analytics" -Method Get
if ($detail.application_id -ne $applicationId -or $analytics.total_applications -lt 1) {
    throw "Persistence or dashboard query smoke check failed."
}

if (-not $SkipDemo) {
    Invoke-RestMethod -Uri "$apiRoot/demo/reset" -Method Post | Out-Null
    $demoRequest = @{ scenario = "mixed"; interval_ms = 50; repeat = 1 } | ConvertTo-Json
    Invoke-RestMethod -Uri "$apiRoot/demo/run" -Method Post `
        -ContentType "application/json" -Body $demoRequest | Out-Null
    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 250
        $demoStatus = Invoke-RestMethod -Uri "$apiRoot/demo/status" -Method Get
        if ($demoStatus.error) {
            throw "Deterministic demo reported an error."
        }
    } while ($demoStatus.running -and (Get-Date) -lt $deadline)
    if ($demoStatus.running -or $demoStatus.processed -ne $demoStatus.total) {
        throw "Deterministic demo did not finish within 30 seconds."
    }
}

Write-Host "Smoke checks passed for $apiRoot (model $($modelInfo.model_version))."
