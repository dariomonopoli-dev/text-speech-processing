param(
    [Parameter(Mandatory = $true)][string]$Python,
    [int[]]$Seeds = @(42, 43, 44),
    [string]$ExperimentName = 'gahd_swissbert_raw_lossfixed',
    [string]$LogPrefix = 'swissbert_lossfixed'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$Python = (Get-Command $Python -ErrorAction Stop).Source
New-Item -ItemType Directory -Force -Path logs | Out-Null
. (Join-Path $PSScriptRoot 'local_python_steps.ps1')

foreach ($seed in $Seeds) {
    Invoke-PythonStep -Name "${LogPrefix}_run$seed" -Arguments @(
        '-u', 'src/exec_scripts.py', '--path_config',
        "configs/${ExperimentName}_run$seed.json"
    )
    $runPath = "runs/$ExperimentName/run$seed"
    $experiment = Get-Content -LiteralPath "$runPath/experiment.json" -Raw | ConvertFrom-Json
    Invoke-PythonStep -Name "${LogPrefix}_predict_run$seed" -Arguments @(
        '-u', 'src/predict_swissbert.py', '--path_checkpoint', $experiment.best_model_checkpoint,
        '--path_output', "$runPath/predictions_swiss_german_text.csv"
    )
}

Invoke-LocalEvaluation -ExperimentName $ExperimentName -Variants @('text') -LogPrefix $LogPrefix
