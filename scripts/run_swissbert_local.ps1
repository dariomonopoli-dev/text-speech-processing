param(
    [Parameter(Mandatory = $true)][string]$Python,
    [int[]]$Seeds = @(42, 43, 44)
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$Python = (Get-Command $Python -ErrorAction Stop).Source
New-Item -ItemType Directory -Force -Path logs | Out-Null
. (Join-Path $PSScriptRoot 'local_python_steps.ps1')

foreach ($seed in $Seeds) {
    Invoke-PythonStep -Name "swissbert_lossfixed_run$seed" -Arguments @(
        '-u', 'src/exec_scripts.py', '--path_config',
        "configs/gahd_swissbert_raw_lossfixed_run$seed.json"
    )
    $runPath = "runs/gahd_swissbert_raw_lossfixed/run$seed"
    $experiment = Get-Content -LiteralPath "$runPath/experiment.json" -Raw | ConvertFrom-Json
    Invoke-PythonStep -Name "swissbert_lossfixed_predict_run$seed" -Arguments @(
        '-u', 'src/predict_swissbert.py', '--path_checkpoint', $experiment.best_model_checkpoint,
        '--path_output', "$runPath/predictions_swiss_german_text.csv"
    )
}

Invoke-LocalEvaluation -ExperimentName 'gahd_swissbert_raw_lossfixed' -Variants @('text') -LogPrefix 'swissbert_lossfixed'
