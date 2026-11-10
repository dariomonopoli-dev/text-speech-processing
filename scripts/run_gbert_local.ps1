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
    Invoke-PythonStep -Name "gbert_local_run$seed" -Arguments @(
        '-u', 'src/exec_scripts.py', '--path_config',
        "configs/gahd_gbert_raw_rtx3080_run$seed.json"
    )
}

Invoke-LocalEvaluation -ExperimentName 'gahd_gbert_raw' -Variants @('text', 'translation_haiku', 'translation_apertus8b') -LogPrefix 'gbert'
