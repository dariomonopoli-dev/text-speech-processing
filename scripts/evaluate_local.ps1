param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$ExperimentName,
    [Parameter(Mandatory = $true)][string]$LogPrefix,
    [string[]]$Variants = @('text'),
    [int[]]$Seeds = @(42, 43, 44)
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$Python = (Get-Command $Python -ErrorAction Stop).Source
New-Item -ItemType Directory -Force -Path logs | Out-Null
. (Join-Path $PSScriptRoot 'local_python_steps.ps1')
Invoke-LocalEvaluation -ExperimentName $ExperimentName -Variants $Variants -LogPrefix $LogPrefix
