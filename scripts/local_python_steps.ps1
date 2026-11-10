function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    $stdoutPath = Join-Path $repoRoot "logs/$Name.stdout.log"
    $stderrPath = Join-Path $repoRoot "logs/$Name.stderr.log"
    if ((Test-Path -LiteralPath $stdoutPath) -or (Test-Path -LiteralPath $stderrPath)) {
        throw "Logs for $Name already exist. Review the previous run before restarting."
    }
    Write-Output "Start $Name"
    $process = Start-Process -FilePath $Python -ArgumentList $Arguments `
        -WorkingDirectory $repoRoot -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    if ($process.ExitCode -ne 0) {
        throw "$Name failed with exit code $($process.ExitCode). See $stderrPath"
    }
    Write-Output "Finished $Name"
}

function Invoke-LocalEvaluation {
    param([string]$ExperimentName, [string[]]$Variants, [string]$LogPrefix)
    foreach ($variant in $Variants) {
        foreach ($subset in @('all', 'shared')) {
            foreach ($seed in $Seeds) {
                Invoke-PythonStep -Name "${LogPrefix}_eval_${variant}_${subset}_run$seed" -Arguments @(
                    '-u', 'src/evaluate_predictions.py',
                    '--path_predictions', "runs/$ExperimentName/run$seed/predictions_swiss_german_$variant.csv",
                    '--path_metrics', "runs/$ExperimentName/run$seed/swiss_german_${variant}_${subset}_metrics.json",
                    '--subset', $subset
                )
            }
            Invoke-PythonStep -Name "${LogPrefix}_aggregate_${variant}_$subset" -Arguments @(
                '-u', 'src/parse_results_into_csv.py',
                '--path_runs', "runs/$ExperimentName",
                '--metrics_pattern', "swiss_german_${variant}_${subset}_metrics.json",
                '--path_output_dir', "runs/$ExperimentName/swiss_german_${variant}_$subset",
                '--predictions_name', "predictions_swiss_german_$variant.csv",
                '--num_runs', "$($Seeds.Count)", '--bootstrap', '--subset', $subset
            )
        }
    }
}
