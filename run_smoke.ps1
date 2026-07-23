# run_smoke.ps1 — run all 6 variants sequentially (HANDOFF section 5)
# Safe to re-run after an interruption: variants with a summary.json are skipped.
$py = Join-Path $PSScriptRoot "zda-env\Scripts\python.exe"
$outRoot = Join-Path $PSScriptRoot "runs"
foreach ($v in @("b0", "b1", "v1", "v2", "v3", "v4")) {
    $done = Join-Path $outRoot "$($v.ToUpper())_seed1337\summary.json"
    if (Test-Path $done) {
        Write-Host "=== $v already complete, skipping ==="
        continue
    }
    Write-Host "=== $v ==="
    & $py (Join-Path $PSScriptRoot "train.py") (Join-Path $PSScriptRoot "configs\$v.json") --out_root $outRoot
    if ($LASTEXITCODE -ne 0) {
        Write-Error "variant $v failed (exit $LASTEXITCODE)"
        exit 1
    }
}
Write-Host "all 6 variants complete"
