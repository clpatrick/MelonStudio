# Quick script to run Olive auto-opt on a quantized model
# Usage: .\run_auto_opt.ps1 -QuantizedModelPath "C:\Models\model_gptq_cuda\quantized\model" -OutputPath "C:\Models\model_gptq_cuda" -Device "gpu" -Provider "cuda"

param(
    [Parameter(Mandatory=$true)]
    [string]$QuantizedModelPath,
    
    [Parameter(Mandatory=$true)]
    [string]$OutputPath,
    
    [Parameter(Mandatory=$false)]
    [string]$Device = "gpu",
    
    [Parameter(Mandatory=$false)]
    [string]$Provider = "cuda",
    
    [Parameter(Mandatory=$false)]
    [switch]$UseOrtGenai = $true
)

# Find the Olive Python executable
$baseDir = Split-Path -Parent $PSScriptRoot
$olivePythonPath = Join-Path $baseDir "MelonStudio\bin\Debug\net8.0-windows\win-x64\scripts\olive\.olive-env\Scripts\python.exe"

if (-not (Test-Path $olivePythonPath)) {
    Write-Host "Error: Olive Python not found at $olivePythonPath" -ForegroundColor Red
    Write-Host "Trying source location..." -ForegroundColor Yellow
    $olivePythonPath = Join-Path $PSScriptRoot ".olive-env\Scripts\python.exe"
    if (-not (Test-Path $olivePythonPath)) {
        Write-Host "Error: Olive Python not found. Please ensure Olive environment is set up." -ForegroundColor Red
        exit 1
    }
}

# Map provider to Olive format
$oliveProvider = switch ($Provider.ToLower()) {
    "cuda" { "CUDAExecutionProvider" }
    "dml" { "DmlExecutionProvider" }
    "tensorrt" { "TensorrtExecutionProvider" }
    default { "CUDAExecutionProvider" }
}

# Build command arguments
$args = @(
    "-m", "olive", "auto-opt",
    "--model_name_or_path", "`"$QuantizedModelPath`"",
    "--output_path", "`"$OutputPath`"",
    "--device", $Device,
    "--provider", $oliveProvider,
    "--log_level", "1"
)

if ($UseOrtGenai) {
    $args += "--use_ort_genai"
}

Write-Host "Running Olive auto-opt..." -ForegroundColor Green
Write-Host "Command: $olivePythonPath $($args -join ' ')" -ForegroundColor Cyan
Write-Host ""

& $olivePythonPath $args

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Optimization completed successfully!" -ForegroundColor Green
} else {
    Write-Host "`n✗ Optimization failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
