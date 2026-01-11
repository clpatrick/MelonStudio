# Quick script for the Phi-4 model that was just quantized
# Just run: .\run_auto_opt_quick.ps1

$quantizedModel = "C:\Models\microsoft_Phi-4-mini-reasoning_gptq_cuda_2\quantized\model"
$outputPath = "C:\Models\microsoft_Phi-4-mini-reasoning_gptq_cuda_2"

# Find the Olive Python executable (runtime location)
$olivePythonPath = "C:\Repos\MelonStudio\MelonStudio\bin\Debug\net8.0-windows\win-x64\scripts\olive\.olive-env\Scripts\python.exe"

if (-not (Test-Path $olivePythonPath)) {
    Write-Host "Error: Olive Python not found at $olivePythonPath" -ForegroundColor Red
    exit 1
}

# Temporarily disable TensorRT provider DLL to avoid CreateEpFactories error
$tensorrtDllPath = "C:\Repos\MelonStudio\MelonStudio\bin\Debug\net8.0-windows\win-x64\scripts\olive\.olive-env\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_tensorrt.dll"
$tensorrtDllBackup = "$tensorrtDllPath.disabled"
$tensorrtDllRenamed = $false

if ((Test-Path $tensorrtDllPath) -and -not (Test-Path $tensorrtDllBackup)) {
    try {
        Move-Item -Path $tensorrtDllPath -Destination $tensorrtDllBackup -Force
        $tensorrtDllRenamed = $true
        Write-Host "Temporarily disabled TensorRT provider DLL to avoid version incompatibility." -ForegroundColor Yellow
    } catch {
        Write-Host "Warning: Could not disable TensorRT DLL: $_" -ForegroundColor Yellow
    }
}

Write-Host "Running Olive auto-opt on Phi-4 quantized model..." -ForegroundColor Green
Write-Host ""

$arguments = @(
    "-m", "olive", "auto-opt",
    "--model_name_or_path", "`"$quantizedModel`"",
    "--output_path", "`"$outputPath`"",
    "--device", "gpu",
    "--provider", "CUDAExecutionProvider",
    "--use_ort_genai",
    "--log_level", "1"
)

& $olivePythonPath $arguments

# Restore TensorRT DLL if it was renamed
if ($tensorrtDllRenamed) {
    Start-Sleep -Milliseconds 500
    try {
        if (Test-Path $tensorrtDllBackup) {
            Move-Item -Path $tensorrtDllBackup -Destination $tensorrtDllPath -Force
            Write-Host "Restored TensorRT provider DLL." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Warning: Could not restore TensorRT DLL: $_" -ForegroundColor Yellow
    }
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Optimization completed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Optimization failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
