using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace MelonStudio.Services
{
    public enum ConversionErrorCategory
    {
        Success,
        UpstreamBug,
        MissingDependency,
        ModelIncompatible,
        NetworkError,
        AuthenticationError,
        DiskSpaceError,
        UnknownError
    }

    public record ConversionDiagnostic(
        ConversionErrorCategory Category,
        string Summary,
        string Details,
        string? SuggestedAction,
        string? IssueUrl
    );

    public record ModelAnalysisResult(
        bool Success,
        int LayerCount,
        double TotalSizeGb,
        double BaseSizeGb,
        double AvgLayerSizeGb,
        string Precision,
        string ErrorMessage,
        string RawOutput = "",
        string ToolName = ""
    );

    public class ModelBuilderService
    {
        public event Action<string>? OnOutputReceived;
        public event Action<string>? OnErrorReceived;
        public event Action<bool>? OnCompleted;
        public event Action<ConversionDiagnostic>? OnDiagnosticGenerated;

        private Process? _currentProcess;
        private CancellationTokenSource? _cts;
        private readonly StringBuilder _stderrBuffer = new();

        public bool IsRunning => _currentProcess != null && !_currentProcess.HasExited;

        public async Task<bool> CheckPythonAvailableAsync()
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = "--version",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(psi);
                if (process == null) return false;
                
                await process.WaitForExitAsync();
                return process.ExitCode == 0;
            }
            catch
            {
                return false;
            }
        }

        public async Task<bool> CheckOnnxRuntimeGenAIAvailableAsync()
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = "-c \"import onnxruntime_genai; print('OK')\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(psi);
                if (process == null) return false;
                
                var output = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();
                return output.Contains("OK");
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Checks if GPU is available for quantization algorithms that require it.
        /// </summary>
        public bool CheckGpuAvailable()
        {
            try
            {
                // Try to import torch and check CUDA availability
                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = "-c \"import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(psi);
                if (process == null) return false;
                
                var output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();
                return output.Contains("CUDA");
            }
            catch
            {
                // If torch is not available, assume no GPU
                return false;
            }
        }

        public async Task<bool> ConvertModelAsync(
            string modelNameOrPath,
            string outputFolder,
            string precision = "int4",
            string executionProvider = "cuda",
            bool enableCudaGraph = false,
            string? huggingFaceToken = null,
            string? cacheDir = null)
        {
            _cts = new CancellationTokenSource();
            _stderrBuffer.Clear();

            var args = new StringBuilder();
            args.Append("-m onnxruntime_genai.models.builder ");
            
            // Model source - resolve HuggingFace cache structure if needed
            if (Directory.Exists(modelNameOrPath))
            {
                var resolvedPath = ResolveModelPath(modelNameOrPath);
                if (resolvedPath != modelNameOrPath)
                {
                    OnOutputReceived?.Invoke($"  📂 Resolved HuggingFace cache: {Path.GetFileName(resolvedPath)}");
                }
                args.Append($"-i \"{resolvedPath}\" ");
            }
            else
            {
                args.Append($"-m \"{modelNameOrPath}\" ");
            }

            args.Append($"-o \"{outputFolder}\" ");
            args.Append($"-p {precision} ");
            args.Append($"-e {executionProvider} ");

            if (!string.IsNullOrEmpty(cacheDir))
            {
                args.Append($"-c \"{cacheDir}\" ");
            }

            var extraOptions = new StringBuilder();
            if (enableCudaGraph)
            {
                extraOptions.Append("enable_cuda_graph=true ");
            }
            if (!string.IsNullOrEmpty(huggingFaceToken))
            {
                extraOptions.Append($"hf_token={huggingFaceToken} ");
            }

            if (extraOptions.Length > 0)
            {
                args.Append($"--extra_options {extraOptions.ToString().Trim()}");
            }

            // Ensure parent directory exists (the builder will create the final folder)
            var parentDir = Path.GetDirectoryName(outputFolder);
            if (!string.IsNullOrEmpty(parentDir) && !Directory.Exists(parentDir))
            {
                Directory.CreateDirectory(parentDir);
            }

            var psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = args.ToString(),
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            OnOutputReceived?.Invoke($"Starting conversion: python {args}");
            OnOutputReceived?.Invoke($"Output folder: {outputFolder}");
            OnOutputReceived?.Invoke("");

            try
            {
                _currentProcess = Process.Start(psi);
                if (_currentProcess == null)
                {
                    OnErrorReceived?.Invoke("Failed to start Python process");
                    OnCompleted?.Invoke(false);
                    return false;
                }

                // Read output asynchronously
                var outputTask = ReadStreamAsync(_currentProcess.StandardOutput, line => 
                {
                    OnOutputReceived?.Invoke(line);
                }, _cts.Token);

                var errorTask = ReadStreamAsync(_currentProcess.StandardError, line => 
                {
                    // Clean and buffer stderr for analysis
                    var cleaned = CleanPythonOutput(line);
                    _stderrBuffer.AppendLine(line); // Keep original for analysis
                    
                    // Only emit non-empty cleaned lines
                    if (!string.IsNullOrWhiteSpace(cleaned))
                    {
                        OnErrorReceived?.Invoke(cleaned);
                    }
                }, _cts.Token);

                await _currentProcess.WaitForExitAsync(_cts.Token);
                await Task.WhenAll(outputTask, errorTask);

                var success = _currentProcess.ExitCode == 0;
                OnOutputReceived?.Invoke("");

                if (success)
                {
                    // Fix any null values in genai_config.json that ONNX Runtime GenAI can't parse
                    SanitizeGenAiConfig(outputFolder);
                    OnOutputReceived?.Invoke("✓ Conversion completed successfully!");
                }
                else
                {
                    OnOutputReceived?.Invoke("✗ Conversion failed");
                    
                    // Generate diagnostic
                    var diagnostic = AnalyzeError(_stderrBuffer.ToString(), modelNameOrPath);
                    OnDiagnosticGenerated?.Invoke(diagnostic);
                    
                    // Output human-readable summary
                    OnOutputReceived?.Invoke("");
                    OnOutputReceived?.Invoke("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    OnOutputReceived?.Invoke($"📋 DIAGNOSIS: {diagnostic.Category}");
                    OnOutputReceived?.Invoke($"📝 {diagnostic.Summary}");
                    if (!string.IsNullOrEmpty(diagnostic.SuggestedAction))
                    {
                        OnOutputReceived?.Invoke($"💡 {diagnostic.SuggestedAction}");
                    }
                    if (!string.IsNullOrEmpty(diagnostic.IssueUrl))
                    {
                        OnOutputReceived?.Invoke($"🔗 Report: {diagnostic.IssueUrl}");
                    }
                    OnOutputReceived?.Invoke("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                }

                OnCompleted?.Invoke(success);
                return success;
            }
            catch (OperationCanceledException)
            {
                OnOutputReceived?.Invoke("Conversion cancelled");
                OnCompleted?.Invoke(false);
                return false;
            }
            catch (Exception ex)
            {
                OnErrorReceived?.Invoke($"Error: {ex.Message}");
                OnCompleted?.Invoke(false);
                return false;
            }
            finally
            {
                _currentProcess = null;
            }
        }

        public void Cancel()
        {
            _cts?.Cancel();
            if (_currentProcess != null && !_currentProcess.HasExited)
            {
                try
                {
                    _currentProcess.Kill(entireProcessTree: true);
                }
                catch { }
            }
        }

        /// <summary>
        /// Maps provider name to Olive format.
        /// </summary>
        private static string MapProviderToOliveFormat(string provider)
        {
            return provider.ToLowerInvariant() switch
            {
                "cuda" => "CUDAExecutionProvider",
                "dml" => "DmlExecutionProvider",
                "tensorrt" => "TensorrtExecutionProvider",
                _ => "CPUExecutionProvider" // Default fallback
            };
        }

        /// <summary>
        /// Maps quantization algorithm to its precision.
        /// For algorithms that support multiple precisions (GPTQ, BitsAndBytes), 
        /// the precision parameter should be provided explicitly.
        /// </summary>
        private static string GetPrecisionForAlgorithm(string algorithm, string? explicitPrecision = null)
        {
            // If explicit precision is provided (for GPTQ, BitsAndBytes), use it
            if (!string.IsNullOrEmpty(explicitPrecision))
            {
                return explicitPrecision.ToLowerInvariant();
            }

            // Fixed precision algorithms
            return algorithm.ToLowerInvariant() switch
            {
                "awq" => "int4",        // AWQ is 4-bit only
                "quarot" => "int4",     // Quarot is 4-bit only
                "olive" => "int4",      // Olive HQQ is 4-bit only
                "spinquant" => "int4",  // Spinquant (typically 4-bit, but check docs)
                "nvmo" => "int4",       // NVMO AWQ is 4-bit
                "gptq" => "int4",       // Default to int4 for GPTQ if not specified
                "bitsandbytes" => "int4", // Default to int4 for BitsAndBytes if not specified
                _ => "int4" // Default fallback
            };
        }

        /// <summary>
        /// Quantizes a model using Olive quantize command.
        /// </summary>
        public async Task<(bool Success, string QuantizedModelPath)> QuantizeModelWithOliveAsync(
            string modelNameOrPath,
            string outputFolder,
            string algorithm,
            string? precision = null,
            string device = "cpu",
            string? huggingFaceToken = null,
            string? cacheDir = null)
        {
            _cts = new CancellationTokenSource();
            _stderrBuffer.Clear();

            // Use intermediate folder for quantized model
            var quantizedOutputFolder = Path.Combine(outputFolder, "quantized");

            // Map algorithm to precision (use explicit precision if provided, otherwise use default)
            var quantizePrecision = GetPrecisionForAlgorithm(algorithm, precision);

            // Determine if algorithm requires GPU
            bool algorithmRequiresGpu = algorithm.ToLowerInvariant() is "awq" or "gptq";
            
            // Check GPU availability for GPU-required algorithms
            if (algorithmRequiresGpu)
            {
                bool hasGpu = CheckGpuAvailable();
                if (!hasGpu)
                {
                    OnErrorReceived?.Invoke($"Error: {algorithm.ToUpperInvariant()} requires GPU, but no GPU is available.");
                    return (false, "");
                }
            }

            // Build arguments for olive quantize command
            // Note: quantize command does NOT support --device parameter, but we can use --save_config_file
            // and then modify the config to specify GPU accelerator, or use a system config file
            var args = new StringBuilder();
            args.Append($"--model_name_or_path \"{modelNameOrPath}\" ");
            args.Append($"--algorithm {algorithm} ");
            
            // Only specify precision for algorithms that support variable precision (GPTQ, BitsAndBytes)
            // Fixed precision algorithms (AWQ, Quarot, Olive HQQ, etc.) don't need --precision
            // Matching Microsoft's example: https://microsoft.github.io/Olive/how-to/cli/cli-quantize.html
            var algorithmLower = algorithm.ToLowerInvariant();
            if (algorithmLower == "gptq" || algorithmLower == "bitsandbytes")
            {
                args.Append($"--precision {quantizePrecision} ");
            }
            // For fixed precision algorithms (AWQ, Quarot, etc.), don't specify precision
            // Olive will use the correct default (int4) for these algorithms
            
            // For GPU-required algorithms, we need to ensure Olive uses GPU
            // Olive quantize doesn't have a direct --device flag, but it should auto-detect GPU
            // The issue is that it's defaulting to CPU. We'll need to check if there's a system config option.
            
            args.Append($"--output_path \"{quantizedOutputFolder}\" ");
            args.Append("--log_level 1");

            // Add trust_remote_code if needed (detect from model analysis or config)
            if (Directory.Exists(modelNameOrPath))
            {
                var configPath = Path.Combine(modelNameOrPath, "config.json");
                if (File.Exists(configPath))
                {
                    try
                    {
                        var configContent = await File.ReadAllTextAsync(configPath);
                        if (configContent.Contains("custom_code") || configContent.Contains("modeling_"))
                        {
                            args.Append(" --trust_remote_code");
                        }
                    }
                    catch { }
                }
            }

            // Find Olive Python (use venv Python with -m olive)
            var baseDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? AppDomain.CurrentDomain.BaseDirectory;
            var oliveEnvPath = Path.Combine(baseDir, "scripts", "olive", ".olive-env", "Scripts");
            string executablePath = "python";
            string commandArgs = $"-m olive quantize {args}";
            
            // Try python -m olive from venv first
            var olivePythonPath = Path.Combine(oliveEnvPath, "python.exe");
            if (File.Exists(olivePythonPath))
            {
                executablePath = olivePythonPath;
            }
            else
            {
                // Fallback to system python -m olive
                OnErrorReceived?.Invoke("Warning: Olive virtual environment not found. Using system Python.");
            }

            // Ensure parent directory exists
            var parentDir = Path.GetDirectoryName(quantizedOutputFolder);
            if (!string.IsNullOrEmpty(parentDir) && !Directory.Exists(parentDir))
            {
                Directory.CreateDirectory(parentDir);
            }

            // Temporarily disable TensorRT provider DLL to prevent CreateEpFactories error
            // ONNX Runtime tries to load all available providers during initialization
            // The TensorRT provider DLL is incompatible with the ONNX Runtime version in Olive
            // We'll disable it for quantize (uses CUDA), and restore it for auto-opt if TensorRT is selected
            var tensorrtDllPath = Path.Combine(baseDir, "scripts", "olive", ".olive-env", "Lib", "site-packages", "onnxruntime", "capi", "onnxruntime_providers_tensorrt.dll");
            var tensorrtDllBackup = tensorrtDllPath + ".disabled";
            
            bool tensorrtDllRenamed = false;
            if (File.Exists(tensorrtDllPath) && !File.Exists(tensorrtDllBackup))
            {
                try
                {
                    File.Move(tensorrtDllPath, tensorrtDllBackup);
                    tensorrtDllRenamed = true;
                    OnOutputReceived?.Invoke("Temporarily disabled TensorRT provider DLL to avoid version incompatibility during quantize step.");
                }
                catch (Exception ex)
                {
                    OnErrorReceived?.Invoke($"Warning: Could not disable TensorRT DLL: {ex.Message}");
                }
            }

            var psi = new ProcessStartInfo
            {
                FileName = executablePath,
                Arguments = commandArgs,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            // Set environment variables for GPU-required algorithms
            if (algorithmRequiresGpu && device.ToLowerInvariant() == "gpu")
            {
                // Force CUDA execution provider only (TensorRT DLL is disabled above)
                psi.EnvironmentVariables["CUDA_VISIBLE_DEVICES"] = "0"; // Use first GPU
                OnOutputReceived?.Invoke("Configuring for GPU acceleration (CUDA only for quantize step)...");
            }
            else
            {
                // For CPU or non-GPU algorithms, use CPU provider only
                psi.EnvironmentVariables["ORT_EXECUTION_PROVIDERS"] = "CPUExecutionProvider";
            }

            OnOutputReceived?.Invoke($"Step 1: Quantizing with algorithm '{algorithm}'...");
            OnOutputReceived?.Invoke($"Command: {executablePath} {commandArgs}");
            OnOutputReceived?.Invoke("");

            try
            {
                _currentProcess = Process.Start(psi);
                if (_currentProcess == null)
                {
                    OnErrorReceived?.Invoke("Failed to start Python process");
                    return (false, "");
                }

                var outputTask = ReadStreamAsync(_currentProcess.StandardOutput, line => 
                {
                    OnOutputReceived?.Invoke(line);
                }, _cts.Token);

                var errorTask = ReadStreamAsync(_currentProcess.StandardError, line => 
                {
                    var cleaned = CleanPythonOutput(line);
                    _stderrBuffer.AppendLine(line);
                    if (!string.IsNullOrWhiteSpace(cleaned))
                    {
                        OnErrorReceived?.Invoke(cleaned);
                    }
                }, _cts.Token);

                await _currentProcess.WaitForExitAsync(_cts.Token);
                await Task.WhenAll(outputTask, errorTask);

                var success = _currentProcess.ExitCode == 0;
                OnOutputReceived?.Invoke("");

                if (success)
                {
                    OnOutputReceived?.Invoke("✓ Quantization completed successfully!");
                    // Find the quantized model path (Olive output structure may vary)
                    // Typically: {outputFolder}/quantized/model or {outputFolder}/quantized
                    var quantizedPath = quantizedOutputFolder;
                    if (Directory.Exists(quantizedPath))
                    {
                        // Check for model files
                        var modelFiles = Directory.GetFiles(quantizedPath, "*.safetensors", SearchOption.AllDirectories);
                        if (modelFiles.Length == 0)
                        {
                            modelFiles = Directory.GetFiles(quantizedPath, "*.bin", SearchOption.AllDirectories);
                        }
                        if (modelFiles.Length > 0)
                        {
                            // Use the directory containing model files
                            quantizedPath = Path.GetDirectoryName(modelFiles[0]) ?? quantizedPath;
                        }
                    }
                    return (true, quantizedPath);
                }
                else
                {
                    OnOutputReceived?.Invoke("✗ Quantization failed");
                    return (false, "");
                }
            }
            catch (OperationCanceledException)
            {
                OnOutputReceived?.Invoke("Quantization cancelled");
                return (false, "");
            }
            catch (Exception ex)
            {
                OnErrorReceived?.Invoke($"Error: {ex.Message}");
                return (false, "");
            }
            finally
            {
                _currentProcess = null;
                // Restore TensorRT provider DLL if it was renamed
                if (tensorrtDllRenamed)
                {
                    // Wait a moment to ensure the process has fully released the file handle
                    await Task.Delay(500);
                    try
                    {
                        if (File.Exists(tensorrtDllBackup))
                        {
                            File.Move(tensorrtDllBackup, tensorrtDllPath);
                            OnOutputReceived?.Invoke("Restored TensorRT provider DLL.");
                        }
                    }
                    catch (Exception ex)
                    {
                        OnErrorReceived?.Invoke($"Warning: Could not restore TensorRT DLL: {ex.Message}");
                    }
                }
            }
        }

        /// <summary>
        /// Optimizes a quantized model using Olive auto-opt command.
        /// </summary>
        public async Task<bool> AutoOptModelWithOliveAsync(
            string quantizedModelPath,
            string outputFolder,
            string device = "gpu",
            string executionProvider = "cuda",
            bool useOrtGenai = true,
            string? huggingFaceToken = null)
        {
            _cts = new CancellationTokenSource();
            _stderrBuffer.Clear();

            // Build arguments for olive auto-opt command
            var args = new StringBuilder();
            args.Append($"--model_name_or_path \"{quantizedModelPath}\" ");
            args.Append($"--output_path \"{outputFolder}\" ");
            args.Append($"--device {device} ");

            // Provider mapping
            if (device == "gpu")
            {
                var oliveProvider = MapProviderToOliveFormat(executionProvider);
                args.Append($"--provider {oliveProvider} ");
            }
            // For CPU, provider defaults to CPUExecutionProvider or can be omitted

            if (useOrtGenai)
            {
                args.Append("--use_ort_genai ");
            }

            args.Append("--log_level 1");

            // Add trust_remote_code if needed
            if (Directory.Exists(quantizedModelPath))
            {
                var configPath = Path.Combine(quantizedModelPath, "config.json");
                if (File.Exists(configPath))
                {
                    try
                    {
                        var configContent = await File.ReadAllTextAsync(configPath);
                        if (configContent.Contains("custom_code") || configContent.Contains("modeling_"))
                        {
                            args.Append(" --trust_remote_code");
                        }
                    }
                    catch { }
                }
            }

            // Find Olive Python (use venv Python with -m olive)
            var baseDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? AppDomain.CurrentDomain.BaseDirectory;
            var oliveEnvPath = Path.Combine(baseDir, "scripts", "olive", ".olive-env", "Scripts");
            string executablePath = "python";
            string commandArgs = $"-m olive auto-opt {args}";
            
            // Try python -m olive from venv first
            var olivePythonPath = Path.Combine(oliveEnvPath, "python.exe");
            if (File.Exists(olivePythonPath))
            {
                executablePath = olivePythonPath;
            }
            else
            {
                // Fallback to system python -m olive
                OnErrorReceived?.Invoke("Warning: Olive virtual environment not found. Using system Python.");
            }

            // Ensure parent directory exists
            var parentDir = Path.GetDirectoryName(outputFolder);
            if (!string.IsNullOrEmpty(parentDir) && !Directory.Exists(parentDir))
            {
                Directory.CreateDirectory(parentDir);
            }

            // Temporarily disable TensorRT provider DLL to prevent CreateEpFactories error
            // Only disable if provider is not TensorRT (if TensorRT is selected, we want to use it)
            var tensorrtDllPath = Path.Combine(baseDir, "scripts", "olive", ".olive-env", "Lib", "site-packages", "onnxruntime", "capi", "onnxruntime_providers_tensorrt.dll");
            var tensorrtDllBackup = tensorrtDllPath + ".disabled";
            
            bool tensorrtDllRenamed = false;
            // Only disable TensorRT if we're not using it as the provider
            if (executionProvider.ToLowerInvariant() != "tensorrt" && File.Exists(tensorrtDllPath) && !File.Exists(tensorrtDllBackup))
            {
                try
                {
                    File.Move(tensorrtDllPath, tensorrtDllBackup);
                    tensorrtDllRenamed = true;
                    OnOutputReceived?.Invoke("Temporarily disabled TensorRT provider DLL to avoid version incompatibility during optimization step.");
                }
                catch (Exception ex)
                {
                    OnErrorReceived?.Invoke($"Warning: Could not disable TensorRT DLL: {ex.Message}");
                }
            }

            var psi = new ProcessStartInfo
            {
                FileName = executablePath,
                Arguments = commandArgs,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            OnOutputReceived?.Invoke($"Step 2: Optimizing with device '{device}'...");
            OnOutputReceived?.Invoke($"Command: {executablePath} {commandArgs}");
            OnOutputReceived?.Invoke("");

            try
            {
                _currentProcess = Process.Start(psi);
                if (_currentProcess == null)
                {
                    OnErrorReceived?.Invoke("Failed to start Python process");
                    OnCompleted?.Invoke(false);
                    return false;
                }

                var outputTask = ReadStreamAsync(_currentProcess.StandardOutput, line => 
                {
                    OnOutputReceived?.Invoke(line);
                }, _cts.Token);

                var errorTask = ReadStreamAsync(_currentProcess.StandardError, line => 
                {
                    var cleaned = CleanPythonOutput(line);
                    _stderrBuffer.AppendLine(line);
                    if (!string.IsNullOrWhiteSpace(cleaned))
                    {
                        OnErrorReceived?.Invoke(cleaned);
                    }
                }, _cts.Token);

                await _currentProcess.WaitForExitAsync(_cts.Token);
                await Task.WhenAll(outputTask, errorTask);

                var success = _currentProcess.ExitCode == 0;
                OnOutputReceived?.Invoke("");

                if (success)
                {
                    OnOutputReceived?.Invoke("✓ Optimization completed successfully!");
                }
                else
                {
                    OnOutputReceived?.Invoke("✗ Optimization failed");
                }

                return success;
            }
            catch (OperationCanceledException)
            {
                OnOutputReceived?.Invoke("Optimization cancelled");
                OnCompleted?.Invoke(false);
                return false;
            }
            catch (Exception ex)
            {
                OnErrorReceived?.Invoke($"Error: {ex.Message}");
                OnCompleted?.Invoke(false);
                return false;
            }
            finally
            {
                _currentProcess = null;
                // Restore TensorRT provider DLL if it was renamed
                if (tensorrtDllRenamed)
                {
                    // Wait a moment to ensure the process has fully released the file handle
                    await Task.Delay(500);
                    try
                    {
                        if (File.Exists(tensorrtDllBackup))
                        {
                            File.Move(tensorrtDllBackup, tensorrtDllPath);
                            OnOutputReceived?.Invoke("Restored TensorRT provider DLL.");
                        }
                    }
                    catch (Exception ex)
                    {
                        OnErrorReceived?.Invoke($"Warning: Could not restore TensorRT DLL: {ex.Message}");
                    }
                }
            }
        }

        /// <summary>
        /// Orchestrates the 2-step Olive conversion process: quantize then auto-opt.
        /// </summary>
        public async Task<bool> ConvertModelWithOliveAsync(
            string modelNameOrPath,
            string outputFolder,
            string quantizationAlgorithm,
            string? precision = null,
            string device = "gpu",
            string executionProvider = "cuda",
            bool useOrtGenai = true,
            string? huggingFaceToken = null,
            string? cacheDir = null)
        {
            // Determine if algorithm requires GPU
            bool requiresGpu = quantizationAlgorithm.ToLowerInvariant() is "awq" or "gptq";

            // Step 1: Quantize
            OnOutputReceived?.Invoke("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            OnOutputReceived?.Invoke("Step 1/2: Quantizing model...");
            OnOutputReceived?.Invoke("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            OnOutputReceived?.Invoke("");

            var (quantizeSuccess, quantizedModelPath) = await QuantizeModelWithOliveAsync(
                modelNameOrPath,
                outputFolder,
                quantizationAlgorithm,
                precision,
                device,  // Pass device parameter (string, not bool)
                huggingFaceToken,
                cacheDir
            );

            if (!quantizeSuccess || string.IsNullOrEmpty(quantizedModelPath))
            {
                OnOutputReceived?.Invoke("");
                OnOutputReceived?.Invoke("✗ Quantization step failed. Aborting conversion.");
                OnCompleted?.Invoke(false);
                return false;
            }

            OnOutputReceived?.Invoke("");
            OnOutputReceived?.Invoke("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            OnOutputReceived?.Invoke("Step 2/2: Optimizing model...");
            OnOutputReceived?.Invoke("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            OnOutputReceived?.Invoke("");

            // Step 2: Auto-opt
            var autoOptSuccess = await AutoOptModelWithOliveAsync(
                quantizedModelPath,
                outputFolder,
                device,
                executionProvider,
                useOrtGenai,
                huggingFaceToken
            );

            if (autoOptSuccess)
            {
                OnOutputReceived?.Invoke("");
                OnOutputReceived?.Invoke("✓ Olive conversion completed successfully!");
                OnCompleted?.Invoke(true);
            }
            else
            {
                OnOutputReceived?.Invoke("");
                OnOutputReceived?.Invoke("✗ Optimization step failed.");
                OnCompleted?.Invoke(false);
            }

            return autoOptSuccess;
        }

        /// <summary>
        /// Runs comprehensive analysis using all available tools.
        /// Returns a list of analysis results from each tool.
        /// </summary>
        public async Task<List<ModelAnalysisResult>> RunComprehensiveAnalysisAsync(string modelPath)
        {
            var results = new List<ModelAnalysisResult>();
            
            // 1. Basic analysis (current tool)
            var basicResult = await InspectModelAsync(modelPath);
            results.Add(basicResult);
            
            // 2. Olive framework analysis
            var oliveResult = await AnalyzeWithOliveAsync(modelPath);
            if (oliveResult != null)
                results.Add(oliveResult);
            
            // 3. ONNX Runtime analysis
            var onnxResult = await AnalyzeWithOnnxRuntimeAsync(modelPath);
            if (onnxResult != null)
                results.Add(onnxResult);
            
            // 4. Enhanced analysis (for safetensors models)
            if (Directory.Exists(modelPath))
            {
                var configPath = Path.Combine(modelPath, "config.json");
                var hasSafetensors = Directory.GetFiles(modelPath, "*.safetensors", SearchOption.TopDirectoryOnly).Length > 0;
                var hasBin = Directory.GetFiles(modelPath, "*.bin", SearchOption.TopDirectoryOnly).Length > 0;
                var hasOnnx = File.Exists(Path.Combine(modelPath, "model.onnx"));
                
                if (File.Exists(configPath) && (hasSafetensors || hasBin) && !hasOnnx)
                {
                    var enhancedResult = await AnalyzeEnhancedAsync(modelPath);
                    if (enhancedResult != null)
                        results.Add(enhancedResult);
                }
            }
            
            return results;
        }

        /// <summary>
        /// Analyzes a model (ONNX or safetensors/PyTorch) to determine layer count and size.
        /// </summary>
        public async Task<ModelAnalysisResult> InspectModelAsync(string modelPath)
        {
            // Determine if this is a folder or file
            string targetPath = modelPath;
            bool isDirectory = Directory.Exists(modelPath);
            bool isFile = File.Exists(modelPath);
            
            if (!isDirectory && !isFile)
            {
                return new ModelAnalysisResult(false, 0, 0, 0, 0, "", "Model path not found", "", "");
            }
            
            // Check if this is a safetensors/PyTorch model (has config.json)
            bool isSafetensorsModel = false;
            if (isDirectory)
            {
                var configPath = Path.Combine(modelPath, "config.json");
                var hasSafetensors = Directory.GetFiles(modelPath, "*.safetensors", SearchOption.TopDirectoryOnly).Length > 0;
                var hasBin = Directory.GetFiles(modelPath, "*.bin", SearchOption.TopDirectoryOnly).Length > 0;
                var hasOnnx = File.Exists(Path.Combine(modelPath, "model.onnx"));
                
                if (File.Exists(configPath) && (hasSafetensors || hasBin) && !hasOnnx)
                {
                    isSafetensorsModel = true;
                    targetPath = modelPath; // Use directory for safetensors
                }
                else if (hasOnnx)
                {
                    // ONNX model in directory
                    targetPath = Path.Combine(modelPath, "model.onnx");
                }
                else
                {
                    return new ModelAnalysisResult(false, 0, 0, 0, 0, "", "No recognized model files found (need model.onnx, *.safetensors, or *.bin with config.json)", "", "");
                }
            }
            else if (isFile)
            {
                // Single file - check extension
                if (modelPath.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
                {
                    targetPath = modelPath;
                    isSafetensorsModel = false;
                }
                else
                {
                    return new ModelAnalysisResult(false, 0, 0, 0, 0, "", "Unsupported file type. Expected .onnx file or model directory.", "", "");
                }
            }
            
            // Select appropriate analysis script
            string? baseScriptPath = FindSplitModelScript();
            if (string.IsNullOrEmpty(baseScriptPath))
            {
                return new ModelAnalysisResult(false, 0, 0, 0, 0, "", "Base script directory not found", "", "");
            }
            
            string scriptPath;
            string toolName;
            if (isSafetensorsModel)
            {
                // Use safetensors analyzer
                scriptPath = baseScriptPath.Replace("split_model.py", "analyze_safetensors.py");
                toolName = "SafeTensors Analyzer";
                if (!File.Exists(scriptPath))
                {
                    return new ModelAnalysisResult(false, 0, 0, 0, 0, "", "Safetensors analysis script not found", "", toolName);
                }
            }
            else
            {
                // Use ONNX analyzer
                scriptPath = baseScriptPath.Replace("split_model.py", "analyze_onnx_size.py");
                toolName = "ONNX Analyzer";
                if (!File.Exists(scriptPath))
                {
                    return new ModelAnalysisResult(false, 0, 0, 0, 0, "", "ONNX analysis script not found", "", toolName);
                }
            }

            var psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"\"{scriptPath}\" \"{targetPath}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            try
            {
                using var process = Process.Start(psi);
                if (process == null) return new ModelAnalysisResult(false, 0, 0, 0, 0, "", "Failed to start analysis process", "", toolName);

                var output = await process.StandardOutput.ReadToEndAsync();
                var error = await process.StandardError.ReadToEndAsync();
                await process.WaitForExitAsync();

                // Combine output and error for full log
                var fullOutput = new StringBuilder();
                if (!string.IsNullOrEmpty(output))
                    fullOutput.AppendLine(output);
                if (!string.IsNullOrEmpty(error))
                    fullOutput.AppendLine(error);

                if (process.ExitCode != 0)
                {
                    return new ModelAnalysisResult(false, 0, 0, 0, 0, "", $"Analysis failed: {error}", fullOutput.ToString(), toolName);
                }

                // Parse output
                // Expected format:
                // Total Calculated Params Size: 4.5000 GB
                // Base Size: 0.5000 GB
                // Average Layer Size: 0.1250 GB
                // Layers Found: 32
                // Dominant Precision: FP16

                double totalSize = 0, baseSize = 0, avgLayerSize = 0;
                int layers = 0;
                string precision = "Unknown";

                foreach (var line in output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
                {
                    if (line.Contains("Total Calculated Params Size:") && double.TryParse(Regex.Match(line, @"[\d\.]+").Value, out double t))
                        totalSize = t;
                    else if (line.Contains("Base Size:") && double.TryParse(Regex.Match(line, @"[\d\.]+").Value, out double b))
                        baseSize = b;
                    else if (line.Contains("Average Layer Size:") && double.TryParse(Regex.Match(line, @"[\d\.]+").Value, out double a))
                        avgLayerSize = a;
                    else if (line.Contains("Layers Found:") && int.TryParse(Regex.Match(line, @"\d+").Value, out int l))
                        layers = l;
                    else if (line.Contains("Dominant Precision:"))
                        precision = line.Split(':')[1].Trim();
                }

                return new ModelAnalysisResult(true, layers, totalSize, baseSize, avgLayerSize, precision, "", fullOutput.ToString(), toolName);
            }
            catch (Exception ex)
            {
                return new ModelAnalysisResult(false, 0, 0, 0, 0, "", $"Exception: {ex.Message}", ex.ToString(), toolName);
            }
        }

        /// <summary>
        /// Analyzes model using Olive framework.
        /// </summary>
        private async Task<ModelAnalysisResult?> AnalyzeWithOliveAsync(string modelPath)
        {
            if (!Directory.Exists(modelPath))
                return null;
            
            var baseScriptPath = FindSplitModelScript();
            if (string.IsNullOrEmpty(baseScriptPath))
                return null;
            
            var scriptPath = baseScriptPath.Replace("split_model.py", "analyze_with_olive.py");
            if (!File.Exists(scriptPath))
                return null;
            
            // Try to use Olive virtual environment Python (Olive is installed there)
            var baseDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? AppDomain.CurrentDomain.BaseDirectory;
            var olivePythonPath = Path.Combine(baseDir, "scripts", "olive", ".olive-env", "Scripts", "python.exe");
            string pythonPath = "python"; // Fallback to system Python
            if (File.Exists(olivePythonPath))
            {
                pythonPath = olivePythonPath;
            }
            
            var psi = new ProcessStartInfo
            {
                FileName = pythonPath,
                Arguments = $"\"{scriptPath}\" \"{modelPath}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            
            try
            {
                using var process = Process.Start(psi);
                if (process == null) return null;
                
                var output = await process.StandardOutput.ReadToEndAsync();
                var error = await process.StandardError.ReadToEndAsync();
                await process.WaitForExitAsync();
                
                var fullOutput = new StringBuilder();
                if (!string.IsNullOrEmpty(output))
                    fullOutput.AppendLine(output);
                if (!string.IsNullOrEmpty(error))
                    fullOutput.AppendLine(error);
                
                return new ModelAnalysisResult(
                    process.ExitCode == 0,
                    0, 0, 0, 0,
                    "",
                    process.ExitCode != 0 ? error : "",
                    fullOutput.ToString(),
                    "Olive Framework"
                );
            }
            catch
            {
                return null; // Silently fail - Olive may not be available
            }
        }

        /// <summary>
        /// Analyzes model using ONNX Runtime tools.
        /// </summary>
        private async Task<ModelAnalysisResult?> AnalyzeWithOnnxRuntimeAsync(string modelPath)
        {
            var baseScriptPath = FindSplitModelScript();
            if (string.IsNullOrEmpty(baseScriptPath))
                return null;
            
            var scriptPath = baseScriptPath.Replace("split_model.py", "analyze_with_onnxruntime.py");
            if (!File.Exists(scriptPath))
                return null;
            
            // Try to use Olive virtual environment Python (ONNX might be installed there)
            var baseDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? AppDomain.CurrentDomain.BaseDirectory;
            var olivePythonPath = Path.Combine(baseDir, "scripts", "olive", ".olive-env", "Scripts", "python.exe");
            string pythonPath = "python"; // Fallback to system Python
            if (File.Exists(olivePythonPath))
            {
                pythonPath = olivePythonPath;
            }
            
            var psi = new ProcessStartInfo
            {
                FileName = pythonPath,
                Arguments = $"\"{scriptPath}\" \"{modelPath}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            
            try
            {
                using var process = Process.Start(psi);
                if (process == null) return null;
                
                var output = await process.StandardOutput.ReadToEndAsync();
                var error = await process.StandardError.ReadToEndAsync();
                await process.WaitForExitAsync();
                
                var fullOutput = new StringBuilder();
                if (!string.IsNullOrEmpty(output))
                    fullOutput.AppendLine(output);
                if (!string.IsNullOrEmpty(error))
                    fullOutput.AppendLine(error);
                
                return new ModelAnalysisResult(
                    process.ExitCode == 0,
                    0, 0, 0, 0,
                    "",
                    process.ExitCode != 0 ? error : "",
                    fullOutput.ToString(),
                    "ONNX Runtime"
                );
            }
            catch
            {
                return null; // Silently fail - ONNX may not be available
            }
        }

        /// <summary>
        /// Runs enhanced analysis with layer type mapping, MoE/Mamba-2/GQA analysis, etc.
        /// </summary>
        private async Task<ModelAnalysisResult?> AnalyzeEnhancedAsync(string modelPath)
        {
            if (!Directory.Exists(modelPath))
                return null;
            
            var baseScriptPath = FindSplitModelScript();
            if (string.IsNullOrEmpty(baseScriptPath))
                return null;
            
            var scriptPath = baseScriptPath.Replace("split_model.py", "analyze_model_enhanced.py");
            if (!File.Exists(scriptPath))
                return null;
            
            // Try to use Olive virtual environment Python (transformers might be installed there)
            var baseDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? AppDomain.CurrentDomain.BaseDirectory;
            var olivePythonPath = Path.Combine(baseDir, "scripts", "olive", ".olive-env", "Scripts", "python.exe");
            string pythonPath = "python"; // Fallback to system Python
            if (File.Exists(olivePythonPath))
            {
                pythonPath = olivePythonPath;
            }
            
            var psi = new ProcessStartInfo
            {
                FileName = pythonPath,
                Arguments = $"\"{scriptPath}\" \"{modelPath}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            
            try
            {
                using var process = Process.Start(psi);
                if (process == null) return null;
                
                var output = await process.StandardOutput.ReadToEndAsync();
                var error = await process.StandardError.ReadToEndAsync();
                await process.WaitForExitAsync();
                
                var fullOutput = new StringBuilder();
                if (!string.IsNullOrEmpty(output))
                    fullOutput.AppendLine(output);
                if (!string.IsNullOrEmpty(error))
                    fullOutput.AppendLine(error);
                
                return new ModelAnalysisResult(
                    process.ExitCode == 0,
                    0, 0, 0, 0,
                    "",
                    process.ExitCode != 0 ? error : "",
                    fullOutput.ToString(),
                    "Enhanced Analysis"
                );
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Export hybrid CPU/GPU partitions using split_model.py.
        /// </summary>
        public async Task ExportHybridPartitionsAsync(
            string onnxModelPath,
            string outputFolder,
            int splitLayer,
            string? huggingFaceToken = null,
            string? cacheDir = null)
        {
            _cts = new CancellationTokenSource();
            _stderrBuffer.Clear();

            // Find split_model.py - should be in the Scripts folder
            var scriptPath = FindSplitModelScript();
            if (string.IsNullOrEmpty(scriptPath))
            {
                OnErrorReceived?.Invoke("split_model.py not found. Cannot create hybrid partitions.");
                OnCompleted?.Invoke(false);
                return;
            }

            // Ensure output directory exists (script might not create parent)
            if (!Directory.Exists(outputFolder))
            {
                Directory.CreateDirectory(outputFolder);
            }

            var args = new StringBuilder();
            // Usage: split_model.py [-h] --split-layer SPLIT_LAYER [--output OUTPUT] model
            args.Append($"\"{scriptPath}\" ");
            args.Append($"\"{onnxModelPath}\" ");
            args.Append($"--split-layer {splitLayer} ");
            args.Append($"--output \"{outputFolder}\" ");

            // Set environment variable for HuggingFace token
            var env = new Dictionary<string, string>();
            if (!string.IsNullOrEmpty(huggingFaceToken))
            {
                env["HF_TOKEN"] = huggingFaceToken;
            }
            if (!string.IsNullOrEmpty(cacheDir))
            {
                env["HF_HOME"] = cacheDir;
            }

            var psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = args.ToString(),
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            foreach (var (key, value) in env)
            {
                psi.EnvironmentVariables[key] = value;
            }

            OnOutputReceived?.Invoke($"Creating hybrid partitions (Split at Layer {splitLayer})...");
            OnOutputReceived?.Invoke($"Command arguments: {args}");
            OnOutputReceived?.Invoke($"Script: {scriptPath}");
            OnOutputReceived?.Invoke($"Model: {onnxModelPath}");
            OnOutputReceived?.Invoke($"Output: {outputFolder}");
            OnOutputReceived?.Invoke("");

            try
            {
                _currentProcess = Process.Start(psi);
                if (_currentProcess == null)
                {
                    OnErrorReceived?.Invoke("Failed to start Python process");
                    OnCompleted?.Invoke(false);
                    return;
                }

                var outputTask = ReadStreamAsync(_currentProcess.StandardOutput, line =>
                {
                    OnOutputReceived?.Invoke(line);
                }, _cts.Token);

                var errorTask = ReadStreamAsync(_currentProcess.StandardError, line =>
                {
                    var cleaned = CleanPythonOutput(line);
                    _stderrBuffer.AppendLine(line);
                    if (!string.IsNullOrWhiteSpace(cleaned))
                    {
                        OnErrorReceived?.Invoke(cleaned);
                    }
                }, _cts.Token);

                await _currentProcess.WaitForExitAsync(_cts.Token);
                await Task.WhenAll(outputTask, errorTask);

                var success = _currentProcess.ExitCode == 0;

                if (success)
                {
                    OnOutputReceived?.Invoke("");
                    OnOutputReceived?.Invoke("✓ Hybrid partitions created successfully!");
                    OnOutputReceived?.Invoke($"  GPU partition: {Path.Combine(outputFolder, "gpu_part.onnx")}");
                    OnOutputReceived?.Invoke($"  CPU partition: {Path.Combine(outputFolder, "cpu_part.onnx")}");
                }
                else
                {
                    OnOutputReceived?.Invoke("✗ Hybrid partition creation failed");
                    var diagnostic = AnalyzeError(_stderrBuffer.ToString(), onnxModelPath);
                    OnDiagnosticGenerated?.Invoke(diagnostic);
                }

                OnCompleted?.Invoke(success);
            }
            catch (OperationCanceledException)
            {
                OnOutputReceived?.Invoke("Export cancelled");
                OnCompleted?.Invoke(false);
            }
            catch (Exception ex)
            {
                OnErrorReceived?.Invoke($"Error: {ex.Message}");
                OnCompleted?.Invoke(false);
            }
            finally
            {
                _currentProcess = null;
            }
        }

        /// <summary>
        /// Find the split_model.py script in common locations.
        /// </summary>
        private static string? FindSplitModelScript()
        {
            // Check possible locations - prefer Scripts folder
            var candidates = new[]
            {
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "scripts", "split_model.py"),
            };

            foreach (var path in candidates)
            {
                var fullPath = Path.GetFullPath(path);
                if (File.Exists(fullPath))
                {
                    return fullPath;
                }
            }

            return null;
        }

        /// <summary>
        /// Clean Python stderr output by removing noise prefixes.
        /// Python writes progress bars, INFO logs, and warnings to stderr.
        /// </summary>
        private static string CleanPythonOutput(string line)
        {
            if (string.IsNullOrEmpty(line)) return line;

            // Remove [ERROR] prefix that we add (it's stderr, not necessarily errors)
            line = line.TrimStart();
            if (line.StartsWith("[ERROR]"))
                line = line.Substring(7).TrimStart();

            // Filter out httpx INFO logs (they're not useful for users)
            if (line.Contains("httpx") && line.Contains("[INFO]"))
                return "";

            // Filter out timestamp-prefixed info messages
            if (Regex.IsMatch(line, @"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} httpx"))
                return "";

            // Remove progress bar remnants that aren't useful
            if (line.Contains("|") && (line.Contains("it/s") || line.Contains("s/it")))
            {
                // Keep just the last progress update
                var match = Regex.Match(line, @"(\d+%.*?\|\s*\d+/\d+)");
                if (match.Success && line.Contains("100%"))
                    return $"Progress: {match.Groups[1].Value}";
                return ""; // Skip intermediate progress
            }

            return line;
        }

        /// <summary>
        /// Analyze stderr to categorize the error and provide actionable diagnostics.
        /// </summary>
        private ConversionDiagnostic AnalyzeError(string stderr, string modelName)
        {
            // Upstream bugs - assertion errors in ONNX Runtime GenAI
            if (stderr.Contains("AssertionError"))
            {
                if (stderr.Contains("make_rotary_embedding") || stderr.Contains("make_inv_freq"))
                {
                    return new ConversionDiagnostic(
                        ConversionErrorCategory.UpstreamBug,
                        "ONNX Runtime GenAI has a bug in rotary embedding calculation for this model",
                        "The builder's assertion '0 < low < high < d_half - 1' failed during rotary embedding cache creation. This is likely due to an unsupported rope_scaling configuration.",
                        "Try a different model, or wait for an onnxruntime-genai update",
                        "https://github.com/microsoft/onnxruntime-genai/issues"
                    );
                }

                return new ConversionDiagnostic(
                    ConversionErrorCategory.UpstreamBug,
                    "ONNX Runtime GenAI builder encountered an assertion error",
                    ExtractTraceback(stderr),
                    "This appears to be a bug in the builder. Report it with the traceback.",
                    "https://github.com/microsoft/onnxruntime-genai/issues"
                );
            }

            // Model architecture not supported
            if (stderr.Contains("KeyError") && stderr.Contains("architectures"))
            {
                return new ConversionDiagnostic(
                    ConversionErrorCategory.ModelIncompatible,
                    "This model's architecture is not supported by ONNX Runtime GenAI",
                    "The builder doesn't recognize this model type. Supported: Phi, Llama, Mistral, Qwen, Gemma, etc.",
                    "Try a model with a supported architecture",
                    null
                );
            }

            // Incomplete model folder (missing config.json or architectures field)
            if (stderr.Contains("TypeError") && stderr.Contains("'NoneType'") && stderr.Contains("architectures"))
            {
                return new ConversionDiagnostic(
                    ConversionErrorCategory.ModelIncompatible,
                    "The model folder is incomplete or corrupted",
                    "The config.json file is missing the 'architectures' field. This often happens when using a HuggingFace cache folder instead of a proper model.",
                    "Use the HuggingFace model ID (e.g., 'meta-llama/Llama-3.2-1B-Instruct') instead of a local temp/cache folder",
                    null
                );
            }

            // MXFP4/Triton warning (not fatal, but informative)
            if (stderr.Contains("MXFP4 quantization requires Triton"))
            {
                // This is a warning, not the main error - continue analysis
            }

            // Missing dependencies
            if (stderr.Contains("ModuleNotFoundError") || stderr.Contains("ImportError"))
            {
                var missingModule = ExtractMissingModule(stderr);
                return new ConversionDiagnostic(
                    ConversionErrorCategory.MissingDependency,
                    $"Missing Python dependency: {missingModule}",
                    $"The module '{missingModule}' is required but not installed.",
                    $"Install with: pip install {missingModule.Replace("_", "-")}",
                    null
                );
            }

            // Authentication errors
            if (stderr.Contains("401") || stderr.Contains("Unauthorized") || 
                stderr.Contains("Access to model") && stderr.Contains("restricted"))
            {
                return new ConversionDiagnostic(
                    ConversionErrorCategory.AuthenticationError,
                    "Authentication required or access denied",
                    "This model requires authentication or you don't have access.",
                    "Check your HuggingFace token and ensure you've accepted the model's license",
                    null
                );
            }

            // Network errors
            if (stderr.Contains("ConnectionError") || stderr.Contains("TimeoutError") ||
                stderr.Contains("HTTPError"))
            {
                return new ConversionDiagnostic(
                    ConversionErrorCategory.NetworkError,
                    "Network connection failed",
                    "Could not connect to HuggingFace to download the model.",
                    "Check your internet connection and try again",
                    null
                );
            }

            // Disk space
            if (stderr.Contains("No space left") || stderr.Contains("disk quota"))
            {
                return new ConversionDiagnostic(
                    ConversionErrorCategory.DiskSpaceError,
                    "Insufficient disk space",
                    "There isn't enough disk space to complete the conversion.",
                    "Free up disk space or use a different output folder",
                    null
                );
            }

            // CUDA/GPU errors
            if (stderr.Contains("CUDA out of memory") || stderr.Contains("RuntimeError: CUDA"))
            {
                return new ConversionDiagnostic(
                    ConversionErrorCategory.ModelIncompatible,
                    "GPU memory insufficient for this model",
                    "The model is too large for your GPU's VRAM.",
                    "Try INT4 quantization, use CPU provider, or try a smaller model",
                    null
                );
            }

            // Default: unknown error
            return new ConversionDiagnostic(
                ConversionErrorCategory.UnknownError,
                "Conversion failed with an unexpected error",
                ExtractTraceback(stderr),
                "Check the full output above for details",
                null
            );
        }

        private static string ExtractMissingModule(string stderr)
        {
            var match = Regex.Match(stderr, @"No module named '([^']+)'");
            if (match.Success) return match.Groups[1].Value;

            match = Regex.Match(stderr, @"ModuleNotFoundError: ([^\n]+)");
            if (match.Success) return match.Groups[1].Value;

            return "unknown module";
        }

        private static string ExtractTraceback(string stderr)
        {
            // Find the last traceback section
            var tracebackIndex = stderr.LastIndexOf("Traceback (most recent call last)");
            if (tracebackIndex >= 0)
            {
                var traceback = stderr.Substring(tracebackIndex);
                // Limit to 500 chars
                if (traceback.Length > 500)
                    traceback = traceback.Substring(0, 500) + "...";
                return traceback;
            }
            return stderr.Length > 300 ? stderr.Substring(stderr.Length - 300) : stderr;
        }

        private static async Task ReadStreamAsync(StreamReader reader, Action<string>? callback, CancellationToken ct)
        {
            while (!ct.IsCancellationRequested)
            {
                var line = await reader.ReadLineAsync();
                if (line == null) break;
                callback?.Invoke(line);
            }
        }

        /// <summary>
        /// Fixes values in config files that ONNX Runtime GenAI cannot parse.
        /// - genai_config.json: "top_k": null -> "top_k": 50
        /// - tokenizer_config.json: "tokenizer_class": "TokenizersBackend" -> "PreTrainedTokenizerFast"
        /// </summary>
        private void SanitizeGenAiConfig(string modelFolder)
        {
            try
            {
                // Fix genai_config.json
                var genaiConfigPath = Path.Combine(modelFolder, "genai_config.json");
                if (File.Exists(genaiConfigPath))
                {
                    var content = File.ReadAllText(genaiConfigPath);
                    var modified = false;

                    // Fix "top_k": null -> "top_k": 50 (reasonable default)
                    if (content.Contains("\"top_k\": null") || content.Contains("\"top_k\":null"))
                    {
                        content = System.Text.RegularExpressions.Regex.Replace(
                            content, 
                            @"""top_k"":\s*null", 
                            "\"top_k\": 50");
                        modified = true;
                    }

                    if (modified)
                    {
                        File.WriteAllText(genaiConfigPath, content);
                        OnOutputReceived?.Invoke("  ⚡ Fixed genai_config.json (null values sanitized)");
                    }
                }

                // Fix tokenizer_config.json
                var tokenizerConfigPath = Path.Combine(modelFolder, "tokenizer_config.json");
                if (File.Exists(tokenizerConfigPath))
                {
                    var content = File.ReadAllText(tokenizerConfigPath);
                    var modified = false;

                    // Fix unsupported tokenizer class
                    if (content.Contains("\"tokenizer_class\": \"TokenizersBackend\"") || 
                        content.Contains("\"tokenizer_class\":\"TokenizersBackend\""))
                    {
                        content = System.Text.RegularExpressions.Regex.Replace(
                            content, 
                            @"""tokenizer_class"":\s*""TokenizersBackend""", 
                            "\"tokenizer_class\": \"PreTrainedTokenizerFast\"");
                        modified = true;
                    }

                    if (modified)
                    {
                        File.WriteAllText(tokenizerConfigPath, content);
                        OnOutputReceived?.Invoke("  ⚡ Fixed tokenizer_config.json (tokenizer class sanitized)");
                    }
                }
            }
            catch (Exception ex)
            {
                OnErrorReceived?.Invoke($"Warning: Could not sanitize config files: {ex.Message}");
            }
        }

        /// <summary>
        /// Resolves a model path, detecting HuggingFace cache structure and returning the actual model folder.
        /// HuggingFace cache structure: {path}/models--{org}--{name}/snapshots/{hash}/
        /// Git clone structure: {path}/ (with config.json at root)
        /// </summary>
        private static string ResolveModelPath(string path)
        {
            // If config.json exists at root, it's a proper model folder
            if (File.Exists(Path.Combine(path, "config.json")))
            {
                return path;
            }

            // Check for HuggingFace cache structure: models--{org}--{name}/snapshots/{hash}/
            var subdirs = Directory.GetDirectories(path);
            foreach (var subdir in subdirs)
            {
                var dirName = Path.GetFileName(subdir);
                if (dirName.StartsWith("models--"))
                {
                    // Found HF cache structure, look for snapshots
                    var snapshotsDir = Path.Combine(subdir, "snapshots");
                    if (Directory.Exists(snapshotsDir))
                    {
                        // Get the first snapshot (usually there's only one)
                        var snapshots = Directory.GetDirectories(snapshotsDir);
                        if (snapshots.Length > 0)
                        {
                            // Use the most recently modified snapshot
                            var latestSnapshot = snapshots
                                .OrderByDescending(d => Directory.GetLastWriteTime(d))
                                .First();
                            
                            // Verify it has config.json
                            if (File.Exists(Path.Combine(latestSnapshot, "config.json")))
                            {
                                return latestSnapshot;
                            }
                        }
                    }
                }
            }

            // Also check if path itself IS the snapshot folder (one level up from models--)
            if (path.Contains("snapshots") && File.Exists(Path.Combine(path, "config.json")))
            {
                return path;
            }

            // No HF cache structure found, return original path
            return path;
        }

        /// <summary>
        /// Verifies the split models by analyzing them individually.
        /// </summary>
        public async Task<(ModelAnalysisResult GpuResult, ModelAnalysisResult CpuResult)> VerifySplitModelsAsync(string outputFolder)
        {
            var gpuPath = Path.Combine(outputFolder, "gpu_part.onnx");
            var cpuPath = Path.Combine(outputFolder, "cpu_part.onnx");

            OnOutputReceived?.Invoke("");
            OnOutputReceived?.Invoke("🔍 Verifying split models...");

            var gpuResult = await InspectModelAsync(gpuPath);
            OnOutputReceived?.Invoke($"  GPU Part: {gpuResult.LayerCount} layers, {gpuResult.Precision}, {gpuResult.TotalSizeGb:F2} GB");

            var cpuResult = await InspectModelAsync(cpuPath);
            OnOutputReceived?.Invoke($"  CPU Part: {cpuResult.LayerCount} layers, {cpuResult.Precision}, {cpuResult.TotalSizeGb:F2} GB");
            OnOutputReceived?.Invoke("");

            return (gpuResult, cpuResult);
        }
    }
}
