using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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

        public async Task ConvertModelAsync(
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
            
            // Model source
            if (Directory.Exists(modelNameOrPath))
            {
                args.Append($"-i \"{modelNameOrPath}\" ");
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
                    return;
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
            }
            catch (OperationCanceledException)
            {
                OnOutputReceived?.Invoke("Conversion cancelled");
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
    }
}

