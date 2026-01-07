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
        /// Export hybrid CPU/GPU partitions using split_model.py.
        /// </summary>
        public async Task ExportHybridPartitionsAsync(
            string modelNameOrPath,
            string outputFolder,
            int gpuLayers,
            string precision = "fp16",
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

            var args = new StringBuilder();
            args.Append($"\"{scriptPath}\" export ");
            args.Append($"\"{modelNameOrPath}\" ");
            args.Append($"--split-layer {gpuLayers} ");
            args.Append($"--output-dir \"{outputFolder}\" ");
            args.Append($"--precision {precision} ");
            args.Append("--json ");

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

            OnOutputReceived?.Invoke($"Creating hybrid partitions (GPU: {gpuLayers} layers)...");
            OnOutputReceived?.Invoke($"Script: {scriptPath}");
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
                    // Parse JSON diagnostics from split_model.py
                    if (line.StartsWith("[DIAG]"))
                    {
                        // Extract diagnostic message
                        OnOutputReceived?.Invoke(line.Substring(7));
                    }
                    else
                    {
                        OnOutputReceived?.Invoke(line);
                    }
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
                    OnOutputReceived?.Invoke($"  GPU partition: {outputFolder}/gpu_part.onnx");
                    OnOutputReceived?.Invoke($"  CPU partition: {outputFolder}/cpu_part.onnx");
                    OnOutputReceived?.Invoke($"  Config: {outputFolder}/hybrid_config.json");
                }
                else
                {
                    OnOutputReceived?.Invoke("✗ Hybrid partition creation failed");
                    var diagnostic = AnalyzeError(_stderrBuffer.ToString(), modelNameOrPath);
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
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Scripts", "split_model.py"),
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "Scripts", "split_model.py"),
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "Scripts", "split_model.py"),
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "MelonStudio", "Scripts", "split_model.py"),
                "split_model.py",  // Current directory fallback
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
    }
}
