using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MelonStudio.Services
{
    public class ModelBuilderService
    {
        public event Action<string>? OnOutputReceived;
        public event Action<string>? OnErrorReceived;
        public event Action<bool>? OnCompleted;

        private Process? _currentProcess;
        private CancellationTokenSource? _cts;

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

            var psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = args.ToString(),
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = outputFolder
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
                var outputTask = ReadStreamAsync(_currentProcess.StandardOutput, OnOutputReceived, _cts.Token);
                var errorTask = ReadStreamAsync(_currentProcess.StandardError, OnErrorReceived, _cts.Token);

                await _currentProcess.WaitForExitAsync(_cts.Token);
                await Task.WhenAll(outputTask, errorTask);

                var success = _currentProcess.ExitCode == 0;
                OnOutputReceived?.Invoke("");
                OnOutputReceived?.Invoke(success ? "✓ Conversion completed successfully!" : "✗ Conversion failed");
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
