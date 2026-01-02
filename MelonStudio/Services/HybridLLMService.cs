using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace MelonStudio.Services
{
    /// <summary>
    /// Partition metadata from split_model.py
    /// </summary>
    public class PartitionInfo
    {
        public string? OriginalModel { get; set; }
        public int SplitLayer { get; set; }
        public string? SplitTensor { get; set; }
        public string? GpuPartition { get; set; }
        public string? CpuPartition { get; set; }
        public int TotalLayers { get; set; }
        public int GpuLayers { get; set; }
        public int CpuLayers { get; set; }
    }

    /// <summary>
    /// Hybrid CPU/GPU inference service for running large models
    /// that exceed GPU VRAM by partitioning across devices.
    /// 
    /// Milestone 1: Static 2-part partition, sequential execution.
    /// </summary>
    public class HybridLLMService : IDisposable
    {
        private Model? _gpuModel;
        private Model? _cpuModel;
        private Tokenizer? _tokenizer;
        private PartitionInfo? _partitionInfo;
        private bool _isInitialized;
        private bool _disposed;

        private int _maxLength = 1024;
        private float _temperature = 0.7f;
        private float _topP = 0.9f;

        public event Action<string>? OnStatusChanged;
        public event Action<double>? OnProgressChanged;

        public bool IsHybridMode => _partitionInfo != null && _cpuModel != null;
        public int GpuLayers => _partitionInfo?.GpuLayers ?? 0;
        public int CpuLayers => _partitionInfo?.CpuLayers ?? 0;
        public int TotalLayers => _partitionInfo?.TotalLayers ?? 0;

        public async Task<bool> InitializeHybridAsync(string partitionDir)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HybridLLMService));

            try
            {
                OnStatusChanged?.Invoke("Loading partition info...");

                var metadataPath = Path.Combine(partitionDir, "partition_info.json");
                if (!File.Exists(metadataPath))
                    throw new FileNotFoundException("Partition metadata not found", metadataPath);

                var json = await File.ReadAllTextAsync(metadataPath);
                _partitionInfo = JsonSerializer.Deserialize<PartitionInfo>(json, 
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                if (_partitionInfo == null)
                    throw new InvalidOperationException("Failed to parse partition metadata");

                var gpuPath = Path.Combine(partitionDir, _partitionInfo.GpuPartition ?? "gpu_part.onnx");
                OnStatusChanged?.Invoke($"Loading GPU partition ({_partitionInfo.GpuLayers} layers)...");
                OnProgressChanged?.Invoke(0.25);
                await Task.Run(() => { _gpuModel = new Model(gpuPath); });

                var cpuPath = Path.Combine(partitionDir, _partitionInfo.CpuPartition ?? "cpu_part.onnx");
                OnStatusChanged?.Invoke($"Loading CPU partition ({_partitionInfo.CpuLayers} layers)...");
                OnProgressChanged?.Invoke(0.5);
                await Task.Run(() => { _cpuModel = new Model(cpuPath); });

                OnStatusChanged?.Invoke("Loading tokenizer...");
                OnProgressChanged?.Invoke(0.75);
                await Task.Run(() => { _tokenizer = new Tokenizer(_gpuModel!); });

                _isInitialized = true;
                OnProgressChanged?.Invoke(1.0);
                OnStatusChanged?.Invoke($"Hybrid ready: {_partitionInfo.GpuLayers} GPU + {_partitionInfo.CpuLayers} CPU layers");
                return true;
            }
            catch (Exception ex)
            {
                OnStatusChanged?.Invoke($"Hybrid init failed: {ex.Message}");
                Cleanup();
                throw;
            }
        }

        public async Task<int> InitializeStandardAsync(string modelPath)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HybridLLMService));

            try
            {
                OnStatusChanged?.Invoke("Loading model (standard mode)...");
                await Task.Run(() =>
                {
                    _gpuModel = new Model(modelPath);
                    _tokenizer = new Tokenizer(_gpuModel);
                });

                _cpuModel = null;
                _partitionInfo = null;
                _isInitialized = true;
                OnStatusChanged?.Invoke("Model loaded (standard mode)");
                return 0;
            }
            catch (Exception ex)
            {
                OnStatusChanged?.Invoke($"Load failed: {ex.Message}");
                throw;
            }
        }

        public void UpdateSettings(int maxLength, float temperature, float topP)
        {
            _maxLength = maxLength;
            _temperature = temperature;
            _topP = topP;
        }

        public async IAsyncEnumerable<string> GenerateResponseAsync(
            string userMessage,
            string systemPrompt,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (!_isInitialized || _gpuModel == null || _tokenizer == null)
                throw new InvalidOperationException("Service not initialized");

            // Build prompt (simplified - real impl would use chat template)
            var prompt = string.IsNullOrWhiteSpace(systemPrompt)
                ? $"User: {userMessage}\nAssistant:"
                : $"System: {systemPrompt}\nUser: {userMessage}\nAssistant:";

            // Tokenize
            var sequences = _tokenizer.Encode(prompt);

            // Configure generator
            using var generatorParams = new GeneratorParams(_gpuModel);
            generatorParams.SetSearchOption("max_length", _maxLength);
            generatorParams.SetSearchOption("temperature", _temperature);
            generatorParams.SetSearchOption("top_p", _topP);
            generatorParams.SetInputSequences(sequences);

            // Note: In Milestone 1, we use standard generation on the full model.
            // True hybrid execution with separate GPU/CPU partitions requires
            // raw ONNX Runtime sessions and manual tensor passing (Milestone 2+).
            using var generator = new Generator(_gpuModel, generatorParams);

            await Task.CompletedTask; // Make async

            while (!generator.IsDone())
            {
                if (cancellationToken.IsCancellationRequested)
                    yield break;

                generator.ComputeLogits();
                generator.GenerateNextToken();

                var outputTokens = generator.GetSequence(0);
                var newToken = outputTokens[outputTokens.Length - 1];
                var tokenText = _tokenizer.Decode(new ReadOnlySpan<int>(new[] { (int)newToken }));

                yield return tokenText;
            }
        }

        private void Cleanup()
        {
            _gpuModel?.Dispose();
            _cpuModel?.Dispose();
            _tokenizer?.Dispose();
            _gpuModel = null;
            _cpuModel = null;
            _tokenizer = null;
            _partitionInfo = null;
            _isInitialized = false;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                Cleanup();
                _disposed = true;
            }
        }
    }
}
