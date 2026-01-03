using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MelonStudio.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace MelonStudio.Services
{
    /// <summary>
    /// Hybrid CPU/GPU inference service for running large models
    /// that exceed GPU VRAM by partitioning across devices.
    /// 
    /// Uses raw ONNX Runtime InferenceSession for partition orchestration,
    /// with ONNX Runtime GenAI only for tokenization.
    /// </summary>
    public class HybridLLMService : IDisposable
    {
        // Raw ONNX Runtime sessions for partition execution
        private InferenceSession? _gpuSession;
        private InferenceSession? _cpuSession;
        
        // GenAI model and tokenizer (for tokenization only)
        private Model? _tokenizerModel;
        private Tokenizer? _tokenizer;
        
        // Configuration
        private HybridConfig? _hybridConfig;
        private string? _modelDir;
        private bool _isHybridMode;
        private bool _isInitialized;
        private bool _disposed;

        // Generation settings
        private int _maxLength = 1024;
        private float _temperature = 0.7f;
        private float _topP = 0.9f;

        // Events
        public event Action<string>? OnStatusChanged;
        public event Action<double>? OnProgressChanged;
        public event Action<string>? OnDiagnostic;

        // Properties
        public bool IsHybridMode => _isHybridMode;
        public int GpuLayers => _hybridConfig?.GpuPartition?.NumLayers ?? 0;
        public int CpuLayers => _hybridConfig?.CpuPartition?.NumLayers ?? 0;
        public int TotalLayers => _hybridConfig?.TotalLayers ?? 0;
        public string Summary => _hybridConfig?.Summary ?? "No model loaded";

        /// <summary>
        /// Initialize hybrid inference from a partition directory.
        /// </summary>
        public async Task<bool> InitializeHybridAsync(string partitionDir)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HybridLLMService));

            try
            {
                _modelDir = partitionDir;
                OnStatusChanged?.Invoke("Loading hybrid configuration...");
                OnDiagnostic?.Invoke($"Partition directory: {partitionDir}");

                // Load hybrid config
                var configPath = Path.Combine(partitionDir, "hybrid_config.json");
                if (!File.Exists(configPath))
                {
                    throw new FileNotFoundException("hybrid_config.json not found", configPath);
                }

                var json = await File.ReadAllTextAsync(configPath);
                _hybridConfig = JsonSerializer.Deserialize<HybridConfig>(json);
                
                if (_hybridConfig == null || !_hybridConfig.IsValid)
                {
                    throw new InvalidOperationException("Invalid hybrid configuration");
                }

                OnDiagnostic?.Invoke($"Config: {_hybridConfig.Architecture}, {_hybridConfig.Summary}");
                OnProgressChanged?.Invoke(0.1);

                // Create GPU session with CUDA EP
                var gpuPath = Path.Combine(partitionDir, _hybridConfig.GpuPartition.OnnxPath);
                if (!File.Exists(gpuPath))
                {
                    throw new FileNotFoundException("GPU partition not found", gpuPath);
                }

                OnStatusChanged?.Invoke($"Loading GPU partition ({_hybridConfig.GpuPartition.NumLayers} layers)...");
                OnDiagnostic?.Invoke($"GPU partition: {gpuPath}");

                await Task.Run(() =>
                {
                    var gpuOptions = new SessionOptions();
                    gpuOptions.AppendExecutionProvider_CUDA(0);
                    gpuOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    _gpuSession = new InferenceSession(gpuPath, gpuOptions);
                });

                OnDiagnostic?.Invoke($"GPU session inputs: {string.Join(", ", _gpuSession!.InputMetadata.Keys)}");
                OnDiagnostic?.Invoke($"GPU session outputs: {string.Join(", ", _gpuSession.OutputMetadata.Keys)}");
                OnProgressChanged?.Invoke(0.4);

                // Create CPU session
                var cpuPath = Path.Combine(partitionDir, _hybridConfig.CpuPartition.OnnxPath);
                if (!File.Exists(cpuPath))
                {
                    throw new FileNotFoundException("CPU partition not found", cpuPath);
                }

                OnStatusChanged?.Invoke($"Loading CPU partition ({_hybridConfig.CpuPartition.NumLayers} layers)...");
                OnDiagnostic?.Invoke($"CPU partition: {cpuPath}");

                await Task.Run(() =>
                {
                    var cpuOptions = new SessionOptions();
                    cpuOptions.AppendExecutionProvider_CPU(0);
                    cpuOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    _cpuSession = new InferenceSession(cpuPath, cpuOptions);
                });

                OnDiagnostic?.Invoke($"CPU session inputs: {string.Join(", ", _cpuSession!.InputMetadata.Keys)}");
                OnDiagnostic?.Invoke($"CPU session outputs: {string.Join(", ", _cpuSession.OutputMetadata.Keys)}");
                OnProgressChanged?.Invoke(0.7);

                // Load tokenizer (use GenAI for this - need a GenAI-compatible model)
                // For now, look for a tokenizer in the partition directory
                OnStatusChanged?.Invoke("Loading tokenizer...");
                await LoadTokenizerAsync(partitionDir);

                OnProgressChanged?.Invoke(1.0);

                _isHybridMode = true;
                _isInitialized = true;
                OnStatusChanged?.Invoke($"Hybrid ready: {_hybridConfig.Summary}");

                return true;
            }
            catch (Exception ex)
            {
                OnStatusChanged?.Invoke($"Hybrid init failed: {ex.Message}");
                OnDiagnostic?.Invoke($"Error: {ex}");
                Cleanup();
                throw;
            }
        }

        /// <summary>
        /// Load tokenizer from the partition directory.
        /// Falls back to sentinel model if no tokenizer files found.
        /// </summary>
        private async Task LoadTokenizerAsync(string partitionDir)
        {
            // Check for tokenizer files
            var tokenizerPath = Path.Combine(partitionDir, "tokenizer.json");
            var tokenizerConfigPath = Path.Combine(partitionDir, "tokenizer_config.json");
            
            if (!File.Exists(tokenizerPath))
            {
                OnDiagnostic?.Invoke("Warning: tokenizer.json not found in partition directory");
                OnDiagnostic?.Invoke("Tokenization may fail - ensure tokenizer files are present");
                return;
            }

            // For ONNX Runtime GenAI tokenizer, we need a full model folder structure
            // The tokenizer is loaded from the model folder, not standalone files
            // We'll try to use the partition directory if it has the right structure
            try
            {
                await Task.Run(() =>
                {
                    // GenAI expects a model folder with genai_config.json
                    // If we have gpu_part.onnx, try loading tokenizer from there
                    _tokenizerModel = new Model(partitionDir);
                    _tokenizer = new Tokenizer(_tokenizerModel);
                });
                OnDiagnostic?.Invoke("Tokenizer loaded from GenAI model");
            }
            catch (Exception ex)
            {
                OnDiagnostic?.Invoke($"GenAI tokenizer failed: {ex.Message}");
                OnDiagnostic?.Invoke("Attempting standalone tokenizer load...");
                
                // TODO: Implement standalone HuggingFace tokenizer loading
                // This would require either:
                // 1. A C# tokenizer library (like TokenizerNet)
                // 2. Or keeping a small GenAI-compatible model just for tokenization
                throw new InvalidOperationException(
                    "Tokenizer not available. Ensure the partition directory contains " +
                    "a genai_config.json for ONNX Runtime GenAI compatibility.", ex);
            }
        }

        /// <summary>
        /// Initialize standard (non-hybrid) inference from a single model.
        /// </summary>
        public async Task<int> InitializeStandardAsync(string modelPath)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HybridLLMService));

            try
            {
                OnStatusChanged?.Invoke("Loading model (standard mode)...");
                _modelDir = modelPath;

                await Task.Run(() =>
                {
                    _tokenizerModel = new Model(modelPath);
                    _tokenizer = new Tokenizer(_tokenizerModel);
                });

                _gpuSession = null;
                _cpuSession = null;
                _hybridConfig = null;
                _isHybridMode = false;
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

        /// <summary>
        /// Update generation settings.
        /// </summary>
        public void UpdateSettings(int maxLength, float temperature, float topP)
        {
            _maxLength = maxLength;
            _temperature = temperature;
            _topP = topP;
        }

        /// <summary>
        /// Generate response using hybrid or standard inference.
        /// </summary>
        public async IAsyncEnumerable<string> GenerateResponseAsync(
            string userMessage,
            string systemPrompt,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (!_isInitialized || _tokenizer == null)
            {
                throw new InvalidOperationException("Service not initialized");
            }

            // Build prompt
            var prompt = string.IsNullOrWhiteSpace(systemPrompt)
                ? $"User: {userMessage}\nAssistant:"
                : $"System: {systemPrompt}\nUser: {userMessage}\nAssistant:";

            if (_isHybridMode && _gpuSession != null && _cpuSession != null)
            {
                // Hybrid inference path
                await foreach (var token in GenerateHybridAsync(prompt, cancellationToken))
                {
                    yield return token;
                }
            }
            else if (_tokenizerModel != null)
            {
                // Standard GenAI path
                await foreach (var token in GenerateStandardAsync(prompt, cancellationToken))
                {
                    yield return token;
                }
            }
            else
            {
                throw new InvalidOperationException("No inference engine available");
            }
        }

        /// <summary>
        /// Hybrid inference: GPU partition -> transfer -> CPU partition -> sample.
        /// </summary>
        private async IAsyncEnumerable<string> GenerateHybridAsync(
            string prompt,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            // Tokenize
            var inputSequence = _tokenizer!.Encode(prompt);
            var inputIds = new long[inputSequence.Length];
            for (int i = 0; i < inputSequence.Length; i++)
            {
                inputIds[i] = inputSequence[i];
            }

            int currentLength = inputIds.Length;
            var generatedIds = inputIds.ToList();
            
            OnDiagnostic?.Invoke($"Starting hybrid generation with {currentLength} input tokens");

            while (currentLength < _maxLength && !cancellationToken.IsCancellationRequested)
            {
                // Run one forward pass through both partitions
                var nextToken = await RunHybridForwardPassAsync(generatedIds.ToArray());
                
                if (nextToken == null) // EOS or error
                {
                    OnDiagnostic?.Invoke("Generation complete (EOS or error)");
                    break;
                }

                generatedIds.Add(nextToken.Value);
                currentLength++;

                // Decode and yield the new token
                var tokenText = _tokenizer.Decode(new[] { (int)nextToken.Value });
                yield return tokenText;
            }

            OnDiagnostic?.Invoke($"Generated {currentLength - inputIds.Length} new tokens");
        }

        /// <summary>
        /// Execute one forward pass through GPU and CPU partitions.
        /// </summary>
        private async Task<long?> RunHybridForwardPassAsync(long[] inputIds)
        {
            // Create input tensors
            int batchSize = 1;
            int seqLength = inputIds.Length;

            // Input IDs tensor
            var inputIdsTensor = new DenseTensor<long>(inputIds, new[] { batchSize, seqLength });
            
            // Attention mask (all 1s)
            var attentionMask = Enumerable.Repeat(1L, seqLength).ToArray();
            var attentionMaskTensor = new DenseTensor<long>(attentionMask, new[] { batchSize, seqLength });
            
            // Position IDs
            var positionIds = Enumerable.Range(0, seqLength).Select(i => (long)i).ToArray();
            var positionIdsTensor = new DenseTensor<long>(positionIds, new[] { batchSize, seqLength });

            return await Task.Run(() =>
            {
                try
                {
                    // Step 1: Run GPU partition
                    var gpuInputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                        NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
                        NamedOnnxValue.CreateFromTensor("position_ids", positionIdsTensor),
                    };

                    using var gpuResults = _gpuSession!.Run(gpuInputs);
                    
                    // Get hidden states from GPU output
                    var hiddenStatesResult = gpuResults.FirstOrDefault(r => 
                        r.Name == "hidden_states" || r.Name.Contains("hidden"));
                    
                    if (hiddenStatesResult == null)
                    {
                        OnDiagnostic?.Invoke($"GPU outputs: {string.Join(", ", gpuResults.Select(r => r.Name))}");
                        throw new InvalidOperationException("GPU partition did not produce hidden_states");
                    }

                    var hiddenStatesTensor = hiddenStatesResult.AsTensor<float>();

                    // Step 2: Run CPU partition
                    var cpuInputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("hidden_states", hiddenStatesTensor),
                        NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
                        NamedOnnxValue.CreateFromTensor("position_ids", positionIdsTensor),
                    };

                    using var cpuResults = _cpuSession!.Run(cpuInputs);
                    
                    // Get logits from CPU output
                    var logitsResult = cpuResults.FirstOrDefault(r => 
                        r.Name == "logits" || r.Name.Contains("logits"));
                    
                    if (logitsResult == null)
                    {
                        OnDiagnostic?.Invoke($"CPU outputs: {string.Join(", ", cpuResults.Select(r => r.Name))}");
                        throw new InvalidOperationException("CPU partition did not produce logits");
                    }

                    var logitsTensor = logitsResult.AsTensor<float>();
                    
                    // Step 3: Sample next token from logits
                    // Get logits for the last position
                    var vocabSize = _hybridConfig!.VocabSize;
                    var lastPositionLogits = new float[vocabSize];
                    for (int i = 0; i < vocabSize; i++)
                    {
                        lastPositionLogits[i] = logitsTensor[0, seqLength - 1, i];
                    }

                    // Apply temperature and sample
                    var nextToken = SampleToken(lastPositionLogits, _temperature, _topP);
                    
                    // Check for EOS (common EOS token IDs)
                    if (nextToken == 2 || nextToken == 0 || nextToken == 32000 || nextToken == 151643)
                    {
                        return null; // End of sequence
                    }

                    return nextToken;
                }
                catch (Exception ex)
                {
                    OnDiagnostic?.Invoke($"Hybrid forward pass error: {ex.Message}");
                    throw;
                }
            });
        }

        /// <summary>
        /// Sample next token using temperature and top-p sampling.
        /// </summary>
        private long SampleToken(float[] logits, float temperature, float topP)
        {
            // Apply temperature
            if (temperature > 0)
            {
                for (int i = 0; i < logits.Length; i++)
                {
                    logits[i] /= temperature;
                }
            }

            // Softmax
            var maxLogit = logits.Max();
            var expLogits = logits.Select(l => Math.Exp(l - maxLogit)).ToArray();
            var sumExp = expLogits.Sum();
            var probs = expLogits.Select(e => e / sumExp).ToArray();

            // Top-p (nucleus) sampling
            var sortedIndices = Enumerable.Range(0, probs.Length)
                .OrderByDescending(i => probs[i])
                .ToArray();

            double cumsum = 0;
            var candidates = new List<(int index, double prob)>();
            
            foreach (var idx in sortedIndices)
            {
                cumsum += probs[idx];
                candidates.Add((idx, probs[idx]));
                if (cumsum >= topP) break;
            }

            // Renormalize and sample
            var candidateSum = candidates.Sum(c => c.prob);
            var rand = new Random().NextDouble() * candidateSum;
            
            double acc = 0;
            foreach (var (idx, prob) in candidates)
            {
                acc += prob;
                if (rand <= acc)
                {
                    return idx;
                }
            }

            return candidates.Last().index;
        }

        /// <summary>
        /// Standard GenAI inference (fallback path).
        /// </summary>
        private async IAsyncEnumerable<string> GenerateStandardAsync(
            string prompt,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var sequences = _tokenizer!.Encode(prompt);

            using var generatorParams = new GeneratorParams(_tokenizerModel!);
            generatorParams.SetSearchOption("max_length", _maxLength);
            generatorParams.SetSearchOption("temperature", _temperature);
            generatorParams.SetSearchOption("top_p", _topP);

            using var generator = new Generator(_tokenizerModel!, generatorParams);
            generator.AppendTokenSequences(sequences);

            while (!generator.IsDone())
            {
                if (cancellationToken.IsCancellationRequested)
                    yield break;

                await Task.Run(() => generator.GenerateNextToken(), cancellationToken);

                var outputTokens = generator.GetSequence(0);
                var newToken = (int)outputTokens[outputTokens.Length - 1];
                var tokenText = _tokenizer.Decode(new[] { newToken });

                yield return tokenText;
            }
        }

        /// <summary>
        /// Validate a hybrid model partition directory.
        /// </summary>
        public async Task<(bool IsValid, List<string> Issues)> ValidatePartitionAsync(string partitionDir)
        {
            var issues = new List<string>();

            // Check hybrid_config.json
            var configPath = Path.Combine(partitionDir, "hybrid_config.json");
            if (!File.Exists(configPath))
            {
                issues.Add("hybrid_config.json not found");
                return (false, issues);
            }

            try
            {
                var json = await File.ReadAllTextAsync(configPath);
                var config = JsonSerializer.Deserialize<HybridConfig>(json);
                
                if (config == null)
                {
                    issues.Add("Failed to parse hybrid_config.json");
                    return (false, issues);
                }

                // Check GPU partition
                var gpuPath = Path.Combine(partitionDir, config.GpuPartition.OnnxPath);
                if (!File.Exists(gpuPath))
                {
                    issues.Add($"GPU partition not found: {config.GpuPartition.OnnxPath}");
                }

                // Check CPU partition
                var cpuPath = Path.Combine(partitionDir, config.CpuPartition.OnnxPath);
                if (!File.Exists(cpuPath))
                {
                    issues.Add($"CPU partition not found: {config.CpuPartition.OnnxPath}");
                }

                // Check tokenizer
                if (!File.Exists(Path.Combine(partitionDir, "tokenizer.json")))
                {
                    issues.Add("tokenizer.json not found (may fail at runtime)");
                }

                // Validate config values
                if (config.SplitLayer <= 0 || config.SplitLayer >= config.TotalLayers)
                {
                    issues.Add($"Invalid split layer: {config.SplitLayer} (total: {config.TotalLayers})");
                }

                if (config.HiddenSize <= 0)
                {
                    issues.Add("Invalid hidden size");
                }

                return (issues.Count == 0, issues);
            }
            catch (Exception ex)
            {
                issues.Add($"Validation error: {ex.Message}");
                return (false, issues);
            }
        }

        /// <summary>
        /// Check if a directory contains a hybrid model.
        /// </summary>
        public static bool IsHybridModelDirectory(string path)
        {
            return File.Exists(Path.Combine(path, "hybrid_config.json"));
        }

        private void Cleanup()
        {
            _gpuSession?.Dispose();
            _cpuSession?.Dispose();
            _tokenizerModel?.Dispose();
            _tokenizer?.Dispose();
            
            _gpuSession = null;
            _cpuSession = null;
            _tokenizerModel = null;
            _tokenizer = null;
            _hybridConfig = null;
            _isHybridMode = false;
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
