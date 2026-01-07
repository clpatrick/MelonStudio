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
    public class HybridLLMService : IDisposable
    {
        private InferenceSession? _gpuSession;
        private InferenceSession? _cpuSession;
        
        // I/O Binding infrastructure for reduced allocation overhead
        private OrtIoBinding? _gpuIoBinding;
        private OrtIoBinding? _cpuIoBinding;
        private RunOptions? _runOptions;
        
        private Model? _tokenizerModel;
        private Tokenizer? _tokenizer;
        
        private HybridConfig? _hybridConfig;
        private string? _modelDir;
        private bool _isHybridMode;
        private bool _isInitialized;
        private bool _disposed;

        private int _maxLength = 1024;
        private float _temperature = 0.7f;
        private float _topP = 0.9f;
        private int _topK = 50;

        public event Action<string>? OnStatusChanged;
        public event Action<double>? OnProgressChanged;
        public event Action<string>? OnDiagnostic;

        public bool IsHybridMode => _isHybridMode;
        public int GpuLayers => _hybridConfig?.GpuPartition?.NumLayers ?? 0;
        public int CpuLayers => _hybridConfig?.CpuPartition?.NumLayers ?? 0;
        public int TotalLayers => _hybridConfig?.TotalLayers ?? 0;
        public string Summary => _hybridConfig?.Summary ?? "No model loaded";

        public async Task<bool> InitializeHybridAsync(string partitionDir)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HybridLLMService));

            try
            {
                _modelDir = partitionDir;
                OnStatusChanged?.Invoke("Loading hybrid configuration...");
                
                var configPath = Path.Combine(partitionDir, "hybrid_config.json");
                if (!File.Exists(configPath)) throw new FileNotFoundException("hybrid_config.json not found", configPath);

                var json = await File.ReadAllTextAsync(configPath);
                _hybridConfig = JsonSerializer.Deserialize<HybridConfig>(json);
                if (_hybridConfig == null || !_hybridConfig.IsValid) throw new InvalidOperationException("Invalid hybrid configuration");

                OnProgressChanged?.Invoke(0.1);

                var gpuPath = Path.Combine(partitionDir, _hybridConfig.GpuPartition.OnnxPath);
                if (!File.Exists(gpuPath)) throw new FileNotFoundException("GPU partition not found", gpuPath);

                OnStatusChanged?.Invoke($"Loading GPU partition ({_hybridConfig.GpuPartition.NumLayers} layers)...");
                await Task.Run(() =>
                {
                    var gpuOptions = new SessionOptions();
                    gpuOptions.AppendExecutionProvider_CUDA(0);
                    gpuOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    _gpuSession = new InferenceSession(gpuPath, gpuOptions);
                });
                OnProgressChanged?.Invoke(0.4);

                var cpuPath = Path.Combine(partitionDir, _hybridConfig.CpuPartition.OnnxPath);
                if (!File.Exists(cpuPath)) throw new FileNotFoundException("CPU partition not found", cpuPath);

                OnStatusChanged?.Invoke($"Loading CPU partition ({_hybridConfig.CpuPartition.NumLayers} layers)...");
                await Task.Run(() =>
                {
                    var cpuOptions = new SessionOptions();
                    cpuOptions.AppendExecutionProvider_CPU(0);
                    cpuOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    _cpuSession = new InferenceSession(cpuPath, cpuOptions);
                });
                OnProgressChanged?.Invoke(0.7);

                OnStatusChanged?.Invoke("Loading tokenizer...");
                await LoadTokenizerAsync(partitionDir);
                OnProgressChanged?.Invoke(0.85);

                // Initialize I/O Bindings for reduced per-step allocation
                OnStatusChanged?.Invoke("Initializing I/O bindings...");
                _gpuIoBinding = _gpuSession!.CreateIoBinding();
                _cpuIoBinding = _cpuSession!.CreateIoBinding();
                _runOptions = new RunOptions();
                OnProgressChanged?.Invoke(1.0);

                _isHybridMode = true;
                _isInitialized = true;
                OnStatusChanged?.Invoke($"Hybrid ready: {_hybridConfig.Summary}");
                return true;
            }
            catch (Exception ex)
            {
                OnStatusChanged?.Invoke($"Hybrid init failed: {ex.Message}");
                Cleanup();
                throw;
            }
        }

        private async Task LoadTokenizerAsync(string partitionDir)
        {
            string loadDir = partitionDir;
            if (_hybridConfig != null && !string.IsNullOrEmpty(_hybridConfig.SourceModel))
            {
                try 
                {
                    var sourcePath = _hybridConfig.SourceModel;
                    if (!Path.IsPathRooted(sourcePath)) sourcePath = Path.GetFullPath(sourcePath);
                    var sourceDir = Path.GetDirectoryName(sourcePath);
                    if (Directory.Exists(sourceDir)) loadDir = sourceDir;
                }
                catch {}
            }
            await Task.Run(() =>
            {
                _tokenizerModel = new Model(loadDir);
                _tokenizer = new Tokenizer(_tokenizerModel);
            });
        }

        public async Task<int> InitializeStandardAsync(string modelPath, string executionProvider = "cuda")
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HybridLLMService));
            try
            {
                OnStatusChanged?.Invoke($"Loading model (standard mode, EP={executionProvider})...");
                _modelDir = modelPath;
                await Task.Run(() =>
                {
                    try {
                        var config = new Config(modelPath);
                        if (executionProvider.ToLower() == "cuda") {
                            config.ClearProviders();
                            config.AppendProvider("cuda");
                        } else {
                            config.ClearProviders();
                        }
                        _tokenizerModel = new Model(config);
                    } catch {
                         _tokenizerModel = new Model(modelPath);
                    }
                    _tokenizer = new Tokenizer(_tokenizerModel);
                });
                _gpuSession = null;
                _cpuSession = null;
                _hybridConfig = null;
                _isHybridMode = false;
                _isInitialized = true;
                return 0;
            }
            catch (Exception ex) {
                OnStatusChanged?.Invoke($"Load failed: {ex.Message}");
                throw;
            }
        }

        public void UpdateSettings(int maxLength, float temperature, float topP, int topK = 50)
        {
            _maxLength = maxLength;
            _temperature = temperature;
            _topP = topP;
            _topK = topK;
        }

        public async IAsyncEnumerable<string> GenerateResponseAsync(
            string userMessage,
            string systemPrompt,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (!_isInitialized || _tokenizer == null) throw new InvalidOperationException("Service not initialized");

            var prompt = string.IsNullOrWhiteSpace(systemPrompt)
                ? $"User: {userMessage}\nAssistant:"
                : $"System: {systemPrompt}\nUser: {userMessage}\nAssistant:";

            if (_isHybridMode)
            {
                await foreach (var token in GenerateHybridAsync(prompt, cancellationToken)) yield return token;
            }
            else
            {
                await foreach (var token in GenerateStandardAsync(prompt, cancellationToken)) yield return token;
            }
        }

        /// <summary>
        /// Hybrid generation with KV Cache using I/O Binding (optimized).
        /// </summary>
        private async IAsyncEnumerable<string> GenerateHybridAsync(
            string prompt,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var sequences = _tokenizer!.Encode(prompt);
            var tokenIds = sequences[0];
            var tokenCount = (int)tokenIds.Length;
            var inputIds = new long[tokenCount];
            for (int i = 0; i < tokenCount; i++) inputIds[i] = tokenIds[i];

            var kvCache = new Dictionary<string, OrtValue>();
            
            long[] currentInputIds = inputIds; 
            int pastSequenceLength = 0; // How many tokens are in KV cache
            
            OnDiagnostic?.Invoke($"Starting KV-Cache Generation with I/O Binding. Input Len: {inputIds.Length}");

            try
            {
                int tokensGenerated = 0;
                while (tokensGenerated < _maxLength && !cancellationToken.IsCancellationRequested)
                {
                    int batchSize = 1;
                    int inputLen = currentInputIds.Length;
                    int totalSequenceLength = pastSequenceLength + inputLen;

                    // Attention Mask: all 1s for (past + current)
                    var maskData = Enumerable.Repeat(1L, totalSequenceLength).ToArray();
                    var attentionMaskTensor = new DenseTensor<long>(maskData, new[] { batchSize, totalSequenceLength });
                    
                    // Position IDs: [pastSequenceLength, pastSequenceLength+1, ...]
                    var positionIdsData = Enumerable.Range(pastSequenceLength, inputLen).Select(i => (long)i).ToArray();
                    var positionIdsTensor = new DenseTensor<long>(positionIdsData, new[] { batchSize, inputLen });
                    
                    var inputIdsTensor = new DenseTensor<long>(currentInputIds, new[] { batchSize, inputLen });

                    // Run Step
                    var (nextTokenId, newKvValues) = await RunHybridStepAsync(
                        inputIdsTensor, attentionMaskTensor, positionIdsTensor, kvCache, pastSequenceLength);
                    
                    // Update Cache - dispose old values before replacing
                    foreach (var kvp in newKvValues)
                    {
                        if (kvCache.TryGetValue(kvp.Key, out var oldValue)) oldValue.Dispose();
                        kvCache[kvp.Key] = kvp.Value;
                    }

                    if (nextTokenId == null) break;

                    long nextToken = nextTokenId.Value;
                    var tokenText = _tokenizer.Decode(new[] { (int)nextToken });
                    yield return tokenText;

                    currentInputIds = new[] { nextToken };
                    pastSequenceLength = totalSequenceLength; // New past = old past + what we just processed
                    tokensGenerated++;
                }
            }
            finally
            {
                foreach (var val in kvCache.Values) val.Dispose();
            }
        }

        // Cache for Pinned Interface Tensors (used for decode step len=1)
        private Dictionary<string, OrtValue> _pinnedInterfaceValues = new();
        private int _hiddenSize = 3072; // Default for Phi-3.5-mini

        private async Task<(long? NextToken, Dictionary<string, OrtValue> NewKvCache)> RunHybridStepAsync(
            DenseTensor<long> inputIds,
            DenseTensor<long> attentionMask,
            DenseTensor<long> positionIds,
            Dictionary<string, OrtValue> currentKvCache,
            int pastSequenceLength)
        {
            var newKvCache = new Dictionary<string, OrtValue>();
            var tempOrtValues = new List<OrtValue>(); // Track OrtValues for disposal after binding
            
            int inputLen = (int)inputIds.Dimensions[1];
            
            // --- GPU Partition with I/O Binding ---
            _gpuIoBinding!.ClearBoundInputs();
            _gpuIoBinding.ClearBoundOutputs();
            
            // Bind input_ids
            var inputIdsOrt = OrtValue.CreateTensorValueFromMemory(inputIds.ToArray(), 
                new long[] { 1, inputIds.Dimensions[1] });
            tempOrtValues.Add(inputIdsOrt);
            _gpuIoBinding.BindInput("input_ids", inputIdsOrt);
            
            // Bind attention_mask
            var maskOrt = OrtValue.CreateTensorValueFromMemory(attentionMask.ToArray(), 
                new long[] { 1, attentionMask.Dimensions[1] });
            tempOrtValues.Add(maskOrt);
            _gpuIoBinding.BindInput("attention_mask", maskOrt);
            
            // Bind position_ids if needed
            if (_gpuSession!.InputMetadata.ContainsKey("position_ids"))
            {
                var posOrt = OrtValue.CreateTensorValueFromMemory(positionIds.ToArray(), 
                    new long[] { 1, positionIds.Dimensions[1] });
                tempOrtValues.Add(posOrt);
                _gpuIoBinding.BindInput("position_ids", posOrt);
            }

            // Bind KV cache inputs (circular binding - reuse previous outputs)
            foreach (var inputName in _hybridConfig!.GpuPartition.Inputs)
            {
                if (inputName.StartsWith("past_key_values"))
                {
                    if (currentKvCache.TryGetValue(inputName, out var existingKv))
                    {
                        // Circular binding - reuse the OrtValue from previous step
                        _gpuIoBinding.BindInput(inputName, existingKv);
                    }
                    else
                    {
                        // First run: EMPTY KV with seq_len=0
                        var emptyData = Array.Empty<Float16>();
                        var emptyKv = OrtValue.CreateTensorValueFromMemory(emptyData, new long[] { 1, 32, 0, 96 });
                        tempOrtValues.Add(emptyKv);
                        _gpuIoBinding.BindInput(inputName, emptyKv);
                    }
                }
            }
            
            // Prepare Interface Tensor Bindings (Pinned Memory Optimization)
            if (inputLen == 1)
            {
                // FAST PATH: Reuse pinned buffers for single-token generation
                foreach (var outputName in _hybridConfig.InterfaceTensors.Values)
                {
                    if (!_pinnedInterfaceValues.ContainsKey(outputName))
                    {
                         // Allocate Pinned Memory: [1, 1, HiddenSize]
                         // We use Array.Empty or new[3072] but CreateTensorValueFromMemory PINS it.
                         // Phi-3.5 hidden_size is 3072. We assume this based on inspection.
                         // TODO: Dynamically detect hidden_size if possible.
                         var pinnedData = new Float16[_hiddenSize];
                         var pinnedOrt = OrtValue.CreateTensorValueFromMemory(pinnedData, new long[] { 1, 1, _hiddenSize });
                         _pinnedInterfaceValues[outputName] = pinnedOrt;
                    }
                    // Bind as Output for GPU
                    _gpuIoBinding.BindOutput(outputName, _pinnedInterfaceValues[outputName]);
                }
            }
            else
            {
                // SLOW PATH: Prefill or batch > 1. Let ORT allocate (OutputToDevice).
                // Clear any existing pinned buffers to be safe (or keep them?)
                // We keep them re-usable for later decode steps.
                foreach (var outputName in _hybridConfig.InterfaceTensors.Values)
                {
                     // Do NOT bind output. ORT will allocate device/host memory as needed.
                     _gpuIoBinding.BindOutputToDevice(outputName, OrtMemoryInfo.DefaultInstance); 
                }
            }
            
            // Bind other outputs (KV cache) to device
            foreach (var outputName in _hybridConfig.GpuPartition.Outputs)
            {
                // Skip interface tensors if we already bound them
                if (_hybridConfig.InterfaceTensors.ContainsValue(outputName) && inputLen == 1) continue;
                
                _gpuIoBinding.BindOutputToDevice(outputName, OrtMemoryInfo.DefaultInstance);
            }

            _gpuIoBinding.SynchronizeBoundInputs();
            
            // OnDiagnostic?.Invoke($"GPU IoBinding: ids={inputIds.Dimensions[1]}, mask={attentionMask.Dimensions[1]}, past={pastSequenceLength}");

            // Run GPU with binding
            IDisposableReadOnlyCollection<OrtValue> gpuResults;
            await Task.Run(() => 
            {
                _gpuSession.RunWithBinding(_runOptions!, _gpuIoBinding);
            });
            gpuResults = _gpuIoBinding.GetOutputValues();
            _gpuIoBinding.SynchronizeBoundOutputs();
            
            // Process GPU outputs
            var gpuOutputList = gpuResults.ToList();
            var capturedInterfaceTensors = new Dictionary<string, OrtValue>();
            
            // Mapping logic for outputs
            // NOTE: If we bound Interface Tensors explicitly (Fast Path), they might NOT be in gpuResults list depending on ORT version/flags.
            // Check if they are. GetOutputValues() usually returns all bound outputs.
            // BUT, if we provided the buffer, we have the OrtValue in _pinnedInterfaceValues.
            
            // We iterate _hybridConfig.GpuPartition.Outputs to be deterministic
            for (int i = 0; i < gpuOutputList.Count; i++)
            {
                var outputName = _hybridConfig.GpuPartition.Outputs[i];
                var ortValue = gpuOutputList[i];
                
                if (outputName.StartsWith("present"))
                {
                    var pastName = outputName.Replace("present.", "past_key_values.");
                    newKvCache[pastName] = ortValue; // Store for circular binding
                }
                else if (_hybridConfig.InterfaceTensors.ContainsValue(outputName))
                {
                    if (inputLen == 1)
                    {
                        // Fast Path: We used our pinned buffer.
                        // The 'ortValue' returned might be a reference to the same buffer.
                        // We rely on _pinnedInterfaceValues[outputName] for CPU binding.
                        // We should NOT dispose the one in _pinnedInterfaceValues.
                        // But 'ortValue' from GetOutputValues might need disposal if it's a separate handle?
                        // Actually, if we bound it, GetOutputValues returns the bound OrtValue.
                        // We stored it in _pinnedInterfaceValues.
                        // We track it in capturedInterfaceTensors to facilitate the CPU loop below.
                        capturedInterfaceTensors[outputName] = ortValue; 
                    }
                    else
                    {
                        // Slow Path: ORT allocated it. We capture it normally.
                        capturedInterfaceTensors[outputName] = ortValue;
                    }
                }
                else
                {
                   // Unused output
                   ortValue.Dispose();
                }
            }
            
            foreach (var ov in tempOrtValues) ov.Dispose();
            tempOrtValues.Clear();

            // --- CPU Partition with I/O Binding ---
            _cpuIoBinding!.ClearBoundInputs();
            _cpuIoBinding.ClearBoundOutputs();
            
            // Bind attention_mask (reuse from above, need fresh OrtValue)
            var cpuMaskOrt = OrtValue.CreateTensorValueFromMemory(attentionMask.ToArray(), 
                new long[] { 1, attentionMask.Dimensions[1] });
            tempOrtValues.Add(cpuMaskOrt);
            _cpuIoBinding.BindInput("attention_mask", cpuMaskOrt);
            
            // Bind position_ids if needed
            if (_cpuSession!.InputMetadata.ContainsKey("position_ids"))
            {
                var cpuPosOrt = OrtValue.CreateTensorValueFromMemory(positionIds.ToArray(), 
                    new long[] { 1, positionIds.Dimensions[1] });
                tempOrtValues.Add(cpuPosOrt);
                _cpuIoBinding.BindInput("position_ids", cpuPosOrt);
            }
            
            // Bind interface tensors from GPU
            foreach(var kvp in capturedInterfaceTensors)
            {
                 // Fast Path: This 'kvp.Value' is our Pinned _pinnedInterfaceValues[...]
                 // Slow Path: This is a device tensor (that needs copy?).
                 // If Slow Path (prefill), 'kvp.Value' is typically Device Memory (GPU).
                 // CPU Session cannot read GPU memory directly typically.
                 // ORT handles the copy if we bind it as input.
                 // Pinned Memory Win: In Fast Path, it is Host Pinned Memory. CPU can read it directly!
                _cpuIoBinding.BindInput(kvp.Key, kvp.Value);
            }
            
            // Bind CPU KV cache inputs
            foreach (var inputName in _hybridConfig.CpuPartition.Inputs)
            {
                if (inputName.StartsWith("past_key_values"))
                {
                    if (currentKvCache.TryGetValue(inputName, out var existingKv))
                    {
                        _cpuIoBinding.BindInput(inputName, existingKv);
                    }
                    else
                    {
                        var emptyData = Array.Empty<Float16>();
                        var emptyKv = OrtValue.CreateTensorValueFromMemory(emptyData, new long[] { 1, 32, 0, 96 });
                        tempOrtValues.Add(emptyKv);
                        _cpuIoBinding.BindInput(inputName, emptyKv);
                    }
                }
            }
            
            // Bind CPU outputs to device
            foreach (var outputName in _hybridConfig.CpuPartition.Outputs)
            {
                _cpuIoBinding.BindOutputToDevice(outputName, OrtMemoryInfo.DefaultInstance);
            }
            
            _cpuIoBinding.SynchronizeBoundInputs();
            
            // Run CPU
            IDisposableReadOnlyCollection<OrtValue> cpuResults;
            await Task.Run(() =>
            {
                _cpuSession.RunWithBinding(_runOptions!, _cpuIoBinding);
            });
            cpuResults = _cpuIoBinding.GetOutputValues();
            _cpuIoBinding.SynchronizeBoundOutputs();
            
            // Process CPU outputs
            var cpuOutputList = cpuResults.ToList();
            long? nextTokenId = null;
            
            for (int i = 0; i < cpuOutputList.Count; i++)
            {
                var outputName = _hybridConfig.CpuPartition.Outputs[i];
                var ortValue = cpuOutputList[i];
                
                if (outputName.StartsWith("present"))
                {
                    var pastName = outputName.Replace("present.", "past_key_values.");
                    newKvCache[pastName] = ortValue;
                }
                else if (outputName == "logits")
                {
                    // Extract logits data using GetTensorDataAsSpan
                    var typeInfo = ortValue.GetTensorTypeAndShape();
                    var vocabSize = (int)typeInfo.Shape[2];
                    var lastTokenIndex = (int)inputIds.Dimensions[1] - 1;
                    
                    var logitsSpan = ortValue.GetTensorDataAsSpan<Float16>();
                    var lastLogits = new float[vocabSize];
                    var offset = lastTokenIndex * vocabSize;
                    for (int j = 0; j < vocabSize; j++)
                    {
                        lastLogits[j] = (float)logitsSpan[offset + j];
                    }
                    
                    nextTokenId = SampleToken(lastLogits, _temperature, _topP, _topK);
                    ortValue.Dispose();
                }
                else
                {
                    ortValue.Dispose();
                }
            }
            
            // Dispose temporary values
            foreach (var ov in tempOrtValues) ov.Dispose();
            
            // Dispose Interface Tensors (Slow Path ONLY)
            // Fast Path: The OrtValue in capturedInterfaceTensors is ALIASING _pinnedInterfaceValues.
            // We MUST NOT dispose it, because we reuse it next step.
            if (inputLen > 1) 
            {
                foreach (var ov in capturedInterfaceTensors.Values) ov.Dispose();
            }

            return (nextTokenId, newKvCache);
        }

        private long SampleToken(float[] logits, float temperature, float topP, int topK)
        {
            if (temperature <= 0.0f)
            {
                float maxVal = float.NegativeInfinity;
                int maxIdx = 0;
                for (int i = 0; i < logits.Length; i++)
                {
                    if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; }
                }
                return maxIdx;
            }

            for (int i = 0; i < logits.Length; i++) logits[i] /= temperature;

            var maxLogit = logits.Max();
            var expLogits = logits.Select(l => Math.Exp(l - maxLogit)).ToArray();
            var sumExp = expLogits.Sum();
            var probs = expLogits.Select(e => e / sumExp).ToArray();

            var sortedIndices = Enumerable.Range(0, probs.Length).OrderByDescending(i => probs[i]).ToList();
            if (topK > 0 && topK < sortedIndices.Count) sortedIndices = sortedIndices.Take(topK).ToList();

            double cumsum = 0;
            var candidates = new List<(int index, double prob)>();
            foreach (var idx in sortedIndices)
            {
                cumsum += probs[idx];
                candidates.Add((idx, probs[idx]));
                if (cumsum >= topP) break;
            }

            var candidateSum = candidates.Sum(c => c.prob);
            var rand = new Random().NextDouble() * candidateSum;
            double acc = 0;
            foreach (var (idx, prob) in candidates)
            {
                acc += prob;
                if (rand <= acc) return idx;
            }
            return candidates.Last().index;
        }

        private IAsyncEnumerable<string> GenerateStandardAsync(string prompt, CancellationToken cancellationToken)
        {
             return GenerateStandardGenAiAsync(prompt, cancellationToken);
        }
        
        private async IAsyncEnumerable<string> GenerateStandardGenAiAsync(string prompt, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var sequences = _tokenizer!.Encode(prompt);
            using var generatorParams = new GeneratorParams(_tokenizerModel!);
            generatorParams.SetSearchOption("max_length", _maxLength);
            generatorParams.SetSearchOption("temperature", _temperature);
            generatorParams.SetSearchOption("top_p", _topP);
            generatorParams.SetSearchOption("top_k", _topK);

            using var generator = new Generator(_tokenizerModel!, generatorParams);
            generator.AppendTokenSequences(sequences);

            while (!generator.IsDone())
            {
                if (cancellationToken.IsCancellationRequested) yield break;
                await Task.Run(() => generator.GenerateNextToken(), cancellationToken);
                var outputTokens = generator.GetSequence(0);
                var newToken = outputTokens[outputTokens.Length - 1];
                yield return _tokenizer.Decode(new[] { newToken });
            }
        }

        public static bool IsHybridModelDirectory(string path) => File.Exists(Path.Combine(path, "hybrid_config.json"));

        private void Cleanup()
        {
            // Dispose I/O Binding resources
            _runOptions?.Dispose();
            _gpuIoBinding?.Dispose();
            _cpuIoBinding?.Dispose();
            _runOptions = null;
            _gpuIoBinding = null;
            _cpuIoBinding = null;
            
            // Dispose Pinned Interface Values
            foreach (var ov in _pinnedInterfaceValues.Values) ov.Dispose();
            _pinnedInterfaceValues.Clear();
            
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
