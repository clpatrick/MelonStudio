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
using System.Runtime.Intrinsics;
using System.Numerics.Tensors; 
using System.Runtime.InteropServices;

namespace MelonStudio.Services
{
    public class HybridLLMService : IDisposable
    {
        private InferenceSession? _gpuSession;
        private InferenceSession? _cpuSession;
        
        // I/O Binding infrastructure for reduced allocation overhead
        private OrtIoBinding? _gpuIoBindingA;
        private OrtIoBinding? _gpuIoBindingB;
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
        public string Summary => _hybridConfig == null 
            ? "No model loaded" 
            : $"{_hybridConfig.Summary} | Speculation: {SpeculativeHits} Hits / {SpeculativeMisses} Misses ({(SpeculativeHits + SpeculativeMisses > 0 ? (100.0 * SpeculativeHits / (SpeculativeHits + SpeculativeMisses)).ToString("F0") : "0")}%)";

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
                    gpuOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
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
                    cpuOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
                    cpuOptions.AppendExecutionProvider_CPU(0);
                    cpuOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

                    // Quantization Loading Priority: INT8 (Best Logic) > INT4-HQ > INT4 (Best Speed) > FP16
                    var int8CpuPath = cpuPath.Replace(".onnx", ".int8.onnx");
                    var int4HqPath = cpuPath.Replace(".onnx", ".int4_hq.onnx");
                    var int4CpuPath = cpuPath.Replace(".onnx", ".int4.onnx");

                    if (File.Exists(int8CpuPath))
                    {
                         OnStatusChanged?.Invoke($"[Perf] Loading Quantized INT8 CPU Model: {Path.GetFileName(int8CpuPath)}");
                         cpuPath = int8CpuPath;
                    }
                    else if (File.Exists(int4HqPath))
                    {
                         OnStatusChanged?.Invoke($"[Perf] Loading High-Quality INT4 CPU Model: {Path.GetFileName(int4HqPath)}");
                         cpuPath = int4HqPath;
                    }
                    else if (File.Exists(int4CpuPath))
                    {
                         OnStatusChanged?.Invoke($"[Perf] Loading Quantized INT4 CPU Model: {Path.GetFileName(int4CpuPath)}");
                         cpuPath = int4CpuPath;
                    }

                    _cpuSession = new InferenceSession(cpuPath, cpuOptions);
                });
                OnProgressChanged?.Invoke(0.7);

                OnStatusChanged?.Invoke("Loading tokenizer...");
                await LoadTokenizerAsync(partitionDir);
                OnProgressChanged?.Invoke(0.85);

                // Initialize I/O Bindings for reduced per-step allocation
                OnStatusChanged?.Invoke("Initializing I/O bindings...");
                _gpuIoBindingA = _gpuSession!.CreateIoBinding();
                _gpuIoBindingB = _gpuSession!.CreateIoBinding();
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

            // Speculative Pipeline State
            Task<(Dictionary<string, OrtValue>, Dictionary<string, OrtValue>, List<OrtValue>)>? speculativeGpuTask = null;
            long speculativeToken = -1;
            DenseTensor<long>? specInputIdsTensor = null;

            try
            {
                // Reset Diagnostics
                SpeculativeHits = 0;
                SpeculativeMisses = 0;
                
                // Initialize N-gram State
                // Note: We DO NOT clear caches here to allow learning across turns!
                // _bigramCache.Clear(); 
                // _trigramCache.Clear();
                
                InitializeCommonTokenIds();
                Console.WriteLine($"[Speculation] Predictor State: {_trigramCache.Count} trigrams, {_bigramCache.Count} bigrams known.");

                int tokensGenerated = 0;
                
                while (tokensGenerated < _maxLength && !cancellationToken.IsCancellationRequested)
                {
                    int batchSize = 1;
                    int inputLen = currentInputIds.Length;
                    int totalSequenceLength = pastSequenceLength + inputLen;

                    // Attention Mask
                    var maskData = Enumerable.Repeat(1L, totalSequenceLength).ToArray();
                    var attentionMaskTensor = new DenseTensor<long>(maskData, new[] { batchSize, totalSequenceLength });
                    
                    // Position IDs
                    var positionIdsData = Enumerable.Range(pastSequenceLength, inputLen).Select(i => (long)i).ToArray();
                    var positionIdsTensor = new DenseTensor<long>(positionIdsData, new[] { batchSize, inputLen });
                    
                    var inputIdsTensor = new DenseTensor<long>(currentInputIds, new[] { batchSize, inputLen });

                    // 1. Resolve GPU Step
                    Dictionary<string, OrtValue> interfaceTensors;
                    Dictionary<string, OrtValue> gpuKv;
                    List<OrtValue> tempGpuValues;

                    if (speculativeGpuTask != null)
                    {
                         // Must await
                         var result = await speculativeGpuTask;
                         
                         bool match = (currentInputIds.Length == 1 && currentInputIds[0] == speculativeToken);
                         if (match)
                         {
                             SpeculativeHits++;
                             (interfaceTensors, gpuKv, tempGpuValues) = result;
                         }
                         else
                         {
                             SpeculativeMisses++;
                             // Cleanup Speculative Results
                             foreach(var ov in result.Item3) ov.Dispose(); // inputs
                             // Interface tensors from speculative run
                             bool wasPinned = (specInputIdsTensor?.Dimensions[1] == 1);
                             if (!wasPinned) foreach(var kvp in result.Item1) kvp.Value.Dispose();
                             // Speculative KV outputs
                             foreach(var kvp in result.Item2) kvp.Value.Dispose();

                             // Re-run GPU Real
                             (interfaceTensors, gpuKv, tempGpuValues) = await RunGpuStepAsync(
                                 inputIdsTensor, attentionMaskTensor, positionIdsTensor, kvCache, (tokensGenerated % 2 == 0 ? _gpuIoBindingA! : _gpuIoBindingB!));
                         }
                         speculativeGpuTask = null;
                    }
                    else
                    {
                        (interfaceTensors, gpuKv, tempGpuValues) = await RunGpuStepAsync(
                            inputIdsTensor, attentionMaskTensor, positionIdsTensor, kvCache, (tokensGenerated % 2 == 0 ? _gpuIoBindingA! : _gpuIoBindingB!));
                    }

                    // 2. Commit GPU KV
                    foreach (var kvp in gpuKv)
                    {
                        if (kvCache.TryGetValue(kvp.Key, out var oldValue)) oldValue.Dispose();
                        kvCache[kvp.Key] = kvp.Value;
                    }

                    // 3. Start CPU
                    var cpuTask = RunCpuStepAsync(
                        attentionMaskTensor, positionIdsTensor, interfaceTensors, kvCache, inputLen);

                    // 4. Speculate (Overlap)
                    if (inputLen == 1 && tokensGenerated < _maxLength - 1)
                    {
                        long currentToken = currentInputIds[0];
                        
                        // Update N-gram Predictor
                        UpdateNgramCaches(currentToken);

                        // Predict Next Token
                        // previous token for trigram is actually _prevToken (shifted in UpdateNgramCaches)
                        // wait! UpdateNgramCaches shifts the window.
                        // So after UpdateNgramCaches(current), _lastToken is current, and _prevToken is previous.
                        // Ideally we want to predict NEXT based on (prev, current).
                        // Since UpdateNgramCaches sets _lastToken = current, we ask Predict(_prevToken, _lastToken).
                        
                        speculativeToken = Predict(_prevToken, _lastToken);
                        
                        // Setup Inputs
                        var specInputIds = new long[] { speculativeToken };
                        specInputIdsTensor = new DenseTensor<long>(specInputIds, new[] { 1, 1 });
                        
                        var specMaskData = Enumerable.Repeat(1L, totalSequenceLength + 1).ToArray();
                        var specMaskTensor = new DenseTensor<long>(specMaskData, new[] { 1, totalSequenceLength + 1 });
                        
                        var specPosData = new long[] { totalSequenceLength };
                        var specPosTensor = new DenseTensor<long>(specPosData, new[] { 1, 1 });
                        
                        speculativeGpuTask = RunGpuStepAsync(
                            specInputIdsTensor, specMaskTensor, specPosTensor, kvCache, (tokensGenerated % 2 == 0 ? _gpuIoBindingB! : _gpuIoBindingA!));
                    }
                    else
                    {
                         // Seed N-gram from Prompt
                         for (int i = 0; i < currentInputIds.Length; i++)
                         {
                             UpdateNgramCaches(currentInputIds[i]);
                         }
                    }

                    // 5. Wait CPU
                    var (nextTokenId, cpuKv, tempCpuValues) = await cpuTask;

                    // 6. Cleanup Steps
                    foreach(var d in tempGpuValues) d.Dispose();
                    foreach(var d in tempCpuValues) d.Dispose();
                    bool isPinned = (inputLen == 1);
                    if (!isPinned) foreach(var kvp in interfaceTensors) kvp.Value.Dispose();

                    // 7. Update CPU KV
                    foreach (var kvp in cpuKv)
                    {
                         if (kvCache.TryGetValue(kvp.Key, out var oldValue)) oldValue.Dispose();
                         kvCache[kvp.Key] = kvp.Value;
                    }

                    if (nextTokenId == null) break;

                    long nextToken = nextTokenId.Value;
                    var tokenText = _tokenizer.Decode(new[] { (int)nextToken });
                    yield return tokenText;


                    currentInputIds = new[] { nextToken };
                    pastSequenceLength = totalSequenceLength; 
                    tokensGenerated++;
                }
            }
            finally
            {
                // Cleanup Speculative Task if orphaned
                if (speculativeGpuTask != null)
                {
                    try 
                    { 
                        // We must wait for it to finish to safely dispose ORT values
                        var result = await speculativeGpuTask;
                        foreach(var d in result.Item3) d.Dispose();
                        // Interface Tensors
                        if (specInputIdsTensor != null)
                        {
                            bool wasPinned = (specInputIdsTensor.Dimensions[1] == 1);
                            if (!wasPinned) foreach(var kvp in result.Item1) kvp.Value.Dispose();
                        }
                        else 
                        {
                            // Fallback safe disposal
                            foreach(var kvp in result.Item1) kvp.Value.Dispose();
                        }

                        foreach(var kvp in result.Item2) kvp.Value.Dispose();
                    } 
                    catch { /* Verify exception or ignore */ }
                }

                foreach (var val in kvCache.Values) val.Dispose();
            }
        }

        // Cache for Pinned Interface Tensors (used for decode step len=1)
        private Dictionary<string, OrtValue> _pinnedInterfaceValues = new();
        private int _hiddenSize = 3072; // Default for Phi-3.5-mini

        // Speculation State - N-gram Caches
        private Dictionary<long, long> _bigramCache = new();
        private Dictionary<(long, long), long> _trigramCache = new();
        private long _prevToken = -1;
        private long _lastToken = -1;
        
        // Common Token IDs (Phi-3.5 tokenizer)
        private int _spaceTokenId = -1;
        private int _newlineTokenId = -1;
        private bool _tokenIdsInitialized = false;
        
        // Sampling RNG (reused to avoid repeated seeding)
        private readonly Random _samplingRng = new();
        
        // Diagnostics
        public int SpeculativeHits { get; private set; }
        public int SpeculativeMisses { get; private set; }

        /// <summary>
        /// Hybrid N-gram + Token Class predictor for speculation.
        /// </summary>
        private long Predict(long prevToken, long currentToken)
        {
            // 1. Try 3-gram (highest precision)
            if (prevToken >= 0 && _trigramCache.TryGetValue((prevToken, currentToken), out var triPred))
                return triPred;
            
            // 2. Try 2-gram
            if (_bigramCache.TryGetValue(currentToken, out var biPred))
                return biPred;
            
            // 3. Token class heuristics
            if (_tokenIdsInitialized)
            {
                try
                {
                    var text = _tokenizer!.Decode(new[] { (int)currentToken });
                    var trimmed = text.TrimEnd();
                    
                    // After sentence-ending punctuation, predict space
                    if (trimmed.EndsWith(".") || trimmed.EndsWith("!") || trimmed.EndsWith("?"))
                        return _spaceTokenId;
                    
                    // After comma/semicolon, predict space
                    if (trimmed.EndsWith(",") || trimmed.EndsWith(";"))
                        return _spaceTokenId;
                    
                    // After colon, predict newline or space
                    if (trimmed.EndsWith(":"))
                        return _newlineTokenId > 0 ? _newlineTokenId : _spaceTokenId;
                    
                    // After newline token, could be another newline or start of word
                    if (text.Contains('\n'))
                        return _spaceTokenId;
                }
                catch { /* Fallback on decode errors */ }
            }
            
            // 4. Fallback: repeat current token
            return currentToken;
        }

        /// <summary>
        /// Initialize common token IDs for heuristic prediction.
        /// </summary>
        private void InitializeCommonTokenIds()
        {
            if (_tokenizer == null || _tokenIdsInitialized) return;
            
            try
            {
                // Find space token
                var spaceSeq = _tokenizer.Encode(" ");
                if (spaceSeq != null && spaceSeq.NumSequences > 0)
                {
                    var ids = spaceSeq[0];
                    if (ids.Length > 0) _spaceTokenId = (int)ids[0];
                }
                
                // Find newline token
                var nlSeq = _tokenizer.Encode("\n");
                if (nlSeq != null && nlSeq.NumSequences > 0)
                {
                    var ids = nlSeq[0];
                    if (ids.Length > 0) _newlineTokenId = (int)ids[0];
                }
                
                _tokenIdsInitialized = true;
                OnDiagnostic?.Invoke($"Predictor initialized: space={_spaceTokenId}, newline={_newlineTokenId}");
            }
            catch (Exception ex)
            {
                OnDiagnostic?.Invoke($"Failed to initialize common token IDs: {ex.Message}");
            }
        }

        /// <summary>
        /// Update N-gram caches with observed token sequence.
        /// </summary>
        private void UpdateNgramCaches(long newToken)
        {
            // Update bigram: lastToken -> newToken
            if (_lastToken >= 0)
            {
                _bigramCache[_lastToken] = newToken;
            }
            
            // Update trigram: (prevToken, lastToken) -> newToken
            if (_prevToken >= 0 && _lastToken >= 0)
            {
                _trigramCache[(_prevToken, _lastToken)] = newToken;
            }
            
            // Shift window
            _prevToken = _lastToken;
            _lastToken = newToken;
        }



        private long SampleToken(float[] logits, float temperature, float topP, int topK)
        {
            // Use TensorPrimitives for SIMD acceleration (AVX-512/AVX2)
            
            // 1. Temperature Scaling
            if (temperature > 0.0f)
            {
               // In-place division: logits = logits / temp
               TensorPrimitives.Divide(logits, temperature, logits);
            }
            else
            {
               // Greedy: ArgMax
               return TensorPrimitives.IndexOfMax(logits);
            }

            // 2. Softmax (TensorPrimitives doesn't have Softmax directly in .NET 8? 
            // It was added in .NET 9 Preview. Check .NET 8 api availability.
            // If not, we do Max/Exp/Sum manually with TensorPrimitives).
            // 2. Softmax 
            float maxLogit = TensorPrimitives.Max(logits);
            
            // exp(x - max)
            TensorPrimitives.Add(logits, -maxLogit, logits);
            TensorPrimitives.Exp(logits, logits);
            
            // Sum
            float sumExp = TensorPrimitives.Sum(logits);
            
            // Normalize: probs = exp / sum
            TensorPrimitives.Divide(logits, sumExp, logits);
            
            // logits now holds probabilities
            var probs = logits;

            // 3. TopK / TopP (Sampling - scalar logic remains)
            // Find indices
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
            var rand = _samplingRng.NextDouble() * candidateSum;
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

        private async Task<(Dictionary<string, OrtValue> InterfaceTensors, Dictionary<string, OrtValue> NewKv, List<OrtValue> TempValues)> RunGpuStepAsync(
            DenseTensor<long> inputIds,
            DenseTensor<long> attentionMask,
            DenseTensor<long> positionIds,
            Dictionary<string, OrtValue> currentKvCache,
            OrtIoBinding ioBinding)
        {
            var newKvCache = new Dictionary<string, OrtValue>();
            var tempOrtValues = new List<OrtValue>();
            int inputLen = (int)inputIds.Dimensions[1];

            ioBinding.ClearBoundInputs();
            ioBinding.ClearBoundOutputs();

            // Bind Inputs
            var inputIdsOrt = OrtValue.CreateTensorValueFromMemory(inputIds.ToArray(), new long[] { 1, inputLen });
            tempOrtValues.Add(inputIdsOrt);
            ioBinding.BindInput("input_ids", inputIdsOrt);

            var maskOrt = OrtValue.CreateTensorValueFromMemory(attentionMask.ToArray(), new long[] { 1, attentionMask.Dimensions[1] });
            tempOrtValues.Add(maskOrt);
            ioBinding.BindInput("attention_mask", maskOrt);

            if (_gpuSession!.InputMetadata.ContainsKey("position_ids"))
            {
                var posOrt = OrtValue.CreateTensorValueFromMemory(positionIds.ToArray(), new long[] { 1, positionIds.Dimensions[1] });
                tempOrtValues.Add(posOrt);
                ioBinding.BindInput("position_ids", posOrt);
            }

            // Bind KV Cache Inputs
            foreach (var inputName in _hybridConfig!.GpuPartition.Inputs)
            {
                if (inputName.StartsWith("past_key_values"))
                {
                    if (currentKvCache.TryGetValue(inputName, out var existingKv))
                    {
                        ioBinding.BindInput(inputName, existingKv);
                    }
                    else
                    {
                        var emptyData = Array.Empty<Float16>();
                        var emptyKv = OrtValue.CreateTensorValueFromMemory(emptyData, new long[] { 1, 32, 0, 96 });
                        tempOrtValues.Add(emptyKv);
                        ioBinding.BindInput(inputName, emptyKv);
                    }
                }
            }

            // Bind Outputs (Strict Order to match Processing Loop)
            foreach (var outputName in _hybridConfig.GpuPartition.Outputs)
            {
                if (_hybridConfig.InterfaceTensors.ContainsValue(outputName) && inputLen == 1)
                {
                    // Fast Path: Bind Pinned Interface Tensor
                    if (!_pinnedInterfaceValues.ContainsKey(outputName))
                    {
                         var pinnedData = new Float16[_hiddenSize];
                         var pinnedOrt = OrtValue.CreateTensorValueFromMemory(pinnedData, new long[] { 1, 1, _hiddenSize });
                         _pinnedInterfaceValues[outputName] = pinnedOrt;
                    }
                    ioBinding.BindOutput(outputName, _pinnedInterfaceValues[outputName]);
                }
                else
                {
                    // Slow Path or Standard Output: Bind to Device
                    ioBinding.BindOutputToDevice(outputName, OrtMemoryInfo.DefaultInstance); 
                }
            }

            ioBinding.SynchronizeBoundInputs();

            // Execute
            await Task.Run(() => { _gpuSession.RunWithBinding(_runOptions!, ioBinding); });
            
            var gpuResults = ioBinding.GetOutputValues();
            ioBinding.SynchronizeBoundOutputs();

            // Process Outputs
            var capturedInterfaceTensors = new Dictionary<string, OrtValue>();
            var gpuOutputList = gpuResults.ToList();

            for (int i = 0; i < gpuOutputList.Count; i++)
            {
                var outputName = _hybridConfig.GpuPartition.Outputs[i];
                var ortValue = gpuOutputList[i];

                if (outputName.StartsWith("present"))
                {
                    var pastName = outputName.Replace("present.", "past_key_values.");
                    newKvCache[pastName] = ortValue;
                }
                else if (_hybridConfig.InterfaceTensors.ContainsValue(outputName))
                {
                    capturedInterfaceTensors[outputName] = ortValue;
                }
                else
                {
                    ortValue.Dispose();
                }
            }

            return (capturedInterfaceTensors, newKvCache, tempOrtValues);
        }

        private async Task<(long? NextToken, Dictionary<string, OrtValue> NewKv, List<OrtValue> TempValues)> RunCpuStepAsync(
            DenseTensor<long> attentionMask,
            DenseTensor<long> positionIds,
            Dictionary<string, OrtValue> interfaceTensors,
            Dictionary<string, OrtValue> currentKvCache,
            int inputLenForDecoder)
        {
            var newKvCache = new Dictionary<string, OrtValue>();
            var tempOrtValues = new List<OrtValue>();

            _cpuIoBinding!.ClearBoundInputs();
            _cpuIoBinding.ClearBoundOutputs();

            // Bind Inputs
            var cpuMaskOrt = OrtValue.CreateTensorValueFromMemory(attentionMask.ToArray(), new long[] { 1, attentionMask.Dimensions[1] });
            tempOrtValues.Add(cpuMaskOrt);
            _cpuIoBinding.BindInput("attention_mask", cpuMaskOrt);

            if (_cpuSession!.InputMetadata.ContainsKey("position_ids"))
            {
                var cpuPosOrt = OrtValue.CreateTensorValueFromMemory(positionIds.ToArray(), new long[] { 1, positionIds.Dimensions[1] });
                tempOrtValues.Add(cpuPosOrt);
                _cpuIoBinding.BindInput("position_ids", cpuPosOrt);
            }

            // Bind Interface Tensors
            foreach(var kvp in interfaceTensors)
            {
                _cpuIoBinding.BindInput(kvp.Key, kvp.Value);
            }

            // Bind KV Cache
            // Note: CPU KV cache input names are different from GPU
            foreach (var inputName in _hybridConfig!.CpuPartition.Inputs)
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

            // Bind Outputs
            foreach (var outputName in _hybridConfig.CpuPartition.Outputs)
            {
                _cpuIoBinding.BindOutputToDevice(outputName, OrtMemoryInfo.DefaultInstance);
            }

            _cpuIoBinding.SynchronizeBoundInputs();

            // Execute
            IDisposableReadOnlyCollection<OrtValue> cpuResults;
            await Task.Run(() => { _cpuSession.RunWithBinding(_runOptions!, _cpuIoBinding); });
            cpuResults = _cpuIoBinding.GetOutputValues();
            _cpuIoBinding.SynchronizeBoundOutputs();

            // Process Results
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
                    var typeInfo = ortValue.GetTensorTypeAndShape();
                    var vocabSize = (int)typeInfo.Shape[2];
                    
                    // We need input length to find last token index?
                    var lastTokenIndex = inputLenForDecoder - 1;

                    var logitsSpan = ortValue.GetTensorDataAsSpan<Float16>();
                    var lastLogits = new float[vocabSize];
                    var offset = lastTokenIndex * vocabSize;
                    
                    // SIMD Optimization for Float16 -> Float32
                    // ORT Float16 is binary compatible with System.Half
                    var sourceHalf = MemoryMarshal.Cast<Float16, Half>(logitsSpan.Slice(offset, vocabSize));
                    
                    // Safe Fallback / JIT Vectorizable Loop
                    for (int j = 0; j < vocabSize; j++) lastLogits[j] = (float)sourceHalf[j];

                    nextTokenId = SampleToken(lastLogits, _temperature, _topP, _topK);
                    ortValue.Dispose();
                }
                else
                {
                    ortValue.Dispose();
                }
            }

            return (nextTokenId, newKvCache, tempOrtValues);
        }

        private void Cleanup()
        {
            // Dispose I/O Binding resources
            _runOptions?.Dispose();
            _gpuIoBindingA?.Dispose();
            _gpuIoBindingB?.Dispose();
            _cpuIoBinding?.Dispose();
            _runOptions = null;
            _gpuIoBindingA = null;
            _gpuIoBindingB = null;
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
            
            // Reset N-gram predictor state to avoid stale predictions
            _prevToken = -1;
            _lastToken = -1;
            _bigramCache.Clear();
            _trigramCache.Clear();
            _tokenIdsInitialized = false;
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
