using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Text.Json;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace MelonStudio.Services
{
    public enum ModelType
    {
        Unknown,
        Llama,
        Phi,
        Mistral,
        Qwen,
        Gemma
    }

    public class LLMService : IDisposable
    {
        private Model? _model;
        private Tokenizer? _tokenizer;
        private bool _isInitialized;
        private ModelType _modelType = ModelType.Unknown;
        private string _modelPath = "";

        private int _maxLength = 8192;
        private double _temperature = 0.7;
        private double _topP = 0.9;

        public bool IsInitialized => _isInitialized;
        public int ModelContextLength { get; private set; } = 0;
        public ModelType DetectedModelType => _modelType;

        public void UpdateSettings(int maxLength, double temperature, double topP)
        {
            _maxLength = maxLength;
            _temperature = temperature;
            _topP = topP;
        }

        public async Task<int> InitializeAsync(string modelPath, string? onnxFileName = null)
        {
            // If already initialized, dispose the old model first to allow hot-swapping
            if (_isInitialized)
            {
                Dispose();
            }

            _modelPath = modelPath;

            int contextLength = await Task.Run(() =>
            {
                if (!Directory.Exists(modelPath))
                    throw new DirectoryNotFoundException($"Model directory not found: {modelPath}");

                // Fix: Check if we are in the 'onnx' subfolder (common mistake)
                // If genai_config.json is missing here, check the parent
                if (!File.Exists(Path.Combine(modelPath, "genai_config.json")) && 
                    File.Exists(Path.Combine(Directory.GetParent(modelPath)?.FullName ?? "", "genai_config.json")))
                {
                    modelPath = Directory.GetParent(modelPath)!.FullName;
                    _modelPath = modelPath; // Update the class field too
                }

                var (ctxLen, modelType) = ReadModelConfig(modelPath);
                _modelType = modelType;

                try 
                {
                    // If a specific ONNX filename was provided, update genai_config.json FIRST
                    // (before Config reads the file)
                    if (!string.IsNullOrEmpty(onnxFileName))
                    {
                        var configPath = Path.Combine(modelPath, "genai_config.json");
                        if (File.Exists(configPath))
                        {
                            try
                            {
                                var json = File.ReadAllText(configPath);
                                
                                // Determine the relative path to the ONNX file
                                var relativePath = File.Exists(Path.Combine(modelPath, onnxFileName))
                                    ? onnxFileName
                                    : $"onnx/{onnxFileName}";
                                
                                // Use JsonNode for proper mutable JSON editing
                                var jsonNode = System.Text.Json.Nodes.JsonNode.Parse(json);
                                if (jsonNode != null)
                                {
                                    var modelNode = jsonNode["model"];
                                    if (modelNode != null)
                                    {
                                        var decoderNode = modelNode["decoder"];
                                        if (decoderNode != null && decoderNode is System.Text.Json.Nodes.JsonObject decoderObj)
                                        {
                                            // Add or update the filename property
                                            decoderObj["filename"] = relativePath;
                                            
                                            var options = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
                                            File.WriteAllText(configPath, jsonNode.ToJsonString(options));
                                        }
                                    }
                                }
                            }
                            catch { /* If config update fails, continue with defaults */ }
                        }
                    }
                    
                    // Now create Config (it will read the updated genai_config.json)
                    using var config = new Config(modelPath);
                    config.ClearProviders();
                    config.AppendProvider("cuda");
                    
                    _model = new Model(config);
                    _tokenizer = new Tokenizer(_model);
                    _isInitialized = true;
                    return ctxLen;
                }
                catch (Exception ex)
                {
                    throw new Exception($"Failed to load model from {modelPath}. Error: {ex.Message}", ex);
                }
            });

            ModelContextLength = contextLength;
            if (contextLength > 0)
            {
                _maxLength = contextLength;
            }
            
            return contextLength;
        }

        private (int contextLength, ModelType modelType) ReadModelConfig(string modelPath)
        {
            int contextLength = 0;
            var modelType = ModelType.Unknown;

            try
            {
                var configPath = Path.Combine(modelPath, "genai_config.json");
                if (File.Exists(configPath))
                {
                    var json = File.ReadAllText(configPath);
                    using var doc = JsonDocument.Parse(json);
                    
                    if (doc.RootElement.TryGetProperty("model", out var modelSection))
                    {
                        if (modelSection.TryGetProperty("context_length", out var ctxLen))
                        {
                            contextLength = ctxLen.GetInt32();
                        }

                        if (modelSection.TryGetProperty("type", out var typeElement))
                        {
                            var typeStr = typeElement.GetString()?.ToLowerInvariant() ?? "";
                            modelType = typeStr switch
                            {
                                "llama" => ModelType.Llama,
                                "phi" or "phi3" => ModelType.Phi,
                                "mistral" => ModelType.Mistral,
                                "qwen" or "qwen2" => ModelType.Qwen,
                                "gemma" or "gemma2" => ModelType.Gemma,
                                _ => ModelType.Unknown
                            };
                        }
                    }
                }
            }
            catch { }
            
            return (contextLength, modelType);
        }

        private string FormatPrompt(string userMessage, string systemPrompt)
        {
            return _modelType switch
            {
                ModelType.Llama => 
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
                    systemPrompt + 
                    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" +
                    userMessage + 
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",

                ModelType.Phi => 
                    "<|system|>\n" + systemPrompt + "<|end|>\n" +
                    $"<|system|>\n{systemPrompt}<|end|>\n<|user|>\n{userMessage}<|end|>\n<|assistant|>\n",

                ModelType.Mistral =>
                    $"<s>[INST] {systemPrompt}\n\n{userMessage} [/INST]",

                ModelType.Qwen =>
                    $"<|im_start|>system\n{systemPrompt}<|im_end|>\n<|im_start|>user\n{userMessage}<|im_end|>\n<|im_start|>assistant\n",

                ModelType.Gemma =>
                    $"<start_of_turn>user\n{systemPrompt}\n\n{userMessage}<end_of_turn>\n<start_of_turn>model\n",

                _ => // Default/Unknown
                     $"System: {systemPrompt}\nUser: {userMessage}\nAssistant:"
            };
        }

        public async IAsyncEnumerable<string> GenerateResponseAsync(
            string userMessage, 
            string systemPrompt, 
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (!_isInitialized || _tokenizer == null || _model == null)
            {
                yield break;
            }

            var prompt = FormatPrompt(userMessage, systemPrompt);
            var sequences = _tokenizer.Encode(prompt);

            using var generatorParams = new GeneratorParams(_model);
            generatorParams.SetSearchOption("max_length", _maxLength);
            generatorParams.SetSearchOption("temperature", _temperature);
            generatorParams.SetSearchOption("top_p", _topP);

            using var generator = new Generator(_model, generatorParams);
            generator.AppendTokenSequences(sequences);

            while (!generator.IsDone())
            {
                if (cancellationToken.IsCancellationRequested)
                    yield break;

                await Task.Run(() => generator.GenerateNextToken(), cancellationToken);

                var outputTokens = generator.GetSequence(0);
                var newToken = outputTokens[outputTokens.Length - 1];
                var tokenText = _tokenizer.Decode(new[] { newToken });

                yield return tokenText;
            }
        }

        public void Dispose()
        {
            _model?.Dispose();
            _tokenizer?.Dispose();
            _model = null;
            _tokenizer = null;
            _isInitialized = false;
        }
    }
}
