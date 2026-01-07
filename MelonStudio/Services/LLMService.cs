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

        public async Task<int> InitializeAsync(string modelPath)
        {
            if (_isInitialized) return ModelContextLength;

            _modelPath = modelPath;

            int contextLength = await Task.Run(() =>
            {
                if (!Directory.Exists(modelPath))
                    throw new DirectoryNotFoundException($"Model directory not found: {modelPath}");

                var (ctxLen, modelType) = ReadModelConfig(modelPath);
                _modelType = modelType;

                try 
                {
                    _model = new Model(modelPath);
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
