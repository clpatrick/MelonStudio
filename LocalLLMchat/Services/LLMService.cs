using System;
using System.IO;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace LocalLLMchat.Services
{
    public class LLMService : IDisposable
    {
        private Model _model;
        private Tokenizer _tokenizer;
        private bool _isInitialized;

        public bool IsInitialized => _isInitialized;

        public async Task InitializeAsync(string modelPath)
        {
            if (_isInitialized) return;

            await Task.Run(() =>
            {
                if (!Directory.Exists(modelPath))
                    throw new DirectoryNotFoundException($"Model directory not found: {modelPath}");

                // Initialize the model
                // The library automatically detects config.json in the folder
                // For TensorRT, we ensure the native libs are present.
                // If using the .Cuda package, it usually tries CUDA/TensorRT first.
                try 
                {
                    _model = new Model(modelPath);
                    _tokenizer = new Tokenizer(_model);
                    _isInitialized = true;
                }
                catch (Exception ex)
                {
                    // Fallback or specific error handling
                    throw new Exception($"Failed to load model from {modelPath}. Ensure ONNX files are present.", ex);
                }
            });
        }

        public async IAsyncEnumerable<string> GenerateResponseAsync(string prompt, string systemPrompt = "You are a helpful AI assistant.")
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model not initialized.");

            // Construct the full prompt based on typical chat templates (e.g. Phi-3)
            // Ideally this should use the model's chat template, but for now we approximate.
            // Phi-3 format: <|user|>\n{prompt}<|end|>\n<|assistant|>
            string fullPrompt = $"<|user|>\n{systemPrompt}\n\n{prompt}<|end|>\n<|assistant|>";

            var sequences = _tokenizer.Encode(fullPrompt);

            using var generatorParams = new GeneratorParams(_model);
            generatorParams.SetSearchOption("max_length", 2048);
            generatorParams.SetInputSequences(sequences);
            // generatorParams.SetSearchOption("past_present_share_buffer", false); // Optional tuning

            using var generator = new Generator(_model, generatorParams);

            while (!generator.IsDone())
            {
                // Compute logits and generate next token
                // We wrap this in Task.Run to avoid blocking the UI thread during heavy compute
                await Task.Run(() => 
                {
                    generator.ComputeLogits();
                    generator.GenerateNextToken();
                });

                // Decode the new token
                // Note: GetSequence(0) returns the entire sequence so far or just the new token?
                // The C# API usually works by decoding the *new* tokens.
                // Let's rely on standard loop pattern for C# GenAI.
                
                // Optimized approach: keep track of previous token count or use specific API if available.
                // Standard example uses tokenizer.Decode on the last generated token.
                
                // Get the last generated token id
                // There isn't a direct "GetLastToken" simple API in all versions, 
                // but usually we can get the sequence.
                 var outputCallback = generator.GetSequence(0);
                 var newTokenId = outputCallback[outputCallback.Length - 1];
                 
                 var decodedToken = _tokenizer.Decode(new[] { newTokenId });
                 
                 yield return decodedToken;
            }
        }

        public void Dispose()
        {
            _tokenizer?.Dispose();
            _model?.Dispose();
            _isInitialized = false;
        }
    }
}
