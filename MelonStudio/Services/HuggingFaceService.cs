using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace MelonStudio.Services
{
    public class HuggingFaceModel
    {
        [JsonPropertyName("id")]
        public string Id { get; set; } = "";

        [JsonPropertyName("author")]
        public string? Author { get; set; }

        [JsonPropertyName("downloads")]
        public int Downloads { get; set; }

        [JsonPropertyName("likes")]
        public int Likes { get; set; }

        [JsonPropertyName("pipeline_tag")]
        public string? PipelineTag { get; set; }

        [JsonPropertyName("tags")]
        public List<string>? Tags { get; set; }

        [JsonPropertyName("lastModified")]
        public DateTime? LastModified { get; set; }

        public bool IsOnnxCompatible => Tags?.Contains("onnx") == true || 
                                        Tags?.Contains("onnxruntime") == true;

        public string DisplayName => Id.Contains("/") ? Id.Split('/')[1] : Id;
    }

    public class HuggingFaceService
    {
        private readonly HttpClient _httpClient;
        private const string BaseUrl = "https://huggingface.co/api";

        // Model architectures supported by ONNX Runtime GenAI
        public static readonly string[] SupportedArchitectures = new[]
        {
            "phi", "llama", "mistral", "qwen", "gemma", "deepseek",
            "granite", "nemotron", "smollm", "chatglm", "olmo"
        };

        public HuggingFaceService()
        {
            _httpClient = new HttpClient();
            _httpClient.DefaultRequestHeaders.Add("User-Agent", "MelonStudio/1.0");
        }

        public async Task<List<HuggingFaceModel>> SearchModelsAsync(
            string query = "",
            string? author = null,
            bool textGenerationOnly = true,
            int limit = 20)
        {
            var url = $"{BaseUrl}/models?";
            
            if (!string.IsNullOrWhiteSpace(query))
            {
                url += $"search={Uri.EscapeDataString(query)}&";
            }

            if (!string.IsNullOrWhiteSpace(author))
            {
                url += $"author={Uri.EscapeDataString(author)}&";
            }

            if (textGenerationOnly)
            {
                url += "pipeline_tag=text-generation&";
            }

            url += $"limit={limit}&sort=downloads&direction=-1";

            try
            {
                var response = await _httpClient.GetAsync(url);
                response.EnsureSuccessStatusCode();

                var models = await response.Content.ReadFromJsonAsync<List<HuggingFaceModel>>();
                return models ?? new List<HuggingFaceModel>();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"HuggingFace API error: {ex.Message}");
                return new List<HuggingFaceModel>();
            }
        }

        public async Task<List<HuggingFaceModel>> GetRecommendedModelsAsync()
        {
            // Get popular models from authors known for ONNX-compatible models
            var results = new List<HuggingFaceModel>();

            // Microsoft models (Phi series)
            var msModels = await SearchModelsAsync("phi onnx", "microsoft", limit: 10);
            results.AddRange(msModels);

            // Meta models (Llama)
            var metaModels = await SearchModelsAsync("llama", "meta-llama", limit: 5);
            results.AddRange(metaModels);

            // Mistral
            var mistralModels = await SearchModelsAsync("", "mistralai", limit: 5);
            results.AddRange(mistralModels);

            // Qwen
            var qwenModels = await SearchModelsAsync("", "Qwen", limit: 5);
            results.AddRange(qwenModels);

            return results;
        }

        public async Task<List<HuggingFaceModel>> SearchOnnxReadyModelsAsync(string query = "")
        {
            // Search specifically for ONNX models
            var searchQuery = string.IsNullOrWhiteSpace(query) ? "onnx genai" : $"{query} onnx";
            return await SearchModelsAsync(searchQuery, limit: 30);
        }

        public bool IsModelArchitectureSupported(HuggingFaceModel model)
        {
            var modelIdLower = model.Id.ToLower();
            foreach (var arch in SupportedArchitectures)
            {
                if (modelIdLower.Contains(arch))
                    return true;
            }

            if (model.Tags != null)
            {
                foreach (var tag in model.Tags)
                {
                    var tagLower = tag.ToLower();
                    foreach (var arch in SupportedArchitectures)
                    {
                        if (tagLower.Contains(arch))
                            return true;
                    }
                }
            }

            return false;
        }
    }
}
