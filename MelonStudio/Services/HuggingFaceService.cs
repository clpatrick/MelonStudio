using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using System.Linq;

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

        [JsonPropertyName("modelId")]
        public string? ModelId { get; set; }

        public bool IsOnnxCompatible => Tags?.Any(t => 
            t.Contains("onnx", StringComparison.OrdinalIgnoreCase)) == true;

        public bool IsCudaCompatible => Tags?.Any(t => 
            t.Contains("cuda", StringComparison.OrdinalIgnoreCase) ||
            t.Contains("tensorrt", StringComparison.OrdinalIgnoreCase)) == true;

        public bool IsInt4 => Tags?.Any(t => 
            t.Contains("int4", StringComparison.OrdinalIgnoreCase) ||
            t.Contains("awq", StringComparison.OrdinalIgnoreCase) ||
            t.Contains("gptq", StringComparison.OrdinalIgnoreCase)) == true;

        public bool IsFp16 => Tags?.Any(t => 
            t.Contains("fp16", StringComparison.OrdinalIgnoreCase) ||
            t.Contains("float16", StringComparison.OrdinalIgnoreCase)) == true;

        public string DisplayName => Id.Contains("/") ? Id.Split('/')[1] : Id;
        
        public string DownloadsFormatted => FormatNumber(Downloads);
        public string LikesFormatted => FormatNumber(Likes);
        
        public string LastModifiedFormatted => LastModified.HasValue 
            ? GetTimeAgo(LastModified.Value) 
            : "";

        // Compatibility detection
        public string CompatibilityStatus
        {
            get
            {
                var (status, _, _) = GetCompatibilityInfo();
                return status;
            }
        }

        public string CompatibilityIcon
        {
            get
            {
                var (status, _, _) = GetCompatibilityInfo();
                return status switch
                {
                    "Compatible" => "✓",
                    "Warning" => "⚠",
                    "Incompatible" => "✗",
                    _ => "?"
                };
            }
        }

        public string CompatibilityTooltip
        {
            get
            {
                var (_, tooltip, _) = GetCompatibilityInfo();
                return tooltip;
            }
        }

        public string CompatibilityColor
        {
            get
            {
                var (status, _, _) = GetCompatibilityInfo();
                return status switch
                {
                    "Compatible" => "#28A745",
                    "Warning" => "#FFC107",
                    "Incompatible" => "#DC3545",
                    _ => "#808080"
                };
            }
        }

        private (string status, string tooltip, string arch) GetCompatibilityInfo()
        {
            var lowerId = Id.ToLowerInvariant();
            var lowerTags = Tags?.Select(t => t.ToLowerInvariant()).ToList() ?? new List<string>();

            // Supported architectures
            string[] supportedArchs = { "phi", "llama", "mistral", "qwen", "gemma", "deepseek", 
                                        "granite", "nemotron", "smollm", "chatglm", "olmo", "gpt-oss", "ernie" };
            var detectedArch = supportedArchs.FirstOrDefault(a => lowerId.Contains(a)) ?? "";

            // Check for incompatible formats
            if (lowerId.Contains("mlx-community") || lowerTags.Contains("mlx"))
                return ("Incompatible", "MLX format (Apple Silicon only) - not compatible with ONNX Runtime", "");

            if (lowerTags.Contains("exl2") || lowerId.Contains("-exl2"))
                return ("Incompatible", "EXL2 format (ExLlamaV2) - not compatible with ONNX Runtime", "");

            // GGUF handling - quantized GGUF not supported
            if (lowerTags.Contains("gguf") || lowerId.Contains("-gguf"))
            {
                if (lowerId.Contains("f16") || lowerId.Contains("fp16") || lowerId.Contains("f32") || lowerId.Contains("fp32"))
                    return ("Warning", $"GGUF float16/32 format - can be converted but may have issues. Arch: {detectedArch}", detectedArch);
                else
                    return ("Incompatible", "Quantized GGUF format - only float16/32 GGUF is supported for conversion", "");
            }

            // GPTQ/AWQ - supported via AutoGPTQ/AutoAWQ but with caveats
            if (lowerTags.Contains("gptq") || lowerId.Contains("-gptq"))
                return ("Warning", $"GPTQ format - supported via AutoGPTQ if in HuggingFace format. Arch: {detectedArch}", detectedArch);

            if (lowerTags.Contains("awq") || lowerId.Contains("-awq"))
                return ("Warning", $"AWQ format - supported via AutoAWQ if in HuggingFace format. Arch: {detectedArch}", detectedArch);

            // Already ONNX format
            if (lowerTags.Contains("onnx") || lowerId.Contains("-onnx") || lowerTags.Contains("onnxruntime"))
                return ("Compatible", $"ONNX format - ready to download and use directly! Arch: {detectedArch}", detectedArch);

            // Standard PyTorch/SafeTensors
            if (string.IsNullOrEmpty(detectedArch))
                return ("Warning", "Unknown architecture - may not be supported by ONNX Runtime GenAI builder", "");

            return ("Compatible", $"PyTorch/SafeTensors format - compatible for conversion. Arch: {detectedArch}", detectedArch);
        }

        private static string FormatNumber(int num)
        {
            if (num >= 1000000) return $"{num / 1000000.0:0.#}M";
            if (num >= 1000) return $"{num / 1000.0:0.#}K";
            return num.ToString();
        }

        private static string GetTimeAgo(DateTime date)
        {
            var span = DateTime.UtcNow - date;
            if (span.TotalDays >= 365) return $"{(int)(span.TotalDays / 365)}y ago";
            if (span.TotalDays >= 30) return $"{(int)(span.TotalDays / 30)}mo ago";
            if (span.TotalDays >= 1) return $"{(int)span.TotalDays}d ago";
            return "today";
        }
    }

    public class HuggingFaceModelDetails
    {
        [JsonPropertyName("id")]
        public string Id { get; set; } = "";

        [JsonPropertyName("author")]
        public string? Author { get; set; }

        [JsonPropertyName("downloads")]
        public int Downloads { get; set; }

        [JsonPropertyName("likes")]
        public int Likes { get; set; }

        [JsonPropertyName("tags")]
        public List<string>? Tags { get; set; }

        [JsonPropertyName("lastModified")]
        public DateTime? LastModified { get; set; }

        [JsonPropertyName("cardData")]
        public JsonElement? CardData { get; set; }

        [JsonPropertyName("siblings")]
        public List<HuggingFaceFile>? Siblings { get; set; }

        public string? Description { get; set; }
        
        public string DisplayName => Id.Contains("/") ? Id.Split('/')[1] : Id;
        
        public string HuggingFaceUrl => $"https://huggingface.co/{Id}";

        public List<string> QuantizationOptions => Siblings?
            .Where(f => f.Filename?.EndsWith(".onnx") == true || 
                        f.Filename?.Contains("int4") == true ||
                        f.Filename?.Contains("fp16") == true)
            .Select(f => f.Filename ?? "")
            .Take(10)
            .ToList() ?? new List<string>();

        public string TotalSizeFormatted
        {
            get
            {
                var totalBytes = Siblings?.Where(f => f.IsFile).Sum(f => f.Size) ?? 0;
                if (totalBytes <= 0) return "--";
                if (totalBytes >= 1073741824) return $"{totalBytes / 1073741824.0:0.##} GB";
                if (totalBytes >= 1048576) return $"{totalBytes / 1048576.0:0.#} MB";
                return $"{totalBytes / 1024.0:0.#} KB";
            }
        }

        // ONNX Detection properties
        public bool HasOnnxFiles => Siblings?.Any(f => 
            f.IsFile && f.Filename?.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) == true) == true;

        public bool HasGenAiConfig => Siblings?.Any(f => 
            f.Filename == "genai_config.json") == true;

        public bool HasPyTorchWeights => Siblings?.Any(f => 
            f.IsFile && (f.Filename?.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) == true ||
                        f.Filename?.EndsWith(".bin", StringComparison.OrdinalIgnoreCase) == true ||
                        f.Filename == "pytorch_model.bin" ||
                        f.Filename?.Contains("model.safetensors") == true)) == true;

        public bool IsOnnxModel => HasOnnxFiles || HasGenAiConfig;

        public string ModelFormat
        {
            get
            {
                if (HasGenAiConfig) return "⚡ ONNX GenAI Ready";
                if (HasOnnxFiles) return "⚡ ONNX Format";
                if (HasPyTorchWeights) return "🔧 Needs Conversion";
                return "❓ Unknown Format";
            }
        }

        public string ActionLabel => IsOnnxModel ? "Download" : "Convert to ONNX";
    }

    public class HuggingFaceFile
    {
        [JsonPropertyName("path")]
        public string? Path { get; set; }

        [JsonPropertyName("rfilename")]
        public string? RFilename { get; set; }

        public string? Filename => Path ?? RFilename;

        [JsonPropertyName("size")]
        public long Size { get; set; }

        [JsonPropertyName("type")]
        public string? Type { get; set; }

        public bool IsFile => Type == "file" || Type == null;

        public string SizeFormatted
        {
            get
            {
                if (Size >= 1073741824) return $"{Size / 1073741824.0:0.##} GB";
                if (Size >= 1048576) return $"{Size / 1048576.0:0.#} MB";
                return $"{Size / 1024.0:0.#} KB";
            }
        }
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

        // Sort options
        public static readonly string[] SortOptions = new[] 
        { 
            "downloads", "likes", "lastModified", "trending" 
        };

        public HuggingFaceService()
        {
            _httpClient = new HttpClient();
            _httpClient.DefaultRequestHeaders.Add("User-Agent", "MelonStudio/1.0");
        }

        public async Task<List<HuggingFaceModel>> SearchModelsAsync(
            string query = "",
            string sortBy = "downloads",
            bool filterOnnx = false,
            bool filterCuda = false,
            bool filterInt4 = false,
            bool filterFp16 = false,
            bool filterSourceModels = false,
            int limit = 50)
        {
            var url = $"{BaseUrl}/models?pipeline_tag=text-generation&";

            // Build search query with filters
            var searchTerms = new List<string>();
            if (!string.IsNullOrWhiteSpace(query))
                searchTerms.Add(query);

            // Source Models filter: search for safetensors (unconverted source models)
            // This is mutually exclusive with ONNX filter
            if (filterSourceModels)
            {
                searchTerms.Add("safetensors");
                // Don't add onnx search term when filtering for source models
            }
            else if (filterOnnx)
            {
                searchTerms.Add("onnx");
            }

            if (filterInt4)
                searchTerms.Add("int4");
            if (filterFp16)
                searchTerms.Add("fp16");

            if (searchTerms.Count > 0)
            {
                url += $"search={Uri.EscapeDataString(string.Join(" ", searchTerms))}&";
            }

            // Request more to allow for filtering
            url += $"limit={limit * 2}&sort={sortBy}&direction=-1";

            try
            {
                var response = await _httpClient.GetAsync(url);
                response.EnsureSuccessStatusCode();

                var models = await response.Content.ReadFromJsonAsync<List<HuggingFaceModel>>();
                if (models == null) return new List<HuggingFaceModel>();

                // Filter out incompatible models
                var filtered = models
                    .Where(m => IsCompatibleModel(m, filterSourceModels))
                    .Take(limit)
                    .ToList();

                return filtered;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"HuggingFace API error: {ex.Message}");
                return new List<HuggingFaceModel>();
            }
        }

        private bool IsCompatibleModel(HuggingFaceModel model, bool sourceModelsOnly = false)
        {
            if (string.IsNullOrEmpty(model.Id)) return false;

            // Use the model's built-in compatibility detection
            // Allow Compatible and Warning models, exclude only Incompatible
            if (model.CompatibilityStatus == "Incompatible")
                return false;

            // When filtering for source models, exclude already-converted ONNX/GenAI models
            // These have KV cache routing embedded that breaks simple graph splitting
            if (sourceModelsOnly)
            {
                var lowerId = model.Id.ToLowerInvariant();
                var lowerTags = model.Tags?.Select(t => t.ToLowerInvariant()).ToList() ?? new List<string>();

                // Exclude ONNX-converted models
                if (lowerTags.Contains("onnx") || lowerTags.Contains("onnxruntime") ||
                    lowerId.Contains("-onnx") || lowerId.Contains("_onnx") ||
                    lowerTags.Contains("genai") || lowerId.Contains("-genai"))
                {
                    return false;
                }

                // Exclude already-quantized formats that aren't source models
                if (lowerTags.Contains("gguf") || lowerId.Contains("-gguf") ||
                    lowerTags.Contains("exl2") || lowerId.Contains("-exl2") ||
                    lowerTags.Contains("mlx") || lowerId.Contains("mlx-community"))
                {
                    return false;
                }
            }

            return true;
        }

        public async Task<HuggingFaceModelDetails?> GetModelDetailsAsync(string modelId)
        {
            try
            {
                var url = $"{BaseUrl}/models/{modelId}";
                var response = await _httpClient.GetAsync(url);
                response.EnsureSuccessStatusCode();

                var details = await response.Content.ReadFromJsonAsync<HuggingFaceModelDetails>();
                
                if (details != null)
                {
                    // Fetch files with sizes from tree API
                    details.Siblings = await GetModelFilesAsync(modelId);
                    
                    // Try to get description from README
                    details.Description = await GetModelDescriptionAsync(modelId);
                }
                
                return details;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error fetching model details: {ex.Message}");
                return null;
            }
        }

        public async Task<List<HuggingFaceFile>> GetModelFilesAsync(string modelId)
        {
            try
            {
                var url = $"{BaseUrl}/models/{modelId}/tree/main";
                var response = await _httpClient.GetAsync(url);
                response.EnsureSuccessStatusCode();

                var files = await response.Content.ReadFromJsonAsync<List<HuggingFaceFile>>();
                return files ?? new List<HuggingFaceFile>();
            }
            catch
            {
                return new List<HuggingFaceFile>();
            }
        }

        private async Task<string?> GetModelDescriptionAsync(string modelId)
        {
            try
            {
                var url = $"https://huggingface.co/{modelId}/raw/main/README.md";
                var response = await _httpClient.GetAsync(url);
                
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    // Extract first paragraph after title
                    var lines = content.Split('\n');
                    var descLines = new List<string>();
                    bool foundContent = false;
                    
                    foreach (var line in lines)
                    {
                        if (string.IsNullOrWhiteSpace(line)) 
                        {
                            if (foundContent && descLines.Count > 0) break;
                            continue;
                        }
                        if (line.StartsWith("#")) 
                        {
                            foundContent = true;
                            continue;
                        }
                        if (foundContent && !line.StartsWith("[") && !line.StartsWith("!"))
                        {
                            descLines.Add(line.Trim());
                            if (descLines.Count >= 3) break;
                        }
                    }
                    
                    return string.Join(" ", descLines);
                }
            }
            catch { }
            
            return null;
        }

        public async Task<List<HuggingFaceModel>> GetRecommendedModelsAsync()
        {
            // Get popular ONNX-ready text generation models
            return await SearchModelsAsync("onnx genai", sortBy: "downloads", limit: 30);
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
