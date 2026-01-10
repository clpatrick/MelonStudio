using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MelonStudio.Models;

namespace MelonStudio.Services
{
    public class LocalModelService
    {
        public string ModelsFolder { get; }

        public LocalModelService(string modelsFolder)
        {
            ModelsFolder = modelsFolder;
        }

        /// <summary>
        /// Scans the models folder and returns all detected models with their status.
        /// </summary>
        public async Task<List<LocalModelInfo>> ScanModelsFolderAsync()
        {
            var models = new List<LocalModelInfo>();

            if (!Directory.Exists(ModelsFolder))
                return models;

            await Task.Run(() =>
            {
                foreach (var dir in Directory.GetDirectories(ModelsFolder))
                {
                    // Skip temp folder
                    if (Path.GetFileName(dir).Equals("temp", StringComparison.OrdinalIgnoreCase))
                        continue;

                    var model = DetectModelInfo(dir);
                    if (model != null)
                    {
                        models.Add(model);
                    }
                }
            });

            return models.OrderBy(m => m.Name).ToList();
        }

        /// <summary>
        /// Detects model type and status from a folder.
        /// </summary>
        private LocalModelInfo? DetectModelInfo(string modelPath)
        {
            var name = Path.GetFileName(modelPath);
            
            // Calculate folder size
            var files = Directory.GetFiles(modelPath, "*", SearchOption.AllDirectories);
            var sizeBytes = files.Sum(f => new FileInfo(f).Length);
            var lastModified = files.Length > 0 
                ? files.Max(f => File.GetLastWriteTimeUtc(f)) 
                : DateTime.UtcNow;

            // Detect model format
            var hasGenAiConfig = File.Exists(Path.Combine(modelPath, "genai_config.json"));
            var hasOnnxFiles = Directory.GetFiles(modelPath, "*.onnx", SearchOption.TopDirectoryOnly).Length > 0 ||
                              Directory.GetFiles(modelPath, "*.onnx", SearchOption.AllDirectories).Length > 0;
            var hasSafeTensors = Directory.GetFiles(modelPath, "*.safetensors", SearchOption.AllDirectories).Length > 0;
            var hasPytorchBin = Directory.GetFiles(modelPath, "*.bin", SearchOption.AllDirectories)
                                         .Any(f => Path.GetFileName(f).Contains("pytorch") || Path.GetFileName(f).Contains("model"));
            var hasConfig = File.Exists(Path.Combine(modelPath, "config.json"));

            // Determine format and status
            string format;
            string precision = "";
            ModelStatus status;

            if (hasGenAiConfig && hasOnnxFiles)
            {
                // Fully converted ONNX model
                format = "onnx";
                status = ModelStatus.Converted;
                precision = DetectOnnxPrecision(modelPath);
            }
            else if (HybridLLMService.IsHybridModelDirectory(modelPath))
            {
                // Hybrid split model (gpu_part.onnx + cpu_part.onnx)
                format = "hybrid";
                status = ModelStatus.Converted;
                precision = DetectOnnxPrecision(modelPath);
            }
            else if (hasOnnxFiles && !hasGenAiConfig)
            {
                // Has ONNX but missing config - partial conversion
                format = "onnx";
                status = ModelStatus.Failed;
            }
            else if (hasSafeTensors)
            {
                // SafeTensors source model
                format = "safetensors";
                status = ModelStatus.Downloaded;
                precision = DetectSourcePrecision(modelPath);
            }
            else if (hasPytorchBin)
            {
                // PyTorch binary source model
                format = "pytorch";
                status = ModelStatus.Downloaded;
                precision = "fp32"; // Usually FP32
            }
            else if (hasConfig)
            {
                // Has config but no model files - incomplete download
                format = "unknown";
                status = ModelStatus.Failed;
            }
            else
            {
                // Not a recognized model folder
                return null;
            }

            return new LocalModelInfo
            {
                Name = name,
                Path = modelPath,
                Status = status,
                Format = format,
                Precision = precision,
                SizeBytes = sizeBytes,
                LastModified = lastModified
            };
        }

        /// <summary>
        /// Detect precision from ONNX model folder name or config.
        /// </summary>
        private string DetectOnnxPrecision(string modelPath)
        {
            var name = Path.GetFileName(modelPath).ToLowerInvariant();
            
            if (name.Contains("int4") || name.Contains("_i4_"))
                return "int4";
            if (name.Contains("fp16") || name.Contains("float16") || name.Contains("_f16_"))
                return "fp16";
            if (name.Contains("fp32") || name.Contains("float32") || name.Contains("_f32_"))
                return "fp32";

            // Try to read from genai_config.json
            var configPath = Path.Combine(modelPath, "genai_config.json");
            if (File.Exists(configPath))
            {
                try
                {
                    var content = File.ReadAllText(configPath).ToLowerInvariant();
                    if (content.Contains("int4")) return "int4";
                    if (content.Contains("fp16") || content.Contains("float16")) return "fp16";
                }
                catch { }
            }

            return "unknown";
        }

        /// <summary>
        /// Detect precision from source model files.
        /// </summary>
        private string DetectSourcePrecision(string modelPath)
        {
            var name = Path.GetFileName(modelPath).ToLowerInvariant();
            
            if (name.Contains("fp16") || name.Contains("float16"))
                return "fp16";
            if (name.Contains("bf16") || name.Contains("bfloat16"))
                return "bf16";
            
            return "fp32"; // Default assumption
        }

        /// <summary>
        /// Update model status (called during conversion).
        /// </summary>
        public void UpdateModelStatus(string modelPath, ModelStatus status, string? errorMessage = null)
        {
            // This would update a status file in the model folder
            var statusFile = Path.Combine(modelPath, ".melon_status.json");
            var statusData = new
            {
                Status = status.ToString(),
                ErrorMessage = errorMessage,
                LastUpdated = DateTime.UtcNow
            };
            
            try
            {
                var json = System.Text.Json.JsonSerializer.Serialize(statusData);
                File.WriteAllText(statusFile, json);
            }
            catch { }
        }
    }
}
