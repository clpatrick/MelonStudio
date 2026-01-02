using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace MelonStudio.Models
{
    public enum ModelStatus
    {
        Downloaded,   // Original format, not converted
        Converting,   // Conversion in progress
        Converted,    // ONNX ready to use
        Failed        // Conversion failed, can retry
    }

    public class LocalModelInfo
    {
        public string Name { get; set; } = "";
        public string Path { get; set; } = "";
        public ModelStatus Status { get; set; } = ModelStatus.Downloaded;
        public string Format { get; set; } = "";  // "onnx", "safetensors", "pytorch"
        public string Precision { get; set; } = "";  // "fp32", "fp16", "int4"
        public long SizeBytes { get; set; }
        public DateTime LastModified { get; set; }
        public string? ErrorMessage { get; set; }

        // UI properties
        public string StatusIcon => Status switch
        {
            ModelStatus.Downloaded => "📥",
            ModelStatus.Converting => "⏳",
            ModelStatus.Converted => "✓",
            ModelStatus.Failed => "⚠",
            _ => "?"
        };

        public string StatusColor => Status switch
        {
            ModelStatus.Downloaded => "#FFC107",
            ModelStatus.Converting => "#17A2B8",
            ModelStatus.Converted => "#28A745",
            ModelStatus.Failed => "#DC3545",
            _ => "#808080"
        };

        public string StatusText => Status switch
        {
            ModelStatus.Downloaded => "Downloaded - needs conversion",
            ModelStatus.Converting => "Converting...",
            ModelStatus.Converted => "Ready to use",
            ModelStatus.Failed => $"Conversion failed: {ErrorMessage ?? "Unknown error"}",
            _ => "Unknown"
        };

        public string SizeFormatted
        {
            get
            {
                if (SizeBytes >= 1073741824) return $"{SizeBytes / 1073741824.0:0.##} GB";
                if (SizeBytes >= 1048576) return $"{SizeBytes / 1048576.0:0.#} MB";
                return $"{SizeBytes / 1024.0:0.#} KB";
            }
        }

        public bool CanConvert => Status == ModelStatus.Downloaded || Status == ModelStatus.Failed;
        
        public bool IsOnnxReady => Status == ModelStatus.Converted && Format == "onnx";

        // Available conversion options based on current format
        public string[] AvailableConversions
        {
            get
            {
                if (Status == ModelStatus.Converted && Format == "onnx")
                {
                    // Already ONNX - can convert to different quantizations
                    return Precision switch
                    {
                        "fp32" => new[] { "int4-cuda", "int4-cpu", "fp16-cuda" },
                        "fp16" => new[] { "int4-cuda", "int4-cpu" },
                        _ => Array.Empty<string>()
                    };
                }
                else if (Format == "safetensors" || Format == "pytorch")
                {
                    // Source format - can convert to any ONNX format
                    return new[] { "int4-cuda", "int4-cpu", "fp16-cuda", "fp32-cpu" };
                }
                return Array.Empty<string>();
            }
        }

        public string ConversionTooltip
        {
            get
            {
                var conversions = AvailableConversions;
                if (conversions.Length == 0)
                    return "No additional conversions available";
                
                return $"Can convert to:\n• {string.Join("\n• ", conversions.Select(FormatConversionOption))}";
            }
        }

        private static string FormatConversionOption(string opt) => opt switch
        {
            "int4-cuda" => "INT4 CUDA (smallest, fast)",
            "int4-cpu" => "INT4 CPU (smallest, no GPU)",
            "fp16-cuda" => "FP16 CUDA (balanced)",
            "fp32-cpu" => "FP32 CPU (original quality)",
            _ => opt
        };
    }
}
