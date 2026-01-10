using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using MelonStudio.Services;

namespace MelonStudio.ViewModels
{
    public class ModelInfo
    {
        public string Name { get; set; } = "";
        public string Path { get; set; } = "";
    }

    public partial class LoadModelViewModel : ObservableObject
    {
        private readonly AppSettings _settings;

        [ObservableProperty]
        private string _modelsFolder = @"C:\models";

        [ObservableProperty]
        private ModelInfo? _selectedModel;

        // Analysis panel properties
        [ObservableProperty]
        private bool _showAnalysisPanel;

        [ObservableProperty]
        private bool _isHybridModel;

        [ObservableProperty]
        private string _analyzedModelPath = "";

        [ObservableProperty]
        private string _modelName = "";

        [ObservableProperty]
        private int _gpuLayers;

        [ObservableProperty]
        private int _cpuLayers;

        [ObservableProperty]
        private int _totalLayers;

        [ObservableProperty]
        private double _gpuSizeGb;

        [ObservableProperty]
        private double _cpuSizeGb;

        [ObservableProperty]
        private double _totalSizeGb;

        [ObservableProperty]
        private string _precision = "";

        [ObservableProperty]
        private bool _isAnalyzing;

        [ObservableProperty]
        private string _selectedOnnxFileName = "";

        public ObservableCollection<ModelInfo> AvailableModels { get; } = new();

        public LoadModelViewModel()
        {
            _settings = AppSettings.Load();
            _modelsFolder = _settings.DefaultOutputFolder;
            RefreshModels();
        }

        public void RefreshModels()
        {
            AvailableModels.Clear();

            if (!Directory.Exists(ModelsFolder))
                return;

            try
            {
                foreach (var dir in Directory.GetDirectories(ModelsFolder))
                {
                    // Skip temp folder
                    if (Path.GetFileName(dir).Equals("temp", System.StringComparison.OrdinalIgnoreCase))
                        continue;

                    // Check if it looks like an ONNX model folder
                    var hasConfig = File.Exists(Path.Combine(dir, "genai_config.json")) ||
                                    File.Exists(Path.Combine(dir, "config.json"));
                    var hasOnnx = Directory.GetFiles(dir, "*.onnx", SearchOption.AllDirectories).Length > 0;

                    if (hasConfig || hasOnnx)
                    {
                        AvailableModels.Add(new ModelInfo
                        {
                            Name = Path.GetFileName(dir),
                            Path = dir
                        });
                    }
                }
            }
            catch { }
        }

        public async Task AnalyzeSelectedFileAsync(string onnxPath)
        {
            IsAnalyzing = true;
            ShowAnalysisPanel = false;

            try
            {
                var folder = Path.GetDirectoryName(onnxPath) ?? "";
                var gpuPath = Path.Combine(folder, "gpu_part.onnx");
                var cpuPath = Path.Combine(folder, "cpu_part.onnx");

                ModelName = Path.GetFileName(folder);
                AnalyzedModelPath = folder;
                SelectedOnnxFileName = Path.GetFileName(onnxPath);

                if (File.Exists(gpuPath) && File.Exists(cpuPath))
                {
                    // Hybrid model
                    IsHybridModel = true;

                    // Analyze both parts
                    var gpuResult = await AnalyzeOnnxFileAsync(gpuPath);
                    var cpuResult = await AnalyzeOnnxFileAsync(cpuPath);

                    GpuLayers = gpuResult.Layers;
                    CpuLayers = cpuResult.Layers;
                    TotalLayers = GpuLayers + CpuLayers;
                    GpuSizeGb = gpuResult.SizeGb;
                    CpuSizeGb = cpuResult.SizeGb;
                    TotalSizeGb = GpuSizeGb + CpuSizeGb;
                    Precision = gpuResult.Precision;
                }
                else
                {
                    // Standard model - analyze the selected file or model.onnx
                    IsHybridModel = false;
                    var modelFile = File.Exists(Path.Combine(folder, "model.onnx")) 
                        ? Path.Combine(folder, "model.onnx") 
                        : onnxPath;

                    var result = await AnalyzeOnnxFileAsync(modelFile);
                    TotalLayers = result.Layers;
                    TotalSizeGb = result.SizeGb;
                    Precision = result.Precision;
                    GpuLayers = 0;
                    CpuLayers = 0;
                    GpuSizeGb = 0;
                    CpuSizeGb = 0;
                }

                ShowAnalysisPanel = true;
            }
            finally
            {
                IsAnalyzing = false;
            }
        }

        private async Task<(int Layers, double SizeGb, string Precision)> AnalyzeOnnxFileAsync(string onnxPath)
        {
            var scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "scripts", "analyze_onnx_size.py");
            
            if (!File.Exists(scriptPath))
                return (0, 0, "Unknown");

            var psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"\"{scriptPath}\" \"{onnxPath}\"",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            try
            {
                using var process = Process.Start(psi);
                if (process == null) return (0, 0, "Unknown");

                var output = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                // Parse output
                int layers = 0;
                double sizeGb = 0;
                string precision = "";

                foreach (var line in output.Split('\n'))
                {
                    if (line.Contains("Layers Found:"))
                    {
                        var match = System.Text.RegularExpressions.Regex.Match(line, @"Layers Found:\s*(\d+)");
                        if (match.Success) layers = int.Parse(match.Groups[1].Value);
                    }
                    else if (line.Contains("Total Calculated Params Size:"))
                    {
                        var match = System.Text.RegularExpressions.Regex.Match(line, @"Size:\s*([\d.]+)\s*GB");
                        if (match.Success) sizeGb = double.Parse(match.Groups[1].Value);
                    }
                    else if (line.Contains("Dominant Precision:"))
                    {
                        precision = line.Split(':').Last().Trim();
                    }
                }

                return (layers, sizeGb, precision);
            }
            catch
            {
                return (0, 0, "Unknown");
            }
        }

        public void SaveSettings()
        {
            _settings.DefaultOutputFolder = ModelsFolder;
            _settings.Save();
        }
    }
}
