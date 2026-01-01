using System.Collections.ObjectModel;
using System.IO;
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

        public void SaveSettings()
        {
            _settings.DefaultOutputFolder = ModelsFolder;
            _settings.Save();
        }
    }
}
