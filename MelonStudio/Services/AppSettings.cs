using System;
using System.IO;
using System.Text.Json;

namespace MelonStudio.Services
{
    public class AppSettings
    {
        public string HuggingFaceToken { get; set; } = "";
        public string DefaultOutputFolder { get; set; } = @"C:\models";
        public string DefaultPrecision { get; set; } = "int4";
        public string DefaultProvider { get; set; } = "cuda";
        public bool EnableCudaGraph { get; set; } = false; // Not all models support this
        public string LastModelPath { get; set; } = "";
        public string LastConversionBrowseFolder { get; set; } = "";

        // Generation settings
        public int MaxLength { get; set; } = 8192;
        public double Temperature { get; set; } = 0.7;
        public double TopP { get; set; } = 0.9;
        public string SystemPrompt { get; set; } = "You are a helpful AI assistant.";

        private static readonly string SettingsPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "MelonStudio",
            "settings.json");

        public static AppSettings Load()
        {
            try
            {
                if (File.Exists(SettingsPath))
                {
                    var json = File.ReadAllText(SettingsPath);
                    return JsonSerializer.Deserialize<AppSettings>(json) ?? new AppSettings();
                }
            }
            catch { }
            
            return new AppSettings();
        }

        public void Save()
        {
            try
            {
                var dir = Path.GetDirectoryName(SettingsPath);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }

                var json = JsonSerializer.Serialize(this, new JsonSerializerOptions 
                { 
                    WriteIndented = true 
                });
                File.WriteAllText(SettingsPath, json);
            }
            catch { }
        }
    }
}
