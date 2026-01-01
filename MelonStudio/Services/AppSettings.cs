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
        public bool EnableCudaGraph { get; set; } = true;
        public string LastModelPath { get; set; } = "";

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
