using System;

namespace LocalLLMchat.Services
{
    public class SettingsService
    {
        // simplistic settings for MVP
        public string ModelPath { get; set; } = @"C:\AI\Models\Phi-3-mini-4k-instruct-onnx"; // Default placeholder
        public string SystemPrompt { get; set; } = "You are a helpful AI assistant.";
    }
}
