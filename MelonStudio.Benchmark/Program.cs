using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace MelonStudio.Benchmark;

public class BenchmarkResult
{
    public string Timestamp { get; set; } = DateTime.UtcNow.ToString("o");
    public string ModelPath { get; set; } = "";
    public string MachineName { get; set; } = Environment.MachineName;
    public List<PromptResult> Results { get; set; } = new();
    public double AverageTokensPerSecond { get; set; }
    public double TotalElapsedSeconds { get; set; }
    public int TotalTokensGenerated { get; set; }
    public string? Error { get; set; }
}

public class PromptResult
{
    public string Prompt { get; set; } = "";
    public int TokensGenerated { get; set; }
    public double ElapsedSeconds { get; set; }
    public double TokensPerSecond { get; set; }
    public string Response { get; set; } = "";
}

public class Program
{
    private static readonly string[] TestPrompts = new[]
    {
        "Hello, how are you?",
        "Explain quantum computing in 3 sentences.",
        "Write a haiku about programming.",
        "What is the capital of France?",
        "List 5 benefits of exercise."
    };

    public static async Task<int> Main(string[] args)
    {
        // Default settings
        string modelPath = @"models\phi3.5-hybrid";
        string outputPath = "benchmarks/results.json";
        string executionProvider = "cuda";
        float temperature = 0.7f;
        float topP = 0.9f;
        int topK = 50;
        int maxLen = 1024;

        // Parse arguments
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLower())
            {
                case "--model":
                    if (i + 1 < args.Length) modelPath = args[++i];
                    break;
                case "--ep":
                case "--provider":
                    if (i + 1 < args.Length) executionProvider = args[++i].ToLower();
                    break;
                case "--out":
                case "--output":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                case "--temp":
                case "--temperature":
                    if (i + 1 < args.Length) float.TryParse(args[++i], out temperature);
                    break;
                case "--topp":
                    if (i + 1 < args.Length) float.TryParse(args[++i], out topP);
                    break;
                case "--topk":
                    if (i + 1 < args.Length) int.TryParse(args[++i], out topK);
                    break;
                case "--max":
                case "--maxlen":
                    if (i + 1 < args.Length) int.TryParse(args[++i], out maxLen);
                    break;
                default:
                    if (!args[i].StartsWith("-"))
                    {
                        if (i == 0 && string.IsNullOrEmpty(args.FirstOrDefault(a => a.StartsWith("--") || a == args[i]))) modelPath = args[i]; // weak check, but okay for pos 0
                        else if (i == 1 && !args[i-1].StartsWith("-")) outputPath = args[i];
                    }
                    break;
            }
        }
        
        // Handle positional args legacy override if not set by flags (simplified)
        if (args.Length > 0 && !args[0].StartsWith("-")) modelPath = args[0];
        if (args.Length > 1 && !args[1].StartsWith("-")) outputPath = args[1];

        Console.WriteLine("=== MelonStudio Benchmark ===");
        Console.WriteLine($"Model Path: {modelPath}");
        Console.WriteLine($"Output: {outputPath}");
        Console.WriteLine($"Settings: EP={executionProvider}, Temp={temperature}, TopP={topP}, TopK={topK}, MaxLen={maxLen}");
        Console.WriteLine();

        var result = new BenchmarkResult { ModelPath = modelPath };

        try
        {
            if (!Directory.Exists(modelPath))
            {
                throw new DirectoryNotFoundException($"Model directory not found: {modelPath}");
            }

            Console.WriteLine("Loading model...");
            var loadStart = Stopwatch.StartNew();
            
            // service will be either HybridLLMService or standard logic wrapped
            using var service = new MelonStudio.Services.HybridLLMService();

            // Hook into diagnostic events to print metadata
            service.OnDiagnostic += (msg) => Console.WriteLine($"[DIAG] {msg}");
            
            // Check if directory is hybrid
            bool isHybrid = MelonStudio.Services.HybridLLMService.IsHybridModelDirectory(modelPath);
            if (isHybrid)
            {
                Console.WriteLine("Detected Hybrid Model Configuration");
                await service.InitializeHybridAsync(modelPath);
            }
            else
            {
                Console.WriteLine($"Detected Standard Model (using {executionProvider})");
                await service.InitializeStandardAsync(modelPath, executionProvider);
            }
            
            // Apply settings
            service.UpdateSettings(maxLen, temperature, topP, topK);
            
            loadStart.Stop();
            Console.WriteLine($"Model loaded in {loadStart.Elapsed.TotalSeconds:F2}s");
            Console.WriteLine(service.Summary);
            Console.WriteLine();

            var totalStopwatch = Stopwatch.StartNew();

            foreach (var prompt in TestPrompts)
            {
                Console.WriteLine($"Prompt: \"{prompt}\"");
                
                var promptResult = await RunPromptAsync(service, prompt);
                result.Results.Add(promptResult);
                
                Console.WriteLine($"  Tokens: {promptResult.TokensGenerated}, Time: {promptResult.ElapsedSeconds:F2}s, Speed: {promptResult.TokensPerSecond:F1} tok/s");
                Console.WriteLine($"  Response: {Truncate(promptResult.Response, 100)}");
                Console.WriteLine();
            }

            totalStopwatch.Stop();

            result.TotalElapsedSeconds = totalStopwatch.Elapsed.TotalSeconds;
            result.TotalTokensGenerated = result.Results.Sum(r => r.TokensGenerated);
            result.AverageTokensPerSecond = result.TotalTokensGenerated / result.TotalElapsedSeconds;

            Console.WriteLine("=== Summary ===");
            Console.WriteLine($"Total Tokens: {result.TotalTokensGenerated}");
            Console.WriteLine($"Total Time: {result.TotalElapsedSeconds:F2}s");
            Console.WriteLine($"Average Speed: {result.AverageTokensPerSecond:F1} tokens/second");
            Console.WriteLine(service.Summary);
        }
        catch (Exception ex)
        {
            result.Error = ex.Message;
            Console.WriteLine($"ERROR: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }

        // Save results
        try
        {
            var outputDir = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(outputDir) && !Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            var json = JsonSerializer.Serialize(result, new JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
            
            await File.WriteAllTextAsync(outputPath, json);
            Console.WriteLine($"\nResults saved to: {outputPath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to save results: {ex.Message}");
            return 1;
        }

        return result.Error != null ? 1 : 0;
    }

    private static async Task<PromptResult> RunPromptAsync(MelonStudio.Services.HybridLLMService service, string prompt)
    {
        var result = new PromptResult { Prompt = prompt };
        
        var stopwatch = Stopwatch.StartNew();
        var response = new System.Text.StringBuilder();
        int tokenCount = 0;
        
        await foreach (var token in service.GenerateResponseAsync(prompt, ""))
        {
            response.Append(token);
            tokenCount++;
            Console.Write(token); // Stream to console
        }
        Console.WriteLine();
        
        stopwatch.Stop();
        
        result.TokensGenerated = tokenCount;
        result.ElapsedSeconds = stopwatch.Elapsed.TotalSeconds;
        result.TokensPerSecond = tokenCount / result.ElapsedSeconds;
        result.Response = response.ToString().Trim();
        
        return result;
    }

    private static string Truncate(string text, int maxLength)
    {
        if (string.IsNullOrEmpty(text)) return "";
        text = text.Replace("\n", " ").Replace("\r", "");
        return text.Length <= maxLength ? text : text.Substring(0, maxLength) + "...";
    }
}
