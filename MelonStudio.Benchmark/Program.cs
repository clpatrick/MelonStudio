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
        var modelPath = args.Length > 0 ? args[0] : @"C:\AI\Models\Phi-3-mini-4k-instruct-onnx";
        var outputPath = args.Length > 1 ? args[1] : "benchmarks/results.json";

        Console.WriteLine("=== MelonStudio Benchmark ===");
        Console.WriteLine($"Model Path: {modelPath}");
        Console.WriteLine($"Output: {outputPath}");
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
            
            using var model = new Model(modelPath);
            using var tokenizer = new Tokenizer(model);
            
            loadStart.Stop();
            Console.WriteLine($"Model loaded in {loadStart.Elapsed.TotalSeconds:F2}s");
            Console.WriteLine();

            var totalStopwatch = Stopwatch.StartNew();

            foreach (var prompt in TestPrompts)
            {
                Console.WriteLine($"Prompt: \"{prompt}\"");
                
                var promptResult = await RunPromptAsync(model, tokenizer, prompt);
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

    private static async Task<PromptResult> RunPromptAsync(Model model, Tokenizer tokenizer, string prompt)
    {
        var result = new PromptResult { Prompt = prompt };
        
        // Phi-3 chat format
        var fullPrompt = $"<|user|>\n{prompt}<|end|>\n<|assistant|>";
        var sequences = tokenizer.Encode(fullPrompt);
        
        using var generatorParams = new GeneratorParams(model);
        generatorParams.SetSearchOption("max_length", 512);
        generatorParams.SetInputSequences(sequences);
        
        using var generator = new Generator(model, generatorParams);
        
        var stopwatch = Stopwatch.StartNew();
        var response = new System.Text.StringBuilder();
        int tokenCount = 0;
        
        while (!generator.IsDone())
        {
            await Task.Run(() =>
            {
                generator.ComputeLogits();
                generator.GenerateNextToken();
            });
            
            var outputSequence = generator.GetSequence(0);
            var newTokenId = outputSequence[outputSequence.Length - 1];
            var decodedToken = tokenizer.Decode(new[] { newTokenId });
            
            response.Append(decodedToken);
            tokenCount++;
        }
        
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
