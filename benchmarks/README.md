# Benchmark Results

This directory contains benchmark results from running MelonStudio on various hardware configurations.

## Running the Benchmark

```bash
cd MelonStudio.Benchmark
dotnet run --configuration Release -- "C:\path\to\your\onnx\model" "../benchmarks/results.json"
```

## Results

The `results.json` file is automatically updated when benchmarks are run. Key metrics:

- **AverageTokensPerSecond**: Overall throughput
- **TotalTokensGenerated**: Sum of all tokens across test prompts
- **TotalElapsedSeconds**: Total inference time (excluding model load)

## Test Prompts

1. "Hello, how are you?"
2. "Explain quantum computing in 3 sentences."
3. "Write a haiku about programming."
4. "What is the capital of France?"
5. "List 5 benefits of exercise."

## Committing Results

After running the benchmark, commit the updated `results.json`:

```bash
git add benchmarks/results.json
git commit -m "Update benchmark results from [YOUR_MACHINE]"
git push
```
