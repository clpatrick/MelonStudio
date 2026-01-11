# Download Progress Implementation Options

## Current Implementation Analysis

### How `snapshot_download` Works
- **With `local_dir` and `local_dir_use_symlinks=False`**: Files are written **directly** to the output directory
- **No intermediate caching**: Files appear in the output directory as they download
- **Internal progress**: Uses `tqdm` for console progress, but no programmatic callback API

### File Monitoring Feasibility
✅ **YES - We CAN monitor file sizes** because:
- Files are written directly to `output_dir` (not cached elsewhere first)
- Files appear incrementally as they download
- We can poll the directory and sum file sizes

---

## Option 1: File Size Monitoring (Recommended)

### How It Works
1. Get total size from metadata (already available)
2. Poll the output directory periodically (every 0.5-1 second)
3. Sum sizes of all files in the directory
4. Calculate: `progress = (downloaded_bytes / total_bytes) * 100`

### Pros
- ✅ Simple to implement
- ✅ Works with current `snapshot_download` approach
- ✅ No changes to Python script needed
- ✅ Accurate for most cases
- ✅ Works with resume downloads

### Cons
- ⚠️ Slight delay (polling interval)
- ⚠️ May show 100% before all files complete (if some files are small/quick)
- ⚠️ Doesn't show per-file progress

### Implementation
```csharp
// In ViewModel, start a background task that polls:
private async Task MonitorDownloadProgress(string outputDir, long totalBytes)
{
    while (IsConverting)
    {
        await Task.Delay(500); // Poll every 500ms
        
        var downloadedBytes = Directory.GetFiles(outputDir, "*", SearchOption.AllDirectories)
            .Sum(f => new FileInfo(f).Length);
        
        var progress = totalBytes > 0 ? (downloadedBytes * 100.0 / totalBytes) : 0;
        
        App.Current.Dispatcher.Invoke(() =>
        {
            DownloadProgress = Math.Min(100, progress);
            IsDownloadProgressIndeterminate = false;
        });
    }
}
```

---

## Option 2: Capture tqdm Output from Python

### How It Works
1. `huggingface_hub` uses `tqdm` internally for progress
2. Redirect stderr and parse tqdm output
3. Extract progress percentage from tqdm's formatted output

### Pros
- ✅ Real-time progress from HuggingFace Hub
- ✅ Shows per-file progress
- ✅ More accurate than file monitoring

### Cons
- ⚠️ Requires parsing tqdm output (fragile)
- ⚠️ tqdm format may change
- ⚠️ Need to handle different tqdm output formats
- ⚠️ May need to disable tqdm's auto-detection

### Implementation
```python
# In olive_download.py, we could:
import os
os.environ["HF_HUB_DISABLE_TQDM"] = "0"  # Keep tqdm enabled
# Then capture stderr and parse lines like:
# "Downloading: 45%|████▌     | 2.1G/4.7G [00:30<00:35, 73.2MB/s]"
```

```csharp
// Parse tqdm output from stderr:
// Pattern: "45%|████▌     | 2.1G/4.7G"
var match = Regex.Match(line, @"(\d+)%");
if (match.Success)
{
    DownloadProgress = int.Parse(match.Groups[1].Value);
}
```

---

## Option 3: Use `download_file` with Progress Callbacks

### How It Works
1. Get file list from metadata (already have this)
2. Download each file individually using `download_file` with `tqdm` callback
3. Track progress per file

### Pros
- ✅ Most accurate progress
- ✅ Can show per-file progress
- ✅ Full control over download process

### Cons
- ⚠️ More complex implementation
- ⚠️ Need to handle file dependencies
- ⚠️ Need to handle LFS files properly
- ⚠️ More error handling needed
- ⚠️ Slower (sequential downloads vs parallel)

### Implementation
```python
from huggingface_hub import download_file, hf_hub_url
from tqdm import tqdm

def download_with_progress(repo_id, files, output_dir, token=None):
    total_size = sum(f['sizeBytes'] for f in files)
    downloaded = 0
    
    for file_info in files:
        url = hf_hub_url(repo_id, file_info['name'])
        local_path = os.path.join(output_dir, file_info['name'])
        
        # Create directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download with progress callback
        def update_progress(bytes_downloaded, total_bytes):
            nonlocal downloaded
            downloaded += bytes_downloaded
            progress = (downloaded / total_size) * 100
            print(f"PROGRESS: {progress:.1f}")  # C# can parse this
        
        download_file(
            repo_id=repo_id,
            filename=file_info['name'],
            local_dir=output_dir,
            token=token,
            # tqdm callback would go here
        )
```

---

## Option 4: Direct HTTP Downloads (C#)

### How It Works
1. Get file URLs from HuggingFace API
2. Download files directly using C# `HttpClient` with progress callbacks
3. Handle authentication, LFS files, etc. in C#

### Pros
- ✅ Full control over progress reporting
- ✅ No Python dependency for downloads
- ✅ Real-time progress updates
- ✅ Can show per-file progress

### Cons
- ⚠️ Need to implement HuggingFace API authentication
- ⚠️ Need to handle Git LFS files
- ⚠️ Need to handle file dependencies
- ⚠️ More complex error handling
- ⚠️ Lose HuggingFace Hub's built-in features (resume, validation, etc.)

### Implementation
```csharp
// Use HttpClient with IProgress<double>
var response = await httpClient.GetAsync(fileUrl, HttpCompletionOption.ResponseHeadersRead);
var totalBytes = response.Content.Headers.ContentLength;

using (var stream = await response.Content.ReadAsStreamAsync())
using (var fileStream = new FileStream(localPath, FileMode.Create))
{
    var buffer = new byte[8192];
    long totalRead = 0;
    int bytesRead;
    
    while ((bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length)) > 0)
    {
        await fileStream.WriteAsync(buffer, 0, bytesRead);
        totalRead += bytesRead;
        
        if (totalBytes.HasValue)
        {
            var progress = (totalRead * 100.0) / totalBytes.Value;
            progressReporter.Report(progress);
        }
    }
}
```

---

## Recommendation

**Option 1 (File Size Monitoring)** is the best balance of:
- ✅ Simplicity
- ✅ Accuracy
- ✅ No breaking changes
- ✅ Works with current implementation
- ✅ Low maintenance

**Option 2 (tqdm parsing)** could be added as an enhancement later if more accuracy is needed.

**Option 3 & 4** are overkill for current needs and add significant complexity.

---

## Implementation Plan for Option 1

1. **Store total bytes** from metadata in ViewModel
2. **Start monitoring task** when download begins
3. **Poll output directory** every 500ms
4. **Calculate progress** and update UI
5. **Stop monitoring** when download completes or fails
