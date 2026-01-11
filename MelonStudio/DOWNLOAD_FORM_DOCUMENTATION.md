# Download Form GUI Documentation

## Overview

The download form is part of the `ModelManagerControl` user control, which provides a comprehensive interface for discovering, downloading, and converting HuggingFace models.

---

## UI Structure

### Main Layout (`ModelManagerControl.xaml`)

The form is split into two main panels:

#### **Left Panel** - Model Browser & Search
- **Search Bar** with filters and sort options
- **Model List** showing search results
- **Conversion Log** at the bottom

#### **Right Panel** - Model Details & Download Form
- **Model Information** (name, author, stats)
- **Conversion Settings** section
- **Download/Convert Buttons**

---

## Download Form Components

### 1. Model ID Input (Lines 294-309)

```xml
<StackPanel Grid.Row="5" Margin="0,0,0,12">
    <TextBlock Text="Model ID" Foreground="#808080" Margin="0,0,0,5"/>
    <TextBox Text="{Binding SelectedModelId, UpdateSourceTrigger=PropertyChanged}"
             Background="#3C3C3C" Foreground="#D4D4D4"
             BorderBrush="#555555" Padding="10,8"/>
             
    <!-- Variant Selector for Multi-Variant ONNX Models -->
    <StackPanel Margin="0,12,0,0" Visibility="{Binding HasVariants, Converter={StaticResource BoolToVis}}">
         <TextBlock Text="Select Variant (Subfolder)" Foreground="#FFC107" FontWeight="SemiBold" Margin="0,0,0,5"/>
         <ComboBox ItemsSource="{Binding Variants}" 
                   SelectedItem="{Binding SelectedVariant}"
                   Background="#3C3C3C" Foreground="#D4D4D4" Padding="10,8"/>
         <TextBlock Text="Note: This will download ONLY the selected variant." 
                    Foreground="#808080" FontSize="10" Margin="0,2,0,0"/>
    </StackPanel>
</StackPanel>
```

**Features:**
- Text input for HuggingFace model ID (e.g., `microsoft/Phi-3.5-mini-instruct`)
- Conditional variant selector (appears when model has multiple variants/subfolders)
- Warning message about selective download

**Binding:** `SelectedModelId` → ViewModel property

---

### 2. Precision & Provider Selection (Lines 311-333)

```xml
<Grid Grid.Row="6" Margin="0,0,0,12">
    <Grid.ColumnDefinitions>
        <ColumnDefinition Width="*"/>
        <ColumnDefinition Width="*"/>
    </Grid.ColumnDefinitions>

    <StackPanel Margin="0,0,5,0">
        <TextBlock Text="Precision" Foreground="#808080" Margin="0,0,0,5"/>
        <ComboBox ItemsSource="{Binding PrecisionOptions}"
                  SelectedItem="{Binding SelectedPrecision}"
                  Background="#3C3C3C" Foreground="#D4D4D4"
                  Padding="10,8"/>
    </StackPanel>

    <StackPanel Grid.Column="1" Margin="5,0,0,0">
        <TextBlock Text="Provider" Foreground="#808080" Margin="0,0,0,5"/>
        <ComboBox ItemsSource="{Binding ProviderOptions}"
                  SelectedItem="{Binding SelectedProvider}"
                  Background="#3C3C3C" Foreground="#D4D4D4"
                  Padding="10,8"/>
    </StackPanel>
</Grid>
```

**Options:**
- **Precision:** `fp32`, `fp16`, `int4` (default: `int4`)
- **Provider:** `cuda`, `dml`, `cpu` (default: `cuda`)

**Note:** These are primarily for conversion, but may affect download behavior for some models.

---

### 3. Output Folder (Lines 335-341)

```xml
<StackPanel Grid.Row="7" Margin="0,0,0,12">
    <TextBlock Text="Output Folder" Foreground="#808080" Margin="0,0,0,5"/>
    <TextBox Text="{Binding OutputFolder, UpdateSourceTrigger=PropertyChanged}"
             Background="#3C3C3C" Foreground="#D4D4D4"
             BorderBrush="#555555" Padding="10,8"/>
</StackPanel>
```

**Default:** `C:\Repos\MelonStudio\models`

**Behavior:** 
- Creates subfolder based on model ID (replaces `/` and `\` with `_`)
- Example: `microsoft/Phi-3.5-mini-instruct` → `microsoft_Phi-3.5-mini-instruct`

---

### 4. HuggingFace Token (Lines 343-351)

```xml
<StackPanel Grid.Row="8" Margin="0,0,0,12">
    <TextBlock Text="HuggingFace Token (optional)" 
               Foreground="#808080" Margin="0,0,0,5"/>
    <PasswordBox x:Name="HfTokenBox"
                 Background="#3C3C3C" Foreground="#D4D4D4"
                 BorderBrush="#555555" Padding="10,8"
                 PasswordChanged="HfTokenBox_PasswordChanged"/>
</StackPanel>
```

**Features:**
- Password box for secure token entry
- Optional (only needed for private/gated models)
- Token is saved and loaded from settings

**Code-behind:** `HfTokenBox_PasswordChanged` updates ViewModel property

---

### 5. Download Button (Lines 361-386)

```xml
<Button Command="{Binding DownloadOnnxModelCommand}"
        IsEnabled="{Binding IsConverting, Converter={StaticResource InverseBoolConverter}}"
        Padding="15,12" Margin="0,0,0,8"
        Background="#6C5CE7" Foreground="White"
        FontWeight="Bold" FontSize="14" BorderThickness="0">
    <Button.Style>
        <Style TargetType="Button">
            <Setter Property="Visibility" Value="Collapsed"/>
            <Style.Triggers>
                <DataTrigger Binding="{Binding SelectedModelDetails.IsOnnxModel}" Value="True">
                    <Setter Property="Visibility" Value="Visible"/>
                </DataTrigger>
            </Style.Triggers>
        </Style>
    </Button.Style>
    <StackPanel Orientation="Horizontal">
        <TextBlock Text="⬇ Download"/>
        <TextBlock Text=" (" 
                   Visibility="{Binding SelectedModelDetails, Converter={StaticResource NullToVis}}"/>
        <TextBlock Text="{Binding SelectedModelDetails.TotalSizeFormatted}"
                   Visibility="{Binding SelectedModelDetails, Converter={StaticResource NullToVis}}"/>
        <TextBlock Text=")" 
                   Visibility="{Binding SelectedModelDetails, Converter={StaticResource NullToVis}}"/>
    </StackPanel>
</Button>
```

**Features:**
- Only visible for ONNX models (`IsOnnxModel == True`)
- Shows model size in button text
- Disabled during download (`IsConverting`)
- Purple color (`#6C5CE7`)

**Command:** `DownloadOnnxModelCommand` → `DownloadOnnxModelAsync()`

---

### 6. Status Message (Lines 355-359)

```xml
<Border Background="#1A1A1A" CornerRadius="4" Padding="10" Margin="0,0,0,12">
    <TextBlock Text="{Binding StatusMessage}" 
               Foreground="#808080" TextWrapping="Wrap"/>
</Border>
```

**Updates:**
- "Ready" (initial state)
- "Downloading {model} with Olive..."
- "Downloading {count} files ({size})..." (from metadata)
- "✓ Downloaded to {path}" (success)
- "✗ Download failed" (error)

---

## Download Flow

### 1. User Actions
1. User searches/browses models
2. Selects a model (clicks on model item)
3. Model details load (including variants if available)
4. User optionally:
   - Selects variant (if available)
   - Enters/updates HuggingFace token
   - Changes output folder
5. Clicks "⬇ Download" button

### 2. ViewModel Processing (`DownloadOnnxModelAsync`)

```csharp
// Validation
- Checks if model is selected
- Verifies model is ONNX (not source model)
- Creates output directory if needed

// Python Script Execution
- Constructs path to olive_download.py
- Finds Python executable (.olive-env or system)
- Builds command arguments:
  * --model_id {SelectedModelId}
  * --output_dir {modelOutputFolder}
  * --token {HuggingFaceToken} (if provided)
  * --subfolder {SelectedVariant} (if selected)

// Process Management
- Starts Python process
- Redirects stdout/stderr
- Parses METADATA: JSON output
- Updates status message with file count and size
- Logs all output to ConversionLog
```

### 3. Python Script (`olive_download.py`)

**Output Format:**
```
Initializing download for {model_id}...
Fetching model metadata...
METADATA: {"total_bytes": 1234567890, "file_count": 42, "files": [...]}
Downloading subset: {subfolder} (if applicable)
[OK] Model successfully saved to {path}
Validating with Olive HfModelHandler...
[OK] Olive acceptance check passed
```

**Exit Codes:**
- `0` = Success
- `1` = Error

---

## Key Features

### ✅ Variant Support
- Detects models with multiple variants/subfolders
- Shows dropdown to select specific variant
- Downloads only selected variant (saves space/time)

### ✅ Metadata Parsing
- Parses `METADATA:` JSON output from Python script
- Updates status with file count and total size
- Provides progress feedback before download starts

### ✅ Token Management
- Secure password box input
- Saved to settings
- Automatically passed to Python script

### ✅ Error Handling
- Validates model selection
- Checks output folder permissions
- Handles Python script errors
- Shows error messages in status/log

### ✅ Progress Feedback
- Status message updates
- Conversion log shows all output
- Real-time updates from Python script

---

## UI States

### **Initial State**
- Status: "Ready"
- Download button: Enabled (if ONNX model selected)
- Log: Empty

### **Downloading State**
- Status: "Downloading {count} files ({size})..."
- Download button: Disabled
- Log: Shows Python script output
- Cancel button: Visible (if conversion in progress)

### **Success State**
- Status: "✓ Downloaded to {path}"
- Download button: Enabled
- Log: Shows completion message

### **Error State**
- Status: "✗ Download failed" or error message
- Download button: Enabled
- Log: Shows error details

---

## Integration Points

### ViewModel Properties
- `SelectedModelId` - Model ID from HuggingFace
- `SelectedVariant` - Optional variant/subfolder
- `OutputFolder` - Base output directory
- `HuggingFaceToken` - Authentication token
- `StatusMessage` - Current status text
- `ConversionLog` - Output log text
- `IsConverting` - Download in progress flag
- `SelectedModelDetails` - Model metadata

### Commands
- `DownloadOnnxModelCommand` - Triggers download
- `CancelConversionCommand` - Cancels download (if supported)

### Services
- `HuggingFaceService` - Model search/details
- `ModelBuilderService` - Model conversion (not used for download)

---

## Improvements Made

### Recent Changes
1. ✅ **Fixed Python script path** - Uses `olive_download.py` (not obsolete scripts)
2. ✅ **Variant support** - Added subfolder selection for multi-variant models
3. ✅ **Metadata parsing** - Shows file count and size before download
4. ✅ **Better error handling** - Improved error messages from Python script

### Potential Future Enhancements
- [ ] Progress bar (percentage complete)
- [ ] Download speed indicator
- [ ] Pause/resume functionality
- [ ] Queue multiple downloads
- [ ] Verify download integrity after completion
- [ ] Browse button for output folder
- [ ] Remember last used output folder

---

## Code References

- **XAML:** `MelonStudio/ModelManagerControl.xaml` (lines 210-425)
- **ViewModel:** `MelonStudio/ViewModels/ModelManagerViewModel.cs` (lines 362-511)
- **Code-behind:** `MelonStudio/ModelManagerControl.xaml.cs`
- **Python Script:** `scripts/olive/olive_download.py`
