using System.Windows;
using Microsoft.Win32;
using MelonStudio.ViewModels;

namespace MelonStudio
{
    public partial class LoadModelWindow : Window
    {
        public LoadModelViewModel ViewModel { get; }
        public string? SelectedModelPath { get; private set; }
        public string? SelectedOnnxFileName { get; private set; }

        public LoadModelWindow()
        {
            InitializeComponent();
            ViewModel = new LoadModelViewModel();
            DataContext = ViewModel;
        }

        private async void BrowseFile_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog
            {
                Title = "Select ONNX Model",
                Filter = "ONNX Models|*.onnx;*.onnx.data|All Files|*.*",
                InitialDirectory = ViewModel.ModelsFolder
            };
            
            if (dialog.ShowDialog() == true)
            {
                var path = dialog.FileName;
                if (path.EndsWith(".onnx.data", System.StringComparison.OrdinalIgnoreCase))
                {
                    path = path.Substring(0, path.Length - 5); // Remove .data
                }
                await ViewModel.AnalyzeSelectedFileAsync(path);
            }
        }

        private void Refresh_Click(object sender, RoutedEventArgs e)
        {
            ViewModel.RefreshModels();
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }

        private void Load_Click(object sender, RoutedEventArgs e)
        {
            // Prefer analyzed model path (from file browse), then selected from list
            if (!string.IsNullOrEmpty(ViewModel.AnalyzedModelPath))
            {
                SelectedModelPath = ViewModel.AnalyzedModelPath;
                SelectedOnnxFileName = ViewModel.SelectedOnnxFileName;
            }
            else if (ViewModel.SelectedModel != null)
            {
                SelectedModelPath = ViewModel.SelectedModel.Path;
                SelectedOnnxFileName = null; // Will use default discovery
            }
            else
            {
                System.Windows.MessageBox.Show("Please select a model to load.", "No Model Selected",
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            ViewModel.SaveSettings();
            DialogResult = true;
            Close();
        }
    }
}
