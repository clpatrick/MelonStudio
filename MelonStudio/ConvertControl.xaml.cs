using System.Windows;
using System.Windows.Controls;
using MelonStudio.ViewModels;

namespace MelonStudio
{
    public partial class ConvertControl : System.Windows.Controls.UserControl
    {
        public ConvertViewModel ViewModel { get; }

        public ConvertControl()
        {
            InitializeComponent();
            ViewModel = new ConvertViewModel();
            DataContext = ViewModel;
            Loaded += OnLoaded;
        }

        private async void OnLoaded(object sender, RoutedEventArgs e)
        {
            await ViewModel.LoadLocalModelsAsync();
            await ViewModel.LoadTempModelsAsync();
            
            // Load saved token
            if (!string.IsNullOrEmpty(ViewModel.HuggingFaceToken))
            {
                HfTokenBox.Password = ViewModel.HuggingFaceToken;
            }
        }

        private void HfTokenBox_PasswordChanged(object sender, RoutedEventArgs e)
        {
            if (sender is PasswordBox pb)
            {
                ViewModel.HuggingFaceToken = pb.Password;
            }
        }

        /// <summary>
        /// Sets up the view for converting a specific model.
        /// </summary>
        public void SetModelForConversion(string modelPath, string modelName)
        {
            ViewModel.LocalModelPath = modelPath;
            ViewModel.SelectedSourceType = ModelSourceType.LocalPath;
            ViewModel.StatusMessage = $"Ready to convert: {modelName}";
        }
    }
}
