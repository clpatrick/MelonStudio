using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using MelonStudio.Services;
using MelonStudio.ViewModels;

namespace MelonStudio
{
    public partial class ModelManagerControl : System.Windows.Controls.UserControl
    {
        public ModelManagerViewModel ViewModel { get; }

        public ModelManagerControl()
        {
            InitializeComponent();
            ViewModel = new ModelManagerViewModel();
            DataContext = ViewModel;
            Loaded += OnLoaded;
            KeyDown += ModelManagerControl_KeyDown;
        }

        private void ModelManagerControl_KeyDown(object sender, System.Windows.Input.KeyEventArgs e)
        {
            if (e.Key == Key.Escape)
            {
                ClearModelSelection();
                e.Handled = true;
            }
        }

        private void ClearModelSelection()
        {
            ViewModel.SelectedModelDetails = null;
            ViewModel.SelectedModelId = "";
            ViewModel.ShowConvertButton = false;
            ViewModel.DownloadedModelPath = null;
            ViewModel.Variants.Clear();
            ViewModel.SelectedVariant = null;
            ViewModel.HasVariants = false;
        }

        private async void OnLoaded(object sender, RoutedEventArgs e)
        {
            await ViewModel.InitializeAsync();
            
            // Load saved token into password box
            if (!string.IsNullOrEmpty(ViewModel.HuggingFaceToken))
            {
                HfTokenBox.Password = ViewModel.HuggingFaceToken;
            }
        }

        private async void ModelItem_Click(object sender, MouseButtonEventArgs e)
        {
            if (sender is System.Windows.Controls.Border border && border.DataContext is HuggingFaceModel model)
            {
                await ViewModel.SelectModelCommand.ExecuteAsync(model);
            }
        }

        private void HfTokenBox_PasswordChanged(object sender, RoutedEventArgs e)
        {
            if (sender is PasswordBox pb)
            {
                ViewModel.HuggingFaceToken = pb.Password;
            }
        }

        private void SearchBox_KeyDown(object sender, System.Windows.Input.KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                ViewModel.SearchModelsCommand.Execute(null);
            }
        }

        /// <summary>
        /// Sets up the view for converting a local model.
        /// </summary>
        public void SetLocalModelForConversion(string modelPath, string modelName)
        {
            ViewModel.SetLocalModelForConversion(modelPath, modelName);
        }
    }
}
