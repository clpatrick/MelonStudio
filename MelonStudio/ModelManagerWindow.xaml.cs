using System.Windows;
using System.Windows.Controls;
using MelonStudio.Services;
using MelonStudio.ViewModels;

namespace MelonStudio
{
    public partial class ModelManagerWindow : Window
    {
        public ModelManagerViewModel ViewModel { get; }

        public ModelManagerWindow()
        {
            InitializeComponent();
            ViewModel = new ModelManagerViewModel();
            DataContext = ViewModel;
            Loaded += OnLoaded;
            Closing += OnClosing;
        }

        private async void OnLoaded(object sender, RoutedEventArgs e)
        {
            await ViewModel.InitializeAsync();
            
            // Load saved token into password box (can't bind PasswordBox directly)
            if (!string.IsNullOrEmpty(ViewModel.HuggingFaceToken))
            {
                HfTokenBox.Password = ViewModel.HuggingFaceToken;
            }
        }

        private void OnClosing(object? sender, System.ComponentModel.CancelEventArgs e)
        {
            ViewModel.SaveSettings();
        }

        private void ModelItem_DoubleClick(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            if (sender is ListBoxItem item && item.DataContext is HuggingFaceModel model)
            {
                ViewModel.SelectedModelId = model.Id;
            }
        }

        private void HfTokenBox_PasswordChanged(object sender, RoutedEventArgs e)
        {
            if (sender is PasswordBox pb)
            {
                ViewModel.HuggingFaceToken = pb.Password;
            }
        }
    }
}
