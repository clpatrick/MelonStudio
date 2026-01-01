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
        }

        private async void OnLoaded(object sender, RoutedEventArgs e)
        {
            await ViewModel.InitializeAsync();
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
