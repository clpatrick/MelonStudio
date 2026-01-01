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

        private void ModelItem_Click(object sender, MouseButtonEventArgs e)
        {
            if (sender is System.Windows.Controls.Border border && border.DataContext is HuggingFaceModel model)
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
