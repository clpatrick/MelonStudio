using System.Windows;
using MelonStudio.ViewModels;

namespace MelonStudio
{
    public partial class MainWindow : Window
    {
        public ChatViewModel ViewModel { get; }

        public MainWindow()
        {
            InitializeComponent();
            ViewModel = new ChatViewModel();
            DataContext = ViewModel;
        }

        private async void LoadModel_Click(object sender, RoutedEventArgs e)
        {
            var loadDialog = new LoadModelWindow();
            loadDialog.Owner = this;
            
            if (loadDialog.ShowDialog() == true && !string.IsNullOrEmpty(loadDialog.SelectedModelPath))
            {
                await ViewModel.LoadModelFromPathAsync(loadDialog.SelectedModelPath);
            }
        }

        private void OpenModelManager_Click(object sender, RoutedEventArgs e)
        {
            var modelManager = new ModelManagerWindow();
            modelManager.Owner = this;
            modelManager.ShowDialog();
        }
    }
}
