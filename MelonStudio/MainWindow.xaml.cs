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

        private void OpenModelManager_Click(object sender, RoutedEventArgs e)
        {
            var modelManager = new ModelManagerWindow();
            modelManager.Owner = this;
            modelManager.ShowDialog();
        }
    }
}
