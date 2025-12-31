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
    }
}
