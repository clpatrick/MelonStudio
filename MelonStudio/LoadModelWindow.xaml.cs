using System.Windows;
using Microsoft.Win32;
using MelonStudio.ViewModels;
using System.Windows.Forms;

namespace MelonStudio
{
    public partial class LoadModelWindow : Window
    {
        public LoadModelViewModel ViewModel { get; }
        public string? SelectedModelPath { get; private set; }

        public LoadModelWindow()
        {
            InitializeComponent();
            ViewModel = new LoadModelViewModel();
            DataContext = ViewModel;
        }

        private void BrowseFolder_Click(object sender, RoutedEventArgs e)
        {
            using var dialog = new FolderBrowserDialog();
            dialog.SelectedPath = ViewModel.ModelsFolder;
            dialog.Description = "Select Models Folder";
            
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                ViewModel.ModelsFolder = dialog.SelectedPath;
                ViewModel.RefreshModels();
                ViewModel.SaveSettings();
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
            if (ViewModel.SelectedModel == null)
            {
                System.Windows.MessageBox.Show("Please select a model to load.", "No Model Selected",
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            SelectedModelPath = ViewModel.SelectedModel.Path;
            ViewModel.SaveSettings();
            DialogResult = true;
            Close();
        }
    }
}
