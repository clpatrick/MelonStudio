using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using MelonStudio.Services;
using MelonStudio.ViewModels;

namespace MelonStudio
{
    public partial class MainWindow : System.Windows.Window
    {
        public ChatViewModel ViewModel { get; }
        public ObservableCollection<ModelInfo> LocalModels { get; } = new();
        
        private readonly AppSettings _settings;

        public MainWindow()
        {
            InitializeComponent();
            ViewModel = new ChatViewModel();
            DataContext = ViewModel;
            
            _settings = AppSettings.Load();
            ModelsFolderBox.Text = _settings.DefaultOutputFolder;
            SettingsHfToken.Password = _settings.HuggingFaceToken;
            
            MyModelsList.ItemsSource = LocalModels;
        }

        // Navigation
        private void NavChat_Checked(object sender, RoutedEventArgs e)
        {
            ShowView("Chat");
        }

        private void NavModels_Checked(object sender, RoutedEventArgs e)
        {
            ShowView("Models");
            RefreshLocalModels();
        }

        private void NavDiscover_Checked(object sender, RoutedEventArgs e)
        {
            ShowView("Discover");
            LoadDiscoverView();
        }

        private void NavSettings_Checked(object sender, RoutedEventArgs e)
        {
            ShowView("Settings");
        }

        private void ShowView(string view)
        {
            if (ChatView == null) return; // Not yet initialized
            
            ChatView.Visibility = view == "Chat" ? Visibility.Visible : Visibility.Collapsed;
            ModelsView.Visibility = view == "Models" ? Visibility.Visible : Visibility.Collapsed;
            DiscoverView.Visibility = view == "Discover" ? Visibility.Visible : Visibility.Collapsed;
            SettingsView.Visibility = view == "Settings" ? Visibility.Visible : Visibility.Collapsed;
        }

        // Chat View
        private async void LoadModel_Click(object sender, RoutedEventArgs e)
        {
            var loadDialog = new LoadModelWindow();
            loadDialog.Owner = this;
            
            if (loadDialog.ShowDialog() == true && !string.IsNullOrEmpty(loadDialog.SelectedModelPath))
            {
                await ViewModel.LoadModelFromPathAsync(loadDialog.SelectedModelPath);
            }
        }

        // Models View
        private void RefreshLocalModels()
        {
            LocalModels.Clear();
            var folder = _settings.DefaultOutputFolder;
            
            if (!Directory.Exists(folder)) return;

            try
            {
                foreach (var dir in Directory.GetDirectories(folder))
                {
                    var name = Path.GetFileName(dir);
                    if (name.Equals("temp", StringComparison.OrdinalIgnoreCase)) continue;

                    var hasConfig = File.Exists(Path.Combine(dir, "genai_config.json")) ||
                                    File.Exists(Path.Combine(dir, "config.json"));
                    var hasOnnx = Directory.GetFiles(dir, "*.onnx", SearchOption.AllDirectories).Length > 0;

                    if (hasConfig || hasOnnx)
                    {
                        LocalModels.Add(new ModelInfo { Name = name, Path = dir });
                    }
                }
            }
            catch { }
        }

        private void RefreshModels_Click(object sender, RoutedEventArgs e)
        {
            RefreshLocalModels();
        }

        private void OpenModelsFolder_Click(object sender, RoutedEventArgs e)
        {
            var folder = _settings.DefaultOutputFolder;
            if (Directory.Exists(folder))
            {
                Process.Start("explorer.exe", folder);
            }
        }

        private async void ModelItem_Click(object sender, MouseButtonEventArgs e)
        {
            // Double-click to load
            if (e.ClickCount == 2 && sender is System.Windows.Controls.Border border && border.DataContext is ModelInfo model)
            {
                await LoadModelAndSwitchToChat(model.Path);
            }
        }

        private async void LoadModelFromList_Click(object sender, RoutedEventArgs e)
        {
            if (sender is System.Windows.Controls.Button btn && btn.Tag is string path)
            {
                await LoadModelAndSwitchToChat(path);
            }
        }

        private async Task LoadModelAndSwitchToChat(string path)
        {
            await ViewModel.LoadModelFromPathAsync(path);
            NavChat.IsChecked = true;
            ShowView("Chat");
        }

        // Discover View
        private void LoadDiscoverView()
        {
            if (DiscoverView.Children.Count == 0)
            {
                // Embed the ModelManager content into Discover view
                var modelManager = new ModelManagerControl();
                DiscoverView.Children.Add(modelManager);
            }
        }

        // Settings View
        private void BrowseModelsFolder_Click(object sender, RoutedEventArgs e)
        {
            using var dialog = new System.Windows.Forms.FolderBrowserDialog();
            dialog.SelectedPath = ModelsFolderBox.Text;
            
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                ModelsFolderBox.Text = dialog.SelectedPath;
            }
        }

        private void SaveSettings_Click(object sender, RoutedEventArgs e)
        {
            _settings.DefaultOutputFolder = ModelsFolderBox.Text;
            _settings.HuggingFaceToken = SettingsHfToken.Password;
            _settings.Save();
            
            System.Windows.MessageBox.Show("Settings saved!", "MelonStudio", 
                MessageBoxButton.OK, MessageBoxImage.Information);
        }

        private void OpenModelManager_Click(object sender, RoutedEventArgs e)
        {
            // Legacy - switch to Discover view instead
            NavDiscover.IsChecked = true;
            ShowView("Discover");
        }
    }
}
