using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using MelonStudio.Models;
using MelonStudio.Services;
using MelonStudio.ViewModels;

namespace MelonStudio
{
    public partial class MainWindow : System.Windows.Window
    {
        public ChatViewModel ViewModel { get; }
        public ObservableCollection<LocalModelInfo> LocalModels { get; } = new();
        private LocalModelService? _localModelService;
        
        private readonly AppSettings _settings;

        public MainWindow()
        {
            InitializeComponent();
            ViewModel = new ChatViewModel();
            DataContext = ViewModel;
            
            _settings = AppSettings.Load();
            LoadSettingsToUI();
            
            MyModelsList.ItemsSource = LocalModels;
            
            // Auto-scroll chat when messages change
            ViewModel.Messages.CollectionChanged += (s, e) =>
            {
                Dispatcher.BeginInvoke(System.Windows.Threading.DispatcherPriority.Background, new Action(() =>
                {
                    ChatScrollViewer.ScrollToEnd();
                }));
            };
        }

        private void LoadSettingsToUI()
        {
            ModelsFolderBox.Text = _settings.DefaultOutputFolder;
            SettingsHfToken.Password = _settings.HuggingFaceToken;
            MaxLengthBox.Text = _settings.MaxLength.ToString();
            TemperatureBox.Text = _settings.Temperature.ToString();
            TopPBox.Text = _settings.TopP.ToString();
            SystemPromptBox.Text = _settings.SystemPrompt;
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

        private void NavConvert_Checked(object sender, RoutedEventArgs e)
        {
            ShowView("Convert");
            LoadConvertView();
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
            ConvertView.Visibility = view == "Convert" ? Visibility.Visible : Visibility.Collapsed;
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

        private void InputTextBox_PreviewKeyDown(object sender, System.Windows.Input.KeyEventArgs e)
        {
            if (e.Key == Key.Enter || e.Key == Key.Return)
            {
                if (Keyboard.Modifiers.HasFlag(ModifierKeys.Shift))
                {
                    // Shift+Enter: Insert newline
                    var textBox = (System.Windows.Controls.TextBox)sender;
                    var caretIndex = textBox.CaretIndex;
                    textBox.Text = textBox.Text.Insert(caretIndex, Environment.NewLine);
                    textBox.CaretIndex = caretIndex + Environment.NewLine.Length;
                    e.Handled = true;
                }
                else
                {
                    // Enter: Send message
                    if (ViewModel.SendMessageCommand.CanExecute(null))
                    {
                        ViewModel.SendMessageCommand.Execute(null);
                    }
                    e.Handled = true;
                }
            }
        }

        // Models View
        private async void RefreshLocalModels()
        {
            LocalModels.Clear();
            var folder = _settings.DefaultOutputFolder;
            
            if (!Directory.Exists(folder)) return;

            try
            {
                _localModelService ??= new LocalModelService(folder);
                var models = await _localModelService.ScanModelsFolderAsync();
                
                foreach (var model in models)
                {
                    LocalModels.Add(model);
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

        private void ConvertLocalModel_Click(object sender, RoutedEventArgs e)
        {
            if (sender is System.Windows.Controls.Button btn && btn.Tag is string path)
            {
                // Switch to Convert view and set the model for conversion
                NavConvert.IsChecked = true;
                ShowView("Convert");
                LoadConvertView();
                
                // Get the convert control and set up for local conversion
                if (ConvertView.Children[0] is ConvertControl convertControl)
                {
                    var modelName = Path.GetFileName(path);
                    convertControl.SetModelForConversion(path, modelName);
                }
            }
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

        // Convert View
        private void LoadConvertView()
        {
            if (ConvertView.Children.Count == 0)
            {
                var convertControl = new ConvertControl();
                ConvertView.Children.Add(convertControl);
            }
        }

        // Settings View
        private void BrowseModelsFolder_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Microsoft.Win32.OpenFolderDialog
            {
                Title = "Select Models Folder",
                InitialDirectory = ModelsFolderBox.Text
            };
            
            if (dialog.ShowDialog() == true)
            {
                ModelsFolderBox.Text = dialog.FolderName;
            }
        }

        private void SaveSettings_Click(object sender, RoutedEventArgs e)
        {
            _settings.DefaultOutputFolder = ModelsFolderBox.Text;
            _settings.HuggingFaceToken = SettingsHfToken.Password;
            
            if (int.TryParse(MaxLengthBox.Text, out int maxLength))
                _settings.MaxLength = maxLength;
            if (double.TryParse(TemperatureBox.Text, out double temp))
                _settings.Temperature = temp;
            if (double.TryParse(TopPBox.Text, out double topP))
                _settings.TopP = topP;
            _settings.SystemPrompt = SystemPromptBox.Text;
            
            _settings.Save();
            
            // Update ViewModel with new settings
            ViewModel.UpdateSettings(_settings);
            
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
