using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;
using MelonStudio.Models;

namespace MelonStudio.Converters
{
    /// <summary>
    /// Converts ModelStatus to a brush color for display.
    /// </summary>
    public class StatusToColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is ModelStatus status)
            {
                var colorHex = status switch
                {
                    ModelStatus.Downloaded => "#FFC107",
                    ModelStatus.Converting => "#17A2B8",
                    ModelStatus.Converted => "#28A745",
                    ModelStatus.Failed => "#DC3545",
                    _ => "#808080"
                };
                return new SolidColorBrush((System.Windows.Media.Color)System.Windows.Media.ColorConverter.ConvertFromString(colorHex));
            }
            return new SolidColorBrush(Colors.Gray);
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
