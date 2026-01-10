using System;
using System.Globalization;
using System.Windows.Data;

namespace MelonStudio.Converters
{
    public class HybridIconConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is bool isHybrid)
            {
                return isHybrid ? "🔀" : "📦";
            }
            return "📦";
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
