using System;
using System.Globalization;
using System.Windows.Data;

namespace MelonStudio.Converters
{
    /// <summary>
    /// Converts between an enum value and a boolean for RadioButton binding.
    /// Usage: IsChecked="{Binding EnumProperty, Converter={StaticResource EnumToBoolConverter}, ConverterParameter=EnumValue}"
    /// </summary>
    public class EnumToBoolConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null || parameter == null) return false;
            
            var enumValue = value.ToString();
            var targetValue = parameter.ToString();
            return enumValue?.Equals(targetValue, StringComparison.OrdinalIgnoreCase) == true;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is bool isChecked && isChecked && parameter != null)
            {
                return Enum.Parse(targetType, parameter.ToString()!);
            }
            return System.Windows.Data.Binding.DoNothing;
        }
    }
}
