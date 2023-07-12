import data_visualization_module.data_visualization as vis
import data_exploration as de
import data_set_optimization as dso


# Get Data
data = de.read_data("House_Rent_Dataset.csv",parse_dates=True)
data_with_normal_deviation = dso.get_data_with_normal_deviation(data, "Rent")


print(data_with_normal_deviation.head())

#vis.scatter_plot_colored_with_multiple_regression_lines('BHK', 'Rent', hue='Area Type', data=data_with_normal_deviation)
vis.barchart(data=data_with_normal_deviation, x='Floor', y='Rent', title='Floor vs Rent')

# Conclusion: BHK, City, Furnishing Status, Point of Contact, Area Type are the most important
# Family intermediate, Size
# Strange: Bathroom, Floor
# Doesnt make sense: Posted On,

