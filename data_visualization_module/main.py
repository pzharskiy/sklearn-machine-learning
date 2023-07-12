import data_exploration as de
import data_visualization as dv

'''
fifa_data = de.read_data(filename="fifa.csv", index_col="Date", parse_dates=True)
dv.linechart(data=fifa_data,
             figsize=(14, 6))

ign_scores_data = de.read_data("ign_scores.csv", index_col="Platform")
dv.barchart(data=ign_scores_data,
            x=ign_scores_data.index,
            y=ign_scores_data["Racing"],
            title="Bar Chart for Average Score Of Racing Games on every platform")

ign_data = de.read_data("ign_scores.csv", index_col="Platform")

dv.heatmap(data=ign_data,
        figsize=(10, 10),
        horizontal_label="Genre",
        title="Average Game Score, by Platform and Genre",
        annotations=True)
'''


insurance_data = de.read_data("insurance.csv")
#dv.scutter_plot(x=insurance_data['bmi'], y=insurance_data['charges'])
#dv.regression_scutter_plot(x=insurance_data['bmi'], y=insurance_data['charges'])
#dv.scutter_plot_colored(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
#dv.scatter_plot_colored_with_multiple_regression_lines(x="bmi", y="charges", hue="smoker", data=insurance_data)
#dv.categorical_scatter_plot(x=insurance_data['smoker'],y=insurance_data['charges'])
#dv.categorical_strip_plot(x=insurance_data['smoker'],y=insurance_data['charges'])

iris_data = de.read_data("iris.csv")
#dv.histogram(iris_data['Petal Length (cm)'])
#dv.kernel_density_estimate(iris_data['Petal Length (cm)'])
dv.two_dimensional_kernel_density_estimate(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'])
#dv.colored_histogram(data=iris_data, x='Petal Length (cm)', hue='Species', title="Histogram of Petal Lengths, by Species")
#dv.colored_kernel_density_estimate(data=iris_data, x='Petal Length (cm)', hue='Species', title="Disctribution of Petal Lengths, by Species")