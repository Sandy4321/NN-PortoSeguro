import pandas as pd
import data_visualize_helper as hlp
import dataset_utils as dsu

if __name__ == "__main__":
	#IMPORTING DATASET
	dataframe = pd.read_csv('../data/train.csv')

	#DESCRIBING DATASET
	dsu.describe_dataset(dataframe)

	# HOW MANY CLASSES DO WE HAVE?
	dsu.print_classes(dataframe)

	# WHAT IS THE CLASSES DISTRIBUTION
	hlp.plot_classes_distribution(dataframe)

	# ARE THERE NULL VALUES ON DATASET?
	dsu.check_null_values(dataframe)

	# WHAT DATA TYPES DO WE HAVE ON DATASET?
	dsu.check_data_types(dataframe)

	# LET'S CHECK THE CORRELATION BETWEEN FEATURES
	hlp.plot_features_corr(dataframe)

	# AS THE DATASET IS IMBALANCED, WE'LL RESAMPLE IT
	rdataframe = dsu.undersample(dataframe, 25000)

	# CLASSES DISTRIBUTION AFTER RESAMPLING
	hlp.plot_classes_distribution(rdataframe)
	
