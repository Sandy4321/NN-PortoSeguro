import pandas as pd
import numpy as np
from collections import Counter

def check_data_types(dataframe):
	"""

	"""
	print('\nDatatypes inside this dataset:')
	print('    ' + str(Counter(dataframe.dtypes.values)))

def check_null_values(dataframe):
	"""

	"""
	print("Null values on dataset? " + str(dataframe.isnull().any().any()))


	total = dataframe.shape[0]
	df_nulls = dataframe.replace(-1, np.NaN).isnull().sum()

	print("    Check if null values were replaced by -1")

	colnames = dataframe.columns.tolist()

	print('\n    Features containing null values:')
	nulls = []
	for i in range(len(df_nulls)):
		col = df_nulls[i]
		per = (col/total)*100

		if per > 0:
			print('        %s ==> %0.2f%%' % (colnames[i], per))

def print_classes(dataframe):
	"""

	"""
	print("Classes found on dataset: " + str(dataframe['target'].unique()))


def describe_dataset(dataframe):
	"""
	
	"""
	rows = dataframe.shape[0]
	features = dataframe.shape[1]
	print("The dataset contains {0} rows and {1} features.".format(rows, features))

def undersample(dataframe, samples):
	"""

	"""
	zeros = dataframe[dataframe['target']==0].sample(n=samples)
	ones = dataframe[dataframe['target']==1]
	rdf = pd.concat([zeros,ones],axis=0)
	return rdf

