import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

def plot_classes_distribution(dataframe):
	"""

	Displays how the classes are distributed

	"""
	thist = dataframe.groupby(['target'], as_index=False).count()['id']

	print('Class value = 0 --> %s examples.' % (thist[0]))
	print('Class value = 1 --> %s examples.' % (thist[1]))
	print('Classes ratio is %d:1.' % (math.floor(thist[0]/thist[1])))

	x_axis = np.arange(2)
	fig, ax = plt.subplots()

	ax.set_ylabel('Labels Distribution')
	ax.set_title('Scores by Labels')
	plt.bar(x_axis, thist)
	plt.xticks(x_axis, ('Negative', 'Positive'))
	plt.show()

def plot_features_corr(dataframe):
	"""
	
	Displays the features Pearson correlation value

	"""
	df_float = dataframe.select_dtypes(include=['float64'])
	df_int = dataframe.select_dtypes(include=['int64'])

	colormap = plt.cm.afmhot
	plt.figure(figsize=(16,12))
	plt.title('Pearson correlation of continuous features', y=1.05, size=15)
	sns.heatmap(df_float.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
	plt.show()