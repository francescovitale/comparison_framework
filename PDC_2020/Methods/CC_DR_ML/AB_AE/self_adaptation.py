import os
import sys
import numpy as np
import math

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

import pandas as pd

input_sacycle_dir = "Input/SACycle/SA/"
input_sacycle_diagnoses_dir = input_sacycle_dir + "Diagnoses/"


output_sacycle_dir = "Output/SACycle/SA/"
output_sacycle_normallabels_dir = output_sacycle_dir + "NormalLabels/"

clustering_technique = ""
n_components = 2
normalization_technique="zscore"
n_clusters = -1
max_clusters = -1


def read_diagnoses():

	diagnoses = pd.read_csv(input_sacycle_diagnoses_dir + "diagnoses.csv")

	return diagnoses
	
def compute_wcss(clustered_dataset, centroids):

	wcss = []
	
	for cluster in centroids:
		temp = []
		cluster_data_points = clustered_dataset.loc[(clustered_dataset["Cluster"] == cluster)]
		cluster_data_points = cluster_data_points.drop(labels=["Cluster"], axis=1)
		centroid_coordinates = np.array([float(i) for i in centroids[cluster]])
		for idx,data_point in cluster_data_points.iterrows():
			data_point = np.array(data_point)
			temp.append(np.linalg.norm(data_point-centroid_coordinates))
		wcss.append(sum([i ** 2 for i in temp]))
	
	
	
	return sum(wcss)
	
def compute_bcss(clustered_dataset, centroids):

	bcss = []
	
	for cluster in centroids:
		temp = 0.0
		cluster_data_points = clustered_dataset.loc[(clustered_dataset["Cluster"] == cluster)]
		cluster_data_points = cluster_data_points.drop(labels=["Cluster"], axis=1)
		centroid_coordinates = np.array([float(i) for i in centroids[cluster]])
		cluster_average = []
		for column in cluster_data_points:
			cluster_average.append(sum(list(cluster_data_points[column]))/len(list(cluster_data_points[column])))
		
		temp = np.linalg.norm(cluster_average-centroid_coordinates)
		temp = temp*temp
		
		bcss.append(len(cluster_data_points) * temp)


	return sum(bcss)
	
def get_optimal_n_clusters(dataset, clustering_algorithm, max_clusters):

	best_clustered_dataset = None
	best_clustering_parameters = None
	
	
	
	per_iteration_clustered_dataset = []
	per_iteration_clustering_parameters = []
	per_iteration_wcss = []
	per_iteration_bcss = []
	

	for i in range(2, max_clusters+1):
		
		clustered_dataset, clustering_parameters = cluster_dataset(dataset, 0, i, None)
		
		per_iteration_clustered_dataset.append(clustered_dataset.copy())
		per_iteration_clustering_parameters.append(clustering_parameters.copy())
		
		per_iteration_wcss.append(compute_wcss(clustered_dataset, clustering_parameters))
		per_iteration_bcss.append(compute_bcss(clustered_dataset, clustering_parameters))
	
	
	
	
	temp_per_iteration_bcss = per_iteration_bcss.copy()
	temp_per_iteration_wcss = per_iteration_wcss.copy()
	temp_per_iteration_wcss.sort()
	
	# the median of per_iteration_wcss is evaluated and used to evaluate the best choice
	wcss_median = temp_per_iteration_wcss[math.ceil(len(temp_per_iteration_wcss)/2)]
	
	temp_per_iteration_wcss = per_iteration_wcss.copy()
	
	found = False
	i = 0
	
	while found != True and i < len(per_iteration_wcss):
		i = i+1
		max_bcss_value = max(temp_per_iteration_bcss)
		max_bcss_idx = temp_per_iteration_bcss.index(max_bcss_value)
		if temp_per_iteration_wcss[max_bcss_idx] <= wcss_median:
			found = True
		else:
			del temp_per_iteration_wcss[max_bcss_idx]
			del temp_per_iteration_bcss[max_bcss_idx]
	
	# evaluate the best choice for n_clusters
	if found == True:
		max_bcss_idx = per_iteration_bcss.index(max_bcss_value)
		wcss_value = per_iteration_wcss[max_bcss_idx]
		best_clustered_dataset = per_iteration_clustered_dataset[max_bcss_idx]
		best_clustering_parameters = per_iteration_clustering_parameters[max_bcss_idx]
		n_clusters = max_bcss_idx+1+2 # +1 accounts for indexes starting from 0, whereas +2 accounts for the starting index of the for cycle
		return n_clusters, best_clustered_dataset, best_clustering_parameters
	else:
		raise Exception("Could not find any optimal number of clusters")

def compress_dataset(dataset, reuse_parameters, compression_parameters_in):
	compressed_dataset = dataset.copy()
	compression_parameters = None

	if reuse_parameters == 0:
		compression_parameters = PCA(n_components=n_components)
		compressed_dataset = compression_parameters.fit_transform(compressed_dataset)
		columns = []
		for i in range(0, n_components):
			columns.append("f_"+ str(i))
		compressed_dataset = pd.DataFrame(data=compressed_dataset, columns=columns)
	else:
		compressed_dataset = compression_parameters_in.transform(compressed_dataset)
		columns = []
		for i in range(0, n_components):
			columns.append("f_"+ str(i))
		compressed_dataset = pd.DataFrame(data=compressed_dataset, columns=columns)

	return compressed_dataset, compression_parameters
	
def normalize_dataset(dataset, reuse_parameters, normalization_parameters_in):
	
	normalized_dataset = dataset.copy()
	normalization_parameters = {}

	if reuse_parameters == 0:
		if normalization_technique == "zscore":
			for column in normalized_dataset:
				column_values = normalized_dataset[column].values
				if np.any(column_values) == True:
					column_values_mean = np.mean(column_values)
					column_values_std = np.std(column_values)
					if column_values_std != 0:
						column_values = (column_values - column_values_mean)/column_values_std
				else:
					column_values_mean = 0
					column_values_std = 0
				
				normalized_dataset[column] = column_values
				normalization_parameters[column+"_mean"] = column_values_mean
				normalization_parameters[column+"_std"] = column_values_std

	else:
		if normalization_technique == "zscore":
			for label in normalized_dataset:
				mean = normalization_parameters_in[label+"_mean"]
				std = normalization_parameters_in[label+"_std"]
				parameter_values = normalized_dataset[label].values
				if std != 0:
					parameter_values = (parameter_values - float(mean))/float(std)
				normalized_dataset[label] = parameter_values
	
	return normalized_dataset, normalization_parameters	
	
def cluster_dataset(dataset, reuse_parameters, n_clusters, clustering_parameters_in):
	clustered_dataset = dataset.copy()
	clustering_parameters = {}

	
	if reuse_parameters == 0:
		if clustering_technique == "agglomerative":
			cluster_configuration = AgglomerativeClustering(n_clusters=n_clusters, affinity='cityblock', linkage='average')
			cluster_labels = cluster_configuration.fit_predict(clustered_dataset)
		elif clustering_technique == "kmeans":
			kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(clustered_dataset)
			cluster_labels = kmeans.labels_

		clustered_dataset["Cluster"] = cluster_labels
		cluster_labels = cluster_labels.tolist()
		used = set();
		clusters = [x for x in cluster_labels if x not in used and (used.add(x) or True)]

		instances_sets = {}
		centroids = {}
		
		for cluster in clusters:
			instances_sets[cluster] = []
			centroids[cluster] = []
		
		temp = clustered_dataset
		for index, row in temp.iterrows():
			instances_sets[int(row["Cluster"])].append(row.values.tolist())
		
		n_features_per_instance = len(instances_sets[0][0])-1
		
		for instances_set_label in instances_sets:
			instances = instances_sets[instances_set_label]
			for idx, instance in enumerate(instances):
				instances[idx] = instance[0:n_features_per_instance]
			for i in range(0,n_features_per_instance):
				values = []
				for instance in instances:
					values.append(instance[i])
				centroids[instances_set_label].append(np.mean(values))
				
		clustering_parameters = centroids
			
	else:
		clusters = []
		for index, instance in clustered_dataset.iterrows():
			min_value = float('inf')
			min_centroid = -1
			for centroid in clustering_parameters_in:
				centroid_coordinates = np.array([float(i) for i in clustering_parameters_in[centroid]])
				dist = np.linalg.norm(instance.values-centroid_coordinates)
				if dist<min_value:
					min_value = dist
					min_centroid = centroid
			clusters.append(min_centroid)
		
		clustered_dataset["Cluster"] = clusters
		

	return clustered_dataset, clustering_parameters	
	
def evaluate_threshold_diagnoses(diagnoses, sa_method, threshold_percentage):

	normal_labels = []
	
	threshold = math.ceil(len(diagnoses)*threshold_percentage)
	
	if sa_method == "data_clustering":
		for i in range(0, n_clusters):
			per_cluster_diagnoses = diagnoses.loc[diagnoses["Cluster"]==i]
			if len(per_cluster_diagnoses) >= threshold:
				for label in list(per_cluster_diagnoses["label"]):
					normal_labels.append(label)
					
	elif sa_method == "normal_data_distance_evaluation":
		pass
	

	return normal_labels
	
def get_normal_labels(diagnoses, cycle_number):

	normal_labels = []
	
	labels = list(diagnoses["label"])
	
	if cycle_number == 1:
		for label in labels:
			if label.find("Second_wave") != -1:
				normal_labels.append(label)
				
	elif cycle_number == 2:
		for label in labels:
			if label.find("Third_wave") != -1:
				normal_labels.append(label)
				
	elif cycle_number == 3:
		for label in labels:
			if label.find("Second_wave") != -1:
				normal_labels.append(label)

	elif cycle_number == 4:
		for label in labels:
			if label.find("Second_wave") != -1:
				normal_labels.append(label)
				
	elif cycle_number == 5:
		for label in labels:
			if label.find("First_wave") != -1:
				normal_labels.append(label)				
				
				
	return normal_labels
	
def write_normal_labels(speculated_normal_labels, real_normal_labels, debug_mode):


	if debug_mode==1:
		file = open(output_sacycle_normallabels_dir + "NormalLabels.txt", "w")
		for idx,label in enumerate(real_normal_labels):
			if idx < len(real_normal_labels)-1:
				file.write(label + "\n")
			else:
				file.write(label)
		file.close()
	
	elif debug_mode==0:
		file = open(output_sacycle_normallabels_dir + "Real_NormalLabels.txt", "w")
		for idx,label in enumerate(real_normal_labels):
			if idx < len(real_normal_labels)-1:
				file.write(label + "\n")
			else:
				file.write(label)
		file.close()
		
		file = open(output_sacycle_normallabels_dir + "NormalLabels.txt", "w")
		for idx,label in enumerate(speculated_normal_labels):
			if idx < len(speculated_normal_labels)-1:
				file.write(label + "\n")
			else:
				file.write(label)
		file.close()
		

	return None

try:
	debug_mode = int(sys.argv[1])
	n_trial = int(sys.argv[2])
	clustering_technique = sys.argv[3]
	cycle_number = int(sys.argv[4])
	normalize_compress = int(sys.argv[5])
	if normalize_compress == 1:
		normalization_technique = sys.argv[6]
		n_components = int(sys.argv[7])
		sa_method = sys.argv[8]
		if sa_method == "normal_data_distance_evaluation":
			threshold_percentage = float(sys.argv[9]) # with respect to normal data intra-cluster sum of squares
		elif sa_method == "data_clustering":
			max_clusters = int(sys.argv[9])
			if max_clusters < 2:
				print("Insert a legit number of max clusters.")
				sys.exit()
			threshold_percentage = float(sys.argv[10])
	elif normalize_compress == 0:
		sa_method = sys.argv[6]
		if sa_method == "normal_data_distance_evaluation":
			threshold_percentage = float(sys.argv[7]) # with respect to normal data intra-cluster sum of squares
		elif sa_method == "data_clustering":
			max_clusters = int(sys.argv[7])
			if max_clusters < 2:
				print("Insert a legit number of max clusters.")
				sys.exit()
			threshold_percentage = float(sys.argv[8])
		
except IndexError:
	print("Insert the right number of inputs.")
	sys.exit()
	
threshold_percentage = threshold_percentage + (n_trial-1)*0.05
	
diagnoses = read_diagnoses()	
real_normal_labels = get_normal_labels(diagnoses, cycle_number)
	
if debug_mode == 0:
	if normalize_compress == 1:
		fitness_values = list(diagnoses["PN_first_wave_fitness"])
		labels = list(diagnoses["label"])
		diagnoses = diagnoses.drop(["label"], axis=1)
		diagnoses, ignore = normalize_dataset(diagnoses, 0, None)
		diagnoses, ignore = compress_dataset(diagnoses, 0, None)	
	try:
		n_clusters, clustered_diagnoses, diagnoses_clusters = get_optimal_n_clusters(diagnoses, clustering_technique, max_clusters)
		
	except Exception as e:
		print(e)
	clustered_diagnoses["label"] = labels
	speculated_normal_labels = evaluate_threshold_diagnoses(clustered_diagnoses, sa_method, threshold_percentage)
	write_normal_labels(speculated_normal_labels, real_normal_labels, debug_mode)
elif debug_mode == 1:
	write_normal_labels(None, real_normal_labels, debug_mode)	

	
	
	
	
	
	
	
	