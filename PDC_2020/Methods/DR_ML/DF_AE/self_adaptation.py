import os
import sys
import numpy as np
import math

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
import pm4py.algo.conformance.tokenreplay as tokenreplay
import pm4py.algo.discovery as pdiscovery
import pm4py

import pandas as pd

input_sacycle_dir = "Input/SACycle/SA/"
input_sacycle_model_dir = input_sacycle_dir + "Model/"
input_sacycle_eventlogs_dir = input_sacycle_dir + "EventLogs/"


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
	
def compute_fitness(petri_net, event_log, cc_variant):

	log_fitness = 0.0
	aligned_traces = None
	parameters = {}
	parameters[log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY] = 'CaseID'
	
	if cc_variant == "ALIGNMENT_BASED":
		aligned_traces = alignments.apply_log(event_log, petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
		log_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]
	elif cc_variant == "TOKEN_BASED":
		replay_results = tokenreplay.algorithm.apply(log = event_log, net = petri_net["network"], initial_marking = petri_net["initial_marking"], final_marking = petri_net["final_marking"], parameters = parameters, variant = tokenreplay.algorithm.Variants.TOKEN_REPLAY)
		log_fitness = replay_fitness.evaluate(results = replay_results, variant = replay_fitness.Variants.TOKEN_BASED)["log_fitness"]



	return log_fitness, aligned_traces

def get_activities(transitions):
	
	activities = []
	
	for transition in transitions:
		if transition._Transition__get_label() != None:
			activities.append(transition._Transition__get_label())

	activites = list(set(activities))

	return activities
	
def compute_unaligned_activities(event_log, aligned_traces):
	
	unaligned_activities = {}

	events = {}
	temp = []
	for aligned_trace in aligned_traces:
		temp.append(list(aligned_trace.values())[0])
	aligned_traces = temp
	#tau_model_moves = 0
	for aligned_trace in aligned_traces:
		for move in aligned_trace:
			log_behavior = move[0]
			model_behavior = move[1]
			if log_behavior != model_behavior:
				if log_behavior != None and log_behavior != ">>":
					try:
						events[log_behavior] = events[log_behavior]+1
					except:
						events[log_behavior] = 0
				elif model_behavior != None and model_behavior != ">>":
					try:
						events[model_behavior] = events[model_behavior] + 1
					except:
						events[model_behavior] = 0
			#if model_behavior == None:
			#	tau_model_moves += 1

	#events = dict(sorted(events.items(), key=lambda item: item[1]))
	while bool(events):
		popped_event = events.popitem()
		if popped_event[1] > 0:
			unaligned_activities[popped_event[0]] = popped_event[1]
	#unaligned_activities["tau"] = tau_model_moves		

	return unaligned_activities

def get_model_logs_activities(model_activities, event_logs):

	model_logs_activities = model_activities
	
	for event_log in event_logs:
		event_log_activities = []
		for trace in event_logs[event_log]:
			for event in trace:
				if event["concept:name"] not in event_log_activities:
					event_log_activities.append(event["concept:name"])
		
		for activity in event_log_activities:
			if activity not in model_logs_activities:
				model_logs_activities.append(activity)

	return model_logs_activities
	
def read_model():

	petri_net = {}
	
	petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pnml_importer.apply(input_sacycle_model_dir + "NormativeModel.pnml")
	
	return petri_net	
	
def generate_diagnoses(normative_model, event_logs, cc_variant):

	columns = []
	
	model_activities = []
	model_activities.append(get_activities(list(normative_model["network"].transitions)))
	temp = []
	max_act_set = -1
	for activities_set in model_activities:
		if len(activities_set) > max_act_set:
			max_act_set = len(activities_set)
			temp = activities_set
	model_activities = temp		


	model_diagnoses_columns = []
	model_compressed_activities = []
	model_diagnoses_columns = get_model_logs_activities(model_activities.copy(), event_logs)
	model_activities = model_diagnoses_columns.copy()
	model_diagnoses_columns.sort()
	model_unalignments = []
	model_fitness_diagnoses = []
	model_labels = []
		
	# model unalignments and fitness values are computed next
		
	for event_log in event_logs:
		log = event_logs[event_log]
		if cc_variant == "ALIGNMENT_BASED":
			try:
				fitness, aligned_traces = compute_fitness(normative_model, log, "ALIGNMENT_BASED")
				unaligned_activities = compute_unaligned_activities(log, aligned_traces)
				
				for activity in model_activities:
					try:
						unaligned_activities[activity] = unaligned_activities.pop(activity)
					except:				
						unaligned_activities[activity] = 0
					
					
				temp_list = []
				for sorted_key in sorted(unaligned_activities):
					temp_list.append(unaligned_activities[sorted_key])
				model_unalignments.append(temp_list)
			except Exception as ex:
				fitness, ignore = compute_fitness(normative_model, log, "TOKEN_BASED")
		else:
			fitness, ignore = compute_fitness(normative_model, log, "TOKEN_BASED")
		model_fitness_diagnoses.append(fitness)
		model_labels.append(event_log)
			
		# model_unalignments, model_fitness_diagnoses, and model_labels are used next	
			
	model_diagnoses = []
	if len(model_unalignments) > 0:
		for idx,el in enumerate(model_fitness_diagnoses):
			model_diagnoses.append(model_unalignments[idx] + [model_fitness_diagnoses[idx]] + [model_labels[idx]])
		model_diagnoses_columns.append("fitness")
		model_diagnoses_columns.append("label")
		model_diagnoses = pd.DataFrame(columns = model_diagnoses_columns, data = model_diagnoses)
		labels = model_diagnoses["label"]
		fitness = model_diagnoses["fitness"]
	else:
		for idx,el in enumerate(model_fitness_diagnoses):
			model_diagnoses.append([model_fitness_diagnoses[idx]] + [model_labels[idx]])
		model_diagnoses = pd.DataFrame(columns = ["fitness", "label"], data = model_diagnoses)
	model_diagnoses = model_diagnoses.reindex(sorted(model_diagnoses.columns), axis=1)	
	return model_diagnoses

def read_event_logs():
	
	event_logs = {}
	
	for event_log_filename in os.listdir(input_sacycle_eventlogs_dir):
		event_log_label = event_log_filename.split(".")[0]
		event_logs[event_log_label] = xes_importer.apply(input_sacycle_eventlogs_dir + event_log_filename)
	
	
	return event_logs		

try:
	debug_mode = int(sys.argv[1])
	n_trial = int(sys.argv[2])
	clustering_technique = sys.argv[3]
	cycle_number = int(sys.argv[4])
	normalize_compress = int(sys.argv[5])
	cc_variant = sys.argv[6]
	if normalize_compress == 1:
		normalization_technique = sys.argv[7]
		n_components = int(sys.argv[8])
		sa_method = sys.argv[9]
		if sa_method == "normal_data_distance_evaluation":
			threshold_percentage = float(sys.argv[10]) # with respect to normal data intra-cluster sum of squares
		elif sa_method == "data_clustering":
			max_clusters = int(sys.argv[10])
			if max_clusters < 2:
				print("Insert a legit number of max clusters.")
				sys.exit()
			threshold_percentage = float(sys.argv[11])
	elif normalize_compress == 0:
		sa_method = sys.argv[7]
		if sa_method == "normal_data_distance_evaluation":
			threshold_percentage = float(sys.argv[8]) # with respect to normal data intra-cluster sum of squares
		elif sa_method == "data_clustering":
			max_clusters = int(sys.argv[8])
			if max_clusters < 2:
				print("Insert a legit number of max clusters.")
				sys.exit()
			threshold_percentage = float(sys.argv[9])
		
except IndexError:
	print("Insert the right number of inputs.")
	sys.exit()
	
threshold_percentage = threshold_percentage + (n_trial-1)*0.05
event_logs = read_event_logs()
normative_model = read_model()
diagnoses = generate_diagnoses(normative_model, event_logs, cc_variant)
real_normal_labels = get_normal_labels(diagnoses, cycle_number)
if debug_mode == 0:
	if normalize_compress == 1:
		fitness_values = list(diagnoses["fitness"])
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
	
	
	
	
	
	
	
	