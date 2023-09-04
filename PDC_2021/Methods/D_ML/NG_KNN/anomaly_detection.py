import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import numpy as np
import pandas as pd
import warnings
import random
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
import pm4py.algo.conformance.tokenreplay as tokenreplay
import pm4py.algo.discovery as pdiscovery
import pm4py
import time

from sklearn.neighbors import NearestNeighbors

from sklearn.decomposition import PCA

from itertools import product

import pickle

input_firstrun_dir = "Input/FirstRun/AD/"
output_firstrun_dir = "Output/FirstRun/AD/"

input_firstrun_training_dir = input_firstrun_dir + "Training/"
input_firstrun_training_eventlogs_dir = input_firstrun_training_dir + "EventLogs/"
input_firstrun_inference_dir = input_firstrun_dir + "Inference/"
input_firstrun_inference_model_dir = input_firstrun_inference_dir + "Model/"
input_firstrun_inference_eventlogs_dir = input_firstrun_inference_dir + "EventLogs/"
input_firstrun_inference_normalization_dir = input_firstrun_inference_dir + "Normalization/"
input_firstrun_inference_compression_dir = input_firstrun_inference_dir + "Compression/"

output_firstrun_training_dir = output_firstrun_dir + "Training/"
output_firstrun_training_model_dir = output_firstrun_training_dir + "Model/"
output_firstrun_training_normalization_dir = output_firstrun_training_dir + "Normalization/"
output_firstrun_training_compression_dir = output_firstrun_training_dir + "Compression/"
output_firstrun_inference_dir = output_firstrun_dir + "Inference/"
output_firstrun_inference_timing_dir = output_firstrun_inference_dir + "Timing/"


input_sacycle_dir = "Input/SACycle/AD/"
output_sacycle_dir = "Output/SACycle/AD/"

input_sacycle_training_dir = input_sacycle_dir + "Training/"
input_sacycle_training_eventlogs_dir = input_sacycle_training_dir + "EventLogs/"
input_sacycle_inference_dir = input_sacycle_dir + "Inference/"
input_sacycle_inference_model_dir = input_sacycle_inference_dir + "Model/"
input_sacycle_inference_eventlogs_dir = input_sacycle_inference_dir + "EventLogs/"
input_sacycle_inference_normalization_dir = input_sacycle_inference_dir + "Normalization/"
input_sacycle_inference_compression_dir = input_sacycle_inference_dir + "Compression/"


output_sacycle_training_dir = output_sacycle_dir + "Training/"
output_sacycle_training_model_dir = output_sacycle_training_dir + "Model/"
output_sacycle_training_normalization_dir = output_sacycle_training_dir + "Normalization/"
output_sacycle_training_compression_dir = output_sacycle_training_dir + "Compression/"
output_sacycle_inference_dir = output_sacycle_dir + "Inference/"
output_sacycle_inference_timing_dir = output_sacycle_inference_dir + "Timing/"

def write_timing(tm, run_mode):
	if run_mode == "FirstRun":
		file = open(output_firstrun_inference_timing_dir + "timing.txt", "a+")
		file.write(str(tm) + "\n")
		file.close()
	elif run_mode == "SACycle":
		file = open(output_sacycle_inference_timing_dir + "timing.txt", "a+")
		file.write(str(tm) + "\n")
		file.close()

def join_event_logs(event_logs):
		
	joined_event_log = []	
		
	for event_log in event_logs:
		
		for trace in event_log:
			joined_event_log.append(trace)
		
	return (pm4py.objects.log.obj.EventLog)(joined_event_log)	
	

def read_diagnoses(mode, run_mode):

	if run_mode == "FirstRun":
		if mode == "training":
			reference_dir = input_firstrun_training_dir
		elif mode == "inference":
			reference_dir = input_firstrun_inference_dir
		
		diagnoses = pd.read_csv(reference_dir + "diagnoses.csv")
	
	elif run_mode == "SACycle":
		if mode == "training":
			reference_dir = input_sacycle_training_dir
		elif mode == "inference":
			reference_dir = input_sacycle_inference_dir
		
		diagnoses = pd.read_csv(reference_dir + "diagnoses.csv")
	

	return diagnoses

def get_performance(normal_labels, diagnoses_labels, classifications):

	tp = 0
	tn = 0
	fp = 0
	fn = 0
	
	accuracy = 0.0
	precision = 0.0
	recall = 0.0
	f1 = 0.0

	

	for idx,label in enumerate(diagnoses_labels):
		if (diagnoses_labels[idx] in normal_labels) and classifications[idx] == "N":
			tn = tn + 1
		elif (diagnoses_labels[idx] in normal_labels) and classifications[idx] == "A":
			fp = fp + 1
		elif (diagnoses_labels[idx] not in normal_labels) and classifications[idx] == "N":
			fn = fn + 1
		elif (diagnoses_labels[idx] not in normal_labels) and classifications[idx] == "A":
			tp = tp + 1
			
	try:		
		accuracy = (tp+tn)/(tp+tn+fp+fn)
	except ZeroDivisionError:
		accuracy = 0.0
	try:
		precision = tp/(tp+fp)
	except ZeroDivisionError:
		precision = 0.0
	try:
		recall = tp/(tp+fn)
	except ZeroDivisionError:
		recall = 0.0
	try:
		f1 = 2*(precision*recall)/(precision+recall)
	except ZeroDivisionError:
		f1 = 0.0

	return accuracy, precision, recall, f1, tp, tn, fp, fn

def compress_dataset(dataset, n_pca_components, reuse_parameters, compression_parameters_in):
	compressed_dataset = dataset.copy()
	compression_parameters = None

	if reuse_parameters == 0:
		compression_parameters = PCA(n_components=n_pca_components)
		compressed_dataset = compression_parameters.fit_transform(compressed_dataset)
		columns = []
		for i in range(0, n_pca_components):
			columns.append("f_"+ str(i))
		compressed_dataset = pd.DataFrame(data=compressed_dataset, columns=columns)
	else:
		compressed_dataset = compression_parameters_in.transform(compressed_dataset)
		columns = []
		for i in range(0, n_pca_components):
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
				temp = ""
				for idx,act in enumerate(column):
					if idx<len(column)-1:
						temp = temp + act + " "
					else:
						temp = temp + act
					
					
				normalization_parameters[temp+"_mean"] = column_values_mean
				normalization_parameters[temp+"_std"] = column_values_std

	else:
		if normalization_technique == "zscore":
			for label in normalized_dataset:
				try:
					temp = ""
					for idx,act in enumerate(label):
						if idx<len(label)-1:
							temp = temp + act + " "
						else:
							temp = temp + act
					mean = normalization_parameters_in[temp+"_mean"]
					std = normalization_parameters_in[temp+"_std"]
					parameter_values = normalized_dataset[label].values
					if std != 0:
						parameter_values = (parameter_values - float(mean))/float(std)
					normalized_dataset[label] = parameter_values
				except KeyError:
					print("Label " + label + " was not found among normalization parameters. The corresponding column will be dropped.")
					normalized_dataset = normalized_dataset.drop(labels=label, axis = 1)
					pass
	
	return normalized_dataset, normalization_parameters	

def read_normal_labels(mode, run_mode):
	normal_labels = []
	
	if run_mode == "FirstRun":
		if mode == "training":
			normal_labels_file = open(input_firstrun_training_dir + "NormalLabels.txt", "r")
			normal_labels_lines = normal_labels_file.readlines()
			for line in normal_labels_lines:
				normal_labels.append(line.rstrip())
			normal_labels_file.close()
		
		elif mode == "inference":
			normal_labels_file = open(input_firstrun_inference_dir + "NormalLabels.txt", "r")
			normal_labels_lines = normal_labels_file.readlines()
			for line in normal_labels_lines:
				normal_labels.append(line.rstrip())
			normal_labels_file.close()
			
	elif run_mode == "SACycle":
		if mode == "training":
			normal_labels_file = open(input_sacycle_training_dir + "NormalLabels.txt", "r")
			normal_labels_lines = normal_labels_file.readlines()
			for line in normal_labels_lines:
				normal_labels.append(line.rstrip())
			normal_labels_file.close()
		
		elif mode == "inference":
			normal_labels_file = open(input_sacycle_inference_dir + "NormalLabels.txt", "r")
			normal_labels_lines = normal_labels_file.readlines()
			for line in normal_labels_lines:
				normal_labels.append(line.rstrip())
			normal_labels_file.close()
	
	return normal_labels

def classify_diagnoses(threshold, model, diagnoses):
	classifications = []
	
	distances, indexes = model.kneighbors(diagnoses)
	distances = pd.DataFrame(distances)
	distances = distances.mean(axis = 1)
	distances_np_array = np.array(distances)
	for idx,elem in enumerate(distances_np_array):
		if elem < threshold:
			classifications.append("N")
		else:
			classifications.append("A")

	return classifications
	
def write_metrics(accuracy, precision, recall, f1, tp, tn, fp, fn, run_mode):

	if debug_mode == 1:
		print("The metrics are:\nAccuracy = " + str(accuracy) + "\nPrecision = " + str(precision) + "\nRecall = " + str(recall) + "\nF1 = " + str(f1) + "\nTP = " + str(tp) + "\nTN = " + str(tn) + "\nFP = " + str(fp) + "\nFN = " + str(fn))

	if run_mode == "FirstRun":
		file = open(output_firstrun_inference_dir + "Metrics.txt", "w")
		file.write("Accuracy = " + str(accuracy) + "\nPrecision = " + str(precision) + "\nRecall = " + str(recall) + "\nF1 = " + str(f1) + "\nTP = " + str(tp) + "\nTN = " + str(tn) + "\nFP = " + str(fp) + "\nFN = " + str(fn))
		file.close()
	elif run_mode == "SACycle":
		file = open(output_sacycle_inference_dir + "Metrics.txt", "w")
		file.write("Accuracy = " + str(accuracy) + "\nPrecision = " + str(precision) + "\nRecall = " + str(recall) + "\nF1 = " + str(f1) + "\nTP = " + str(tp) + "\nTN = " + str(tn) + "\nFP = " + str(fp) + "\nFN = " + str(fn))
		file.close()
		
def read_event_logs(run_mode, mode):
	
	event_logs = {}
	
	if run_mode == "FirstRun":
		if mode == "training":
			for event_log_filename in os.listdir(input_firstrun_training_eventlogs_dir):
				event_log_label = event_log_filename.split(".")[0]
				event_logs[event_log_label] = xes_importer.apply(input_firstrun_training_eventlogs_dir + event_log_filename)
		elif mode == "inference":
			for event_log_filename in os.listdir(input_firstrun_inference_eventlogs_dir):
				event_log_label = event_log_filename.split(".")[0]
				event_logs[event_log_label] = xes_importer.apply(input_firstrun_inference_eventlogs_dir + event_log_filename)
				
				
			
	elif run_mode == "SACycle":
		if mode == "training":
			for event_log_filename in os.listdir(input_sacycle_training_eventlogs_dir):
				event_log_label = event_log_filename.split(".")[0]
				event_logs[event_log_label] = xes_importer.apply(input_sacycle_training_eventlogs_dir + event_log_filename)
		elif mode == "inference":
			for event_log_filename in os.listdir(input_sacycle_inference_eventlogs_dir):
				event_log_label = event_log_filename.split(".")[0]
				event_logs[event_log_label] = xes_importer.apply(input_sacycle_inference_eventlogs_dir + event_log_filename)		
	
	
	return event_logs		
		
def get_normal_event_logs(event_logs, normal_labels):

	temp = event_logs.copy()
	
	for event_log in event_logs:
		if event_log not in normal_labels:
			del temp[event_log]

	return temp

def get_activities(event_logs):
	
	activities = []
	
	for event_log in event_logs:
		
		for trace in event_logs[event_log]:
			for event in trace:
				if event["concept:name"] not in activities:
					activities.append(event["concept:name"])	
					
	activites = list(set(activities))

	return activities
		
def read_threshold(run_mode, mode):
	threshold = None

	if run_mode == "FirstRun":
		if mode == "inference":
			threshold_file = open(input_firstrun_inference_model_dir + "threshold.txt","r")
			threshold = float(threshold_file.readline().strip())
			threshold_file.close()
		
	elif run_mode == "SACycle":
		if mode == "inference":
			threshold_file = open(input_sacycle_inference_model_dir + "threshold.txt","r")
			threshold = float(threshold_file.readline().strip())
			threshold_file.close()
			
		
	return threshold	
	
def save_n_grams_list(n_grams_list, run_mode):

	if run_mode == "FirstRun":
		file = open(output_firstrun_training_model_dir + "n_grams_list.txt", "w")
		
		for idx,n_gram in enumerate(n_grams_list):
			if idx < len(n_grams_list)-1:
				for idx_act,activity in enumerate(n_gram):
					if idx_act < len(n_gram)-1:
						file.write(activity + "-")
					else:
						file.write(activity)
				file.write("\n")		
			else:
				for idx_act,activity in enumerate(n_gram):
					if idx_act < len(n_gram)-1:
						file.write(activity + "-")
					else:
						file.write(activity)
				
		file.close()
	elif run_mode == "SACycle":
		file = open(output_sacycle_training_model_dir + "n_grams_list.txt", "w")
		for idx,n_gram in enumerate(n_grams_list):
			if idx < len(n_grams_list)-1:
				file.write(n_gram + "\n")
			else:
				file.write(n_gram)
		file.close()

	return None	
	
def save_model(model, threshold, n_grams_list, run_mode):
	
	if run_mode == "FirstRun":
		pickle.dump(model, open(output_firstrun_training_model_dir + "knn.pkl", 'wb'))
		threshold_file = open(output_firstrun_training_model_dir + "threshold.txt", "w")
		threshold_file.write(str(threshold))
		threshold_file.close()
		
	elif run_mode == "SACycle":
		pickle.dump(model, open(output_sacycle_training_model_dir + "knn.pkl", 'wb'))
		threshold_file = open(output_sacycle_training_model_dir + "threshold.txt", "w")
		threshold_file.write(str(threshold))
		threshold_file.close()
		
	save_n_grams_list(n_grams_list, run_mode)	
		
	return None
	
def save_threshold(threshold, run_mode):
	
	if run_mode == "FirstRun":
		file = open(output_firstrun_training_model_dir + "threshold.txt", "w")
		file.write(str(threshold))
		file.close()
	elif run_mode == "SACycle":
		file = open(output_sacycle_training_model_dir + "threshold.txt", "w")
		file.write(str(threshold))
		file.close()
		
	return None	
	
def read_model(run_mode, mode):

	threshold = None
	model = None
	
	threshold = read_threshold(run_mode, mode)
	n_grams_list = read_n_grams_list(run_mode)
	if run_mode == "FirstRun":
		if mode == "inference":
			model = pickle.load(open(input_firstrun_inference_model_dir + "knn.pkl" , 'rb'))
	elif run_mode == "SACycle":
		if mode == "inference":
			model = pickle.load(open(input_sacycle_inference_model_dir + "knn.pkl" , 'rb'))

	return threshold, model, n_grams_list
	
def read_n_grams_list(run_mode):

	n_grams_list = []

	if run_mode == "FirstRun":
		file = open(input_firstrun_inference_model_dir + "n_grams_list.txt", "r")
		for line in file.readlines():
			n_gram = []
			line = line.strip()
			tokens = line.split("-")
			for token in tokens:
				n_gram.append(token)
			n_grams_list.append(tuple(n_gram))
		file.close()
	
	elif run_mode == "SACycle":
		file = open(input_sacycle_inference_model_dir + "n_grams_list.txt", "r")
		for line in file.readlines():
			n_gram = []
			line = line.strip()
			tokens = line.split("-")
			for token in tokens:
				n_gram.append(token)
			n_grams_list.append(tuple(n_gram))
		file.close()

	return n_grams_list	
	
def get_case_level_attributes(event_logs):
	case_level_attributes = []
	
	for event_log in event_logs:
		for trace in event_logs[event_log]:
			for attribute in trace.attributes:
				if (attribute not in ["creator", "variant", "zuweiser", "concept:name", "remarks", "PulmOther", "Diagnosis", "Cause_Of_Death","ABx","InHospital_cCT_Result","Remarks"]) and (attribute not in case_level_attributes):
					case_level_attributes.append(attribute)
			
	
	return case_level_attributes
	
def save_preprocessing_parameters(normalization_parameters, compression_parameters, run_mode, mode):

	if run_mode == "FirstRun":
		if mode == "training":
			normalization_dir = output_firstrun_training_normalization_dir
			compression_dir = output_firstrun_training_compression_dir
		elif mode == "inference":
			normalization_dir = input_firstrun_inference_normalization_dir
			compression_dir = input_firstrun_inference_compression_dir
			
	elif run_mode == "SACycle":	
		if mode == "training":
			normalization_dir = output_sacycle_training_normalization_dir
			compression_dir = output_sacycle_training_compression_dir
		elif mode == "inference":
			normalization_dir = input_sacycle_inference_normalization_dir
			compression_dir = input_sacycle_inference_compression_dir
			

	file = open(normalization_dir + "normalization_parameters.txt", "w")
	for idx,normalization_parameter in enumerate(normalization_parameters):
		if idx<len(normalization_parameters)-1:
			file.write(normalization_parameter + ":" + str(normalization_parameters[normalization_parameter]) + "\n")
		else:
			file.write(normalization_parameter + ":" + str(normalization_parameters[normalization_parameter]))
	file.close()
	

	pickle.dump(compression_parameters, open(compression_dir + "training_pca_model.pkl","wb"))

	return None
	
def read_preprocessing_parameters(run_mode, mode):

	if run_mode == "FirstRun":
		if mode == "training":
			normalization_dir = output_firstrun_training_normalization_dir
			compression_dir = output_firstrun_training_compression_dir
		elif mode == "inference":
			normalization_dir = input_firstrun_inference_normalization_dir
			compression_dir = input_firstrun_inference_compression_dir
			
	elif run_mode == "SACycle":	
		if mode == "training":
			normalization_dir = output_sacycle_training_normalization_dir
			compression_dir = output_sacycle_training_compression_dir
		elif mode == "inference":
			normalization_dir = input_sacycle_inference_normalization_dir
			compression_dir = input_sacycle_inference_compression_dir

	normalization_parameters = {}
	clustering_parameters = {}
	compression_parameters = None
	
	file = open(normalization_dir + "normalization_parameters.txt", "r")
	lines = file.readlines()
	for line in lines:
		line = line.replace("\n","")
		tokens = line.split(":")
		normalization_parameters[tokens[0]] = float(tokens[1])
	file.close()

	compression_parameters = pickle.load(open(compression_dir + "training_pca_model.pkl",'rb'))

	return normalization_parameters, compression_parameters	
		
				
def train_knn(train, validation, n_neighbors):
	model = NearestNeighbors(n_neighbors = n_neighbors)
	model.fit(train)
	distances, indexes = model.kneighbors(validation)
	distances = pd.DataFrame(distances)
	distances = distances.mean(axis = 1)
	threshold = round(distances.describe()["75%"],3)
	
	return model, threshold
	
def get_event_logs_ngrams(event_logs, n):
	
	possible_n_grams = {}

	activities = get_activities(event_logs)	
	temp = list(product(activities,repeat=n))
	for n_gram in temp:
		possible_n_grams[n_gram] = 0
	per_event_log_traces = get_per_event_log_traces(event_logs)
	possible_n_grams = compute_ngrams(possible_n_grams, per_event_log_traces, n)
	
	return possible_n_grams	

def get_per_event_log_traces(event_logs):

	per_event_log_traces = {}

	for event_log in event_logs:
		traces = []
		for trace in event_logs[event_log]:
			events = []
			for event in trace:
				events.append(event["concept:name"])
				
			traces.append(events)
		per_event_log_traces[event_log] = traces		
			
	
	return per_event_log_traces
	
def compute_ngrams(possible_n_grams, per_event_log_traces, n):

	
	rows = []
	for event_log in per_event_log_traces:
		per_event_log_row = {}
		for key in possible_n_grams:
			per_event_log_row[key] = 0
		for trace in per_event_log_traces[event_log]:
			if len(trace) < n:
				pass
			else:
				for i in range(n-1, len(trace), n):
					current_n_gram = trace[i-n+1:i+1]
					per_event_log_row[tuple(current_n_gram)] = per_event_log_row[tuple(current_n_gram)] + 1
					
		rows.append(list(per_event_log_row.values()) + [event_log])
		
	temp = pd.DataFrame(columns = list(possible_n_grams.keys()) + ["label"], data = rows)
				
	return temp		

	
			
def get_labels(event_logs):
	labels = []
	for event_log in event_logs:
		labels.append(event_log)
	return labels	

def write_features(features, run_mode):
	if run_mode == "FirstRun":
		features.to_csv(output_firstrun_inference_dir + "Features/features.csv", index = False)	

try:
	run_mode = sys.argv[1]
	mode = sys.argv[2]
	debug_mode = int(sys.argv[3])
	sample_time = int(sys.argv[4])
	n = int(sys.argv[5])
	normalization_technique = sys.argv[6]
	n_pca_components = int(sys.argv[7])
	n_neighbors = int(sys.argv[8])
	use_categorical_attributes = int(sys.argv[9])
	if mode == "training":
		validation_split_percentage = float(sys.argv[10])
	
except IndexError:
	print("Insert the right number of input arguments")
	sys.exit()


if mode == "training":
	event_logs = read_event_logs(run_mode, mode)
	normal_labels = read_normal_labels(mode, run_mode)
	
	event_logs = get_normal_event_logs(event_logs, normal_labels)
	event_logs = pd.Series(event_logs)
	training_event_logs, validation_event_logs = [i.to_dict() for i in train_test_split(event_logs, test_size=validation_split_percentage)]
	training_n_grams = get_event_logs_ngrams(training_event_logs, n)
	training_n_grams = training_n_grams.drop(labels="label", axis=1)
	training_n_grams = training_n_grams.reindex(sorted(training_n_grams.columns), axis=1)
	n_grams_list = list(training_n_grams.columns)
	training_n_grams, normalization_parameters = normalize_dataset(training_n_grams, 0, None)
	training_n_grams, compression_parameters = compress_dataset(training_n_grams, n_pca_components, 0, None)
	
	validation_n_grams = get_event_logs_ngrams(validation_event_logs, n)
	validation_n_grams = validation_n_grams.drop(labels="label", axis=1)
	
	columns_to_drop = []
	columns_to_add = []
	
	for column in n_grams_list:
		if column not in list(validation_n_grams.columns):
			columns_to_add.append(column)
	for column in list(validation_n_grams.columns):
		if column not in n_grams_list:
			columns_to_drop.append(column)
			
	zeros_vector = []
	for i in range(0, len(validation_n_grams)):
		zeros_vector.append([0]*len(columns_to_add))
	to_concat_df = pd.DataFrame(columns = columns_to_add, data = zeros_vector)		
	validation_n_grams = validation_n_grams.drop(columns_to_drop, axis=1)			
	validation_n_grams = pd.concat([validation_n_grams, to_concat_df], axis=1)
	
	validation_n_grams = validation_n_grams.reindex(sorted(validation_n_grams.columns), axis=1)
	validation_n_grams, ignore = normalize_dataset(validation_n_grams, 1, normalization_parameters)
	validation_n_grams, ignore = compress_dataset(validation_n_grams, n_pca_components, 1, compression_parameters)
	
	model, threshold = train_knn(training_n_grams, validation_n_grams, n_neighbors)
	save_model(model, threshold, n_grams_list, run_mode)
	save_preprocessing_parameters(normalization_parameters, compression_parameters, run_mode, mode)


elif mode == "inference":
	event_logs = read_event_logs(run_mode, mode)
	labels = get_labels(event_logs)
	normal_labels = read_normal_labels(mode, run_mode)
	threshold, model, training_n_grams = read_model(run_mode, mode)
	normalization_parameters, compression_parameters = read_preprocessing_parameters(run_mode, mode)
	
	if sample_time == 1:
		tm_1 = 0
		tm_1 = time.time()
	possible_n_grams = get_event_logs_ngrams(event_logs, n)
	if sample_time == 1:
		tm_1 = time.time() - tm_1
	write_features(possible_n_grams, run_mode)
	possible_n_grams = 	possible_n_grams.drop(labels="label", axis=1)
		
	columns_to_drop = []
	columns_to_add = []
	all_columns = list(possible_n_grams.columns)
	training_columns = training_n_grams
	
	for column in all_columns:
		if column not in training_columns:
			columns_to_drop.append(column)
			
	possible_n_grams = possible_n_grams.drop(columns_to_drop, axis=1)	
	possible_n_grams = possible_n_grams.reindex(sorted(possible_n_grams.columns), axis=1)
	possible_n_grams, ignore = normalize_dataset(possible_n_grams, 1, normalization_parameters)
	possible_n_grams, ignore = compress_dataset(possible_n_grams, n_pca_components, 1, compression_parameters)
	
	if sample_time == 1:
		tm_2 = 0
		tm_2 = time.time()
	
	classifications = classify_diagnoses(threshold, model, possible_n_grams)
	if sample_time == 1:
		tm_2 = time.time() - tm_2 + tm_1
		write_timing(tm_2, run_mode)
	accuracy, precision, recall, f1, tp, tn, fp, fn = get_performance(normal_labels, labels, classifications)
	write_metrics(accuracy, precision, recall, f1, tp, tn, fp, fn, run_mode)
	
	
	
	
	
	
	
	
	
	