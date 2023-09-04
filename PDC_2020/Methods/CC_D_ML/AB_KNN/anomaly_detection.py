import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import numpy as np
import pandas as pd
import warnings
import random
import time
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

from sklearn.neighbors import NearestNeighbors

from sklearn.decomposition import PCA

import pickle

input_firstrun_dir = "Input/FirstRun/AD/"
output_firstrun_dir = "Output/FirstRun/AD/"

input_firstrun_training_dir = input_firstrun_dir + "Training/"
input_firstrun_inference_dir = input_firstrun_dir + "Inference/"
input_firstrun_inference_model_dir = input_firstrun_inference_dir + "Model/"
input_firstrun_inference_normalization_dir = input_firstrun_inference_dir + "Normalization/"
input_firstrun_inference_compression_dir = input_firstrun_inference_dir + "Compression/"


output_firstrun_training_dir = output_firstrun_dir + "Training/"
output_firstrun_training_model_dir = output_firstrun_training_dir + "Model/"
output_firstrun_inference_dir = output_firstrun_dir + "Inference/"
output_firstrun_training_normalization_dir = output_firstrun_training_dir + "Normalization/"
output_firstrun_training_compression_dir = output_firstrun_training_dir + "Compression/"
output_firstrun_inference_timing_dir = output_firstrun_inference_dir + "Timing/"

input_sacycle_dir = "Input/SACycle/AD/"
output_sacycle_dir = "Output/SACycle/AD/"

input_sacycle_training_dir = input_sacycle_dir + "Training/"
input_sacycle_inference_dir = input_sacycle_dir + "Inference/"
input_sacycle_inference_model_dir = input_sacycle_inference_dir + "Model/"
input_sacycle_inference_normalization_dir = input_sacycle_inference_dir + "Normalization/"
input_sacycle_inference_compression_dir = input_sacycle_inference_dir + "Compression/"

output_sacycle_training_dir = output_sacycle_dir + "Training/"
output_sacycle_training_model_dir = output_sacycle_training_dir + "Model/"
output_sacycle_inference_dir = output_sacycle_dir + "Inference/"
output_sacycle_training_normalization_dir = output_sacycle_training_dir + "Normalization/"
output_sacycle_training_compression_dir = output_sacycle_training_dir + "Compression/"
output_sacycle_inference_timing_dir = output_sacycle_inference_dir + "Timing/"

hidden_neurons = 150
latent_code_dimension = 6
epochs = 9999


def write_timing(tm, run_mode):
	if run_mode == "FirstRun":
		file = open(output_firstrun_inference_timing_dir + "timing.txt", "a+")
		file.write(str(tm) + "\n")
		file.close()
	elif run_mode == "SACycle":
		file = open(output_sacycle_inference_timing_dir + "timing.txt", "a+")
		file.write(str(tm) + "\n")
		file.close()

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
				normalization_parameters[column+"_mean"] = column_values_mean
				normalization_parameters[column+"_std"] = column_values_std

	else:
		if normalization_technique == "zscore":
			for label in normalized_dataset:
				try:
					mean = normalization_parameters_in[label+"_mean"]
					std = normalization_parameters_in[label+"_std"]
					parameter_values = normalized_dataset[label].values
					if std != 0:
						parameter_values = (parameter_values - float(mean))/float(std)
					normalized_dataset[label] = parameter_values
				except KeyError:
					print("Label " + label + " was not found among normalization parameters. The corresponding column will be dropped.")
					normalized_dataset = normalized_dataset.drop(labels=label, axis = 1)
					pass
	
	return normalized_dataset, normalization_parameters		
	
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
	
def save_model(model, threshold, run_mode):
	
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
	if run_mode == "FirstRun":
		if mode == "inference":
			model = pickle.load(open(input_firstrun_inference_model_dir + "knn.pkl" , 'rb'))
	elif run_mode == "SACycle":
		if mode == "inference":
			model = pickle.load(open(input_sacycle_inference_model_dir + "knn.pkl" , 'rb'))

	return threshold, model	
	
	
def train_knn(train, validation, n_neighbors):
	model = NearestNeighbors(n_neighbors = n_neighbors)
	model.fit(train)
	distances, indexes = model.kneighbors(validation)
	distances = pd.DataFrame(distances)
	distances = distances.mean(axis = 1)
	threshold = round(distances.describe()["75%"],3)
	
	return model, threshold	
	
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
		
try:
	run_mode = sys.argv[1]
	mode = sys.argv[2]
	debug_mode = int(sys.argv[3])
	sample_time = int(sys.argv[4])
	normalization_technique = sys.argv[5]
	n_pca_components = int(sys.argv[6])
	n_neighbors = int(sys.argv[7])
	if mode == "training":
		validation_split_percentage = float(sys.argv[8])
	
except IndexError:
	print("Insert the right number of input arguments")


if mode == "training":
	diagnoses = read_diagnoses(mode, run_mode)
	normal_labels = read_normal_labels(mode, run_mode)
	diagnoses = diagnoses[diagnoses["label"].isin(normal_labels)]
	diagnoses = diagnoses.drop(labels = "label", axis = 1)
	train_diagnoses, validation_diagnoses = train_test_split(diagnoses, test_size = validation_split_percentage)
	
	# The following piece of code should be replaced to address the case where diagnoses from the training and validation sets differ. For now, this works, but real implementations need to address this.
	train_diagnoses = train_diagnoses.reindex(sorted(train_diagnoses.columns), axis=1)
	validation_diagnoses = validation_diagnoses.reindex(sorted(validation_diagnoses.columns), axis=1)
	
	train_diagnoses, normalization_parameters = normalize_dataset(train_diagnoses, 0, None)
	train_diagnoses, compression_parameters = compress_dataset(train_diagnoses, n_pca_components, 0, None)
	validation_diagnoses, ignore = normalize_dataset(validation_diagnoses, 1, normalization_parameters)
	validation_diagnoses, ignore = compress_dataset(validation_diagnoses, n_pca_components, 1, compression_parameters)
	model, threshold = train_knn(train_diagnoses, validation_diagnoses, n_neighbors)
	save_model(model, threshold, run_mode)
	save_preprocessing_parameters(normalization_parameters, compression_parameters, run_mode, mode)
	

elif mode == "inference":
	diagnoses = read_diagnoses(mode, run_mode)
	diagnoses_labels = list(diagnoses["label"])
	normal_labels = read_normal_labels(mode, run_mode)
	diagnoses = diagnoses.drop(labels = "label", axis = 1)
	threshold, model = read_model(run_mode, mode)
	
	# The following piece of code should be replaced to address the case where diagnoses from the training and test sets differ. For now, this works, but real implementations need to address this.
	diagnoses = diagnoses.reindex(sorted(diagnoses.columns), axis=1)
	normalization_parameters, compression_parameters = read_preprocessing_parameters(run_mode, mode)
	diagnoses, ignore = normalize_dataset(diagnoses, 1, normalization_parameters)
	diagnoses, ignore = compress_dataset(diagnoses, n_pca_components, 1, compression_parameters)
	if sample_time == 1:
		tm = 0
		tm = time.time()	
	classifications = classify_diagnoses(threshold, model, diagnoses)
	if sample_time == 1:
		tm = time.time() - tm
		write_timing(tm, run_mode)
	accuracy, precision, recall, f1, tp, tn, fp, fn = get_performance(normal_labels, diagnoses_labels, classifications)
	write_metrics(accuracy, precision, recall, f1, tp, tn, fp, fn, run_mode)
		





