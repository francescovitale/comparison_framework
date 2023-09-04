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

from tensorflow.keras.layers import Dense,Input,Concatenate
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

from sklearn.neighbors import NearestNeighbors

from sklearn.decomposition import PCA
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from itertools import permutations

import pickle

input_firstrun_dir = "Input/FirstRun/AD/"
output_firstrun_dir = "Output/FirstRun/AD/"

input_firstrun_training_dir = input_firstrun_dir + "Training/"
input_firstrun_training_eventlogs_dir = input_firstrun_training_dir + "EventLogs/"
input_firstrun_inference_dir = input_firstrun_dir + "Inference/"
input_firstrun_inference_model_dir = input_firstrun_inference_dir + "Model/"
input_firstrun_inference_eventlogs_dir = input_firstrun_inference_dir + "EventLogs/"

output_firstrun_training_dir = output_firstrun_dir + "Training/"
output_firstrun_training_model_dir = output_firstrun_training_dir + "Model/"
output_firstrun_inference_dir = output_firstrun_dir + "Inference/"
output_firstrun_inference_timing_dir = output_firstrun_inference_dir + "Timing/"


input_sacycle_dir = "Input/SACycle/AD/"
output_sacycle_dir = "Output/SACycle/AD/"

input_sacycle_training_dir = input_sacycle_dir + "Training/"
input_sacycle_training_eventlogs_dir = input_sacycle_training_dir + "EventLogs/"
input_sacycle_inference_dir = input_sacycle_dir + "Inference/"
input_sacycle_inference_model_dir = input_sacycle_inference_dir + "Model/"
input_sacycle_inference_eventlogs_dir = input_sacycle_inference_dir + "EventLogs/"


output_sacycle_training_dir = output_sacycle_dir + "Training/"
output_sacycle_training_model_dir = output_sacycle_training_dir + "Model/"
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

def classify_diagnoses(model, threshold, diagnoses):

	classifications = []

	diagnoses_np_array = np.array(diagnoses)
	recon = ae_model.predict(diagnoses_np_array, verbose=0)
	
		
	for idx,elem in enumerate(diagnoses_np_array):
		error = mean_squared_error(diagnoses_np_array[idx],recon[idx])
		if error > threshold:
			classifications.append("A")
		else:
			classifications.append("N")

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
	
def read_model(run_mode):
	
	df_relationships = read_df_relationships(run_mode)
	if run_mode == "FirstRun":
		ae_model = load_model(input_firstrun_inference_model_dir + "ae_model.h5")
		ae_mse = read_mse(run_mode)
		
		return ae_model, ae_mse, df_relationships
	elif run_mode == "SACycle":
		ae_model = load_model(input_sacycle_inference_model_dir + "ae_model.h5")
		ae_mse = read_mse(run_mode)
		
		return ae_model, ae_mse, df_relationships

	
def get_case_level_attributes(event_logs):
	case_level_attributes = []
	
	for event_log in event_logs:
		for trace in event_logs[event_log]:
			for attribute in trace.attributes:
				if (attribute not in ["creator", "variant", "zuweiser", "concept:name", "remarks", "PulmOther", "Diagnosis", "Cause_Of_Death","ABx","InHospital_cCT_Result","Remarks"]) and (attribute not in case_level_attributes):
					case_level_attributes.append(attribute)
			
	
	return case_level_attributes
	
def get_event_logs_statistics(event_logs, use_categorical_attributes):
	
	statistics = {}
	statistics_columns = []

	activities = get_activities(event_logs)
	if use_categorical_attributes == 1:
		case_level_attributes = get_case_level_attributes(event_logs)
	else:
		case_level_attributes = []
	
	statistics_columns = activities
	
	per_event_log_statistics = []
		
		
	for event_log in event_logs:
		statistics[event_log] = {}
		for statistics_column in statistics_columns:
			if statistics_column not in activities:
				statistics[event_log][statistics_column] = []
			else:
				statistics[event_log][statistics_column] = 0
				
		for trace in event_logs[event_log]:
			for attribute in trace.attributes:
				try:
					statistics[event_log][attribute].append(trace.attributes[attribute])
				except:
					pass
			for event in trace:
				try:
					statistics[event_log][event["concept:name"]] = statistics[event_log][event["concept:name"]] + 1
				except:
					pass
		
		per_event_log_statistics.append(list(statistics[event_log].values()))
		
	
	for idx_stat,statistics in enumerate(per_event_log_statistics):
		for idx_elem,elem in enumerate(statistics):
			if elem.__class__ == list and elem:
				per_event_log_statistics[idx_stat][idx_elem] = max(set(elem), key=elem.count)
			elif elem.__class__ == list and (not elem):
				per_event_log_statistics[idx_stat][idx_elem] = "null"
			
		
	statistics = pd.DataFrame(columns=statistics_columns, data = per_event_log_statistics)
	for column in statistics:
		if statistics[column].dtypes != "int64" and statistics[column].dtypes != "float64":
			statistics[column] = statistics[column].astype("category").cat.codes
	
	
	return statistics
				
def read_mse(run_mode):

	if run_mode == "FirstRun":
		file = open(input_firstrun_inference_model_dir + "ae_model_mse.txt", "r")
		mse = float(file.readline())
		file.close()
	elif run_mode == "SACycle":
		file = open(input_sacycle_inference_model_dir + "ae_model_mse.txt", "r")
		mse = float(file.readline())
		file.close()

	return mse
	

	
def autoencoder(hidden_neurons, latent_code_dimension, input_dimension):
	input_layer = Input(shape=(input_dimension,)) # Input
	encoder = Dense(hidden_neurons,activation="relu")(input_layer) # Encoder
	code = Dense(latent_code_dimension)(encoder) # Code
	decoder = Dense(hidden_neurons,activation="relu")(code) # Decoder
	output_layer = Dense(input_dimension,activation="linear")(decoder) # Output
	model = Model(inputs=[input_layer],outputs=[output_layer])
	model.compile(optimizer="adam",loss="mse")
	if debug_mode == 1:
		model.summary()
	return model

def train_autoencoder(normal_data, validation_split_percentage, hidden_neurons, latent_code_dimension, epochs, run_mode):
	input_dimension = len(list(normal_data.columns))
	train_data = np.array(normal_data)
	
	assert latent_code_dimension < input_dimension, print("The autoencoder code layer dimension must be smaller than the input dimension")
	model = autoencoder(hidden_neurons,latent_code_dimension, input_dimension)
	if run_mode == "FirstRun":
		if debug_mode == 1:
			history = model.fit(train_data,train_data,epochs=epochs,shuffle=True,verbose=1,validation_split=validation_split_percentage,callbacks= [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'), ModelCheckpoint(output_firstrun_training_model_dir + "ae_model.h5",monitor='val_loss', save_best_only=True, mode='min', verbose=0)])
		else:
			history = model.fit(train_data,train_data,epochs=epochs,shuffle=True,verbose=0,validation_split=validation_split_percentage,callbacks= [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'), ModelCheckpoint(output_firstrun_training_model_dir + "ae_model.h5",monitor='val_loss', save_best_only=True, mode='min', verbose=0)])
	elif run_mode == "SACycle":
		if debug_mode == 1:
			history = model.fit(train_data,train_data,epochs=epochs,shuffle=True,verbose=1,validation_split=validation_split_percentage,callbacks= [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'), ModelCheckpoint(output_sacycle_training_model_dir + "ae_model.h5",monitor='val_loss', save_best_only=True, mode='min', verbose=0)])
		else:
			history = model.fit(train_data,train_data,epochs=epochs,shuffle=True,verbose=0,validation_split=validation_split_percentage,callbacks= [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'), ModelCheckpoint(output_sacycle_training_model_dir + "ae_model.h5",monitor='val_loss', save_best_only=True, mode='min', verbose=0)])
	
	return model
	
def save_mse(mse, run_mode):

	if run_mode == "FirstRun":
		file = open(output_firstrun_training_model_dir + "ae_model_mse.txt", "w")
		file.write(str(mse))
		file.close()
	elif run_mode == "SACycle":
		file = open(output_sacycle_training_model_dir + "ae_model_mse.txt", "w")
		file.write(str(mse))
		file.close()
	return None	
	
def get_labels(event_logs):
	labels = []
	for event_log in event_logs:
		labels.append(event_log)
	return labels	
	
	
def get_event_logs_ngrams(event_logs, n):
	
	possible_n_grams = {}

	activities = get_activities(event_logs)	
	temp = list(permutations(activities,n))
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
					
		rows.append(list(per_event_log_row.values()))
		
	temp = pd.DataFrame(columns = possible_n_grams, data = rows)
				
	return temp		
	
def save_df_relationships(df_relationships, run_mode):

	if run_mode == "FirstRun":
		file = open(output_firstrun_training_model_dir + "df_relationships.txt", "w")
		
		for idx,n_gram in enumerate(df_relationships):
			if idx < len(df_relationships)-1:
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
		file = open(output_sacycle_training_model_dir + "df_relationships.txt", "w")
		for idx,n_gram in enumerate(df_relationships):
			if idx < len(df_relationships)-1:
				file.write(n_gram + "\n")
			else:
				file.write(n_gram)
		file.close()

	return None		

def read_df_relationships(run_mode):

	df_relationships_list = []

	if run_mode == "FirstRun":
		file = open(input_firstrun_inference_model_dir + "df_relationships.txt", "r")
		for line in file.readlines():
			df_relationship = []
			line = line.strip()
			tokens = line.split("-")
			for token in tokens:
				df_relationship.append(token)
			df_relationships_list.append(tuple(df_relationship))
		file.close()
	
	elif run_mode == "SACycle":
		file = open(input_sacycle_inference_model_dir + "df_relationships.txt", "r")
		for line in file.readlines():
			df_relationship = []
			line = line.strip()
			tokens = line.split("-")
			for token in tokens:
				df_relationship.append(token)
			df_relationships_list.append(tuple(df_relationship))
		file.close()

	return df_relationships_list		
	
def get_df_relationships(event_logs):

	activities = get_activities(event_logs)
	possible_df_relationships = list(permutations(activities,2))
	possible_df_relationships = sorted(possible_df_relationships)
	

	rows = []
	for event_log in event_logs:
		row = {}
		for df_relationship in possible_df_relationships:
			row[df_relationship] = 0
		event_log_df = footprints_discovery.apply(event_logs[event_log], variant=footprints_discovery.Variants.TRACE_BY_TRACE)
		for trace_analysis in event_log_df:
			for df_relationship in trace_analysis["sequence"]:
				row[df_relationship] = row[df_relationship] + 1
				
		rows.append(list(row.values()))
		
	possible_df_relationships = pd.DataFrame(columns=possible_df_relationships, data = rows)	
	
	return possible_df_relationships	

try:
	run_mode = sys.argv[1]
	mode = sys.argv[2]
	debug_mode = int(sys.argv[3])
	sample_time = int(sys.argv[4])
	use_categorical_attributes = int(sys.argv[5])
	if mode == "training":
		validation_split_percentage = float(sys.argv[6])
	if mode == "training":
		hidden_neurons = int(sys.argv[7])
		latent_code_dimension = int(sys.argv[8])
		epochs = int(sys.argv[9])	
	
except IndexError:
	print("Insert the right number of input arguments")
	sys.exit()


if mode == "training":
	event_logs = read_event_logs(run_mode, mode)
	normal_labels = read_normal_labels(mode, run_mode)
	event_logs = get_normal_event_logs(event_logs, normal_labels)
	
	event_logs_df_relationships = get_df_relationships(event_logs)
	event_logs_df_relationships = event_logs_df_relationships.reindex(sorted(event_logs_df_relationships.columns), axis=1)
	df_relationships = list(event_logs_df_relationships.columns)

	ae_model = train_autoencoder(event_logs_df_relationships, validation_split_percentage, hidden_neurons, latent_code_dimension, epochs, run_mode)
	ae_model_recons = ae_model.predict(event_logs_df_relationships)
	ae_model_mse = mean_squared_error(event_logs_df_relationships, ae_model_recons)
	save_mse(ae_model_mse, run_mode)
	save_df_relationships(df_relationships, run_mode)

elif mode == "inference":
	event_logs = read_event_logs(run_mode, mode)
	labels = get_labels(event_logs)
	ae_model, ae_model_mse, df_relationships = read_model(run_mode)
	normal_labels = read_normal_labels(mode, run_mode)
	if sample_time == 1:
		tm_1 = 0
		tm_1 = time.time()
	possible_df_relationships = get_df_relationships(event_logs)
	if sample_time == 1:
		tm_1 = time.time() - tm_1
	columns_to_drop = []
	columns_to_add = []
	all_columns = list(possible_df_relationships.columns)
	
	for column in all_columns:
		if column not in df_relationships:
			columns_to_drop.append(column)
			
	possible_df_relationships = possible_df_relationships.drop(columns_to_drop, axis=1)	
	possible_df_relationships = possible_df_relationships.reindex(sorted(possible_df_relationships.columns), axis=1)
	
	if sample_time == 1:
		tm_2 = 0
		tm_2 = time.time()
	
	classifications = classify_diagnoses(ae_model,ae_model_mse, possible_df_relationships)
	if sample_time == 1:
		tm_2 = time.time() - tm_2 + tm_1
		write_timing(tm_2, run_mode)
	
	accuracy, precision, recall, f1, tp, tn, fp, fn = get_performance(normal_labels, labels, classifications)
	write_metrics(accuracy, precision, recall, f1, tp, tn, fp, fn, run_mode)
	
	
	
	
	
	
	
	
	
	