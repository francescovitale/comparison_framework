from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
import pm4py.algo.conformance.tokenreplay as tokenreplay


import os
import sys
import pandas as pd
import random
import bisect
import time
import numpy as np
from sklearn.decomposition import PCA

input_firstrun_dir = "Input/FirstRun/DG/"
input_firstrun_eventlogs_dir = input_firstrun_dir + "EventLogs/"
input_firstrun_normativemodels_dir = input_firstrun_dir + "NormativeModels/"

output_firstrun_dir = "Output/FirstRun/DG/"
output_firstrun_diagnoses_dir = output_firstrun_dir + "Diagnoses/"
output_firstrun_timing_dir = output_firstrun_dir + "Timing/"

input_sacycle_dir = "Input/SACycle/DG/"
input_sacycle_eventlogs_dir = input_sacycle_dir + "EventLogs/"
input_sacycle_normativemodels_dir = input_sacycle_dir + "NormativeModels/"

output_sacycle_dir = "Output/SACycle/DG/"
output_sacycle_diagnoses_dir = output_sacycle_dir + "Diagnoses/"
output_sacycle_timing_dir = output_sacycle_dir + "Timing/"

input_visualization_dir = "Input/DG/"
input_visualization_eventlogs_dir = input_visualization_dir + "EventLogs/"
input_visualization_normativemodels_dir = input_visualization_dir + "NormativeModels/"



output_visualization_dir = "Output/DG/"
output_visualization_diagnoses_dir = output_visualization_dir + "Diagnoses/"

n_components = -1
normalization_technique = "zscore"

def read_normative_models(run_mode):

	normative_models = {}

	if run_mode == "FirstRun":
		for normative_model_filename in os.listdir(input_firstrun_normativemodels_dir):
			net, im, fm = pnml_importer.apply(input_firstrun_normativemodels_dir + normative_model_filename)
			normative_models[normative_model_filename.split(".")[0]] = {}
			normative_models[normative_model_filename.split(".")[0]]["net"] = net
			normative_models[normative_model_filename.split(".")[0]]["initial_marking"] = im
			normative_models[normative_model_filename.split(".")[0]]["final_marking"] = fm
	
	elif run_mode == "SACycle":
		for normative_model_filename in os.listdir(input_sacycle_normativemodels_dir):
			net, im, fm = pnml_importer.apply(input_sacycle_normativemodels_dir + normative_model_filename)
			normative_models[normative_model_filename.split(".")[0]] = {}
			normative_models[normative_model_filename.split(".")[0]]["net"] = net
			normative_models[normative_model_filename.split(".")[0]]["initial_marking"] = im
			normative_models[normative_model_filename.split(".")[0]]["final_marking"] = fm
			
	elif run_mode == "Visualization":
		for normative_model_filename in os.listdir(input_visualization_normativemodels_dir):
			net, im, fm = pnml_importer.apply(input_visualization_normativemodels_dir + normative_model_filename)
			normative_models[normative_model_filename.split(".")[0]] = {}
			normative_models[normative_model_filename.split(".")[0]]["net"] = net
			normative_models[normative_model_filename.split(".")[0]]["initial_marking"] = im
			normative_models[normative_model_filename.split(".")[0]]["final_marking"] = fm
			
			

	return normative_models
	
def read_event_logs(run_mode):
	
	event_logs = {}
	
	if run_mode == "FirstRun":
		for event_log_filename in os.listdir(input_firstrun_eventlogs_dir):
			event_log_label = event_log_filename.split(".")[0]
			event_logs[event_log_label] = xes_importer.apply(input_firstrun_eventlogs_dir + event_log_filename)
			
	elif run_mode == "SACycle":
		for event_log_filename in os.listdir(input_sacycle_eventlogs_dir):
			event_log_label = event_log_filename.split(".")[0]
			event_logs[event_log_label] = xes_importer.apply(input_sacycle_eventlogs_dir + event_log_filename)
			
	elif run_mode == "Visualization":
		for event_log_filename in os.listdir(input_visualization_eventlogs_dir):
			event_log_label = event_log_filename.split(".")[0]
			event_logs[event_log_label] = xes_importer.apply(input_visualization_eventlogs_dir + event_log_filename)
			
	
	return event_logs
	
	
def compute_fitness(petri_net, event_log, cc_variant):

	log_fitness = 0.0
	aligned_traces = None
	parameters = {}
	parameters[log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY] = 'CaseID'
	
	if cc_variant == "ALIGNMENT_BASED":
		aligned_traces = alignments.apply_log(event_log, petri_net["net"], petri_net["initial_marking"], petri_net["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
		log_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]
	elif cc_variant == "TOKEN_BASED":
		replay_results = tokenreplay.algorithm.apply(log = event_log, net = petri_net["net"], initial_marking = petri_net["initial_marking"], final_marking = petri_net["final_marking"], parameters = parameters, variant = tokenreplay.algorithm.Variants.TOKEN_REPLAY)
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
	
def compress_dataset(dataset, reuse_parameters, columns_names, compression_parameters_in):
	compressed_dataset = dataset.copy()
	compression_parameters = None

	if reuse_parameters == 0:
		compression_parameters = PCA(n_components=n_components)
		compressed_dataset = compression_parameters.fit_transform(compressed_dataset)
		columns = columns_names
		compressed_dataset = pd.DataFrame(data=compressed_dataset, columns=columns)
	else:
		compressed_dataset = compression_parameters_in.transform(compressed_dataset)
		columns = columns_names
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
	
def generate_diagnoses(normative_models, event_logs, apply_pca, n_pca_components):

	diagnoses = {}
	columns = []
	
	model_activities = []
	for normative_model in normative_models:
		model_activities.append(get_activities(list(normative_models[normative_model]["net"].transitions)))
	temp = []
	max_act_set = -1
	for activities_set in model_activities:
		if len(activities_set) > max_act_set:
			max_act_set = len(activities_set)
			temp = activities_set
	model_activities = temp		


	for normative_model in normative_models:

		model_diagnoses_columns = []
		model_compressed_activities = []
		model_diagnoses_columns = get_model_logs_activities(model_activities.copy(), event_logs)
		model_activities = model_diagnoses_columns.copy()
		model_diagnoses_columns.sort()
		if apply_pca == 1:
			for i in range(0, n_pca_components):
				model_compressed_activities.append(normative_model + "_PC_" + str(i))
				columns.append(normative_model + "_PC_" + str(i))
		
		model_unalignments = []
		model_fitness_diagnoses = []
		model_labels = []
		
		# model unalignments and fitness values are computed next
		
		for event_log in event_logs:
			log = event_logs[event_log]
			try:
				fitness, aligned_traces = compute_fitness(normative_models[normative_model], log, "ALIGNMENT_BASED")
				unaligned_activities = compute_unaligned_activities(log, aligned_traces)
				
				for activity in model_activities:
					try:
						unaligned_activities[normative_model + "_" + activity] = unaligned_activities.pop(activity)
					except:
						
						unaligned_activities[normative_model + "_" + activity] = 0
				
				
				temp_list = []
				for sorted_key in sorted(unaligned_activities):
					temp_list.append(unaligned_activities[sorted_key])
				model_unalignments.append(temp_list)
			except Exception as ex:
				fitness, ignore = compute_fitness(normative_models[normative_model], log, "TOKEN_BASED")
			model_fitness_diagnoses.append(fitness)
			model_labels.append(event_log)
			
		# model_unalignments, model_fitness_diagnoses, and model_labels are used next	
			
		model_diagnoses = []
		if len(model_unalignments) > 0:
			for idx,el in enumerate(model_fitness_diagnoses):
				model_diagnoses.append(model_unalignments[idx] + [model_fitness_diagnoses[idx]] + [model_labels[idx]])
			model_diagnoses_columns.append(normative_model + "_fitness")
			model_diagnoses_columns.append("label")
			model_diagnoses = pd.DataFrame(columns = model_diagnoses_columns, data = model_diagnoses)
			labels = model_diagnoses["label"]
			fitness = model_diagnoses[normative_model + "_fitness"]
			
			if normalize == 0 and apply_pca == 0:
				pass
			elif normalize == 1 and apply_pca == 0:
				normalized_unalignments, ignore = normalize_dataset(model_diagnoses.loc[:, model_diagnoses.columns.isin(model_activities)], 0, None)
				model_diagnoses = model_diagnoses.drop(columns=model_activities, axis = 1)
				model_diagnoses = pd.concat([model_diagnoses, normalized_unalignments], axis = 1)
			elif normalize == 0 and apply_pca == 1:
				compressed_unalignments, ignore = compress_dataset(model_diagnoses.loc[:, model_diagnoses.columns.isin(model_activities + [normative_model + "_fitness"])], 0, model_compressed_activities, None)
				model_diagnoses = model_diagnoses.drop(columns=model_activities + [normative_model + "_fitness"], axis = 1)
				model_diagnoses = pd.concat([model_diagnoses, compressed_unalignments], axis = 1)
				if run_mode == "Visualization":
					model_diagnoses[normative_model + "_fitness"] = fitness
			elif normalize == 1 and apply_pca == 1:
				normalized_unalignments, ignore = normalize_dataset(model_diagnoses.loc[:, model_diagnoses.columns.isin(model_activities)], 0, None)
				compressed_unalignments, ignore = compress_dataset(normalized_unalignments, 0, model_compressed_activities, None)
				model_diagnoses = model_diagnoses.drop(columns=model_activities, axis = 1)
				model_diagnoses = pd.concat([model_diagnoses, compressed_unalignments], axis = 1)
				if run_mode == "Visualization":
					model_diagnoses[normative_model + "_fitness"] = fitness
			
		else:
			for idx,el in enumerate(model_fitness_diagnoses):
				model_diagnoses.append([model_fitness_diagnoses[idx]] + [model_labels[idx]])
			model_diagnoses = pd.DataFrame(columns = [normative_model + "_fitness", "label"], data = model_diagnoses)
		
		
		model_diagnoses = model_diagnoses.reindex(sorted(model_diagnoses.columns), axis=1)	
		diagnoses[normative_model] = model_diagnoses
	
	labels = diagnoses[random.choice(list(diagnoses.keys()))]["label"]
	concatenated_diagnoses = pd.DataFrame()
	for normative_model in diagnoses:
		concatenated_diagnoses = pd.concat([concatenated_diagnoses, diagnoses[normative_model].drop(columns="label")], axis = 1)
	concatenated_diagnoses["label"] = labels	
	
			
	return concatenated_diagnoses

def write_diagnoses(diagnoses, run_mode):

	if run_mode == "FirstRun":
		diagnoses.to_csv(output_firstrun_diagnoses_dir + "diagnoses.csv", index = False)
	elif run_mode == "SACycle":
		diagnoses.to_csv(output_sacycle_diagnoses_dir + "diagnoses.csv", index = False)
	elif run_mode == "Visualization":
		diagnoses.to_csv(output_visualization_diagnoses_dir + "diagnoses.csv", index = False)	

	return None
	
def write_timing(tm, run_mode):
	if run_mode == "FirstRun":
		file = open(output_firstrun_timing_dir + "timing.txt", "a+")
		file.write(str(tm) + "\n")
		file.close()
	elif run_mode == "SACycle":
		file = open(output_sacycle_timing_dir + "timing.txt", "a+")
		file.write(str(tm) + "\n")
		file.close()
	
try:
	debug_mode = sys.argv[1]
	run_mode = sys.argv[2]
	sample_time = int(sys.argv[3])
	normalize = int(sys.argv[4])
	apply_pca = int(sys.argv[5])
	if apply_pca == 1:
		n_components = int(sys.argv[6])

except IndexError:
	print("Insert the right number of input arguments.")
	sys.exit()
	

normative_models = read_normative_models(run_mode)
event_logs = read_event_logs(run_mode)
if sample_time == 1:
	tm = 0
	tm = time.time()
diagnoses = generate_diagnoses(normative_models, event_logs, apply_pca, n_components)
if sample_time == 1:
	tm = time.time() - tm
	write_timing(tm, run_mode)

write_diagnoses(diagnoses, run_mode)









