import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import numpy as np
import pandas as pd
import time
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
	
def get_petri_net(event_log, noise_threshold):

	petri_net = {}

	parameters = {}
	parameters[pdiscovery.inductive.algorithm.Variants.IMf.value.Parameters.NOISE_THRESHOLD] = noise_threshold

	petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pdiscovery.inductive.algorithm.apply(event_log, variant=pdiscovery.inductive.algorithm.Variants.IMf, parameters=parameters)
	return petri_net

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

def classify_diagnoses(threshold, diagnoses):

	classifications = []
	diagnoses_np_array = np.array(diagnoses)
	for idx,elem in enumerate(diagnoses_np_array):
		if elem < threshold:
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
	
def save_model(petri_net, threshold, run_mode):
	
	if run_mode == "FirstRun":
		pnml_exporter.apply(petri_net["network"], petri_net["initial_marking"], output_firstrun_training_model_dir + "NormativeModel.pnml",final_marking=petri_net["final_marking"])
		
	elif run_mode == "SACycle":
		pnml_exporter.apply(petri_net["network"], petri_net["initial_marking"], output_sacycle_training_model_dir + "NormativeModel.pnml",final_marking=petri_net["final_marking"])
		
	save_threshold(threshold, run_mode)

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
	petri_net = {}
	
	threshold = read_threshold(run_mode, mode)
	if run_mode == "FirstRun":
		if mode == "inference":
			petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pnml_importer.apply(input_firstrun_inference_model_dir + "NormativeModel.pnml")
	elif run_mode == "SACycle":
		if mode == "inference":
			petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pnml_importer.apply(input_sacycle_inference_model_dir + "NormativeModel.pnml")

	return threshold, petri_net


try:
	run_mode = sys.argv[1]
	mode = sys.argv[2]
	debug_mode = int(sys.argv[3])
	sample_time = int(sys.argv[4])
	noise_threshold = float(sys.argv[5])
	cc_variant = sys.argv[6]
	if mode == "training":
		validation_split_percentage = float(sys.argv[7])
	
except IndexError:
	print("Insert the right number of input arguments")
	sys.exit()

if mode == "training":
	event_logs = read_event_logs(run_mode, mode)
	normal_labels = read_normal_labels(mode, run_mode)
	event_logs = get_normal_event_logs(event_logs, normal_labels)
	event_logs_labels = list(event_logs.keys())
	train_event_logs_labels, validation_event_logs_labels = train_test_split(event_logs_labels, test_size = validation_split_percentage)
	train_event_logs = {}
	validation_event_logs = {}
	for event_log in event_logs:
		if event_log in train_event_logs_labels:
			train_event_logs[event_log] = event_logs[event_log]
		elif event_log in validation_event_logs_labels:
			validation_event_logs[event_log] = event_logs[event_log]
	train_event_log = join_event_logs(list(train_event_logs.values()))
	train_petri_net = get_petri_net(train_event_log, noise_threshold)
	diagnoses = generate_diagnoses(train_petri_net, validation_event_logs, cc_variant)
	threshold = sum(diagnoses["fitness"])/len(diagnoses["fitness"])
	#threshold = max(diagnoses["fitness"])
	save_model(train_petri_net, threshold, run_mode)


elif mode == "inference":
	event_logs = read_event_logs(run_mode, mode)
	threshold, normative_model = read_model(run_mode, mode)
	if sample_time == 1:
		tm = 0
		tm = time.time()
	diagnoses = generate_diagnoses(normative_model, event_logs, cc_variant)
	diagnoses_labels = list(diagnoses["label"])
	normal_labels = read_normal_labels(mode, run_mode)
	diagnoses = diagnoses.drop(labels = "label", axis = 1)
	classifications = classify_diagnoses(threshold, diagnoses["fitness"])
	if sample_time == 1:
		tm = time.time() - tm
		write_timing(tm, run_mode)
	accuracy, precision, recall, f1, tp, tn, fp, fn = get_performance(normal_labels, diagnoses_labels, classifications)
	write_metrics(accuracy, precision, recall, f1, tp, tn, fp, fn, run_mode)
	
	
	
	
	
	
	
	
	
	
	