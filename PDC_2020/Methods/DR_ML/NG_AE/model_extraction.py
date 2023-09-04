from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.algo.conformance.footprints.util import evaluation
from pm4py.algo.conformance.footprints import algorithm as fp_conformance
from pm4py.algo.discovery.footprints import algorithm as fp_discovery
import pm4py.algo.discovery as pd
import pm4py

import os
import sys
import math
import random

input_dir = "Input/"
input_sacycle_dir = input_dir + "SACycle/ME/"
input_sacycle_eventlogs_dir = input_sacycle_dir + "EventLogs/"
input_sacycle_labels_dir = input_sacycle_dir + "Labels/"

output_dir = "Output/"
output_sacycle_dir = output_dir + "SACycle/ME/"
output_sacycle_petrinets_dir = output_sacycle_dir + "PetriNets/"
output_sacycle_statistics_dir = output_sacycle_dir + "Statistics/"

def load_event_logs(cycle_number):

	event_logs = {}
	
	for filename in os.listdir(input_sacycle_eventlogs_dir + str(cycle_number)):
		event_logs[filename.split(".")[0]] = xes_importer.apply(input_sacycle_eventlogs_dir + str(cycle_number) + "/" + filename)


	return event_logs

def load_labels(cycle_number):

	normal_labels = []
	real_normal_labels = []
	
	for filename in os.listdir(input_sacycle_labels_dir):
		if filename.split(".")[0] == "NormalLabels_" + str(cycle_number):
			file = open(input_sacycle_labels_dir + "/" + filename)
			for line in file.readlines():
				normal_labels.append(line.strip())
			file.close()
			
		elif filename.split(".")[0] == "Real_NormalLabels_" + str(cycle_number):
			file = open(input_sacycle_labels_dir + "/" + filename)
			for line in file.readlines():
				real_normal_labels.append(line.strip())
			file.close()
	
	return normal_labels, real_normal_labels
	
	
def split_event_logs(event_logs, normal_labels, real_normal_labels):

	normal_event_logs = []
	real_normal_event_logs = []

	for event_log in event_logs:
		if event_log in normal_labels:
			normal_event_logs.append(event_logs[event_log])
		if event_log in real_normal_labels:
			real_normal_event_logs.append(event_logs[event_log])

	return normal_event_logs, real_normal_event_logs
	
def get_petri_nets(normal_event_log, real_normal_event_log):

	normal_event_logs_petri_net = {}
	real_normal_event_logs_petri_net = {}

	parameters = {}
	parameters[pd.inductive.algorithm.Variants.IMf.value.Parameters.NOISE_THRESHOLD] = 0.0

	normal_event_logs_petri_net["network"], normal_event_logs_petri_net["initial_marking"], normal_event_logs_petri_net["final_marking"] = pd.inductive.algorithm.apply(normal_event_log, variant=pd.inductive.algorithm.Variants.IMf, parameters=parameters)
	real_normal_event_logs_petri_net["network"], real_normal_event_logs_petri_net["initial_marking"], real_normal_event_logs_petri_net["final_marking"] = pd.inductive.algorithm.apply(real_normal_event_log, variant=pd.inductive.algorithm.Variants.IMf, parameters=parameters)
	
	
	return normal_event_logs_petri_net, real_normal_event_logs_petri_net
	
def write_petri_nets(normal_event_logs_petri_net, real_normal_event_logs_petri_net, cycle_number):

	pnml_exporter.apply(normal_event_logs_petri_net["network"], normal_event_logs_petri_net["initial_marking"], output_sacycle_petrinets_dir + "NormalEventLogs_PetriNet_" + str(cycle_number) + ".pnml",final_marking=normal_event_logs_petri_net["final_marking"])
	pnml_exporter.apply(real_normal_event_logs_petri_net["network"], real_normal_event_logs_petri_net["initial_marking"], output_sacycle_petrinets_dir + "Real_NormalEventLogs_PetriNet_" + str(cycle_number) + ".pnml",final_marking=real_normal_event_logs_petri_net["final_marking"])


	return None
	
def join_event_logs(event_logs):
		
	joined_event_log = []	
		
	for event_log in event_logs:
		for trace in event_log:
			joined_event_log.append(trace)
		
	return (pm4py.objects.log.obj.EventLog)(joined_event_log)


def get_footprint(event_log):

	footprint = footprints_discovery.apply(event_log, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
	
	return footprint
	
def compute_footprint_conformance(footprint_to_compare, reference_event_log):

	parameters = {}
	reference_petri_net = {}
	parameters[pd.inductive.algorithm.Variants.IMf.value.Parameters.NOISE_THRESHOLD] = 0.0
	reference_petri_net["network"], reference_petri_net["initial_marking"], reference_petri_net["final_marking"] = pd.inductive.algorithm.apply(reference_event_log, variant=pd.inductive.algorithm.Variants.IMf, parameters=parameters)
	reference_tree = pm4py.convert_to_process_tree(reference_petri_net["network"], reference_petri_net["initial_marking"], reference_petri_net["final_marking"])
	reference_tree_footprint = fp_discovery.apply(reference_tree, variant=fp_discovery.Variants.PROCESS_TREE)
	
	conf_result = fp_conformance.apply(footprint_to_compare, reference_tree_footprint, variant=fp_conformance.Variants.LOG_EXTENSIVE)
	fitness = evaluation.fp_fitness(footprint_to_compare, reference_tree_footprint, conf_result)

	return fitness
	
def write_similarity_statistics(footprint_conformance, cycle_number):

	file = open(output_sacycle_statistics_dir + "similarity_statistics_" + str(cycle_number) + ".txt", "w")
	
	file.write("Footprint conformance:" + str(footprint_conformance))
	
	file.close()

	return None

try:
	cycle_number = int(sys.argv[1])
except IndexError:
	print("Insert the right number of input arguments.")
	sys.exit()
	
event_logs = load_event_logs(cycle_number)
normal_labels, real_normal_labels = load_labels(cycle_number)
normal_event_logs, real_normal_event_logs = split_event_logs(event_logs, normal_labels, real_normal_labels)

normal_event_log = join_event_logs(normal_event_logs)
real_normal_event_log = join_event_logs(real_normal_event_logs)

normal_event_logs_petri_net, real_normal_event_logs_petri_net = get_petri_nets(normal_event_log, real_normal_event_log)
write_petri_nets(normal_event_logs_petri_net, real_normal_event_logs_petri_net, cycle_number)

normal_event_logs_footprint = get_footprint(normal_event_log)
normal_event_log_footprint_conformance = compute_footprint_conformance(normal_event_logs_footprint, real_normal_event_log)

write_similarity_statistics(normal_event_log_footprint_conformance, cycle_number)





	