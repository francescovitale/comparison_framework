from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
import pm4py.algo.discovery as pdiscovery
import pm4py

import os
import sys
import math
import random

from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness


input_dir = "Input/"
input_firstrun_dir = input_dir + "FirstRun/PP/"
input_firstrun_eventlogs_dir = input_firstrun_dir + "EventLogs/"

output_dir = "Output/"
output_firstrun_dir = output_dir + "FirstRun/PP/"
output_firstrun_eventlogs_dir = output_firstrun_dir + "EventLogs/"
output_firstrun_normallabels_dir = output_firstrun_dir + "NormalLabels/"

def read_event_logs():

	event_logs = {}

	for filename in os.listdir(input_firstrun_eventlogs_dir):
		if filename.split(".")[0] == "Normal":
			event_logs["Normal"] = xes_importer.apply(input_firstrun_eventlogs_dir + filename)
		else:
			event_logs["Anomalous"] = xes_importer.apply(input_firstrun_eventlogs_dir + filename)

	return event_logs
	
def even_out_event_logs(event_logs):

	max_n_traces = -99999999999
	event_log_max_n_traces = ""
	
	for event_log in event_logs:
		n_traces = len(event_logs[event_log])
		if n_traces >= max_n_traces:
			max_n_traces = n_traces
			event_log_max_n_traces = event_log
			
	for event_log in event_logs:
		if event_log != event_log_max_n_traces:
			n_traces = len(event_logs[event_log])
			n_traces_to_add = max_n_traces - n_traces
			temp = event_logs[event_log]
			for i in range(0, n_traces_to_add):
				temp.append(random.choice(event_logs[event_log]))
			event_logs[event_log] = temp

	return event_logs
	
def split_event_log(event_log, event_log_name, n_traces_per_event_log):
	event_logs = {}
	n_event_logs = math.ceil(len(event_log)/n_traces_per_event_log)
	
	
	for i in range(0, n_event_logs):
		if i < n_event_logs-1:
			event_logs[event_log_name + "_" + str(i)] = event_log[i*n_traces_per_event_log:i*n_traces_per_event_log+n_traces_per_event_log]
		else:
			event_logs[event_log_name + "_" + str(i)] = event_log[i*n_traces_per_event_log:]
			
	return event_logs
	
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

def add_traces_to_event_logs(event_logs, n_traces_to_reach):


	for event_log in event_logs:
		n_traces_to_add = n_traces_to_reach-len(event_logs[event_log])
		if n_traces_to_add > 0:
			temp = event_logs[event_log]
			for i in range(0, n_traces_to_add):
				temp.append(random.choice(event_logs[event_log]))
			event_logs[event_log] = temp	

	return event_logs

def write_event_logs(event_logs, event_logs_type, cycle):

	try:
		os.mkdir(output_firstrun_eventlogs_dir + str(cycle))
	except FileExistsError:
		pass

	for idx,event_log in enumerate(event_logs):
	
		pm4py.write_xes((pm4py.objects.log.obj.EventLog)(event_log), output_firstrun_eventlogs_dir + str(cycle) + "/" + event_logs_type + "_" + str(idx) + '.xes')

	return None
	
def write_normal_labels(normal_labels, cycle):

	file = open(output_firstrun_normallabels_dir + "NormalLabels_" + str(cycle) + ".txt", "w")
	
	for idx,label in enumerate(normal_labels):	
		if idx < len(normal_labels)-1:
			file.write(label + "\n")
		else:
			file.write(label)
	
	file.close()

	return None
	
def even_out_event_log(event_logs, n_logs_per_type):
	
	if len(event_logs) > n_logs_per_type:
		event_logs = {k: event_logs[k] for k in list(event_logs)[:n_logs_per_type]}
	elif len(event_logs) < n_logs_per_type:
		# to implement
		pass
		
		
	return event_logs	
		
	
	
	
	
try:
	n_traces_per_log = int(sys.argv[1])
	n_logs_per_type = int(sys.argv[2])

except IndexError:
	print("Insert the right number of input arguments")
	sys.exit()

event_logs = read_event_logs()

for event_log in event_logs:
	event_logs[event_log] = split_event_log(event_logs[event_log], event_log, n_traces_per_log)
	event_logs[event_log] = even_out_event_log(event_logs[event_log], n_logs_per_type)
	

per_cycle_event_logs = {}
per_cycle_normal_labels = {}

for i in range(0, 1):

	if i == 0:
		per_cycle_event_logs[i] = {}
		per_cycle_normal_labels[i] = []
			
		per_cycle_event_logs[i]["Normal"] = random.choices(list(event_logs["Normal"].values()), k=math.floor(len(event_logs["Normal"])*1.0))
		per_cycle_event_logs[i]["Anomalous"] = random.choices(list(event_logs["Anomalous"].values()), k=math.floor(len(event_logs["Anomalous"])*1.0))
			
		for idx,log in enumerate(event_logs["Normal"]):
			per_cycle_normal_labels[i].append("Normal_" + str(idx))
			
	for event_log_type in per_cycle_event_logs[i]:
		write_event_logs(per_cycle_event_logs[i][event_log_type], event_log_type, i)			
		
	write_normal_labels(per_cycle_normal_labels[i], i)


