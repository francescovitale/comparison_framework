import os
import sys
import pandas as pd
import math
import random

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
import pm4py


input_dir = "Input/"
input_firstrun_dir = input_dir + "FirstRun/ELE/"
input_firstrun_data_dir = input_firstrun_dir + "Data/"
input_firstrun_training_data_dir = input_firstrun_data_dir + "Training/"
input_firstrun_test_data_dir = input_firstrun_data_dir + "Test/"

output_dir = "Output/"
output_firstrun_dir = output_dir + "FirstRun/ELE/"
output_firstrun_eventlogs_dir = output_firstrun_dir + "EventLogs/"

def timestamp_builder(number):
	SSS = number
	ss = int(math.floor(SSS/1000))
	mm = int(math.floor(ss/60))
	hh = int(math.floor(mm/24))
	
	SSS = SSS % 1000
	ss = ss%60
	mm = mm%60
	hh = hh%24
	
	return "1900-01-01T"+str(hh)+":"+str(mm)+":"+str(ss)+"."+str(SSS)

def read_event_logs(training_trace_percentage, testing_trace_percentage):

	normal_event_log = None
	anomalous_event_log = None
	
	normal_temp = []
	anomalous_temp = []
	idx_timestamp = 0
	idx_to_add = 0
	for event_log_filename in os.listdir(input_firstrun_training_data_dir):
		event_log = pd.read_csv(input_firstrun_training_data_dir + event_log_filename, compression="gzip")
		unique_traces = list(set(event_log["case:concept:name"]))
		unique_traces = random.choices(unique_traces, k=math.floor(len(unique_traces)*training_trace_percentage))
		event_log = event_log.values.tolist()
		inner_temp = []
		for event_idx,event in enumerate(event_log):
			if event_log[event_idx][1] in unique_traces:
				event_log[event_idx][1] = event_log[event_idx][1] + str(idx_to_add)
				event_log[event_idx].append(timestamp_builder(idx_timestamp))
				idx_timestamp += 1
				inner_temp.append(event_log[event_idx])
				
		normal_temp.append(inner_temp)
		idx_to_add = idx_to_add + len(unique_traces) + 1
		
	
	for event_log_filename in os.listdir(input_firstrun_test_data_dir):
		event_log = pd.read_csv(input_firstrun_test_data_dir + event_log_filename, compression="gzip")
		unique_traces = list(set(event_log["case:concept:name"]))
		unique_traces = random.choices(unique_traces, k=math.floor(len(unique_traces)*testing_trace_percentage))
		event_log = event_log.values.tolist()
		inner_temp_normal = []
		inner_temp_anomalous = []
		for event_idx,event in enumerate(event_log):
			if event_log[event_idx][1] in unique_traces:
				event_log[event_idx][1] = event_log[event_idx][1] + str(idx_to_add)
				event_log[event_idx].append(timestamp_builder(idx_timestamp))
				idx_timestamp += 1
				if event_log[event_idx][2] == True:
					inner_temp_normal.append([event_log[event_idx][0], event_log[event_idx][1], event_log[event_idx][3]])
				else:
					inner_temp_anomalous.append([event_log[event_idx][0], event_log[event_idx][1], event_log[event_idx][3]])
				
		normal_temp.append(inner_temp_normal)
		anomalous_temp.append(inner_temp_anomalous)
		idx_to_add = idx_to_add + len(unique_traces) + 1
		
		
	normal_temp = sum(normal_temp, [])
	anomalous_temp = sum(anomalous_temp, [])
	
	
	normal_temp = pd.DataFrame(columns = ["concept:name", "case:concept:name", "time:timestamp"], data = normal_temp)
	anomalous_temp = pd.DataFrame(columns = ["concept:name", "case:concept:name", "time:timestamp"], data = anomalous_temp)
	
	
	normal_temp = dataframe_utils.convert_timestamp_columns_in_df(normal_temp)
	anomalous_temp= dataframe_utils.convert_timestamp_columns_in_df(anomalous_temp)
	
	parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}
	normal_event_log = log_converter.apply(normal_temp, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
	anomalous_event_log = log_converter.apply(anomalous_temp, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
	

	return normal_event_log, anomalous_event_log
	
def write_event_logs(normal_event_log, anomalous_event_log):

	xes_exporter.apply(normal_event_log, output_firstrun_eventlogs_dir + "Normal.xes")
	xes_exporter.apply(anomalous_event_log, output_firstrun_eventlogs_dir + "Anomalous.xes")

	return None
	
try:
	training_trace_percentage = float(sys.argv[1])
	testing_trace_percentage = float(sys.argv[2])

except IndexError:
	print("Input the arguments.")
	sys.exit()
	
normal_event_log, anomalous_event_log = read_event_logs(training_trace_percentage, testing_trace_percentage)

write_event_logs(normal_event_log, anomalous_event_log)





