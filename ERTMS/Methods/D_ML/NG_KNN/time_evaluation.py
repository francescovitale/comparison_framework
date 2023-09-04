import os
import sys
import math
import random

input_dir = "Input/"
input_firstrun_dir = input_dir + "FirstRun/TE/"

output_dir = "Output/"
output_firstrun_dir = output_dir + "FirstRun/TE/"


def load_times():
	
	times = []
	
	file = open(input_firstrun_dir + "AD_timing_FirstRun.txt")
	for line in file.readlines():
		times.append(float(line.strip("\n")))
	file.close()
	
	return times
	
def get_times(dg_times, ad_times):
	times = []
	
	for idx,execution_time in enumerate(dg_times):
		times.append(dg_times[idx] + ad_times[idx])

	return times
	
def write_times(times):
	for idx,time in enumerate(times):
		file = open(output_firstrun_dir + "Timing_FirstRun_" + str(idx+1) + ".txt", "w")
		file.write(str(time))
		file.close()	

def write_statistics(times):

	statistics_file = open(output_firstrun_dir + "Timing_FirstRun.txt", "w")
	
	statistics_file.write("mean = " + str(sum(times)/len(times)))
	
	statistics_file.close()

	return None
	
times = load_times()
write_times(times)
write_statistics(times)



	