import sys
import os

input_dir = "Input/"
input_firstrun_dir = input_dir + "FirstRun/MS/"
input_firstrun_metrics_dir = input_firstrun_dir + "Metrics/"

input_sacycle_dir = input_dir + "SACycle/MS/"
input_sacycle_metrics_dir = input_sacycle_dir + "Metrics/"

output_dir = "Output/"
output_firstrun_dir = output_dir + "FirstRun/MS/"
output_sacycle_dir = output_dir + "SACycle/MS/"


def read_metrics(run_mode):

	metrics = {}
	
	if run_mode == "FirstRun":
		metrics_dir = input_firstrun_metrics_dir
	elif run_mode == "SACycle":
		metrics_dir = input_sacycle_metrics_dir
		
	for metrics_filename in os.listdir(metrics_dir):
		if len(metrics_filename.split(".")[0].split("_")) < 3:
			metrics[metrics_filename.split(".")[0]] = {}
			file = open(metrics_dir + metrics_filename, "r")
			for line in file:
				tokens = line.split("=")
				for idx,token in enumerate(tokens):
					tokens[idx] = tokens[idx].strip()
				metrics[metrics_filename.split(".")[0]][tokens[0]] = float(tokens[1])
		
		
	return metrics
	
def clean_environment(best_accuracy_model, run_mode):

	if run_mode == "FirstRun":
		for model_filename in os.listdir(output_dir + run_mode + "/AD/Training/Model"):
			if model_filename.split(".")[0].split("_")[-1] != str(best_accuracy_model):
				os.remove(output_dir + run_mode + "/AD/Training/Model/" + model_filename)
			else:
				model_filename_tokens = model_filename.split(".")[0].split("_")
				new_model_filename = ""
				for idx,token in enumerate(model_filename_tokens):
					if idx < len(model_filename_tokens)-2:
						new_model_filename = new_model_filename + token + "_"
					elif idx == len(model_filename_tokens)-2:
						new_model_filename = new_model_filename + token

				new_model_filename = new_model_filename + "." + model_filename.split(".")[-1]		
					
				os.rename(output_dir + run_mode + "/AD/Training/Model/" + model_filename, output_dir + run_mode + "/AD/Training/Model/" + new_model_filename)
				
		for model_filename in os.listdir(input_dir + run_mode + "/AD/Inference/Model"):
			if model_filename.split(".")[0].split("_")[-1] != str(best_accuracy_model):
				os.remove(input_dir + run_mode + "/AD/Inference/Model/" + model_filename)
			else:
				model_filename_tokens = model_filename.split(".")[0].split("_")
				new_model_filename = ""
				for idx,token in enumerate(model_filename_tokens):
					if idx < len(model_filename_tokens)-2:
						new_model_filename = new_model_filename + token + "_"
					elif idx == len(model_filename_tokens)-2:
						new_model_filename = new_model_filename + token
				new_model_filename = new_model_filename + "." + model_filename.split(".")[-1]		
					
				os.rename(input_dir + run_mode + "/AD/Inference/Model/" + model_filename, input_dir + run_mode + "/AD/Inference/Model/" + new_model_filename)		

		for metrics_filename in os.listdir(output_dir + run_mode + "/AD/Inference"):
			if metrics_filename != "Timing" and metrics_filename != "Features":
				if metrics_filename.split(".")[0].split("_")[-1] != str(best_accuracy_model):
					os.remove(output_dir + run_mode + "/AD/Inference/" + metrics_filename)
				elif metrics_filename.split(".")[0].split("_")[-1] != "pre":
					metrics_filename_tokens = metrics_filename.split(".")[0].split("_")
					new_metrics_filename = "Metrics_FirstRun.txt"
					os.rename(output_dir + run_mode + "/AD/Inference/" + metrics_filename, output_dir + run_mode + "/AD/Inference/" + new_metrics_filename)				
				
	elif run_mode == "SACycle":
		for model_filename in os.listdir(output_dir + run_mode + "/AD/Training/Model"):
			if model_filename.split(".")[0].split("_")[-1] != str(best_accuracy_model):
				os.remove(output_dir + run_mode + "/AD/Training/Model/" + model_filename)
			else:
				model_filename_tokens = model_filename.split(".")[0].split("_")
				new_model_filename = ""
				for idx,token in enumerate(model_filename_tokens):
					if idx < len(model_filename_tokens)-2:
						new_model_filename = new_model_filename + token + "_"
					elif idx == len(model_filename_tokens)-2:
						new_model_filename = new_model_filename + token
				new_model_filename = new_model_filename + "." + model_filename.split(".")[-1]		
					
				os.rename(output_dir + run_mode + "/AD/Training/Model/" + model_filename, output_dir + run_mode + "/AD/Training/Model/" + new_model_filename)
		
		inference_models = os.listdir(input_dir + run_mode + "/AD/Inference/Model")
		keep_sa_model = False
		for model_filename in os.listdir(input_dir + run_mode + "/AD/Inference/Model"):
			if model_filename.split(".")[0].split("_")[-1] == "SA":
				keep_sa_model = True
		
		if keep_sa_model == True:
			for model_filename in os.listdir(input_dir + run_mode + "/AD/Inference/Model"):
				if model_filename.split(".")[0].split("_")[-1] != "SA":
					os.remove(input_dir + run_mode + "/AD/Inference/Model/" + model_filename)
				else:
					model_filename_tokens = model_filename.split(".")[0].split("_")
					new_model_filename = ""
					for idx,token in enumerate(model_filename_tokens):
						if idx < len(model_filename_tokens)-2:
							new_model_filename = new_model_filename + token + "_"
						elif idx == len(model_filename_tokens)-2:
							new_model_filename = new_model_filename + token
					new_model_filename = new_model_filename + "." + model_filename.split(".")[-1]
					os.rename(input_dir + run_mode + "/AD/Inference/Model/" + model_filename, input_dir + run_mode + "/AD/Inference/Model/" + new_model_filename)
					
		else:
			for model_filename in os.listdir(input_dir + run_mode + "/AD/Inference/Model"):
				if model_filename.split(".")[0].split("_")[-1] != str(best_accuracy_model) and model_filename.split(".")[0].split("_")[-1] != "temp":
					os.remove(input_dir + run_mode + "/AD/Inference/Model/" + model_filename)
				elif model_filename.split(".")[0].split("_")[-1] != "temp":
					model_filename_tokens = model_filename.split(".")[0].split("_")
					new_model_filename = ""
					for idx,token in enumerate(model_filename_tokens):
						if idx < len(model_filename_tokens)-2:
							new_model_filename = new_model_filename + token + "_"
						elif idx == len(model_filename_tokens)-2:
							new_model_filename = new_model_filename + token
					new_model_filename = new_model_filename + "." + model_filename.split(".")[-1]		
						
					os.rename(input_dir + run_mode + "/AD/Inference/Model/" + model_filename, input_dir + run_mode + "/AD/Inference/Model/" + new_model_filename)	
					
		for idx,normal_labels_filename in enumerate(os.listdir(output_dir + run_mode + "/SA/NormalLabels")):
			if normal_labels_filename.split(".")[0].split("_")[-1] != str(best_accuracy_model) and normal_labels_filename.split(".")[0].split("_")[0] != "Real" and len(normal_labels_filename.split(".")[0].split("_")) < 4:
				os.remove(output_dir + run_mode + "/SA/NormalLabels/" + normal_labels_filename)
			elif normal_labels_filename.split(".")[0].split("_")[0] != "Real" and normal_labels_filename.split(".")[0].split("_")[-1] == str(best_accuracy_model):
				normal_labels_filename_tokens = normal_labels_filename.split(".")[0].split("_")
				new_normal_labels_filename = "NormalLabels"
				os.rename(output_dir + run_mode + "/SA/NormalLabels/" + normal_labels_filename, output_dir + run_mode + "/SA/NormalLabels/" + new_normal_labels_filename + ".txt")	
		
		for metrics_filename in os.listdir(output_dir + run_mode + "/AD/Inference"):	
			if metrics_filename != "Timing" and metrics_filename != "Features":
				if metrics_filename.split(".")[0].split("_")[-1] != str(best_accuracy_model) and len(metrics_filename.split(".")[0].split("_")) < 3:
					os.remove(output_dir + run_mode + "/AD/Inference/" + metrics_filename)
				elif len(metrics_filename.split(".")[0].split("_")) < 3 and metrics_filename.split(".")[0].split("_")[-1] == str(best_accuracy_model):
					metrics_filename_tokens = metrics_filename.split(".")[0].split("_")
					new_metrics_filename = "Metrics.txt"
					os.rename(output_dir + run_mode + "/AD/Inference/" + metrics_filename, output_dir + run_mode + "/AD/Inference/" + new_metrics_filename)
	
def evaluate_best_accuracy(metrics):
	best_accuracy_model = ""
	best_accuracy = 0.0
	for label in metrics:
		if metrics[label]["Accuracy"] > best_accuracy:
			best_accuracy = metrics[label]["Accuracy"]
			best_accuracy_model = label.split("_")[-1]
			
	return best_accuracy_model, best_accuracy
	
def write_best_performance(best_accuracy_model, best_accuracy,statistics_filename, run_mode):

	if run_mode == "FirstRun":
		best_performance_output_dir = output_firstrun_dir
	
	elif run_mode == "SACycle":
		best_performance_output_dir = output_sacycle_dir

	file = open(best_performance_output_dir + "best_model_" + statistics_filename + ".txt", "w")
	file.write("The best model was obtained at training run " + str(best_accuracy_model) + " with accuracy: " + str(best_accuracy))
	file.close()

	return None
	
def write_statistics(metrics, statistics_filename, run_mode):

	mean_accuracy = 0.0
	mean_precision = 0.0
	mean_recall = 0.0
	mean_f1 = 0.0
	for run in metrics:
		for figure in metrics[run]:
			if figure == "Accuracy":
				mean_accuracy += metrics[run][figure]
			elif figure == "Precision":
				mean_precision += metrics[run][figure]
			elif figure == "Recall":
				mean_recall += metrics[run][figure]
			elif figure == "F1":
				mean_f1 += metrics[run][figure]	
	
	mean_accuracy = mean_accuracy/len(metrics)
	mean_precision = mean_precision/len(metrics)
	mean_recall = mean_recall/len(metrics)
	mean_f1 = mean_f1/len(metrics)
	
	if run_mode == "FirstRun":
		file = open(output_firstrun_dir + statistics_filename + ".txt", "w")
	elif run_mode == "SACycle":
		file = open(output_sacycle_dir + statistics_filename + ".txt", "w")
	
	file.write("Mean accuracy = " + str(mean_accuracy) + "\n")
	file.write("Mean precision = " + str(mean_precision) + "\n")
	file.write("Mean recall = " + str(mean_recall) + "\n")
	file.write("Mean f1 = " + str(mean_f1))
	
	file.close()
	

	return None
	
try:
	run_mode = sys.argv[1]
	statistics_filename = sys.argv[2]

except IndexError:
	print("Insert the right number of inputs")
	sys.exit()

metrics = read_metrics(run_mode)
best_accuracy_model, best_accuracy = evaluate_best_accuracy(metrics)
#write_best_performance(best_accuracy_model, best_accuracy, statistics_filename, run_mode)
clean_environment(best_accuracy_model, run_mode)
write_statistics(metrics, statistics_filename, run_mode)












