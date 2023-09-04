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

from tensorflow.keras.layers import Dense,Input,Concatenate
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

input_firstrun_dir = "Input/FirstRun/AD/"
output_firstrun_dir = "Output/FirstRun/AD/"

input_firstrun_training_dir = input_firstrun_dir + "Training/"
input_firstrun_inference_dir = input_firstrun_dir + "Inference/"
input_firstrun_inference_model_dir = input_firstrun_inference_dir + "Model/"


output_firstrun_training_dir = output_firstrun_dir + "Training/"
output_firstrun_training_model_dir = output_firstrun_training_dir + "Model/"
output_firstrun_inference_dir = output_firstrun_dir + "Inference/"
output_firstrun_inference_timing_dir = output_firstrun_inference_dir + "Timing/"

input_sacycle_dir = "Input/SACycle/AD/"
output_sacycle_dir = "Output/SACycle/AD/"

input_sacycle_training_dir = input_sacycle_dir + "Training/"
input_sacycle_inference_dir = input_sacycle_dir + "Inference/"
input_sacycle_inference_model_dir = input_sacycle_inference_dir + "Model/"

output_sacycle_training_dir = output_sacycle_dir + "Training/"
output_sacycle_training_model_dir = output_sacycle_training_dir + "Model/"
output_sacycle_inference_dir = output_sacycle_dir + "Inference/"
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
	
def read_model(run_mode):


	if run_mode == "FirstRun":
		ae_model = load_model(input_firstrun_inference_model_dir + "ae_model.h5")
		ae_mse = read_mse(run_mode)
		
		return ae_model, ae_mse
	elif run_mode == "SACycle":
		ae_model = load_model(input_sacycle_inference_model_dir + "ae_model.h5")
		ae_mse = read_mse(run_mode)
		
		return ae_model, ae_mse

	
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
		

try:
	run_mode = sys.argv[1]
	mode = sys.argv[2]
	debug_mode = int(sys.argv[3])
	sample_time = int(sys.argv[4])
	if mode == "training":
		validation_split_percentage = float(sys.argv[5])
	if mode == "training":
		hidden_neurons = int(sys.argv[6])
		latent_code_dimension = int(sys.argv[7])
		epochs = int(sys.argv[8])
	
except IndexError:
	print("Insert the right number of input arguments")


if mode == "training":
	diagnoses = read_diagnoses(mode, run_mode)
	normal_labels = read_normal_labels(mode, run_mode)
	diagnoses = diagnoses[diagnoses["label"].isin(normal_labels)]
	diagnoses = diagnoses.drop(labels = "label", axis = 1)
	
	# The following piece of code should be replaced to address the case where diagnoses from the training and validation sets differ. For now, this works, but real implementations need to address this.
	diagnoses = diagnoses.reindex(sorted(diagnoses.columns), axis=1)
	ae_model = train_autoencoder(diagnoses, validation_split_percentage, hidden_neurons, latent_code_dimension, epochs, run_mode)
	ae_model_recons = ae_model.predict(diagnoses)
	ae_model_mse = mean_squared_error(diagnoses, ae_model_recons)
	save_mse(ae_model_mse, run_mode)
	

elif mode == "inference":
	diagnoses = read_diagnoses(mode, run_mode)
	diagnoses_labels = list(diagnoses["label"])
	normal_labels = read_normal_labels(mode, run_mode)
	diagnoses = diagnoses.drop(labels = "label", axis = 1)
	
	ae_model, ae_model_mse = read_model(run_mode)
	# The following commented piece of code should be replaced with other code taking into account that there could be diagnoses which are not found during training. The commented piece of code does not address this problem properly
	'''
	if len(list(diagnoses.columns)) != ae_model.input_shape[1]:
		dimension_difference = ae_model.input_shape[1] - len(list(diagnoses.columns))
		if dimension_difference < 0:
			for i in range(0, abs(dimension_difference)):
				diagnoses = diagnoses.drop(labels = [random.choice(list(diagnoses.columns))], axis=1)
	'''
	# The following piece of code should be replaced to address the case where diagnoses from the training and test sets differ. For now, this works, but real implementations need to address this.
	diagnoses = diagnoses.reindex(sorted(diagnoses.columns), axis=1)
	
	if sample_time == 1:
		tm = 0
		tm = time.time()
	classifications = classify_diagnoses(ae_model, ae_model_mse, diagnoses)
	if sample_time == 1:
		tm = time.time() - tm
		write_timing(tm, run_mode)
	accuracy, precision, recall, f1, tp, tn, fp, fn = get_performance(normal_labels, diagnoses_labels, classifications)
	write_metrics(accuracy, precision, recall, f1, tp, tn, fp, fn, run_mode)
		





