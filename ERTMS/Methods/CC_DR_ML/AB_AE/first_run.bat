:: %1: nreps

set apply_pca=0
set n_pca_components=3
set debug_mode=1
set validation_split_percentage=0.2
set hidden_neurons=250
set latent_code_dimension=24
set epochs=1000
set normalize=0
set nreps=%1

:: Cleaning all outputs
del /F /Q Output\FirstRun\DG\Diagnoses\diagnoses.csv
del /F /Q Output\FirstRun\AD\Training\Model\*
del /F /Q Output\FirstRun\AD\Inference\*
del /F /Q Output\FirstRun\MS\*
del /F /Q Output\FirstRun\TE\*
del /F /Q Output\FirstRun\AD\Inference\Timing\*
del /F /Q Output\FirstRun\DG\Timing\*

:: Cleaning all inputs
del /F /Q Input\FirstRun\AD\Inference\NormalLabels.txt
del /F /Q Input\FirstRun\AD\Inference\diagnoses.csv
del /F /Q Input\FirstRun\AD\Inference\Model\*
del /F /Q Input\FirstRun\AD\Training\diagnoses.csv
del /F /Q Input\FirstRun\AD\Training\*
del /F /Q Input\FirstRun\DG\EventLogs\*
del /F /Q Input\FirstRun\MS\Metrics\*
del /F /Q Input\FirstRun\TE\*
del /F /Q Input\FirstRun\DG\EventLogs\*
del /F /Q Input\FirstRun\AD\Inference\NormalLabels.txt
del /F /Q Input\FirstRun\AD\Training\NormalLabels.txt

:: The script logic starts here
copy Output\FirstRun\PP\EventLogs\0\* Input\FirstRun\DG\EventLogs
copy Output\FirstRun\PP\NormalLabels\NormalLabels_0.txt Input\FirstRun\AD\Inference
ren Input\FirstRun\AD\Inference\NormalLabels_0.txt NormalLabels.txt
copy Output\FirstRun\PP\NormalLabels\NormalLabels_0.txt Input\FirstRun\AD\Training
ren Input\FirstRun\AD\Training\NormalLabels_0.txt NormalLabels.txt

set sample_time=0
python diagnoses_generation.py %debug_mode% FirstRun %sample_time% %normalize% %apply_pca% %n_pca_components% 

copy Output\FirstRun\DG\Diagnoses\diagnoses.csv Input\FirstRun\AD\Training
copy Output\FirstRun\DG\Diagnoses\diagnoses.csv Input\FirstRun\AD\Inference

:: The training-inference cycle starts here
set sample_time=1
for /l %%x in (1, 1, %nreps%) do (
	
	python anomaly_detection.py FirstRun training %debug_mode% %sample_time% %validation_split_percentage% %hidden_neurons% %latent_code_dimension% %epochs%
		
	copy Output\FirstRun\AD\Training\Model\* Input\FirstRun\AD\Inference\Model


	python diagnoses_generation.py %debug_mode% FirstRun %sample_time% %normalize% %apply_pca% %n_pca_components% 

	python anomaly_detection.py FirstRun inference %debug_mode% %sample_time%
		
	ren Output\FirstRun\AD\Training\Model\ae_model.h5 ae_model_%%x.h5
	ren Output\FirstRun\AD\Training\Model\ae_model_mse.txt ae_model_mse_%%x.txt
	ren Input\FirstRun\AD\Inference\Model\ae_model.h5 ae_model_%%x.h5
	ren Input\FirstRun\AD\Inference\Model\ae_model_mse.txt ae_model_mse_%%x.txt
	ren Output\FirstRun\AD\Inference\Metrics.txt Metrics_%%x.txt
	copy Output\FirstRun\AD\Inference\Metrics_%%x.txt Results_PSA
	copy Output\FirstRun\AD\Inference\Metrics_%%x.txt Results_SA
	ren Results_SA\Metrics_%%x.txt Metrics_FirstRun_%%x.txt
	ren Results_PSA\Metrics_%%x.txt Metrics_FirstRun_%%x.txt
)

ren Output\FirstRun\AD\Inference\Timing\timing.txt AD_timing_FirstRun.txt
ren Output\FirstRun\DG\Timing\timing.txt DG_timing_FirstRun.txt
copy Output\FirstRun\AD\Inference\Timing\AD_timing_FirstRun.txt Input\FirstRun\TE
copy Output\FirstRun\DG\Timing\DG_timing_FirstRun.txt Input\FirstRun\TE

copy Output\FirstRun\AD\Inference\* Input\FirstRun\MS\Metrics

python model_selector.py FirstRun firstrun_statistics

python time_evaluation.py

copy Output\FirstRun\TE\* Results_SA
copy Output\FirstRun\TE\* Results_PSA
copy Output\FirstRun\MS\* Results_SA
ren Results_SA\firstrun_statistics.txt Metrics_FirstRun.txt










