:: %1: nreps

set debug_mode=0
set validation_split_percentage=0.2
set normalization_technique=zscore
set n_pca_components=3
set n_neighbors=3
set use_categorical_labels=0
set nreps=%1

:: Cleaning all outputs
del /F /Q Output\FirstRun\AD\Training\Model\*
del /F /Q Output\FirstRun\AD\Training\Normalization\*
del /F /Q Output\FirstRun\AD\Training\Compression\*
del /F /Q Output\FirstRun\AD\Inference\*
del /F /Q Output\FirstRun\AD\Inference\Timing\*
del /F /Q Output\FirstRun\TE\*
del /F /Q Output\FirstRun\MS\*

:: Cleaning all inputs
del /F /Q Input\FirstRun\AD\Inference\Model\*
del /F /Q Input\FirstRun\AD\Training\*
del /F /Q Input\FirstRun\TE\*
del /F /Q Input\FirstRun\AD\Inference\*
del /F /Q Input\FirstRun\AD\GenerateDiagnoses\*
del /F /Q Input\FirstRun\AD\Training\EventLogs\*
del /F /Q Input\FirstRun\AD\Inference\EventLogs\*
del /F /Q Input\FirstRun\AD\Inference\Normalization\*
del /F /Q Input\FirstRun\AD\Inference\Compression\*
del /F /Q Input\FirstRun\MS\Metrics\*

:: The script logic starts here
copy Output\FirstRun\PP\EventLogs\0\* Input\FirstRun\AD\Training\EventLogs
copy Output\FirstRun\PP\EventLogs\0\* Input\FirstRun\AD\Inference\EventLogs
copy Output\FirstRun\PP\NormalLabels\NormalLabels_0.txt Input\FirstRun\AD\Inference
ren Input\FirstRun\AD\Inference\NormalLabels_0.txt NormalLabels.txt
copy Output\FirstRun\PP\NormalLabels\NormalLabels_0.txt Input\FirstRun\AD\Training
ren Input\FirstRun\AD\Training\NormalLabels_0.txt NormalLabels.txt

:: The training-inference cycle starts here
set sample_time=1
for /l %%x in (1, 1, %nreps%) do (
		
	python anomaly_detection.py FirstRun training %debug_mode% %sample_time% %normalization_technique% %n_pca_components% %n_neighbors% %use_categorical_labels% %validation_split_percentage%

	copy Output\FirstRun\AD\Training\Model\* Input\FirstRun\AD\Inference\Model
	copy Output\FirstRun\AD\Training\Normalization\* Input\FirstRun\AD\Inference\Normalization
	copy Output\FirstRun\AD\Training\Compression\* Input\FirstRun\AD\Inference\Compression

	python anomaly_detection.py FirstRun inference %debug_mode% %sample_time% %normalization_technique% %n_pca_components% %n_neighbors% %use_categorical_labels% %validation_split_percentage%
		
	ren Input\FirstRun\AD\Inference\Model\knn.pkl knn_%%x.pkl	
	ren Input\FirstRun\AD\Inference\Model\threshold.txt threshold_%%x.txt
	ren Input\FirstRun\AD\Inference\Normalization\normalization_parameters.txt normalization_parameters_%%x.txt
	ren Input\FirstRun\AD\Inference\Compression\training_pca_model.pkl training_pca_model_%%x.pkl
	ren Output\FirstRun\AD\Inference\Metrics.txt Metrics_%%x.txt
	copy Output\FirstRun\AD\Inference\Metrics_%%x.txt Results_PSA
	copy Output\FirstRun\AD\Inference\Metrics_%%x.txt Results_SA
	ren Results_SA\Metrics_%%x.txt Metrics_FirstRun_%%x.txt
	ren Results_PSA\Metrics_%%x.txt Metrics_FirstRun_%%x.txt

)

ren Output\FirstRun\AD\Inference\Timing\timing.txt AD_timing_FirstRun.txt
copy Output\FirstRun\AD\Inference\Timing\AD_timing_FirstRun.txt Input\FirstRun\TE
copy Output\FirstRun\AD\Inference\* Input\FirstRun\MS\Metrics

python model_selector.py FirstRun firstrun_statistics

python time_evaluation.py

copy Output\FirstRun\TE\* Results_SA
copy Output\FirstRun\TE\* Results_PSA
copy Output\FirstRun\MS\* Results_SA
ren Results_SA\firstrun_statistics.txt Metrics_FirstRun.txt






