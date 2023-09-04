set debug_mode=0
set sa_debug_mode=0
set clustering_technique=kmeans
set validation_split_percentage=0.2
set hidden_neurons=250
set latent_code_dimension=24
set epochs=1000
set cycle_number=%1
set self_adapt=1
set normal_cluster_threshold_percentage=0.2
set normalize=0
set normalize_compress=1
set max_clusters=5
set apply_pca=0
set n_pca_components=3
set sa_n_pca_components=3
set normalization_technique=zscore
set distance_threshold_percentage=0.2
set sa_technique=data_clustering
set sample_time=0



:: Cleaning all outputs
del /F /Q Output\SACycle\DG\Diagnoses\*
del /F /Q Output\SACycle\AD\Inference\*
del /F /Q Output\SACycle\MS\*
del /F /Q Output\SACycle\SA\NormalLabels\*

:: Cleaning all inputs
del /F /Q Input\SACycle\DG\EventLogs\*
del /F /Q Input\SACycle\AD\Inference\NormalLabels.txt
del /F /Q Input\SACycle\AD\Inference\diagnoses.csv
del /F /Q Input\SACycle\AD\Training\diagnoses.csv
del /F /Q Input\SACycle\AD\Training\NormalLabels.txt
del /F /Q Input\SACycle\MS\Metrics\*

:: The script logic starts here
copy Output\FirstRun\PP\EventLogs\0\* Input\SACycle\DG\EventLogs
python diagnoses_generation.py %debug_mode% SACycle %sample_time% %normalize% %apply_pca% %n_pca_components%
copy Output\SACycle\DG\Diagnoses\diagnoses.csv Input\SACycle\AD\Training
copy Output\SACycle\DG\Diagnoses\diagnoses.csv Input\SACycle\AD\Inference
del /F /Q Input\SACycle\DG\EventLogs\*
del /F /Q Output\SACycle\DG\Diagnoses\*


copy Output\FirstRun\PP\EventLogs\%cycle_number%\* Input\SACycle\DG\EventLogs

if %cycle_number%==1 (copy Output\FirstRun\AD\Training\Model\* Input\SACycle\AD\Inference\Model)

del /F /Q Output\SACycle\AD\Training\Model\*

copy Output\FirstRun\PP\NormalLabels\NormalLabels_%cycle_number%.txt Input\SACycle\AD\Inference
ren Input\SACycle\AD\Inference\NormalLabels_%cycle_number%.txt NormalLabels.txt

python diagnoses_generation.py %debug_mode% SACycle %sample_time% %normalize% %apply_pca% %n_pca_components%

copy Output\SACycle\DG\Diagnoses\diagnoses.csv Input\SACycle\SA\Diagnoses

python anomaly_detection.py SACycle inference %debug_mode% %sample_time% 

ren Output\SACycle\AD\Inference\Metrics.txt Metrics_%cycle_number%_pre.txt

if not %self_adapt%==1 (
	copy Output\SACycle\AD\Inference\Metrics_%cycle_number%_pre.txt Results_SA
)

if %self_adapt%==1 (

	ren Output\SACycle\AD\Inference\Metrics.txt Metrics_%cycle_number%_pre.txt

	del /F /Q Input\SACycle\AD\Inference\Model\*


	for /l %%x in (1, 1, 3) do (
		
		python self_adaptation.py %sa_debug_mode% %%x %clustering_technique% %cycle_number% %normalize_compress% %normalization_technique% %n_pca_components% %sa_technique% %max_clusters% %normal_cluster_threshold_percentage%
		copy Output\SACycle\SA\NormalLabels\NormalLabels.txt Input\SACycle\AD\Training
		
		
		if not %sa_debug_mode%==1 (
			ren Output\SACycle\SA\NormalLabels\NormalLabels.txt NormalLabels_%cycle_number%_%%x_temp.txt
			
			for /l %%x in (1, 1, 3) do (
				python anomaly_detection.py SACycle training %debug_mode% %sample_time% %validation_split_percentage% %hidden_neurons% %latent_code_dimension% %epochs%
				
				copy Output\SACycle\AD\Training\Model\* Input\SACycle\AD\Inference\Model

				python anomaly_detection.py SACycle inference %debug_mode% %sample_time% 
				
				
				ren Output\SACycle\AD\Training\Model\ae_model.h5 ae_model_%%x.h5
				ren Output\SACycle\AD\Training\Model\ae_model_mse.txt ae_model_mse_%%x.txt
				ren Input\SACycle\AD\Inference\Model\ae_model.h5 ae_model_%%x.h5
				ren Input\SACycle\AD\Inference\Model\ae_model_mse.txt ae_model_mse_%%x.txt
				ren Output\SACycle\AD\Inference\Metrics.txt Metrics_%%x.txt
				
				
			)	
			
			copy Output\SACycle\AD\Inference\* Input\SACycle\MS\Metrics

			python model_selector.py SACycle sacycle_%cycle_number%_sa_%%x_statistics
			del /F /Q Input\SACycle\MS\Metrics\*
			
			ren Input\SACycle\AD\Inference\Model\ae_model.h5 ae_model_%%x_temp.h5
			ren Input\SACycle\AD\Inference\Model\ae_model_mse.txt ae_model_mse_%%x_temp.txt

			
			ren Output\SACycle\AD\Inference\Metrics.txt Metrics_%%x_temp.txt
		)
		
		if %sa_debug_mode%==1 (
			ren Output\SACycle\SA\NormalLabels\NormalLabels.txt NormalLabels_%cycle_number%_%%x.txt
			python anomaly_detection.py SACycle training %debug_mode% %sample_time% %validation_split_percentage% %hidden_neurons% %latent_code_dimension% %epochs%
				
			copy Output\SACycle\AD\Training\Model\* Input\SACycle\AD\Inference\Model

			python anomaly_detection.py SACycle inference %debug_mode% %sample_time%
				
				
			ren Output\SACycle\AD\Training\Model\ae_model.h5 ae_model_%%x.h5
			ren Output\SACycle\AD\Training\Model\ae_model_mse.txt ae_model_mse_%%x.txt
			ren Input\SACycle\AD\Inference\Model\ae_model.h5 ae_model_%%x.h5
			ren Input\SACycle\AD\Inference\Model\ae_model_mse.txt ae_model_mse_%%x.txt
			ren Output\SACycle\AD\Inference\Metrics.txt Metrics_%%x.txt
		
		)

	)

	if not %sa_debug_mode%==1 (

		for /l %%x in (1, 1, 3) do (
			ren Output\SACycle\AD\Inference\Metrics_%%x_temp.txt Metrics_%%x.txt
			ren Output\SACycle\SA\NormalLabels\NormalLabels_%cycle_number%_%%x_temp.txt NormalLabels_%cycle_number%_%%x.txt
			ren Input\SACycle\AD\Inference\Model\ae_model_%%x_temp.h5 ae_model_%%x.h5
			ren Input\SACycle\AD\Inference\Model\ae_model_mse_%%x_temp.txt ae_model_mse_%%x.txt
		)

	)

	if not %sa_debug_mode%==1 (ren Output\SACycle\SA\NormalLabels\Real_NormalLabels.txt Real_NormalLabels_%cycle_number%.txt)

	copy Output\SACycle\AD\Inference\* Input\SACycle\MS\Metrics

	python model_selector.py SACycle sacycle_%cycle_number%_sa_overall_statistics
	del /F /Q Input\SACycle\MS\Metrics\*


	if not %sa_debug_mode%==1 (

		ren Input\SACycle\AD\Inference\Model\ae_model.h5 ae_model_SA.h5
		ren Input\SACycle\AD\Inference\Model\ae_model_mse.txt ae_model_mse_SA.txt
		

		ren Output\SACycle\SA\NormalLabels\NormalLabels.txt NormalLabels_%cycle_number%.txt
		copy Output\SACycle\SA\NormalLabels\* Results_SA
		copy Output\SACycle\SA\NormalLabels\* Input\SACycle\ME\Labels

	)


	if %sa_debug_mode%==1 (
	ren Output\SACycle\AD\Inference\Metrics.txt Metrics_%cycle_number%_post.txt
	copy Output\SACycle\AD\Inference\Metrics_%cycle_number%_pre.txt Results_PSA
	copy Output\SACycle\AD\Inference\Metrics_%cycle_number%_post.txt Results_PSA
	ren Output\SACycle\SA\NormalLabels\NormalLabels.txt NormalLabels_%cycle_number%.txt
	copy Output\SACycle\SA\NormalLabels\NormalLabels_%cycle_number%.txt Results_PSA
	)
	if not %sa_debug_mode%==1 (
	ren Output\SACycle\AD\Inference\Metrics.txt Metrics_SA_%cycle_number%_post.txt
	del Input\SACycle\AD\Training\NormalLabels.txt
	copy Output\SACycle\SA\NormalLabels\Real_NormalLabels_%cycle_number%.txt Input\SACycle\AD\Training
	ren Input\SACycle\AD\Training\Real_NormalLabels_%cycle_number%.txt NormalLabels.txt

	for /l %%x in (1, 1, 3) do (
		
		python anomaly_detection.py SACycle training %debug_mode% %sample_time% %validation_split_percentage% %hidden_neurons% %latent_code_dimension% %epochs%
		copy Output\SACycle\AD\Training\Model\* Input\SACycle\AD\Inference\Model
		python anomaly_detection.py SACycle inference %debug_mode% %sample_time%
		
		ren Output\SACycle\AD\Training\Model\ae_model.h5 ae_model_%%x.h5
		ren Output\SACycle\AD\Training\Model\ae_model_mse.txt ae_model_mse_%%x.txt
		ren Input\SACycle\AD\Inference\Model\ae_model.h5 ae_model_%%x.h5
		ren Input\SACycle\AD\Inference\Model\ae_model_mse.txt ae_model_mse_%%x.txt
		ren Output\SACycle\AD\Inference\Metrics.txt Metrics_%%x.txt
		
	)

	copy Output\SACycle\AD\Inference\* Input\SACycle\MS\Metrics

	python model_selector.py SACycle sacycle_%cycle_number%_psa_statistics

	
	python model_extraction.py %cycle_number%

	ren Output\SACycle\AD\Inference\Metrics.txt Metrics_PSA_%cycle_number%_post.txt

	copy Output\SACycle\AD\Inference\Metrics_%cycle_number%_pre.txt Results_SA
	copy Output\SACycle\AD\Inference\Metrics_PSA_%cycle_number%_post.txt Results_SA
	copy Output\SACycle\AD\Inference\Metrics_SA_%cycle_number%_post.txt Results_SA
	
	copy Output\SACycle\ME\PetriNets\NormalEventLogs_PetriNet_%cycle_number%.pnml Results_SA
	copy Output\SACycle\ME\PetriNets\Real_NormalEventLogs_PetriNet_%cycle_number%.pnml Results_SA
	copy Output\SACycle\ME\Statistics\similarity_statistics_%cycle_number%.txt Results_SA
	
	copy Output\SACycle\MS\* Results_SA
	)
)


