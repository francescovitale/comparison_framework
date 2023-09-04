set debug_mode=0
set sa_debug_mode=0
set anomaly_detection_technique=ae
set clustering_technique=kmeans
set validation_split_percentage=0.2
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
set cc_variant=TOKEN_BASED
set cc_variant_sa=ALIGNMENT_BASED
set noise_threshold=0.2
set sample_time=0



:: Cleaning all outputs
del /F /Q Output\SACycle\AD\Inference\*
del /F /Q Output\SACycle\MS\*
del /F /Q Output\SACycle\SA\NormalLabels\*

:: Cleaning all inputs
del /F /Q Input\SACycle\AD\Training\EventLogs\*
del /F /Q Input\SACycle\AD\Inference\EventLogs\*
del /F /Q Input\SACycle\SA\EventLogs\*
del /F /Q Input\SACycle\AD\Inference\NormalLabels.txt
del /F /Q Input\SACycle\AD\Training\NormalLabels.txt
del /F /Q Input\SACycle\MS\Metrics\*

copy Output\FirstRun\PP\EventLogs\0\* Input\SACycle\AD\Inference\EventLogs
copy Output\FirstRun\PP\EventLogs\%cycle_number%\* Input\SACycle\SA\EventLogs
copy Output\FirstRun\PP\EventLogs\0\* Input\SACycle\AD\Training\EventLogs


if %cycle_number%==1 (
	copy Output\FirstRun\AD\Training\Model\* Input\SACycle\AD\Inference\Model
)

del /F /Q Output\SACycle\AD\Training\Model\*

copy Output\FirstRun\PP\NormalLabels\NormalLabels_%cycle_number%.txt Input\SACycle\AD\Inference
ren Input\SACycle\AD\Inference\NormalLabels_%cycle_number%.txt NormalLabels.txt

python anomaly_detection.py SACycle inference %debug_mode% %sample_time% %noise_threshold% %cc_variant%

ren Output\SACycle\AD\Inference\Metrics.txt Metrics_%cycle_number%_pre.txt

if not %self_adapt%==1 (
	copy Output\SACycle\AD\Inference\Metrics_%cycle_number%_pre.txt Results_SA
)

if %self_adapt%==1 (
	

	del /F /Q Input\SACycle\AD\Inference\Model\*


	for /l %%x in (1, 1, 3) do (
		python self_adaptation.py %sa_debug_mode% %%x %clustering_technique% %cycle_number% %normalize_compress% %cc_variant_sa% %normalization_technique% %n_pca_components% %sa_technique% %max_clusters% %normal_cluster_threshold_percentage%
		copy Output\SACycle\SA\NormalLabels\NormalLabels.txt Input\SACycle\AD\Training
	
		if not %sa_debug_mode%==1 (
			ren Output\SACycle\SA\NormalLabels\NormalLabels.txt NormalLabels_%cycle_number%_%%x_temp.txt
				
			for /l %%x in (1, 1, 3) do (
				python anomaly_detection.py SACycle training %debug_mode% %sample_time% %noise_threshold% %cc_variant% %validation_split_percentage%
				
				copy Output\SACycle\AD\Training\Model\* Input\SACycle\AD\Inference\Model

				python anomaly_detection.py SACycle inference %debug_mode% %sample_time% %noise_threshold% %cc_variant%
				
				ren Output\SACycle\AD\Training\Model\threshold.txt threshold_%%x.txt
				ren Output\SACycle\AD\Training\Model\NormativeModel.pnml NormativeModel_%%x.pnml
				ren Input\SACycle\AD\Inference\Model\threshold.txt threshold_%%x.txt
				ren Input\SACycle\AD\Inference\Model\NormativeModel.pnml NormativeModel_%%x.pnml
				
				ren Output\SACycle\AD\Inference\Metrics.txt Metrics_%%x.txt
			)
			
			copy Output\SACycle\AD\Inference\* Input\SACycle\MS\Metrics

			python model_selector.py SACycle sacycle_%cycle_number%_sa_%%x_statistics
			del /F /Q Input\SACycle\MS\Metrics\*
			
			ren Input\SACycle\AD\Inference\Model\threshold.txt threshold_%%x_temp.txt
			ren Input\SACycle\AD\Inference\Model\NormativeModel.pnml NormativeModel_%%x_temp.pnml
			ren Output\SACycle\AD\Inference\Metrics.txt Metrics_%%x_temp.txt
	
		)
	
	

	)
	
	if not %sa_debug_mode%==1 (

		for /l %%x in (1, 1, 3) do (
			ren Output\SACycle\AD\Inference\Metrics_%%x_temp.txt Metrics_%%x.txt
			ren Output\SACycle\SA\NormalLabels\NormalLabels_%cycle_number%_%%x_temp.txt NormalLabels_%cycle_number%_%%x.txt
			ren Input\SACycle\AD\Inference\Model\threshold_%%x_temp.txt threshold_%%x.txt
			ren Input\SACycle\AD\Inference\Model\NormativeModel_%%x_temp.pnml NormativeModel_%%x.pnml
		)

	)
	
	if not %sa_debug_mode%==1 (ren Output\SACycle\SA\NormalLabels\Real_NormalLabels.txt Real_NormalLabels_%cycle_number%.txt)
	
	copy Output\SACycle\AD\Inference\* Input\SACycle\MS\Metrics

	python model_selector.py SACycle sacycle_%cycle_number%_sa_overall_statistics

	del /F /Q Input\SACycle\MS\Metrics\*
	
	
	if not %sa_debug_mode%==1 (
	
		ren Input\SACycle\AD\Inference\Model\threshold.txt threshold_SA.txt
		ren Input\SACycle\AD\Inference\Model\NormativeModel.pnml NormativeModel_SA.pnml

		ren Output\SACycle\SA\NormalLabels\NormalLabels.txt NormalLabels_%cycle_number%.txt
		copy Output\SACycle\SA\NormalLabels\* Results_SA
		copy Output\SACycle\SA\NormalLabels\* Input\SACycle\ME\Labels

	)
	
	if not %sa_debug_mode%==1 (
		ren Output\SACycle\AD\Inference\Metrics.txt Metrics_SA_%cycle_number%_post.txt
		del Input\SACycle\AD\Training\NormalLabels.txt
		copy Output\SACycle\SA\NormalLabels\Real_NormalLabels_%cycle_number%.txt Input\SACycle\AD\Training
		ren Input\SACycle\AD\Training\Real_NormalLabels_%cycle_number%.txt NormalLabels.txt

		for /l %%x in (1, 1, 3) do (
			
			python anomaly_detection.py SACycle training %debug_mode% %sample_time% %noise_threshold% %cc_variant% %validation_split_percentage%
			copy Output\SACycle\AD\Training\Model\* Input\SACycle\AD\Inference\Model
			python anomaly_detection.py SACycle inference %debug_mode% %sample_time% %noise_threshold% %cc_variant%
			
			ren Output\SACycle\AD\Training\Model\threshold.txt threshold_%%x.txt
			ren Output\SACycle\AD\Training\Model\NormativeModel.pnml NormativeModel_%%x.pnml
			ren Input\SACycle\AD\Inference\Model\threshold.txt threshold_%%x.txt
			ren Input\SACycle\AD\Inference\Model\NormativeModel.pnml NormativeModel_%%x.pnml
				
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







