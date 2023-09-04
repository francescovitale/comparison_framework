set debug_mode=1
set anomaly_detection_technique=ae
set validation_split_percentage=0.1
set hidden_neurons=50
set latent_code_dimension=10
set epochs=1000


del /F /Q Input\SACycle\AD\Inference\Model\*

python anomaly_detection.py SACycle training %debug_mode% %anomaly_detection_technique% %validation_split_percentage% %hidden_neurons% %latent_code_dimension% %epochs%

copy Output\SACycle\AD\Training\Model\* Input\SACycle\AD\Inference\Model

python anomaly_detection.py SACycle inference %debug_mode% %anomaly_detection_technique%