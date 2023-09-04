set debug_mode=1
set validation_split_percentage=0.3
set hidden_neurons=150
set latent_code_dimension=24
set epochs=1000

python anomaly_detection.py FirstRun training %debug_mode% %validation_split_percentage% %hidden_neurons% %latent_code_dimension% %epochs%

copy Output\FirstRun\AD\Training\Model\* Input\FirstRun\AD\Inference\Model

python anomaly_detection.py FirstRun inference %debug_mode%