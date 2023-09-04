del /F /Q Output\FirstRun\AD\Inference\Features\*
del /F /Q Input\FirstRun\AD\Inference\EventLogs\*

copy Output\FirstRun\PP\EventLogs\0\* Input\FirstRun\AD\Inference\EventLogs
python anomaly_detection.py FirstRun inference 0 0 2 zscore 2 3 0
copy Output\FirstRun\AD\Inference\Features\features.csv Results_SA
