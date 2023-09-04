del /F /Q Output\FirstRun\DG\Diagnoses\diagnoses.csv
del /F /Q Input\FirstRun\DG\EventLogs\*

copy Output\FirstRun\PP\EventLogs\0\* Input\FirstRun\DG\EventLogs
python diagnoses_generation.py 0 FirstRun 0 0 0 0
copy Output\FirstRun\DG\Diagnoses\diagnoses.csv Results_SA
