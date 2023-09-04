set n_traces_per_log=5
set sa_debug_mode=0
set n_logs_per_type=200
set visualization_purpose=0
set reference_pn=1

if not %sa_debug_mode%==1 (del /F /Q Results_SA\*)
if %sa_debug_mode%==1 (del /F /Q Results_PSA\*)

copy Model\NormativeModel.pnml Input\FirstRun\DG\NormativeModels

python preprocessing.py %n_traces_per_log% %n_logs_per_type%
::xcopy Output\FirstRun\PP\EventLogs Input\SACycle\ME\EventLogs /s /e