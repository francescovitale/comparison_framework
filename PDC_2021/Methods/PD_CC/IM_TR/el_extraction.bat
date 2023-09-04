set training_trace_percentage=0.01
set testing_trace_percentage=0.1

del /F /Q Output\FirstRun\ELE\EventLogs\*
del /F /Q Input\FirstRun\PP\EventLogs\*

python event_logs_extraction.py %training_trace_percentage% %testing_trace_percentage%

copy Output\FirstRun\ELE\EventLogs\* Input\FirstRun\PP\EventLogs
