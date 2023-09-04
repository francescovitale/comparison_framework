for /D %%p IN ("Output\FirstRun\PP\EventLogs\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
del /F /Q Output\FirstRun\PP\NormalLabels\*
del /F /Q Output\FirstRun\PP\NormativeModel\*
del /F /Q Output\FirstRun\DG\Diagnoses\diagnoses.csv
del /F /Q Output\FirstRun\AD\Training\Model\*
del /F /Q Output\FirstRun\AD\Training\Compression\*
del /F /Q Output\FirstRun\AD\Training\Normalization\*
del /F /Q Output\FirstRun\AD\Inference\*
del /F /Q Output\FirstRun\AD\Inference\Timing\*
del /F /Q Output\FirstRun\MS\*
del /F /Q Output\FirstRun\TE\*

::del /F /Q Output\SACycle\MS\*
::del /F /Q Output\SACycle\SA\NormalLabels\*
::del /F /Q Output\SACycle\ME\PetriNets\*
::del /F /Q Output\SACycle\ME\Statistics\*
::del /F /Q Output\SACycle\AD\Inference\*
::del /F /Q Output\SACycle\AD\Inference\Timing\*
::del /F /Q Output\SACycle\AD\Training\*
::del /F /Q Output\SACycle\AD\Training\Model\*
::del /F /Q Output\SACycle\AD\Training\Compression\*
::del /F /Q Output\SACycle\AD\Training\Normalization\*


del /F /Q Input\FirstRun\AD\Inference\*
del /F /Q Input\FirstRun\AD\Inference\Model\*
del /F /Q Input\FirstRun\AD\Inference\Compression\*
del /F /Q Input\FirstRun\AD\Inference\Normalization\*
del /F /Q Input\FirstRun\AD\Training\*
del /F /Q Input\FirstRun\MS\Metrics\*
del /F /Q Input\FirstRun\TE\*
del /F /Q Input\FirstRun\DG\EventLogs\*

::for /D %%p IN ("Input\SACycle\ME\EventLogs\*") DO (
::	del /s /f /q %%p\*.*
::	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
::	rmdir "%%p" /s /q
::)
::del /F /Q Input\SACycle\ME\Labels\*
::del /F /Q Input\SACycle\AD\Inference\*
::del /F /Q Input\SACycle\AD\Inference\EventLogs\*
::del /F /Q Input\SACycle\AD\Inference\Compression\*
::del /F /Q Input\SACycle\AD\Inference\Normalization\*
::del /F /Q Input\SACycle\AD\Inference\Model\*
::del /F /Q Input\SACycle\AD\Training\*
::del /F /Q Input\SACycle\AD\Training\EventLogs\*
::del /F /Q Input\SACycle\MS\Metrics\*
::del /F /Q Input\SACycle\SA\EventLogs\*
::del /F /Q Input\SACycle\SA\Model\*
