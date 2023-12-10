# Requirements and instructions to run the comparison framework for control-flow anomaly detection methods
This repository contains the implementation of a comparison framework for control-flow anomaly detection techniques. The implementation involves both the instantiation of several methods within method categories and the setup of each environment according to the first step of the comparison framework, namely Event Log Preprocessing.

Before delving into the details of the project files, please consider that this project has been executed on a Windows 10 machine with Python 3.8.1. There are a few libraries that have been used within Python modules. Among these, there are:

- scikit-learn 1.0.2
- scipy 1.8.0
- pm4py 2.2.19.1
- pandas 1.4.1
- tensorflow 2.8.0

Please note that the list above is not comprehensive and there could be other requirements for running the project.

The project provides a batch file for each dataset used to evaluate the performance of the control-flow anomaly detection methods employed. For example, autorun_ERTMS.bat contains the directives to run the project against the ERTMS dataset. All the scripts that are called within the .bat file are within the folder with the same name as what follows the underscore in the .bat file. Considering the example, the ERTMS folder contains the scripts for running the project against the ERTMS dataset. The structure of each batch file is the same and as follows:

```
cd X

call clean_all_results
call clean_all
call setup_all
call start_all
call clean_all

cd ..
```
**X** is the dataset against which the project is run. **clean_all_results** cleans the results of all control-flow anomaly detection methods related to **X**. **clean_all** cleans the sub-folders within **X** to delete input/output files from previous executions of the methods. **setup_all** sets up the environment related to **X** according to the specific Event Log Preprocessing implementation. **start_all** executes all methods against the **X** dataset. **clean_all** cleans the sub-folders within **X** to delete input/output files from previous executions of the methods.

Since the execution of the methods is subject to statistical fluctuations as regards the results obtained, one can set the number of repetitions/runs for each method, as follows:

1. Go into the **X** folder;
2. Open the start_all.bat batch file with a text editor;
3. Edit the batch file by specifying the desired number of repetitions/runs per method.

Once autorun_**X**.bat runs through and ends, the results of each method are collected in the Results sub-folder of the **X** folder.

## Sample project run
Let us consider a sample project run by running the project against the ERTMS dataset. We set the number of desired repetitions as described above. Then, we run the autorun_ERTMS.bat batch file. Finally, we navigate the Results sub-folder of the ERTMS folder to, say, check the results of the AB_AE method. This means that we need to go further down the folder tree by selecting CC_DR_ML first and AB_AE second. Once here, we can check the mean performance results (Metrics_FirstRun.txt) or the performance results per run (Metrics_FirstRun_y.txt).

## Further research and incomplete implementations

It is worth mentioning that the expression "FirstRun" in the result files indicates that the project evaluates the anomaly detection method with respect to the first execution in hypothetical production environments. However, real settings could require self-adapting to run-time changes in the operation of such environments. Further research could explore whether control-flow anomaly detection could adapt to new run-time conditions. This also motivates the incomplete self-adaptation files within method folders that further research should address.





