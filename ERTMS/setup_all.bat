cd Methods

cd CC_D_ML
cd AB_KNN
call setup
cd ../..
copy CC_DR_ML\AB_AE\Model\NormativeModel.pnml CC_DR_ML\AB_AE\Input\FirstRun\DG\NormativeModels
xcopy CC_D_ML\AB_KNN\Output\FirstRun\PP\ CC_DR_ML\AB_AE\Output\FirstRun\PP /s /e
xcopy CC_D_ML\AB_KNN\Output\FirstRun\PP\ D_ML\AF_KNN\Output\FirstRun\PP /s /e
xcopy CC_D_ML\AB_KNN\Output\FirstRun\PP\ D_ML\DF_KNN\Output\FirstRun\PP /s /e
xcopy CC_D_ML\AB_KNN\Output\FirstRun\PP\ D_ML\NG_KNN\Output\FirstRun\PP /s /e
xcopy CC_D_ML\AB_KNN\Output\FirstRun\PP\ DR_ML\AF_AE\Output\FirstRun\PP /s /e
xcopy CC_D_ML\AB_KNN\Output\FirstRun\PP\ DR_ML\DF_AE\Output\FirstRun\PP /s /e
xcopy CC_D_ML\AB_KNN\Output\FirstRun\PP\ DR_ML\NG_AE\Output\FirstRun\PP /s /e
xcopy CC_D_ML\AB_KNN\Output\FirstRun\PP\ PD_CC\HM_TR\Output\FirstRun\PP /s /e
xcopy CC_D_ML\AB_KNN\Output\FirstRun\PP\ PD_CC\IM_TR\Output\FirstRun\PP /s /e

cd ..






