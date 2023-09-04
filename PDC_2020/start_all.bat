set nreps=1

cd Methods

cd CC_D_ML
cd AB_KNN
call first_run %nreps%
call get_features
cd ../..
copy CC_D_ML\AB_KNN\Results_SA\* ..\Results\CC_D_ML\AB_KNN

cd CC_DR_ML
cd AB_AE
call first_run %nreps%
cd ../..
copy CC_DR_ML\AB_AE\Results_SA\* ..\Results\CC_DR_ML\AB_AE

cd D_ML
cd AF_KNN
call first_run %nreps%
call get_features
cd ..
cd DF_KNN
call first_run %nreps%
call get_features
cd ..
cd NG_KNN
call first_run %nreps%
call get_features
cd ../..
copy D_ML\AF_KNN\Results_SA\* ..\Results\D_ML\AF_KNN
copy D_ML\DF_KNN\Results_SA\* ..\Results\D_ML\DF_KNN
copy D_ML\NG_KNN\Results_SA\* ..\Results\D_ML\NG_KNN

cd DR_ML
cd NG_AE
call first_run %nreps%
cd ..
cd DF_AE
call first_run %nreps%
cd ..
cd AF_AE
call first_run %nreps%
cd ../..
copy DR_ML\NG_AE\Results_SA\* ..\Results\DR_ML\NG_AE
copy DR_ML\DF_AE\Results_SA\* ..\Results\DR_ML\DF_AE
copy DR_ML\AF_AE\Results_SA\* ..\Results\DR_ML\AF_AE

cd PD_CC
cd HM_TR
call first_run %nreps%
cd ..
cd IM_TR
call first_run %nreps%
cd ../..
copy PD_CC\HM_TR\Results_SA\* ..\Results\PD_CC\HM_TR
copy PD_CC\IM_TR\Results_SA\* ..\Results\PD_CC\IM_TR

cd ..