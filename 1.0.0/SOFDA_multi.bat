REM Batch script for running python modules




echo off
REM Set all directories and variables
SET WORK_DIR=C:\LocalStore\School\UofA\Thesis\04_SOFT\py\ABMRI
SET PY_DIR=C:\LocalStore\School\UofA\Thesis\04_SOFT\py\ABMRI\0.0.2
SET OUT_DIR=C:\LocalStore\School\UofA\Thesis\04_SOFT\py\ABMRI\_outs\_batch


SET PARS_FILENAME=ABMRI274_10b_F6C1U1S2.xls

set PY_MOD=main_cmd.py


REM Change directory and execute the script
echo 'Executing python scripts from dir %WORK_DIR%'

REM cd %WORK_DIR%

REM dir > %PARS_FILENAME%.txt 2>&1


cd %PY_DIR%

echo on

REM redirect output
echo 'run 1'
powershell "python  %PY_MOD% %PARS_FILENAME% -wi | tee %OUT_DIR%\out1b_%PARS_FILENAME%.txt "
echo 'run 2'
powershell "python  %PY_MOD% %PARS_FILENAME% -wi | tee %OUT_DIR%\out2b_%PARS_FILENAME%.txt "
echo 'run 3'
powershell "python  %PY_MOD% %PARS_FILENAME% -wi | tee %OUT_DIR%\out3b_%PARS_FILENAME%.txt "



REM -wi -db all -ls 30 -lcfg logger.conf -ll DEBUG


echo off
REM Wrap up
echo 'finished executing python scripts from batch file'

pause