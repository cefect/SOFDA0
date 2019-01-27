REM Batch script for running python modules
echo off
REM Set all directories and variables

SET WORK_DIR=C:\LocalStore\03_TOOLS\SOFDA\
SET PY_DIR=C:\LocalStore\03_TOOLS\SOFDA\0.0.4
SET OUT_DIR=C:\LocalStore\03_TOOLS\SOFDA\_outs\_batch
set PY_MOD=main_cmd.py

REM Specify control file
SET CONTROL_FN=C:\LocalStore\03_TOOLS\SOFDA\_ins\SOFDA004_test.xls


REM Change directory and execute the script
echo Executing python scripts from dir %WORK_DIR%

REM cd %WORK_DIR%

REM dir > %CONTROL_FN%.txt 2>&1


cd %PY_DIR%



REM redirect output
echo RUN 1
python  %PY_MOD% -cf %CONTROL_FN% -wd home -part -fo

REM -wd 'working directory. \'auto\'= use current . \'gui\' =use gui. home (default)= users home dir'
REM -part 'partial data loading flag\n'
REM -fo  'force open the outputs folder after execution\n'

REM -wi -db all -ls 30 -lcfg logger.conf -ll DEBUG -prof 0



REM Wrap up
echo finished executing python scripts from batch file

pause