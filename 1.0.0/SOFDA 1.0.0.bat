REM Batch script for running python modules
echo off

REM Point to the executable
REM SET EXE_DIR=C:\Users\cef\Downloads\SOFDA004
SET EXE_NM=sofda


REM Define all working directories
SET WORK_DIR=%userprofile%\SOFDA


REM Specify control file
SET CONTROL_FN=%WORK_DIR%\_ins\SOFDA004_test.xls

echo Executing  %EXE_NM%

echo RUN 1
%EXE_NM% -cf %CONTROL_FN% -part -fo -wd %WORK_DIR%

REM basic flags (see /? for more )s
REM -wd 'working directory. \'auto\'= use current . \'gui\' =use gui. home (default)= users home dir'
REM -part 'partial data loading flag\n'
REM -fo  'force open the outputs folder after execution\n'


REM place more run commands here if desired



REM Wrap up
echo finished batch file execution

pause