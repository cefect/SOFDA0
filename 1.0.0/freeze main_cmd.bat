echo off
REM Batch Script to freeze SOFDA's main_cmd.py

set pyinstaller_der=C:\LocalStore\06_SOFT\py\2.7.15\dir\Scripts\
set exe_nm=sofda

set PATH = %PATH%;%pyinstaller_der%

pyinstaller main_cmd.py -n %exe_nm%

echo finished pyinstaller. copying other files

echo trying xcopy with %exe_nm%

REM copy pars file
xcopy /e _pars\*.* dist\%exe_nm%\_pars\

REM copy batch file
set batn="SOFDA 1.0.0.bat"
echo f | xcopy /f /y %batn% dist\%exe_nm%\%batn%

REM copy sample inputs
echo f | xcopy /f /y C:\LocalStore\03_TOOLS\SOFDA\_ins\_sample\*.* dist\%exe_nm%\_sample\

pause