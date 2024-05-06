REM set "dstIPAddress=172.16.183.24"  YF_TB_9000
set "sourceDirectory=%~1"
set "dstIPAddress=172.16.183.184"
set "dstRoot=PythonDL/"
@echo off
setlocal enabledelayedexpansion
REM Check if any arguments were provided
if "%~1"=="" (
  echo No source directory specified.
  exit /b
)
if not "%~2"=="" (
   set "dstIPAddress="%~2"
   echo dst ip address changed to %dstIPAddress% .
)

if not "%~3"=="" (
   set "dstRoot=%~3"
   echo dstRoot changed to %dstRoot% .
)
REM remove last \ character from source dir
REM Extract the last character of the string
set "lastCharacter=%sourceDirectory:~-1%"

REM Check if the last character is a backslash
if "%lastCharacter%"=="\" (
  REM Remove the backslash from the end of the string
  set "sourceDirectory=%sourceDirectory:~0,-1%"
)
echo source diretory is :  %sourceDirectory%
set sourcePath=%sourceDirectory%
set directoryPath=""
echo Extract the directory name and file name from the sourceDirectory path
for %%I in ("!sourcePath!") do (
  set "fileName=%%~nxI"
  set "directoryPath=%%~dpI"
)
echo directoryPath= %directoryPath%
echo fileName= %fileName%
REM Get the current working directory
for /f "usebackq delims=" %%A in (`"cd"`) do set "currentDirectory=%%A\"
echo currentDirectory %currentDirectory%


REM Remove the substring from the string
call set "finalPath=%%directoryPath:%currentDirectory%=%%"
echo result finalPath = %finalPath%


set rootDst=/home/yf/%dstRoot%
if "%fileName%"=="" (
  echo Source directory does not include a path.
  set "destinationDirectory=%rootDst%"
) else (
    echo Source directory include a path: %directoryName%
    set "destinationDirectory=%rootDst%%directoryName%%finalPath%"
)



set substring=\
call set "destinationDirectory=%%destinationDirectory:%substring%=/%%"
echo sourceDirectory %sourceDirectory%
echo destinationDirectory %destinationDirectory%


pscp -r  -pw 123456 %currentDirectory%%sourceDirectory% yf@%dstIPAddress%:%destinationDirectory%



