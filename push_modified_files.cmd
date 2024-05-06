REM set "dstIPAddress=172.16.183.24"  YF_TB_9000
set "dstIPAddress=172.16.183.184"
set "dstRoot=PythonDL/"
@echo off
setlocal enabledelayedexpansion
REM Check if any arguments were provided


REM 获取git status命令的输出信息
for /f "tokens=1,2*" %%A in ('git status -s') do (
    if "%%A"=="M" (
        REM 提取被修改的文件路径
        set "sourceFile=%%B"
        echo sourceFile !sourceFile!
        REM 提取目录路径和文件名
        for %%I in ("!sourceFile!") do (
            set "fileName=%%~nxI"
            set "directoryPath=%%~dpI"
        )
        REM Get the current working directory
        for /f "usebackq delims=" %%A in (`"cd"`) do set "currentDirectory=%%A\"
        echo currentDirectory !currentDirectory!


        REM Remove the substring from the string
        call set "finalPath=%%directoryPath:!currentDirectory!=%%"
        echo result finalPath = !finalPath!


        echo directoryPath !directoryPath!
        REM 构建目标目录路径
        set "destinationDirectory=/home/yf/%dstRoot%!finalPath!"

        REM 替换目标目录路径中的反斜杠为正斜杠
        set "destinationDirectory=!destinationDirectory:\=/!"

        REM 执行文件传输命令
          
        echo sourceFile !sourceFile!
        echo destinationDirectory !destinationDirectory!   
        pscp -r -pw 123456 !sourceFile! yf@%dstIPAddress%:!!destinationDirectory!   
          
    )
)



