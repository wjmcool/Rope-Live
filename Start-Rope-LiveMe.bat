@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem ===========================================================================
rem  Rope-LiveMe  -  source-mode launcher (global runtime)
rem
rem  Interpreter lookup order
rem      1. python_path.txt  (explicit, first non-comment line wins)
rem      2. python.exe / python3.exe found on PATH
rem      3. py launcher  (-3.12 / -3.11 / -3.10)
rem      4. well-known install locations
rem    Pass A accepts only an interpreter whose torch sees CUDA, pass B accepts
rem    any interpreter that simply has the runtime dependencies.
rem
rem  Other inputs
rem      ffmpeg.exe on PATH  -> exported as FFMPEG_PATH
rem
rem  App layout
rem      Rope.py        entry point
rem      bootstrap.py   path / cwd / launch handshake / licence patch
rem      rope\*.py      application package
rem
rem  ext_dependencies is not used by this launcher.
rem ===========================================================================

cd /d "%~dp0"
set "HERE=%~dp0"
set "PYCMD="

echo === Rope-LiveMe source mode ===
echo root    : %HERE%

rem ---------- 1) explicit interpreter ---------------------------------------
set "REQUIRE_CUDA=0"
if exist "%HERE%python_path.txt" call :read_config

rem ---------- 2) pass A: CUDA capable ---------------------------------------
if not defined PYCMD set "REQUIRE_CUDA=1"
if not defined PYCMD call :scan

rem ---------- 3) pass B: dependencies only ---------------------------------
if not defined PYCMD set "REQUIRE_CUDA=0"
if not defined PYCMD call :scan

if defined PYCMD goto :found
echo.
echo [x] no usable python interpreter.
echo     Either write its full path into python_path.txt, or add it to PATH.
echo     Need: CPython 3.10 - 3.12, 64 bit, with torch / onnxruntime-gpu /
echo           tensorrt / opencv / customtkinter installed.
echo.
pause
exit /b 2

:found
set "FFMPEG_PATH="
for /f "delims=" %%F in ('where ffmpeg.exe 2^>nul') do call :take_ffmpeg "%%~dpF"
if defined FFMPEG_PATH (echo ffmpeg  : %FFMPEG_PATH%) else (echo ffmpeg  : [warn] not found on PATH)
echo python  : %PYCMD%
echo.
"%PYCMD%" "%HERE%bootstrap.py" %*
set "RC=%ERRORLEVEL%"
echo.
echo === exited with code %RC% ===
pause
exit /b %RC%

rem ===========================================================================
rem  :read_config  -  first usable line of python_path.txt
rem ===========================================================================
:read_config
for /f "usebackq delims=" %%L in ("%HERE%python_path.txt") do call :take_config "%%L"
exit /b 0

:take_config
if defined PYCMD exit /b 0
set "CFGLINE=%~1"
if not defined CFGLINE exit /b 0
if "%CFGLINE:~0,1%"=="#" exit /b 0
if /i "%CFGLINE:~0,3%"=="rem" exit /b 0
rem  env:NAME  ->  read that environment variable
if /i "%CFGLINE:~0,4%"=="env:" call set "CFGLINE=%%%CFGLINE:~4%%%"
rem  %VAR%  ->  expand environment references inside the value
call set "CFGLINE=%%CFGLINE%%"
if not defined CFGLINE exit /b 0
call :probe "%CFGLINE%"
exit /b 0

rem ===========================================================================
rem  :scan  -  probe every candidate produced by :candidates
rem ===========================================================================
:scan
set "CANDLIST=%TEMP%\rope_live_candidates.txt"
call :candidates > "%CANDLIST%" 2>nul
for /f "usebackq delims=" %%P in ("%CANDLIST%") do call :probe "%%P"
del "%CANDLIST%" >nul 2>nul
exit /b 0

:candidates
for /f "delims=" %%P in ('where python.exe 2^>nul') do echo(%%P
for /f "delims=" %%P in ('where python3.exe 2^>nul') do echo(%%P
where py.exe >nul 2>nul && for %%V in (3.12 3.11 3.10) do call :py_version %%V
for %%D in (312 311 310) do call :known "%%D"
exit /b 0

:py_version
for /f "delims=" %%P in ('py -%~1 -c "import sys;print(sys.executable)" 2^>nul') do echo(%%P
exit /b 0

:known
if exist "%LOCALAPPDATA%\Programs\Python\Python%~1\python.exe" echo(%LOCALAPPDATA%\Programs\Python\Python%~1\python.exe
if exist "C:\Python%~1\python.exe" echo(C:\Python%~1\python.exe
if exist "C:\Program Files\Python%~1\python.exe" echo(C:\Program Files\Python%~1\python.exe
exit /b 0

rem ===========================================================================
rem  :probe <python.exe>
rem ===========================================================================
:probe
if defined PYCMD exit /b 0
if "%~1"=="" exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys,struct;sys.exit(0 if (3,10)<=sys.version_info[:2]<(3,13) and struct.calcsize('P')==8 else 1)" >nul 2>nul
if errorlevel 1 (
    echo       skip [needs cp3.10-3.12 x64]: %~1
    exit /b 0
)
"%~1" -c "import torch,onnxruntime,cv2,customtkinter" >nul 2>nul
if errorlevel 1 (
    echo       skip [runtime deps missing]: %~1
    exit /b 0
)
if "%REQUIRE_CUDA%"=="1" (
    "%~1" -c "import sys,torch;sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
    if errorlevel 1 (
        echo       skip [torch has no CUDA]: %~1
        exit /b 0
    )
)
echo       accepted: %~1
set "PYCMD=%~1"
exit /b 0

rem ===========================================================================
rem  :take_ffmpeg <dir\>
rem ===========================================================================
':take_ffmpeg
if defined FFMPEG_PATH exit /b 0
set "CAND=%~1"
if "%CAND:~-1%"=="\" set "CAND=%CAND:~0,-1%"
if not exist "%CAND%\ffmpeg.exe" exit /b 0
rem  WinGet Links 里是 0 字节 App Execution Alias，必须先跑一次才知道能不能用
"%CAND%\ffmpeg.exe" -hide_banner -version >nul 2>nul
if errorlevel 1 (
    echo       skip ffmpeg [broken binary]: %CAND%
    exit /b 0
)
set "FFMPEG_PATH=%CAND%"
echo       accepted ffmpeg: %CAND%
exit /b 0
