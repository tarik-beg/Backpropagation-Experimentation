@echo off
rem Builds all .cpp files in the workspace root and .vscode\lib into build\main.exe using MSVC.
setlocal
pushd "%~dp0.."

if not exist build mkdir build

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
if errorlevel 1 exit /b 1

cl /std:c++17 /EHsc /Zi /Od /W4 /nologo /Fe:build\main.exe /Fo:build\ /Fd:build\ *.cpp .vscode\lib\*.cpp
set RESULT=%errorlevel%

popd
exit /b %RESULT%
