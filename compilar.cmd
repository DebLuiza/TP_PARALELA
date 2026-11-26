@echo off
setlocal

if "%~1"=="" goto iniciar
if /i "%~1"=="cpu" goto iniciar
if /i "%~1"=="cuda" goto iniciar
echo Uso: %~nx0 [cpu ^| cuda]
exit /b 2

:iniciar
pushd "%~dp0" || exit /b 1
where /q cl.exe
if errorlevel 1 (
    call :configurar_msvc
    if errorlevel 1 goto erro
)
if not exist build\ mkdir build || goto erro

if /i "%~1"=="cuda" goto compilar_cuda
cl.exe /nologo /O2 /std:c11 /FI"scripts\windows_time_compat.h" /Fo"build\kmeans_seq.obj" /Fe"build\kmeans_seq.exe" kmeans_seq.c || goto erro
cl.exe /nologo /O2 /std:c11 /openmp /FI"scripts\windows_time_compat.h" /Fo"build\kmeans_omp.obj" /Fe"build\kmeans_omp.exe" kmeans_omp.c || goto erro
if /i "%~1"=="cpu" goto fim

:compilar_cuda
set "NVCC=nvcc.exe"
if exist "%CUDA_PATH%\bin\nvcc.exe" set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
if exist "%CD%\.toolchain\cuda\bin\nvcc.exe" set "NVCC=%CD%\.toolchain\cuda\bin\nvcc.exe"
if exist "%NVCC%" goto executar_nvcc
where /q "%NVCC%"
if errorlevel 1 (
    echo CUDA Toolkit ausente. Defina CUDA_PATH ou adicione nvcc.exe ao PATH.
    goto erro
)

:executar_nvcc
if not defined CUDA_ARCH set "CUDA_ARCH=sm_89"
"%NVCC%" -O2 "-arch=%CUDA_ARCH%" --pre-include "scripts\windows_time_compat.h" -o "build\kmeans_cuda.exe" kmeans_cuda.cu || goto erro

:fim
popd
exit /b 0

:erro
popd
exit /b 1

:configurar_msvc
set "VS_INSTALLER=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
set "VS_INSTALACAO="
if exist "%VS_INSTALLER%\vswhere.exe" (
    for /f "usebackq delims=" %%I in (`"%VS_INSTALLER%\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_INSTALACAO=%%I"
)
if not defined VS_INSTALACAO (
    echo Visual Studio Build Tools com o compilador C++ x64 nao encontrado.
    exit /b 1
)
set "PATH=%VS_INSTALLER%;%PATH%"
call "%VS_INSTALACAO%\VC\Auxiliary\Build\vcvars64.bat" >nul
exit /b %errorlevel%
