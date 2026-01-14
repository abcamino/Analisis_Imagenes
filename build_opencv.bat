@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

cd /d "C:\Users\luado\Desktop\Claude_Projects\Analisis_Imagenes"

echo Getting Python paths...
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_path('include'))"') do set PYTHON_INCLUDE=%%i
for /f "tokens=*" %%i in ('python -c "import pybind11; print(pybind11.get_include())"') do set PYBIND_INCLUDE=%%i
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"') do set PYTHON_LIB=%%i
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"') do set EXT_SUFFIX=%%i

set OPENCV_DIR=C:\opencv\opencv\build
set OPENCV_INCLUDE=%OPENCV_DIR%\include
set OPENCV_LIB=%OPENCV_DIR%\x64\vc16\lib

echo.
echo Python include: %PYTHON_INCLUDE%
echo pybind11 include: %PYBIND_INCLUDE%
echo OpenCV include: %OPENCV_INCLUDE%
echo OpenCV lib: %OPENCV_LIB%
echo Extension: %EXT_SUFFIX%
echo.

echo Compiling with OpenCV...
cl /O2 /EHsc /MD /std:c++17 ^
    /I"%PYTHON_INCLUDE%" ^
    /I"%PYBIND_INCLUDE%" ^
    /I"%OPENCV_INCLUDE%" ^
    cpp_opencv\preprocessor_opencv.cpp ^
    /link /DLL ^
    /LIBPATH:"%PYTHON_LIB%" ^
    /LIBPATH:"%OPENCV_LIB%" ^
    python314.lib ^
    opencv_world490.lib ^
    /OUT:aneurysm_cpp%EXT_SUFFIX%

echo.
echo Checking output...
dir *.pyd 2>nul

echo.
echo Copying OpenCV DLL...
copy "%OPENCV_DIR%\x64\vc16\bin\opencv_world490.dll" . 2>nul
