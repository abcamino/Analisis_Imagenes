@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

cd /d "C:\Users\luado\Desktop\Claude_Projects\Analisis_Imagenes"

echo Getting Python include path...
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_path('include'))"') do set PYTHON_INCLUDE=%%i
echo Python include: %PYTHON_INCLUDE%

echo Getting pybind11 include path...
for /f "tokens=*" %%i in ('python -c "import pybind11; print(pybind11.get_include())"') do set PYBIND_INCLUDE=%%i
echo pybind11 include: %PYBIND_INCLUDE%

echo Getting Python library path...
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"') do set PYTHON_LIB=%%i
echo Python lib: %PYTHON_LIB%

echo Getting Python extension suffix...
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"') do set EXT_SUFFIX=%%i
echo Extension suffix: %EXT_SUFFIX%

echo.
echo Compiling...
cl /O2 /EHsc /MD /std:c++17 ^
    /I"%PYTHON_INCLUDE%" ^
    /I"%PYBIND_INCLUDE%" ^
    cpp_simple\preprocessor_simple.cpp ^
    /link /DLL ^
    /LIBPATH:"%PYTHON_LIB%" ^
    python314.lib ^
    /OUT:aneurysm_cpp%EXT_SUFFIX%

echo.
echo Checking output...
dir *.pyd 2>nul
