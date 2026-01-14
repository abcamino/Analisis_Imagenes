@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "C:\Users\luado\Desktop\Claude_Projects\Analisis_Imagenes"
python setup_simple.py build_ext --inplace
echo.
echo Build complete. Check for aneurysm_cpp*.pyd file.
dir *.pyd 2>nul
