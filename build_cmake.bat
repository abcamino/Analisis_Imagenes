@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

cd /d "C:\Users\luado\Desktop\Claude_Projects\Analisis_Imagenes\cpp_simple"

if exist build rmdir /s /q build
mkdir build
cd build

"C:\Program Files\CMake\bin\cmake.exe" .. -G "Visual Studio 17 2022" -A x64 -Dpybind11_DIR="%USERPROFILE%\AppData\Roaming\Python\Python314\site-packages\pybind11\share\cmake\pybind11"

"C:\Program Files\CMake\bin\cmake.exe" --build . --config Release

echo.
echo Build complete. Checking for output...
dir /s *.pyd 2>nul
copy /Y Release\*.pyd ..\..\  2>nul
