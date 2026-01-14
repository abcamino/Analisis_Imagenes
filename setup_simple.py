"""
Simplified setup for C++ module without OpenCV dependency.
"""
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "aneurysm_cpp",
        sources=["cpp_simple/preprocessor_simple.cpp"],
        cxx_std=17,
        define_macros=[("VERSION_INFO", "1.0.0")],
    ),
]

setup(
    name="aneurysm_cpp",
    version="1.0.0",
    description="C++ preprocessing module for aneurysm detection",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
)
