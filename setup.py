from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

# Get OpenCV include/lib paths (adjust for your system)
opencv_include = os.environ.get('OPENCV_INCLUDE', '/usr/include/opencv4')
opencv_lib = os.environ.get('OPENCV_LIB', '/usr/lib')

ext_modules = [
    Pybind11Extension(
        "aneurysm_cpp",
        sources=[
            "cpp/src/image_loader.cpp",
            "cpp/src/normalizer.cpp",
            "cpp/src/roi_extractor.cpp",
            "cpp/src/preprocessor.cpp",
            "cpp/bindings/python_bindings.cpp",
        ],
        include_dirs=[
            "cpp/include",
            opencv_include,
        ],
        library_dirs=[opencv_lib],
        libraries=["opencv_core", "opencv_imgproc", "opencv_imgcodecs"],
        cxx_std=17,
        define_macros=[("VERSION_INFO", "1.0.0")],
    ),
]

setup(
    name="aneurysm_detector",
    version="1.0.0",
    author="Usuario",
    description="Hybrid Python/C++ aneurysm detection system",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "onnxruntime>=1.16.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "training": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "timm>=0.9.0",
        ],
        "dev": [
            "pytest>=7.0.0",
        ],
    },
)
