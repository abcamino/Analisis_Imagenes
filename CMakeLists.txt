cmake_minimum_required(VERSION 3.14)
project(aneurysm_detector VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

# Collect source files
set(CPP_SOURCES
    cpp/src/image_loader.cpp
    cpp/src/normalizer.cpp
    cpp/src/roi_extractor.cpp
    cpp/src/preprocessor.cpp
)

# Create Python module
pybind11_add_module(aneurysm_cpp
    ${CPP_SOURCES}
    cpp/bindings/python_bindings.cpp
)

# Link libraries
target_link_libraries(aneurysm_cpp PRIVATE ${OpenCV_LIBS})

# Set output directory
set_target_properties(aneurysm_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Installation
install(TARGETS aneurysm_cpp DESTINATION .)
