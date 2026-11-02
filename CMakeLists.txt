cmake_minimum_required(VERSION 3.19)
project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)

# Benchmarking needs an optimised build; a Debug binary measures the wrong thing.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Target architecture. Override on the command line for a card newer than the
# default, e.g. -DCUDA_COMPUTE_CAPABILITY=89 for Ada / RTX 4090. Building for an
# older architecture still runs, because the driver JIT-compiles the PTX, but the
# kernel it measures is then not the one a native build would execute.
if(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
  set(CUDA_COMPUTE_CAPABILITY 75)
endif()

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# add_compile_options("--ptxas-options=-v")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(sgemm sgemm.cu ${SRC})
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
