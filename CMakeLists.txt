# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.14)

project(tritononnxruntimebackend LANGUAGES C CXX)

#
# Options
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(TRITON_ENABLE_ONNXRUNTIME_TENSORRT
  "Enable TensorRT execution provider for ONNXRuntime backend in server" OFF)
option(TRITON_ENABLE_ONNXRUNTIME_OPENVINO
  "Enable OpenVINO execution provider for ONNXRuntime backend in server" OFF)
set(TRITON_ONNXRUNTIME_INCLUDE_PATHS "" CACHE PATH "Paths to ONNXRuntime includes")

set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-backend)

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_onnxruntime.ldscript libtriton_onnxruntime.ldscript COPYONLY)

add_library(
  triton-onnxruntime-backend SHARED
  src/onnxruntime.cc
  src/onnxruntime_loader.cc
  src/onnxruntime_loader.h
  src/onnxruntime_utils.cc
  src/onnxruntime_utils.h
)

add_library(
  TritonOnnxRuntimeBackend::triton-onnxruntime-backend ALIAS triton-onnxruntime-backend
)

target_include_directories(
  triton-onnxruntime-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TRITON_ONNXRUNTIME_INCLUDE_PATHS}
)

target_compile_features(triton-onnxruntime-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-onnxruntime-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Werror>
)

if(${TRITON_ENABLE_GPU})
target_compile_definitions(
  triton-onnxruntime-backend
  PRIVATE TRITON_ENABLE_GPU=1
)
endif() # TRITON_ENABLE_GPU
if(${TRITON_ENABLE_ONNXRUNTIME_TENSORRT})
target_compile_definitions(
  triton-onnxruntime-backend
  PRIVATE TRITON_ENABLE_ONNXRUNTIME_TENSORRT=1
)
endif() # TRITON_ENABLE_ONNXRUNTIME_TENSORRT
if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
target_compile_definitions(
  triton-onnxruntime-backend
  PRIVATE TRITON_ENABLE_ONNXRUNTIME_OPENVINO=1
)
endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO

target_link_libraries(
  triton-onnxruntime-backend
  PRIVATE
    triton-backend-utils  # from repo-backend
)

set_target_properties(
  triton-onnxruntime-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_onnxruntime
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_onnxruntime.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_onnxruntime.ldscript"
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonOnnxRuntimeBackend)

install(
  TARGETS
    triton-onnxruntime-backend
  EXPORT
    triton-onnxruntime-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  EXPORT
    triton-onnxruntime-backend-targets
  FILE
    TritonOnnxRuntimeBackendTargets.cmake
  NAMESPACE
    TritonOnnxRuntimeBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonOnnxRuntimeBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonOnnxRuntimeBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonOnnxRuntimeBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-onnxruntime-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonOnnxRuntimeBackendTargets.cmake
  NAMESPACE TritonOnnxRuntimeBackend::
)

export(PACKAGE TritonOnnxRuntimeBackend)
