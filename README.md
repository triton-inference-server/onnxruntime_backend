<!--
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
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# ONNX Runtime Backend

The Triton backend for the [ONNX
Runtime](https://github.com/microsoft/onnxruntime). You can learn more
about Triton backends in the [backend
repo](https://github.com/triton-inference-server/backend). Ask
questions or report problems on the [issues
page](https://github.com/triton-inference-server/onnxruntime_backend/issues).

Use a recent cmake to build and install in a local directory.
Typically you will want to build an appropriate ONNX Runtime
implementation as part of the build. You do this by specifying a ONNX
Runtime version and a Triton container version that you want to use
with the backend. You can find the combination of versions used in a
particular Triton release in the TRITON_VERSION_MAP at the top of
build.py in the branch matching the Triton release you are interested
in. For example, to build the ONNX Runtime backend for Triton 21.05,
use the versions from TRITON_VERSION_MAP in the [r21.05 branch of
build.py](https://github.com/triton-inference-server/server/blob/r21.05/build.py#L66).

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BUILD_ONNXRUNTIME_VERSION=1.8.1 -DTRITON_BUILD_CONTAINER_VERSION=21.07 ..
$ make install
```

The resulting install/backends/onnxruntime directory can be added to a
Triton installation as /opt/tritonserver/backends/onnxruntime.

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
* triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]

You can add TensorRT support to the ONNX Runtime backend by using
-DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON. You can add OpenVino support
by using -DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON
-DTRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION=\<version\>, where
\<version\> is the OpenVino version to use and should match the
TRITON_VERSION_MAP entry as described above. So, to build with both
TensorRT and OpenVino support:

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BUILD_ONNXRUNTIME_VERSION=1.8.1 -DTRITON_BUILD_CONTAINER_VERSION=21.07 -DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON -DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON -DTRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION=2021.2.200 ..
$ make install
```


## ONNX Runtime with TensorRT optimization
TensorRT can be used in conjunction with an ONNX model to further optimize the performance. To enable TensorRT optimization you must set the model configuration appropriately. There are several optimizations available for TensorRT, like selection of the compute precision and workspace size. The optimization parameters and their description are as follows.


* `precision_mode`: The precision used for optimization. Allowed values are "FP32" and "FP16". Default value is "FP32".
* `max_workspace_size_bytes`: The maximum GPU memory the model can use temporarily during execution. Default value is 1GB.

The section of model config file specifying these parameters will look like:

```
.
.
.
optimization { execution_accelerators {
  gpu_execution_accelerator : [ {
    name : "tensorrt"
    parameters { key: "precision_mode" value: "FP16" }
    parameters { key: "max_workspace_size_bytes" value: "1073741824" }}]
}}
.
.
.

```
