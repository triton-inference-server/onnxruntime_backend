<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Security Policy

## Reporting a Vulnerability

Please do not open a public GitHub issue for a suspected vulnerability.
Report it through one of the following channels:

1. **NVIDIA Vulnerability Disclosure Program** (preferred):
   <https://www.nvidia.com/en-us/security/>
2. **Email:** [psirt@nvidia.com](mailto:psirt@nvidia.com). Encrypt sensitive
   reports with NVIDIA's public PGP key:
   <https://www.nvidia.com/en-us/security/pgp-key>
3. **GitHub Private Vulnerability Reporting (where enabled):** use the "Report a
   vulnerability" button on the repository's Security tab.

OEM partners should contact their NVIDIA Customer Program Manager.

Please include:

1. Product name and version or branch that contains the vulnerability
2. Type of vulnerability (code execution, denial of service, buffer
   overflow, etc.)
3. Instructions to reproduce the vulnerability
4. Proof-of-concept or exploit code, if available
5. Potential impact, including how an attacker could exploit it

NVIDIA PSIRT acknowledges reports, assesses them, and coordinates fixes and
disclosure with the reporter. See <https://www.nvidia.com/en-us/security/>
for past security bulletins and notices.

## Security Architecture and Context

**Project:** the Triton Inference Server backend for
[ONNX Runtime](https://github.com/microsoft/onnxruntime). It is a C++
shared library (`libtriton_onnxruntime.so`) that Triton loads as a plugin
through the `TRITONBACKEND_*` API and that executes ONNX models using ONNX
Runtime.

**Software classification:** Library (server plugin). It opens no network
listeners and has no authentication layer of its own. It runs inside the
Triton server process with that process's privileges.

**Primary security responsibility:** safely load and execute model artifacts
supplied through the Triton model repository, and map inference request
tensors to and from ONNX Runtime without memory-safety errors.

**Key security boundaries and interfaces:**

- **Model repository to backend:** `model.onnx` (or the file named by
  `default_model_filename`) is resolved under the model repository path and
  version directory in `src/onnxruntime.cc` and loaded through ONNX Runtime
  (`CreateSession` / `CreateSessionFromArray` in `src/onnxruntime_loader.cc`).
- **Model configuration to backend:** `config.pbtxt` parameters, including
  `execution_accelerators` options for the TensorRT and CUDA execution
  providers and backend configuration settings, are parsed in
  `src/onnxruntime.cc` and `src/onnxruntime_utils.cc` and passed to ONNX
  Runtime provider options.
- **Request tensors to backend:** input tensor shapes, data types and
  buffers received from Triton core are validated and passed to ONNX Runtime.
- **Build time:** `cmake/download_onnxruntime.cmake` and
  `tools/gen_ort_dockerfile.py` fetch ONNX Runtime and related components
  (for example OpenVINO and ccache) that become part of the shipped binary.

**Repository Exposure Classification:** Public (basis: the GitHub repository
is publicly visible).

**Service Exposure Classification:** Not determined (low confidence). Basis:
this is a plugin library whose exposure depends entirely on how the hosting
Triton deployment is configured and who can supply models and requests.

## Threat Model

1. **Malicious or untrusted model artifact:** a crafted ONNX model loaded
   through `OnnxLoader::LoadSession` can trigger parsing or execution flaws in
   ONNX Runtime or in the execution provider kernels (memory corruption,
   denial of service, excessive resource use). Impact is code execution or
   crash inside the Triton server process.
2. **Untrusted write access to the model repository or its configuration:**
   anyone who can modify `config.pbtxt` or model files can choose execution
   providers and provider options, and can point provider cache settings
   (for example `trt_engine_cache_path`, `trt_timing_cache_path`) at
   locations they select. This can cause files to be written to unintended
   paths accessible to the server process or load attacker-controlled cached
   engines.
3. **Malformed inference requests:** unexpected tensor shapes, sizes, string
   tensors or data types sent by clients can cause out-of-bounds accesses or
   large allocations in input and output handling, leading to denial of
   service or memory corruption.
4. **Resource exhaustion through model and instance configuration:** models
   with large memory requirements, many instances, or large workspace
   settings can exhaust host or GPU memory on shared servers.
5. **Compromised build-time dependencies:** the ONNX Runtime package and
   other components downloaded during the build (including through
   `TRITON_ONNXRUNTIME_PACKAGE_URL` and the generated Dockerfile) could be
   tampered with if fetched over untrusted channels or without pinned
   integrity checks, affecting every deployment built from them.
6. **Vulnerabilities in bundled ONNX Runtime and execution providers:**
   known flaws in the ONNX Runtime version, CUDA, TensorRT or OpenVINO
   libraries in use are inherited by this backend until the versions are
   updated.

## Critical Security Assumptions

- The model repository is **trusted**. Models and `config.pbtxt` files are
  assumed to come from authorized users, and write access to the repository
  is restricted. The backend does not sandbox model execution.
- The backend relies on the Triton server for **authentication,
  authorization, TLS and rate limiting**. It performs none of these itself.
- Input tensor metadata from clients is assumed to be checked by Triton core
  against the model configuration before it reaches this backend; the backend
  validates what it needs for ONNX Runtime but is not a general input
  firewall.
- The ONNX Runtime and GPU libraries it links against are assumed to be
  obtained from trusted sources and kept up to date.
- Triton runs the backend with the privileges of the server process; any
  isolation (containers, least-privilege users, file system permissions for
  cache directories) is the deployer's responsibility.
- Build inputs (package URLs, base images, Dockerfile arguments) are assumed
  to be supplied by trusted maintainers.
