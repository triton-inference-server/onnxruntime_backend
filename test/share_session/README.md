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

This test verifies that `share_session_between_instances` reuses a single ORT
session for multiple model instances of the same kind and device in the same
instance group.
It checks concurrent requests with distinct inputs, including optional inputs.
For GPU instances, it also checks that shared sessions use ORT-managed compute
streams and unshared sessions retain their instance's compute stream.
Session creation is checked separately for every configured GPU.

The default is shared CPU sessions. Run all variants with:

```bash
for kind in CPU GPU; do
  for sharing in 0 1; do
    INSTANCE_KIND=$kind SHARE_SESSION=$sharing bash test.sh || exit 1
  done
done
```

To cover an instance group spanning two GPUs, run both sharing modes on a
two-GPU machine. Each GPU gets two instances; sharing must create one session
per GPU, not one session for the whole group:

```bash
for sharing in 0 1; do
  CUDA_VISIBLE_DEVICES=0,1 GPU_COUNT=2 INSTANCE_KIND=GPU \
    SHARE_SESSION=$sharing bash test.sh || exit 1
done
```

`GPU_COUNT` defaults to 1 and selects consecutive CUDA device ordinals starting
at 0 after `CUDA_VISIBLE_DEVICES` remapping.

Like other backend tests in this repository, it assumes the Triton Server QA
test environment is set up and that `../common/util.sh` is available.

The cache-key regression test runs without Triton Server, ONNX Runtime, or any
GPU. It checks same-device reuse and isolation between devices, kinds, and
instance groups using the production key type. From this directory:

```bash
cmake -S . -B /tmp/share-session-key-build \
  -DTRITON_CORE_INCLUDE_DIR=/path/to/core/include
cmake --build /tmp/share-session-key-build --parallel 2
ctest --test-dir /tmp/share-session-key-build --output-on-failure
```
