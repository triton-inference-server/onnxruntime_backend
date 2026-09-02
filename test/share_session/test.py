#!/usr/bin/env python
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

import unittest

import numpy as np
import tritonclient.http as httpclient


class ShareSessionTest(unittest.TestCase):
    def setUp(self):
        self.client_ = httpclient.InferenceServerClient("localhost:8000")
        self.model_name_ = "add_with_initializer_shared"
        self.input_data_ = np.zeros((5, 5)).astype(np.float32)

    def test_infer_without_optional_input(self):
        infer_input = httpclient.InferInput("INPUT", self.input_data_.shape, "FP32")
        infer_input.set_data_from_numpy(self.input_data_, binary_data=False)

        results = self.client_.infer(self.model_name_, [infer_input])
        np.testing.assert_allclose(results.as_numpy("OUTPUT"), self.input_data_ + 1)

    def test_infer_with_optional_input(self):
        infer_input = httpclient.InferInput("INPUT", self.input_data_.shape, "FP32")
        infer_input.set_data_from_numpy(self.input_data_, binary_data=False)
        optional_input = httpclient.InferInput(
            "INITIALIZER", self.input_data_.shape, "FP32"
        )
        optional_input.set_data_from_numpy(self.input_data_, binary_data=False)

        results = self.client_.infer(self.model_name_, [infer_input, optional_input])
        np.testing.assert_allclose(results.as_numpy("OUTPUT"), self.input_data_)


if __name__ == "__main__":
    unittest.main()
