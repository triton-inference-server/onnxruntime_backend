// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <onnxruntime_c_api.h>

#include <memory>
#include <mutex>

#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"

namespace triton { namespace backend { namespace onnxruntime {

/// A singleton to load Onnx model because loading models requires
/// Onnx Runtime environment which is unique per process
class OnnxLoader {
 public:
  ~OnnxLoader();

  /// Initialize loader with default environment settings
  static TRITONSERVER_Error* Init(common::TritonJson::Value& backend_config);

  /// Stop loader, and once all Onnx sessions are unloaded via UnloadSession()
  /// the resource it allocated will be released
  static TRITONSERVER_Error* Stop();

  /// Load a Onnx model from a path and return the corresponding
  /// OrtSession.
  ///
  /// \param bool is_path If true 'model' is a path to the model file,
  /// if false 'model' is the serialized model.
  /// \param model The Onnx model or path to the model.
  /// \param session_options The options to use when creating the session
  /// \param session Returns the Onnx model session
  /// \return Error status.
  static TRITONSERVER_Error* LoadSession(
      const bool is_path, const std::string& model,
      const OrtSessionOptions* session_options, OrtSession** session);

  /// Unload a Onnx model session
  ///
  /// \param session The Onnx model session to be unloaded
  static TRITONSERVER_Error* UnloadSession(OrtSession* session);

  /// Returns whether global thread pool is enabled.
  /// If the loader is not initialized it returns false.
  static bool IsGlobalThreadPoolEnabled();

 private:
  OnnxLoader(OrtEnv* env, bool enable_global_threadpool = false)
      : env_(env), global_threadpool_enabled_(enable_global_threadpool),
        live_session_cnt_(0), closing_(false)
  {
  }

  /// Decrease 'live_session_cnt_' if 'decrement_session_cnt' is true, and then
  /// release Onnx Runtime environment if it is closing and no live sessions
  ///
  /// \param decrement_session_cnt Whether to decrease the 'live_session_cnt_'
  static void TryRelease(bool decrement_session_cnt);

  static std::unique_ptr<OnnxLoader> loader;

  OrtEnv* env_;
  bool global_threadpool_enabled_;
  std::mutex mu_;
  size_t live_session_cnt_;
  bool closing_;
};

}}}  // namespace triton::backend::onnxruntime
