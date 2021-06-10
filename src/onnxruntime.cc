// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <stdint.h>
#include <mutex>
#include <vector>
#include "onnxruntime_loader.h"
#include "onnxruntime_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

//
// ONNX Runtime Backend that implements the TRITONBACKEND API.
//

namespace triton { namespace backend { namespace onnxruntime {

/// Deleter for OrtSession.
struct SessionDeleter {
  void operator()(OrtSession* f) { OnnxLoader::UnloadSession(f); }
};

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Load an ONNX model using 'artifact_name' as the name for the ONNX
  // file/directory. If 'instance_group_kind' is not
  // TRITONSERVER_INSTANCEGROUPKIND_AUTO then use it and
  // 'instance_group_device_id' to initialize the appropriate
  // execution providers. Return in 'model_path' the full path to the
  // onnx file, return in 'session' and 'allocator' the ORT session
  // and allocator.
  TRITONSERVER_Error* LoadModel(
      const std::string& artifact_name,
      const TRITONSERVER_InstanceGroupKind instance_group_kind,
      const int32_t instance_group_device_id, std::string* model_path,
      OrtSession** session, OrtAllocator** default_allocator,
      cudaStream_t stream);

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();
  TRITONSERVER_Error* AutoCompleteMaxBatch(
      const OnnxTensorInfoMap& input_tensor_infos,
      const OnnxTensorInfoMap& output_tensor_infos);
  TRITONSERVER_Error* AutoCompleteIO(
      const char* key, const OnnxTensorInfoMap& io_infos);

  // Session options used when creating a ORT session.
  std::unique_ptr<OrtSessionOptions, SessionOptionsDeleter> session_options_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->AutoCompleteConfig());

    triton::common::TritonJson::WriteBuffer json_buffer;
    (*state)->ModelConfig().Write(&json_buffer);

    TRITONSERVER_Message* message;
    RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(
        &message, json_buffer.Base(), json_buffer.Size()));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(
        triton_model, 1 /* config_version */, message));
  }

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{
  // Create session options that will be cloned and used for each
  // instance when creating that instance's session.
  OrtSessionOptions* soptions;
  THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_api->CreateSessionOptions(&soptions));
  session_options_.reset(soptions);

  THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_api->SetIntraOpNumThreads(soptions, 1));

  GraphOptimizationLevel optimization_level =
      GraphOptimizationLevel::ORT_ENABLE_ALL;
  {
    triton::common::TritonJson::Value optimization;
    if (ModelConfig().Find("optimization", &optimization)) {
      triton::common::TritonJson::Value graph;
      if (optimization.Find("graph", &graph)) {
        int64_t graph_level = 0;
        THROW_IF_BACKEND_MODEL_ERROR(graph.MemberAsInt("level", &graph_level));
        if (graph_level == -1) {
          optimization_level = GraphOptimizationLevel::ORT_ENABLE_BASIC;
        } else if (graph_level == 1) {
          optimization_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
        }
      }
    }
  }
  THROW_IF_BACKEND_MODEL_ORT_ERROR(
      ort_api->SetSessionGraphOptimizationLevel(soptions, optimization_level));

  // FIXME. Is it possible to share a single OrtSession across
  // multiple instances? If so then should move loading and validation
  // of the session to here instead of creating a session for each
  // instance in ModelStateInstance::Create().
}

TRITONSERVER_Error*
ModelState::LoadModel(
    const std::string& artifact_name,
    const TRITONSERVER_InstanceGroupKind instance_group_kind,
    const int32_t instance_group_device_id, std::string* model_path,
    OrtSession** session, OrtAllocator** default_allocator, cudaStream_t stream)
{
  // Find the ONNX file that describes the model itself. If the model
  // configuration doesn't have an explicit model file specified then
  // use the default name ("model.onnx").
  std::string cc_model_filename = artifact_name;
  if (cc_model_filename.empty()) {
    cc_model_filename = "model.onnx";
  }

  *model_path = JoinPath(
      {RepositoryPath(), std::to_string(Version()), cc_model_filename});

  // If the model path is a directory then the actual model is
  // <dir>/model.onnx.
  {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(*model_path, &is_dir));
    if (is_dir) {
      *model_path = JoinPath({*model_path, "model.onnx"});
    }
  }

  {
    bool exists;
    RETURN_IF_ERROR(FileExists(*model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + *model_path +
            "' for model instance '" + Name() + "'");
  }

  // Make a clone for the session options for this instance...
  OrtSessionOptions* soptions;
  RETURN_IF_ORT_ERROR(
      ort_api->CloneSessionOptions(session_options_.get(), &soptions));
  std::unique_ptr<OrtSessionOptions, SessionOptionsDeleter> soptions_wrapper(
      soptions);

  bool need_lock = false;

  // Execution providers if they are requested... kind == AUTO if used
  // to indicate that execution providers should not be added (this is
  // just a local convention to this function, not the standard
  // interpretation of AUTO).
  if (instance_group_kind != TRITONSERVER_INSTANCEGROUPKIND_AUTO) {
    // Don't need to ensure uniqueness of the providers, ONNX Runtime
    // will check it.

    // GPU execution providers
#ifdef TRITON_ENABLE_GPU
    if (instance_group_kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
      triton::common::TritonJson::Value optimization;
      if (model_config_.Find("optimization", &optimization)) {
        triton::common::TritonJson::Value eas;
        if (optimization.Find("execution_accelerators", &eas)) {
          triton::common::TritonJson::Value gpu_eas;
          if (eas.Find("gpu_execution_accelerator", &gpu_eas)) {
            for (size_t ea_idx = 0; ea_idx < gpu_eas.ArraySize(); ea_idx++) {
              triton::common::TritonJson::Value ea;
              RETURN_IF_ERROR(gpu_eas.IndexAsObject(ea_idx, &ea));
              std::string name;
              RETURN_IF_ERROR(ea.MemberAsString("name", &name));
#ifdef TRITON_ENABLE_ONNXRUNTIME_TENSORRT
              if (name == kTensorRTExecutionAccelerator) {
                // create tensorrt options with default values
                OrtTensorRTProviderOptions trt_options{
                    instance_group_device_id,
                    stream != nullptr ? 1 : 0,
                    stream != nullptr ? (void*)stream : nullptr,
                    1000,     // trt_max_partition_iterations
                    1,        // trt_min_subgraph_size
                    1 << 30,  // max_workspace_size
                    0,        // trt_fp16_enable
                    0,        // trt_int8_enable
                    nullptr,  // trt_int8_calibration_table_name
                    0,        // trt_int8_calibration_table_name
                    0,        // trt_dla_enable
                    0,        // trt_dla_core
                    0,        // trt_dump_subgraphs
                    0,        // trt_engine_cache_enable
                    nullptr,  // trt_engine_cache_path
                    0,        // trt_engine_decryption_enable
                    nullptr,  // trt_engine_decryption_lib_path
                    0         // trt_force_sequential_engine_build
                };
                // Validate and set parameters
                triton::common::TritonJson::Value params;
                if (ea.Find("parameters", &params)) {
                  std::vector<std::string> param_keys;
                  RETURN_IF_ERROR(params.Members(&param_keys));
                  for (const auto& param_key : param_keys) {
                    std::string value_string;
                    if (param_key == "precision_mode") {
                      RETURN_IF_ERROR(params.MemberAsString(
                          param_key.c_str(), &value_string));
                      if (value_string == "FP16") {
                        trt_options.trt_fp16_enable = 1;
                      } else if (value_string != "FP32") {
                        RETURN_ERROR_IF_FALSE(
                            false, TRITONSERVER_ERROR_INVALID_ARG,
                            std::string("unsupported precision mode '") +
                                value_string + "' is requested");
                      }
                    } else if (param_key == "max_workspace_size_bytes") {
                      RETURN_IF_ERROR(params.MemberAsString(
                          param_key.c_str(), &value_string));
                      int64_t max_workspace_size_bytes;
                      RETURN_IF_ERROR(ParseLongLongValue(
                          value_string, &max_workspace_size_bytes));
                      trt_options.trt_max_workspace_size =
                          static_cast<size_t>(max_workspace_size_bytes);
                    } else {
                      return TRITONSERVER_ErrorNew(
                          TRITONSERVER_ERROR_INVALID_ARG,
                          std::string(
                              "unknown parameter '" + param_key +
                              "' is provided for TensorRT Execution "
                              "Accelerator")
                              .c_str());
                    }
                  }
                }
                RETURN_IF_ORT_ERROR(
                    ort_api->SessionOptionsAppendExecutionProvider_TensorRT(
                        soptions, &trt_options));
                LOG_MESSAGE(
                    TRITONSERVER_LOG_VERBOSE,
                    (std::string(
                         "TensorRT Execution Accelerator is set for '") +
                     Name() + "' on device " +
                     std::to_string(instance_group_device_id))
                        .c_str());
                continue;
              }
#endif  // TRITON_ENABLE_ONNXRUNTIME_TENSORRT
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("unknown Execution Accelerator '") + name +
                   "' is requested")
                      .c_str());
            }
          }
        }
      }

      // Default GPU execution provider.
      // Using default values for evrything other than device id and cuda stream
      OrtCUDAProviderOptions cuda_options{
          instance_group_device_id,
          OrtCudnnConvAlgoSearch::EXHAUSTIVE,  // cudnn_conv_algo_search
          std::numeric_limits<size_t>::max(),  // gpu_mem_limit
          0,                                   // arena_extend_strategy
          true,                                // do_copy_in_default_stream
          stream != nullptr ? 1 : 0,
          stream != nullptr ? (void*)stream : nullptr,
          nullptr  // default_memory_arena_cfg
      };
      RETURN_IF_ORT_ERROR(ort_api->SessionOptionsAppendExecutionProvider_CUDA(
          soptions, &cuda_options));
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("CUDA Execution Accelerator is set for '") + Name() +
           "' on device " + std::to_string(instance_group_device_id))
              .c_str());
    }
#endif  // TRITON_ENABLE_GPU

    // CPU execution providers
    {
      triton::common::TritonJson::Value optimization;
      if (model_config_.Find("optimization", &optimization)) {
        triton::common::TritonJson::Value eas;
        if (optimization.Find("execution_accelerators", &eas)) {
          triton::common::TritonJson::Value cpu_eas;
          if (eas.Find("cpu_execution_accelerator", &cpu_eas)) {
            for (size_t ea_idx = 0; ea_idx < cpu_eas.ArraySize(); ea_idx++) {
              triton::common::TritonJson::Value ea;
              RETURN_IF_ERROR(cpu_eas.IndexAsObject(ea_idx, &ea));
              std::string name;
              RETURN_IF_ERROR(ea.MemberAsString("name", &name));
#ifdef TRITON_ENABLE_ONNXRUNTIME_OPENVINO
              if (name == kOpenVINOExecutionAccelerator) {
                need_lock = true;
                OrtOpenVINOProviderOptions openvino_options;
                openvino_options.device_type =
                    "CPU_FP32";  // device_type default is CPU_FP32

                RETURN_IF_ORT_ERROR(
                    ort_api->SessionOptionsAppendExecutionProvider_OpenVINO(
                        soptions, &openvino_options));

                LOG_MESSAGE(
                    TRITONSERVER_LOG_VERBOSE,
                    (std::string(
                         "OpenVINO Execution Accelerator is set for '") +
                     Name() + "' on CPU")
                        .c_str());
                continue;
              }
#endif  // TRITON_ENABLE_ONNXRUNTIME_OPENVINO
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("unknown Execution Accelerator '") + name +
                   "' is requested")
                      .c_str());
            }
          }
        }
      }
    }
  }

  // Register all op libraries that contain custom operations.
  {
    triton::common::TritonJson::Value model_ops;
    if (model_config_.Find("model_operations", &model_ops)) {
      triton::common::TritonJson::Value op_library_filenames;
      if (model_ops.Find("op_library_filename", &op_library_filenames)) {
        for (size_t op_idx = 0; op_idx < op_library_filenames.ArraySize();
             op_idx++) {
          std::string op_filename;
          RETURN_IF_ERROR(
              op_library_filenames.IndexAsString(op_idx, &op_filename));
          void* library_handle = nullptr;
          RETURN_IF_ORT_ERROR(ort_api->RegisterCustomOpsLibrary(
              soptions, op_filename.c_str(), &library_handle));
        }
      }
    }
  }

  // ONNX session creation with OpenVINO is not thread-safe,
  // so multiple creations are serialized with a global lock.
  static std::mutex global_context_mu;
  std::unique_lock<std::mutex> glock(global_context_mu, std::defer_lock);
  if (need_lock) {
    glock.lock();
  }

  RETURN_IF_ERROR(OnnxLoader::LoadSession(
      true /* is_path */, *model_path, soptions, session));

  // get default cpu allocator
  RETURN_IF_ORT_ERROR(
      ort_api->GetAllocatorWithDefaultOptions(default_allocator));

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // If the model configuration already specifies inputs and outputs
  // then don't perform any auto-completion.
  size_t input_cnt = 0;
  size_t output_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (ModelConfig().Find("input", &inputs)) {
      input_cnt = inputs.ArraySize();
    }

    triton::common::TritonJson::Value config_batch_inputs;
    if (ModelConfig().Find("batch_input", &config_batch_inputs)) {
      input_cnt += config_batch_inputs.ArraySize();
    }

    triton::common::TritonJson::Value outputs;
    if (ModelConfig().Find("output", &outputs)) {
      output_cnt = outputs.ArraySize();
    }
  }

  if ((input_cnt > 0) && (output_cnt > 0)) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("skipping model configuration auto-complete for '") +
         Name() + "': inputs and outputs already specified")
            .c_str());
    return nullptr;  // success
  }

  std::string artifact_name;
  RETURN_IF_ERROR(
      ModelConfig().MemberAsString("default_model_filename", &artifact_name));

  // Must cleanup 'session'.  'allocator' is default allocator which
  // is managed by ONNX Runtime so don't need to free/release
  std::unique_ptr<OrtSession, SessionDeleter> session;
  OrtAllocator* default_allocator;
  std::string model_path;
  {
    OrtSession* sptr = nullptr;
    RETURN_IF_ERROR(LoadModel(
        artifact_name, TRITONSERVER_INSTANCEGROUPKIND_AUTO, 0, &model_path,
        &sptr, &default_allocator, nullptr));
    session.reset(sptr);
  }
  OnnxTensorInfoMap input_tensor_infos;
  RETURN_IF_ERROR(
      InputInfos(session.get(), default_allocator, input_tensor_infos));
  OnnxTensorInfoMap output_tensor_infos;
  RETURN_IF_ERROR(
      OutputInfos(session.get(), default_allocator, output_tensor_infos));
  RETURN_IF_ERROR(
      AutoCompleteMaxBatch(input_tensor_infos, output_tensor_infos));
  if (input_cnt == 0) {
    RETURN_IF_ERROR(AutoCompleteIO("input", input_tensor_infos));
  }
  if (output_cnt == 0) {
    RETURN_IF_ERROR(AutoCompleteIO("output", output_tensor_infos));
  }

  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    triton::common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("post auto-complete:\n") + buffer.Contents()).c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteMaxBatch(
    const OnnxTensorInfoMap& input_tensor_infos,
    const OnnxTensorInfoMap& output_tensor_infos)
{
  // Determine if the model can potentially support batching. All
  // input and output tensors must have a variable first dimension.
  bool can_support_batching = true;
  for (const auto& io_info : input_tensor_infos) {
    const auto& dims = io_info.second.dims_;
    if ((dims.size() == 0) || (dims[0] != -1)) {
      can_support_batching = false;
    }
  }
  for (const auto& io_info : output_tensor_infos) {
    const auto& dims = io_info.second.dims_;
    if ((dims.size() == 0) || (dims[0] != -1)) {
      can_support_batching = false;
    }
  }

  // Set max-batch-size to 1 if we have determined that batching is
  // supported and max-batch-size is not specified. We need to update
  // the configuration itself as well as the cached value we have already
  // initialized in the model state.
  if (can_support_batching) {
    if (MaxBatchSize() == 0) {
      triton::common::TritonJson::Value mbs_value;
      ModelConfig().Find("max_batch_size", &mbs_value);
      mbs_value.SetInt(1);
      SetMaxBatchSize(1);
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("autofilled max_batch_size to 1 for model '") + Name() +
           "' since batching is supporrted but no max_batch_size is specified "
           "in model configuration. Must specify max_batch_size to utilize "
           "autofill with a larger max batch size")
              .c_str());
    }
  } else if (MaxBatchSize() != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("autofill failed for model '") + Name() +
         "': model does not support batching while non-zero max_batch_size"
         " is specified")
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteIO(const char* key, const OnnxTensorInfoMap& io_infos)
{
  triton::common::TritonJson::Value existing_ios;
  bool found_ios = ModelConfig().Find(key, &existing_ios);

  triton::common::TritonJson::Value ios(
      ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
  for (const auto& io_info : io_infos) {
    triton::common::TritonJson::Value io(
        ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_ERROR(io.AddString("name", io_info.first));
    RETURN_IF_ERROR(io.AddString(
        "data_type", OnnxDataTypeToModelConfigDataType(io_info.second.type_)));

    // The model signature supports batching then the first dimension
    // is -1 and should not appear in the model configuration 'dims'
    // that we are creating.
    const auto& io_info_dims = io_info.second.dims_;
    triton::common::TritonJson::Value dims(
        ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
    for (size_t i = (MaxBatchSize() > 0) ? 1 : 0; i < io_info_dims.size();
         ++i) {
      RETURN_IF_ERROR(dims.AppendInt(io_info_dims[i]));
    }

    // If dims are empty then must use a reshape...
    if (dims.ArraySize() == 0) {
      RETURN_IF_ERROR(dims.AppendInt(1));
      triton::common::TritonJson::Value reshape(
          ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
      triton::common::TritonJson::Value reshape_dims(
          ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
      RETURN_IF_ERROR(reshape.Add("shape", std::move(reshape_dims)));
      RETURN_IF_ERROR(io.Add("reshape", std::move(reshape)));
    }
    RETURN_IF_ERROR(io.Add("dims", std::move(dims)));
    RETURN_IF_ERROR(ios.Append(std::move(io)));
  }

  if (found_ios) {
    existing_ios.Swap(ios);
  } else {
    ModelConfig().Add(key, std::move(ios));
  }

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  void ReleaseOrtRunResources();
  TRITONSERVER_Error* ValidateBooleanSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
  TRITONSERVER_Error* ValidateTypedSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
  TRITONSERVER_Error* ValidateInputs(const size_t expected_input_cnt);
  TRITONSERVER_Error* ValidateOutputs();
  void OrtRun(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count,
      const std::vector<const char*>& input_names,
      const std::vector<const char*>& output_names);
  void SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, std::vector<const char*>* input_names,
      bool* cuda_copy);
  void SetStringInputTensor(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses, const char* input_name,
      std::vector<const char*>* string_ptrs, bool* cuda_copy);
  void SetStringInputBuffer(
      const std::string& name, const std::vector<size_t>& expected_byte_sizes,
      const std::vector<size_t>& expected_element_cnts,
      std::vector<TRITONBACKEND_Response*>* responses, char* input_buffer,
      std::vector<const char*>* string_ptrs);
  void FillStringData(std::vector<const char*>* string_ptrs, size_t cnt);
  void ReadOutputTensors(
      size_t total_batch_size, const std::vector<const char*>& output_names,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);
  bool SetStringOutputBuffer(
      const std::string& name, const char* content, const size_t* offsets,
      std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  ModelState* model_state_;

  // The full path to the ONNX model file.
  std::string model_path_;

  // Onnx Runtime variables that are used across runs on this
  // instance.
  OrtSession* session_;
  OrtAllocator* default_allocator_;
  OrtMemoryInfo* cuda_allocator_info_;
  const OrtMemoryInfo* cpu_allocator_info_;
  OrtIoBinding* io_binding_;
  OrtRunOptions* runOptions_;

  // Onnx Runtime variables that will be reset and used for every run
  // on this instance.
  std::vector<OrtValue*> input_tensors_;
  std::vector<OrtValue*> output_tensors_;
  OrtValue** output_buffer_;
  std::vector<BackendMemory*> input_tensor_memories_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), session_(nullptr), default_allocator_(nullptr),
      cuda_allocator_info_(nullptr), cpu_allocator_info_(nullptr),
      io_binding_(nullptr), output_buffer_(nullptr)
{
  THROW_IF_BACKEND_INSTANCE_ERROR(model_state->LoadModel(
      ArtifactFilename(), Kind(), DeviceId(), &model_path_, &session_,
      &default_allocator_, CudaStream()));

  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    THROW_IF_BACKEND_INSTANCE_ORT_ERROR(ort_api->CreateMemoryInfo(
        "Cuda", OrtAllocatorType::OrtArenaAllocator, DeviceId(),
        OrtMemTypeDefault, &cuda_allocator_info_));
  }

  THROW_IF_BACKEND_INSTANCE_ORT_ERROR(
      ort_api->AllocatorGetInfo(default_allocator_, &cpu_allocator_info_));

  THROW_IF_BACKEND_INSTANCE_ORT_ERROR(
      ort_api->CreateIoBinding(session_, &io_binding_));

  THROW_IF_BACKEND_INSTANCE_ORT_ERROR(ort_api->CreateRunOptions(&runOptions_));

  size_t expected_input_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (model_state->ModelConfig().Find("input", &inputs)) {
      expected_input_cnt = inputs.ArraySize();
    }

    triton::common::TritonJson::Value config_batch_inputs;
    if (model_state->ModelConfig().Find("batch_input", &config_batch_inputs)) {
      expected_input_cnt += config_batch_inputs.ArraySize();
    }
  }

  // If this is a sequence model then make sure that the required
  // inputs are present in the model and have the correct shape and
  // datatype.
  triton::common::TritonJson::Value sequence_batching;
  if (model_state->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    bool have_start, have_end, have_ready, have_corrid;
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_START", false /* required */,
        &have_start));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_END", false /* required */,
        &have_end));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_READY", false /* required */,
        &have_ready));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateTypedSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_CORRID", false /* required */,
        &have_corrid));
    if (have_start) {
      expected_input_cnt += 1;
    }
    if (have_end) {
      expected_input_cnt += 1;
    }
    if (have_ready) {
      expected_input_cnt += 1;
    }
    if (have_corrid) {
      expected_input_cnt += 1;
    }
  }

  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateInputs(expected_input_cnt));
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateOutputs());
}

ModelInstanceState::~ModelInstanceState()
{
  ReleaseOrtRunResources();
  ort_api->ReleaseRunOptions(runOptions_);
  ort_api->ReleaseIoBinding(io_binding_);
  ort_api->ReleaseMemoryInfo(cuda_allocator_info_);
  if (session_ != nullptr) {
    OnnxLoader::UnloadSession(session_);
  }
  // 'default_allocator_' is default allocator which is managed by ONNX Runtime
}

void
ModelInstanceState::ReleaseOrtRunResources()
{
  for (auto& tensor : input_tensors_) {
    if (tensor != nullptr) {
      ort_api->ReleaseValue(tensor);
    }
  }
  input_tensors_.clear();

  // first release the Ortvalues
  for (auto& tensor : output_tensors_) {
    if (tensor != nullptr) {
      ort_api->ReleaseValue(tensor);
    }
  }
  output_tensors_.clear();

  // next release the allocated buffer using the specified allocator
  if (output_buffer_) {
    auto free_status =
        ort_api->AllocatorFree(default_allocator_, output_buffer_);
    output_buffer_ = nullptr;
    if (free_status != nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("onnx runtime allocator free error:") +
           std::to_string(ort_api->GetErrorCode(free_status)) +
           ort_api->GetErrorMessage(free_status))
              .c_str());
      ort_api->ReleaseStatus(free_status);
    }
  }

  for (BackendMemory* mem : input_tensor_memories_) {
    delete mem;
  }
  input_tensor_memories_.clear();
}

TRITONSERVER_Error*
ModelInstanceState::ValidateBooleanSequenceControl(
    triton::common::TritonJson::Value& sequence_batching,
    const std::string& control_kind, bool required, bool* have_control)
{
  std::string tensor_name;
  std::string tensor_datatype;
  RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
      sequence_batching, model_state_->Name(), control_kind, required,
      &tensor_name, &tensor_datatype, nullptr, nullptr, nullptr, nullptr));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    OnnxTensorInfoMap input_tensor_infos;
    RETURN_IF_ERROR(
        InputInfos(session_, default_allocator_, input_tensor_infos));
    const auto& iit = input_tensor_infos.find(tensor_name);
    if (iit == input_tensor_infos.end()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("configuration specified sequence control '") +
           tensor_name + "', but model does not provide that input")
              .c_str());
    }

    // Control tensors must have shape [1].
    const int nonbatch_start_idx = (model_state_->MaxBatchSize() > 0) ? 1 : 0;
    std::vector<int64_t> debatched_dims;
    for (size_t i = nonbatch_start_idx; i < iit->second.dims_.size(); i++) {
      debatched_dims.push_back(iit->second.dims_[i]);
    }

    if ((debatched_dims.size() != 1) || (debatched_dims[0] != 1)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', sequence control '" + tensor_name + "' in model has dims " +
           ShapeToString(debatched_dims) + " but dims [1] is expected")
              .c_str());
    }

    if (ModelConfigDataTypeToOnnxDataType(tensor_datatype) !=
        iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', sequence control '" + tensor_name +
           "', the model expects data-type " +
           OnnxDataTypeName(iit->second.type_) +
           " but the model configuration specifies data-type " +
           tensor_datatype)
              .c_str());
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateTypedSequenceControl(
    triton::common::TritonJson::Value& sequence_batching,
    const std::string& control_kind, bool required, bool* have_control)
{
  std::string tensor_name;
  std::string tensor_datatype;
  RETURN_IF_ERROR(GetTypedSequenceControlProperties(
      sequence_batching, model_state_->Name(), control_kind, required,
      &tensor_name, &tensor_datatype));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    OnnxTensorInfoMap input_tensor_infos;
    RETURN_IF_ERROR(
        InputInfos(session_, default_allocator_, input_tensor_infos));
    const auto& iit = input_tensor_infos.find(tensor_name);
    if (iit == input_tensor_infos.end()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("configuration specified sequence control '") +
           tensor_name + "', but model does not provide that input")
              .c_str());
    }

    // Control tensors must have shape [1].
    const int nonbatch_start_idx = (model_state_->MaxBatchSize() > 0) ? 1 : 0;
    std::vector<int64_t> debatched_dims;
    for (size_t i = nonbatch_start_idx; i < iit->second.dims_.size(); i++) {
      debatched_dims.push_back(iit->second.dims_[i]);
    }

    if ((debatched_dims.size() != 1) || (debatched_dims[0] != 1)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', sequence control '" + tensor_name + "' in model has dims " +
           ShapeToString(debatched_dims) + " but dims [1] is expected")
              .c_str());
    }

    if (ModelConfigDataTypeToOnnxDataType(tensor_datatype) !=
        iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', sequence control '" + tensor_name +
           "', the model expects data-type " +
           OnnxDataTypeName(iit->second.type_) +
           " but the model configuration specifies data-type " +
           tensor_datatype)
              .c_str());
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateInputs(const size_t expected_input_cnt)
{
  std::set<std::string> input_tensor_names;
  RETURN_IF_ERROR(InputNames(session_, input_tensor_names));

  OnnxTensorInfoMap input_tensor_infos;
  RETURN_IF_ERROR(InputInfos(session_, default_allocator_, input_tensor_infos));

  if (input_tensor_infos.size() != expected_input_cnt) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to load model '") + model_state_->Name() +
         "', configuration expects " + std::to_string(expected_input_cnt) +
         " inputs, model provides " + std::to_string(input_tensor_infos.size()))
            .c_str());
  }

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("input", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));

    auto iit = input_tensor_infos.find(io_name);
    if (iit == input_tensor_infos.end()) {
      RETURN_IF_ERROR(CheckAllowedModelInput(io, input_tensor_names));
    }

    auto onnx_data_type = ModelConfigDataTypeToOnnxDataType(io_dtype);
    if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype ") + io_dtype + " for input '" +
           io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    } else if (onnx_data_type != iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', configuration expects datatype " + io_dtype + " for input '" +
           io_name + "', model provides TYPE_" +
           TRITONSERVER_DataTypeString(
               ConvertFromOnnxDataType(iit->second.type_)))
              .c_str());
    }

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    std::vector<int64_t> dims;
    triton::common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }

    triton::common::TritonJson::Value allow_ragged_batch_json;
    bool allow_ragged_batch = false;
    if (io.Find("allow_ragged_batch", &allow_ragged_batch_json)) {
      RETURN_IF_ERROR(allow_ragged_batch_json.AsBool(&allow_ragged_batch));
    }
    if (allow_ragged_batch) {
      const std::vector<int64_t>& model_shape = iit->second.dims_;
      // Make sure the input has shpae [-1]
      if ((model_shape.size() != 1) || (model_shape[0] != WILDCARD_DIM)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("unable to load model '") + model_state_->Name() +
             "', configuration expects model provides input with shape [-1]  "
             "for ragged input '" +
             io_name + "', model provides " + ShapeToString(model_shape))
                .c_str());
      }
    } else {
      RETURN_IF_ERROR(CompareDimsSupported(
          model_state_->Name(), io_name, iit->second.dims_, dims,
          model_state_->MaxBatchSize(), false /* compare_exact */));
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateOutputs()
{
  std::set<std::string> output_tensor_names;
  RETURN_IF_ERROR(OutputNames(session_, output_tensor_names));

  OnnxTensorInfoMap output_tensor_infos;
  RETURN_IF_ERROR(
      OutputInfos(session_, default_allocator_, output_tensor_infos));

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("output", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));

    auto iit = output_tensor_infos.find(io_name);
    if (iit == output_tensor_infos.end()) {
      RETURN_IF_ERROR(CheckAllowedModelOutput(io, output_tensor_names));
    }

    auto onnx_data_type = ModelConfigDataTypeToOnnxDataType(io_dtype);
    if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype ") + io_dtype + " for output '" +
           io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    } else if (onnx_data_type != iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', configuration expects datatype " + io_dtype + " for output '" +
           io_name + "', model provides TYPE_" +
           TRITONSERVER_DataTypeString(
               ConvertFromOnnxDataType(iit->second.type_)))
              .c_str());
    }

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    std::vector<int64_t> dims;
    triton::common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }

    // The batch output shape doesn't necessarily match the model
    if (model_state_->FindBatchOutput(io_name) == nullptr) {
      RETURN_IF_ERROR(CompareDimsSupported(
          model_state_->Name(), io_name, iit->second.dims_, dims,
          model_state_->MaxBatchSize(), true /* compare_exact */));
    }
  }

  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = model_state_->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to ONNX Runtime backend for '" + Name() +
                  "'")
                  .c_str()));
      return;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                Name() + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
    return;
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  // Use scoped class to clean up ORT tensors and other resources that
  // need to persist until ORT run completes.
  struct ScopedCleanup {
    ScopedCleanup(ModelInstanceState* ctx) : ctx_(ctx) {}
    ~ScopedCleanup()
    {
      if (ctx_ != nullptr) {
        ctx_->ReleaseOrtRunResources();
      }
    }
    ModelInstanceState* ctx_;
  } io_tensor_wrapper(this);

  std::vector<const char*> input_names;
  bool cuda_copy = false;
  BackendInputCollector collector(
      requests, request_count, &responses, model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedInput(), CudaStream());
  SetInputTensors(
      total_batch_size, requests, request_count, &responses, &collector,
      &input_names, &cuda_copy);

  // Request to retrieve all model outputs. 'output_names' and
  // 'output_tensors_' are parallel vectors and so must be kept in
  // sync. [TODO] should collect only the outputs needed by some
  // request.
  std::vector<const char*> output_names;
  {
    triton::common::TritonJson::Value ios;
    TRITONSERVER_Error* err =
        model_state_->ModelConfig().MemberAsArray("output", &ios);
    if (err == nullptr) {
      for (size_t i = 0; i < ios.ArraySize(); i++) {
        triton::common::TritonJson::Value io;
        err = ios.IndexAsObject(i, &io);
        if (err != nullptr) {
          break;
        }

        // Use names from ModelConfig by reference since the model
        // config will persist longer than this inference execution.
        const char* io_name;
        size_t io_name_len;
        err = io.MemberAsString("name", &io_name, &io_name_len);
        if (err != nullptr) {
          break;
        }

        std::string io_dtype;
        err = io.MemberAsString("data_type", &io_dtype);
        auto onnx_data_type = ModelConfigDataTypeToOnnxDataType(io_dtype);

        output_names.emplace_back(io_name);
        output_tensors_.emplace_back(nullptr);

        bool use_cpu_allocator =
            (cuda_allocator_info_ == nullptr ||
             onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

        RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
            &responses, request_count,
            ort_api->BindOutputToDevice(
                io_binding_, io_name,
                use_cpu_allocator ? cpu_allocator_info_
                                  : cuda_allocator_info_));
      }
    }

    if (err != nullptr) {
      SendErrorForResponses(&responses, request_count, err);
      output_names.clear();
    }
  }

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(CudaStream());
  }
#endif

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  // Run...
  OrtRun(&responses, request_count, input_names, output_names);

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  ReadOutputTensors(
      total_batch_size, output_names, requests, request_count, &responses);

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send onnxruntime backend response");
    }
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Report the entire batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), total_batch_size, exec_start_ns,
          compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");
}

void
ModelInstanceState::OrtRun(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count, const std::vector<const char*>& input_names,
    const std::vector<const char*>& output_names)
{
  OrtStatus* status = nullptr;
  status = ort_api->RunWithBinding(session_, runOptions_, io_binding_);
  if (status != nullptr) {
    OrtErrorCode code = ort_api->GetErrorCode(status);
    std::string msg = ort_api->GetErrorMessage(status);
    SendErrorForResponses(
        responses, response_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("onnxruntime execute failure ") +
             std::to_string(code) + ": " + msg)
                .c_str()));
  }
}

void
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    bool* cuda_copy)
{
  const int max_batch_size = model_state_->MaxBatchSize();

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_InputProperties(
            input, &input_name, &input_datatype, &input_shape,
            &input_dims_count, nullptr, nullptr));

    input_names->emplace_back(input_name);
    input_tensors_.emplace_back(nullptr);

    std::vector<int64_t> batchn_shape;
    // For a ragged input tensor, the tensor shape should be
    // the flatten shape of the whole batch
    if (StateForModel()->IsInputRagged(input_name)) {
      batchn_shape = std::vector<int64_t>{0};
      for (size_t idx = 0; idx < request_count; idx++) {
        TRITONBACKEND_Input* input;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_RequestInput(requests[idx], input_name, &input));
        const int64_t* input_shape;
        uint32_t input_dims_count;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]), TRITONBACKEND_InputProperties(
                                      input, nullptr, nullptr, &input_shape,
                                      &input_dims_count, nullptr, nullptr));

        batchn_shape[0] += GetElementCount(input_shape, input_dims_count);
      }
    }
    // The shape for the entire input batch, [total_batch_size, ...]
    else {
      batchn_shape =
          std::vector<int64_t>(input_shape, input_shape + input_dims_count);
      if (max_batch_size != 0) {
        batchn_shape[0] = total_batch_size;
      }
    }

    if (input_datatype != TRITONSERVER_TYPE_BYTES) {
      // The input must be in contiguous CPU memory. Use appropriate
      // allocator info to bind inputs to the right device. .i.e bind inputs to
      // GPU if they are being provided on GPU.
      const char* input_buffer;
      size_t batchn_byte_size;
      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;
      std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>>
          allowed_input_types;
      if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
        allowed_input_types = {{TRITONSERVER_MEMORY_GPU, DeviceId()},
                               {TRITONSERVER_MEMORY_CPU_PINNED, 0},
                               {TRITONSERVER_MEMORY_CPU, 0}};
      } else {
        allowed_input_types = {{TRITONSERVER_MEMORY_CPU_PINNED, 0},
                               {TRITONSERVER_MEMORY_CPU, 0}};
      }

      RESPOND_ALL_AND_RETURN_IF_ERROR(
          responses, request_count,
          collector->ProcessTensor(
              input_name, nullptr, 0, allowed_input_types, &input_buffer,
              &batchn_byte_size, &memory_type, &memory_type_id));

      // Create ORT Tensor
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->CreateTensorWithDataAsOrtValue(
              memory_type == TRITONSERVER_MEMORY_GPU ? cuda_allocator_info_
                                                     : cpu_allocator_info_,
              (void*)input_buffer, batchn_byte_size, batchn_shape.data(),
              batchn_shape.size(), ConvertToOnnxDataType(input_datatype),
              &input_tensors_.back()));
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->BindInput(io_binding_, input_name, input_tensors_.back()));
    } else {
      // For BYTES input, we need to convert the serialized string
      // representation into what is required for ORT. ORT expects a
      // vector of char*, one for each element. For each tensor we get
      // a copy of the data in a contiguous CPU buffer and then
      // in-place modify that from the Triton
      // <int32_len><bytes><int32_len><bytes>... serialization into a
      // <bytes><null-terminator><bytes><null-terminator>... serialization
      // and then initialize 'string_ptrs' to point to each <bytes>.
      std::vector<const char*> string_ptrs;

      SetStringInputTensor(
          requests, request_count, responses, input_name, &string_ptrs,
          cuda_copy);

      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->CreateTensorAsOrtValue(
              default_allocator_, batchn_shape.data(), batchn_shape.size(),
              ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &input_tensors_.back()));
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->FillStringTensor(
              input_tensors_.back(), string_ptrs.data(), string_ptrs.size()));
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->BindInput(io_binding_, input_name, input_tensors_.back()));
    }
  }

  // Process batch input if any
  for (const auto& batch_input : StateForModel()->BatchInputs()) {
    std::vector<int64_t> shape;
    collector->BatchInputShape(batch_input, &shape);

    for (const auto& input_name : batch_input.TargetNames()) {
      input_names->emplace_back(input_name.c_str());
      input_tensors_.emplace_back(nullptr);

      const char* dst_buffer;
      size_t dst_buffer_byte_size;
      TRITONSERVER_MemoryType dst_memory_type;
      int64_t dst_memory_type_id;

      // Batch inputs are always created on CPU
      RESPOND_ALL_AND_SET_NULL_IF_ERROR(
          (*responses), responses->size(),
          collector->ProcessBatchInput(
              batch_input, nullptr, 0, {{TRITONSERVER_MEMORY_CPU, 0}},
              &dst_buffer, &dst_buffer_byte_size, &dst_memory_type,
              &dst_memory_type_id));

      // Create ORT Tensor
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, responses->size(),
          ort_api->CreateTensorWithDataAsOrtValue(
              cpu_allocator_info_, (void*)dst_buffer, dst_buffer_byte_size,
              shape.data(), shape.size(),
              ConvertToOnnxDataType(batch_input.DataType()),
              &input_tensors_.back()));

      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, responses->size(),
          ort_api->BindInput(
              io_binding_, input_name.c_str(), input_tensors_.back()));
    }
  }

  // Finalize...
  *cuda_copy |= collector->Finalize();
}

void
ModelInstanceState::SetStringInputTensor(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses, const char* input_name,
    std::vector<const char*>* string_ptrs, bool* cuda_copy)
{
  size_t total_byte_size = 0;
  std::vector<size_t> expected_byte_sizes;
  std::vector<size_t> expected_element_cnts;
  expected_byte_sizes.reserve(request_count);
  expected_element_cnts.reserve(request_count);
  for (size_t ridx = 0; ridx < request_count; ++ridx) {
    TRITONBACKEND_Input* in;
    RESPOND_AND_SET_NULL_IF_ERROR(
        &((*responses)[ridx]),
        TRITONBACKEND_RequestInput(requests[ridx], input_name, &in));

    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_InputProperties(
            in, nullptr, nullptr, &input_shape, &input_dims_count,
            &input_byte_size, nullptr));

    // Skip input in this request if error response has already been sent.
    if ((*responses)[ridx] == nullptr) {
      expected_byte_sizes.push_back(0);
      expected_element_cnts.push_back(0);
    } else {
      expected_element_cnts.push_back(
          GetElementCount(input_shape, input_dims_count));
      expected_byte_sizes.push_back(input_byte_size);
    }

    total_byte_size += expected_byte_sizes.back();
  }

  // For string input, the copy to contiguous buffer is needed because ORT
  // expects elements to be C strings thus we need to modify input buffer.
  // Reserve one more byte at the end of input_buffer to ensure last
  // element of String data can become valid C string.
  BackendMemory* input_memory;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      BackendMemory::Create(
          model_state_->TritonMemoryManager(),
          {BackendMemory::AllocationType::CPU_PINNED_POOL,
           BackendMemory::AllocationType::CPU},
          0 /* memory_type_id */, total_byte_size + 1, &input_memory));
  input_tensor_memories_.push_back(input_memory);

  const TRITONSERVER_MemoryType mem_type = input_memory->MemoryType();
  char* input_buffer = input_memory->MemoryPtr();

  size_t buffer_offset = 0;
  for (size_t ridx = 0; ridx < request_count; ++ridx) {
    TRITONBACKEND_Input* in;
    TRITONSERVER_Error* err =
        TRITONBACKEND_RequestInput(requests[ridx], input_name, &in);
    if ((err == nullptr) && ((*responses)[ridx] != nullptr)) {
      uint32_t input_buffer_count;
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          responses, request_count,
          TRITONBACKEND_InputProperties(
              in, nullptr, nullptr, nullptr, nullptr, nullptr,
              &input_buffer_count));

      size_t input_offset = 0;
      for (size_t idx = 0; idx < input_buffer_count; ++idx) {
        const void* src_buffer;
        size_t src_byte_size;
        TRITONSERVER_MemoryType src_memory_type;
        int64_t src_memory_type_id;
        err = TRITONBACKEND_InputBuffer(
            in, idx, &src_buffer, &src_byte_size, &src_memory_type,
            &src_memory_type_id);
        if (err == nullptr) {
          if ((input_offset + src_byte_size) > expected_byte_sizes[ridx]) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("buffer size for input '") + input_name +
                 "' exceeds batch byte size " +
                 std::to_string(expected_byte_sizes[ridx]))
                    .c_str());
          } else {
            bool cuda_used = false;
            err = CopyBuffer(
                input_name, src_memory_type, src_memory_type_id, mem_type, 0,
                src_byte_size, src_buffer,
                input_buffer + buffer_offset + input_offset, CudaStream(),
                &cuda_used);
            *cuda_copy |= cuda_used;
          }
        }

        if (err == nullptr) {
          input_offset += src_byte_size;
        } else {
          break;
        }
      }
    }

    if (err != nullptr) {
      if ((*responses)[ridx] != nullptr) {
        RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[ridx]), err);
      }

      TRITONSERVER_ErrorDelete(err);
    }

    buffer_offset += expected_byte_sizes[ridx];
  }

#ifdef TRITON_ENABLE_GPU
  // Synchronize to ensure the buffer is ready to be modified
  if (*cuda_copy) {
    cudaStreamSynchronize(CudaStream());
    *cuda_copy = false;
  }
#endif  // TRITON_ENABLE_GPU

  // Modify input buffer and set string expected by ORT
  SetStringInputBuffer(
      input_name, expected_byte_sizes, expected_element_cnts, responses,
      input_buffer, string_ptrs);
  input_buffer[total_byte_size] = 0;
}

void
ModelInstanceState::SetStringInputBuffer(
    const std::string& input_name,
    const std::vector<size_t>& expected_byte_sizes,
    const std::vector<size_t>& expected_element_cnts,
    std::vector<TRITONBACKEND_Response*>* responses, char* input_buffer,
    std::vector<const char*>* string_ptrs)
{
  // offset for each response
  size_t buffer_copy_offset = 0;
  for (size_t idx = 0; idx < expected_byte_sizes.size(); idx++) {
    const size_t expected_byte_size = expected_byte_sizes[idx];
    const size_t expected_element_cnt = expected_element_cnts[idx];

    size_t element_cnt = 0;
    if ((*responses)[idx] != nullptr) {
      size_t remaining_bytes = expected_byte_size;
      char* data_content = input_buffer + buffer_copy_offset;
      // Continue if the remaining bytes may still contain size info
      while (remaining_bytes >= sizeof(uint32_t)) {
        if (element_cnt >= expected_element_cnt) {
          RESPOND_AND_SET_NULL_IF_ERROR(
              &((*responses)[idx]),
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("unexpected number of string elements ") +
                   std::to_string(element_cnt + 1) + " for inference input '" +
                   input_name + "', expecting " +
                   std::to_string(expected_element_cnt))
                      .c_str()));
          break;
        }

        const uint32_t len = *(reinterpret_cast<const uint32_t*>(data_content));
        remaining_bytes -= sizeof(uint32_t);
        // Make first byte of size info 0, so that if there is string data
        // in front of it, the data becomes valid C string.
        *data_content = 0;
        data_content = data_content + sizeof(uint32_t);
        if (len > remaining_bytes) {
          RESPOND_AND_SET_NULL_IF_ERROR(
              &((*responses)[idx]),
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("incomplete string data for inference input '") +
                   input_name + "', expecting string of length " +
                   std::to_string(len) + " but only " +
                   std::to_string(remaining_bytes) + " bytes available")
                      .c_str()));
          break;
        } else {
          string_ptrs->push_back(data_content);
          element_cnt++;
          data_content = data_content + len;
          remaining_bytes -= len;
        }
      }
    }

    FillStringData(string_ptrs, expected_element_cnt - element_cnt);
    buffer_copy_offset += expected_byte_size;
  }
}

void
ModelInstanceState::FillStringData(
    std::vector<const char*>* string_ptrs, size_t cnt)
{
  static const char* empty = "";
  for (size_t c = 0; c < cnt; c++) {
    string_ptrs->push_back(empty);
  }
}

void
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, const std::vector<const char*>& output_names,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->MaxBatchSize(),
      model_state_->TritonMemoryManager(), model_state_->EnablePinnedInput(),
      CudaStream());

  // Use to hold string output contents
  bool cuda_copy = false;
  std::pair<TRITONSERVER_MemoryType, int64_t> alloc_perference;
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    alloc_perference = {TRITONSERVER_MEMORY_GPU, DeviceId()};
  } else {
    alloc_perference = {TRITONSERVER_MEMORY_CPU, 0};
  }

  size_t output_count = 0;
  RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
      responses, request_count,
      ort_api->GetBoundOutputValues(
          io_binding_, default_allocator_, &output_buffer_, &output_count));
  if (output_count != output_names.size()) {
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("Retrieved output count is not equal to expected count.")));
  }
  std::vector<std::vector<char>> string_buffers;
  for (size_t idx = 0; idx < output_names.size(); idx++) {
    OrtValue* output_tensor = output_tensors_[idx] = output_buffer_[idx];
    std::string name = output_names[idx];

    const BatchOutput* batch_output = StateForModel()->FindBatchOutput(name);
    if (batch_output == nullptr) {
      if (output_tensor == nullptr) {
        RESPOND_ALL_AND_RETURN_IF_ERROR(
            responses, request_count,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                (std::string("output tensor '") + name + "' is not found")
                    .c_str()));
      }

      // Get output type and shape
      OrtTypeInfo* typeinfo;
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->GetTypeInfo(output_tensor, &typeinfo));
      std::unique_ptr<OrtTypeInfo, TypeInfoDeleter> typeinfo_wrapper(typeinfo);

      const OrtTensorTypeAndShapeInfo* type_and_shape;
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->CastTypeInfoToTensorInfo(typeinfo, &type_and_shape));

      size_t num_dims;
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->GetDimensionsCount(type_and_shape, &num_dims));

      std::vector<int64_t> batchn_shape(num_dims);
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->GetDimensions(
              type_and_shape, batchn_shape.data(), batchn_shape.size()));

      ONNXTensorElementDataType type;
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->GetTensorElementType(type_and_shape, &type));

      if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        const size_t element_count = GetElementCount(batchn_shape);
        size_t total_length = 0;
        RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
            responses, request_count,
            ort_api->GetStringTensorDataLength(output_tensor, &total_length));

        string_buffers.emplace_back(std::vector<char>(total_length));
        auto content = string_buffers.back().data();
        std::vector<size_t> offsets(element_count + 1);
        RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
            responses, request_count,
            ort_api->GetStringTensorContent(
                output_tensor, content, total_length, offsets.data(),
                element_count));
        // Mark "passed end byte offset"
        offsets[element_count] = total_length;

        cuda_copy |= SetStringOutputBuffer(
            name, content, offsets.data(), &batchn_shape, requests,
            request_count, responses);
      } else {
        // Fixed size data type...
        char* output_buffer = nullptr;
        RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
            responses, request_count,
            ort_api->GetTensorMutableData(
                output_tensor, (void**)&output_buffer));

        responder.ProcessTensor(
            name, ConvertFromOnnxDataType(type), batchn_shape, output_buffer,
            alloc_perference.first, alloc_perference.second);
      }
    } else {
      char* output_buffer = nullptr;
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->GetTensorMutableData(output_tensor, (void**)&output_buffer));
      responder.ProcessBatchOutput(
          name, *batch_output, output_buffer, alloc_perference.first,
          alloc_perference.second);
    }
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRITON_ENABLE_GPU
}

bool
ModelInstanceState::SetStringOutputBuffer(
    const std::string& name, const char* content, const size_t* offsets,
    std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  size_t element_idx = 0;
  bool cuda_copy = false;
  for (size_t ridx = 0; ridx < request_count; ++ridx) {
    const auto& request = requests[ridx];
    auto& response = (*responses)[ridx];

    // batchn_shape holds the shape of the entire tensor batch. When
    // batching is enabled override the first batch dimension with each
    // requests batch size (reusing for efficiency).
    if (model_state_->MaxBatchSize() > 0) {
      TRITONBACKEND_Input* input;
      TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input);
      const int64_t* shape;
      TRITONBACKEND_InputProperties(
          input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
      (*batchn_shape)[0] = shape[0];
    }

    const size_t expected_element_cnt = GetElementCount(*batchn_shape);

    // If 'request' requested this output then copy it from
    // 'content'. If it did not request this output then just skip it
    // in the 'content'.
    bool need_output = false;
    if (response != nullptr) {
      uint32_t output_count;
      RESPOND_AND_SET_NULL_IF_ERROR(
          &response, TRITONBACKEND_RequestOutputCount(request, &output_count));
      if (response != nullptr) {
        for (uint32_t output_idx = 0; output_idx < output_count; output_idx++) {
          const char* req_output_name;
          RESPOND_AND_SET_NULL_IF_ERROR(
              &response, TRITONBACKEND_RequestOutputName(
                             request, output_idx, &req_output_name));
          if ((response != nullptr) && (req_output_name == name)) {
            need_output = true;
            break;
          }
        }
      }
    }

    if (need_output) {
      TRITONBACKEND_Output* response_output;
      TRITONSERVER_Error* err = TRITONBACKEND_ResponseOutput(
          response, &response_output, name.c_str(), TRITONSERVER_TYPE_BYTES,
          batchn_shape->data(), batchn_shape->size());
      if (err == nullptr) {
        // Calculate expected byte size in advance using string offsets
        const size_t data_byte_size =
            offsets[element_idx + expected_element_cnt] - offsets[element_idx];
        const size_t expected_byte_size =
            data_byte_size + sizeof(uint32_t) * expected_element_cnt;

        TRITONSERVER_MemoryType actual_memory_type =
            TRITONSERVER_MEMORY_CPU_PINNED;
        int64_t actual_memory_type_id = 0;
        void* buffer;
        err = TRITONBACKEND_OutputBuffer(
            response_output, &buffer, expected_byte_size, &actual_memory_type,
            &actual_memory_type_id);
        if (err == nullptr) {
          bool cuda_used = false;
          size_t copied_byte_size = 0;
          for (size_t e = 0; e < expected_element_cnt; ++e) {
            const uint32_t len =
                offsets[element_idx + e + 1] - offsets[element_idx + e];
            // Prepend size of the string
            err = CopyBuffer(
                name, TRITONSERVER_MEMORY_CPU /* src_memory_type */,
                0 /* src_memory_type_id */, actual_memory_type,
                actual_memory_type_id, sizeof(uint32_t),
                static_cast<const void*>(&len),
                static_cast<char*>(buffer) + copied_byte_size, stream_,
                &cuda_used);
            if (err != nullptr) {
              break;
            }

            cuda_copy |= cuda_used;
            copied_byte_size += sizeof(uint32_t);

            // Copy raw string content
            err = CopyBuffer(
                name, TRITONSERVER_MEMORY_CPU /* src_memory_type */,
                0 /* src_memory_type_id */, actual_memory_type,
                actual_memory_type_id, len, content + offsets[element_idx + e],
                static_cast<char*>(buffer) + copied_byte_size, stream_,
                &cuda_used);
            if (err != nullptr) {
              break;
            }

            cuda_copy |= cuda_used;
            copied_byte_size += len;
          }
        }
      }

      RESPOND_AND_SET_NULL_IF_ERROR(&response, err);
    }

    element_idx += expected_element_cnt;
  }

  return cuda_copy;
}

/////////////

extern "C" {

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Triton TRITONBACKEND API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor) + " does not support '" + name +
         "' TRITONBACKEND API version: " +
         std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
         std::to_string(TRITONBACKEND_API_VERSION_MINOR))
            .c_str());
  }

  // Onetime initialization for the onnxruntime loader.
  RETURN_IF_ERROR(OnnxLoader::Init());

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  LOG_IF_ERROR(OnnxLoader::Stop(), "failed to stop OnnxLoader");
  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::onnxruntime
