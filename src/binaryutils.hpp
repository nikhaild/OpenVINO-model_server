//*****************************************************************************
// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#pragma once

#include <memory>

#include "status.hpp"
#include "tensorinfo.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "kfs_grpc_inference_service.hpp"
#pragma GCC diagnostic pop

namespace ovms {
template <typename TensorType>
Status convertBinaryRequestTensorToOVTensor(const TensorType& src, ov::Tensor& tensor, const std::shared_ptr<TensorInfo>& tensorInfo);

extern template Status convertBinaryRequestTensorToOVTensor<tensorflow::TensorProto>(const tensorflow::TensorProto& src, ov::Tensor& tensor, const std::shared_ptr<TensorInfo>& tensorInfo);
extern template Status convertBinaryRequestTensorToOVTensor<::inference::ModelInferRequest::InferInputTensor>(const ::inference::ModelInferRequest::InferInputTensor& src, ov::Tensor& tensor, const std::shared_ptr<TensorInfo>& tensorInfo);
}  // namespace ovms
