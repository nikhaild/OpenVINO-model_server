//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include "model.h"

namespace ovms {

Status Model::addVersion(   const std::string& name,
                            const std::string& path,
                            const std::string& backend,
                            const model_version_t& version,
                            const size_t batchSize,
                            const shapesMap& shapes,
                            const layoutsMap& layouts) {
    std::shared_ptr<ModelInstance> modelInstance = std::make_shared<ModelInstance>();
    auto status = modelInstance->loadModel(path, backend, version, batchSize, shapes, layouts);
    if (status != Status::OK) {
        return status;
    }
    this->name = name;
    if (this->defaultVersion < version)
        this->defaultVersion = version;
    modelVersions[version] = std::move(modelInstance);
    
    return Status::OK;
}

Status Model::dropVersion(const model_version_t& version) {
    std::map<model_version_t, std::shared_ptr<ModelInstance>>::iterator it = modelVersions.find(version);
    if (it != modelVersions.end()) {
        return Status::MODELINSTANCE_NOT_FOUND;
    }
    modelVersions.erase(version);

    return Status::OK;
}

} // namespace ovms
