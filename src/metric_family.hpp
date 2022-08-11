//*****************************************************************************
// Copyright 2022 Intel Corporation
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

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace prometheus {
class Registry;
}

namespace ovms {

class MetricRegistry;

class MetricFamilyBase {
public:
    virtual ~MetricFamilyBase() = default;
};

template <typename MetricType>
class MetricFamily : public MetricFamilyBase {
    std::string name, description;
    //std::vector<std::shared_ptr<MetricType>> metrics;

public:
    MetricFamily(const std::string& name, const std::string& description, prometheus::Registry& registryImplRef) :
        name(name),
        description(description),
        registryImplRef(registryImplRef) {}

    const std::string& getName() const { return this->name; }
    const std::string& getDesc() const { return this->description; }

    std::shared_ptr<MetricType> addMetric(const std::map<std::string, std::string>& labels = {}, const std::vector<double>& bucketBoundaries = {});

    bool remove(std::shared_ptr<MetricType> metric);

private:
    // Prometheus internals
    prometheus::Registry& registryImplRef;
    void* familyImplRef;

    friend class MetricRegistry;
};

}  // namespace ovms
