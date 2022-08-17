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

template <typename MetricType>
class MetricFamily {
public:
    MetricFamily(const std::string& name, const std::string& description, prometheus::Registry& registryImplRef);
    MetricFamily(const MetricFamily&) = delete;
    MetricFamily(MetricFamily&&) = delete;

    std::shared_ptr<MetricType> addMetric(const std::map<std::string, std::string>& labels = {}, const std::vector<double>& bucketBoundaries = {});

    void remove(std::shared_ptr<MetricType> metric);

private:
    prometheus::Registry& registryImplRef;
    void* familyImplRef;  // This is reference to prometheus::Family<T> where T is prometheus::Counter/Gauge/Histogram depending on MetricType.

    friend class MetricRegistry;
};

}  // namespace ovms
