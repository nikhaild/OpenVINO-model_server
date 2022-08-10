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

#include "metric_kind.hpp"

namespace prometheus {
class Registry;
}

namespace ovms {

class Metric;
class Labels;
class MetricFamily {
    MetricKind kind;
    std::string name, description;
    std::vector<std::shared_ptr<Metric>> metrics;

public:
    MetricFamily(MetricKind kind, const std::string& name, const std::string& description, prometheus::Registry& registryImplRef);

    MetricKind getKind() const;
    const std::string& getName() const;
    const std::string& getDesc() const;

    std::shared_ptr<Metric> addMetric(const std::map<std::string, std::string>& labels = {}, const std::vector<double>& bucketBoundaries = {});

private:
    // Prometheus internals
    prometheus::Registry& registryImplRef;
};

}  // namespace ovms
