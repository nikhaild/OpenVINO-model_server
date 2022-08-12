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
#include "metric_family.hpp"

#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include "metric.hpp"

namespace ovms {

template <>
std::shared_ptr<MetricCounter> MetricFamily<MetricCounter>::addMetric(const Metric::Labels& labels, const Metric::BucketBoundaries& bucketBoundaries) {
    prometheus::Family<prometheus::Counter>& familyImpl = prometheus::BuildCounter()
                                                              .Name(this->getName())
                                                              .Help(this->getDesc())
                                                              .Register(this->registryImplRef);
    this->familyImplRef = static_cast<void*>(&familyImpl);
    prometheus::Counter& counterImpl = familyImpl.Add(labels);
    return std::make_shared<MetricCounter>(labels, counterImpl);
}

template <>
std::shared_ptr<MetricGauge> MetricFamily<MetricGauge>::addMetric(const Metric::Labels& labels, const Metric::BucketBoundaries& bucketBoundaries) {
    prometheus::Family<prometheus::Gauge>& familyImpl = prometheus::BuildGauge()
                                                            .Name(this->getName())
                                                            .Help(this->getDesc())
                                                            .Register(this->registryImplRef);
    this->familyImplRef = static_cast<void*>(&familyImpl);
    prometheus::Gauge& gaugeImpl = familyImpl.Add(labels);
    return std::make_shared<MetricGauge>(labels, gaugeImpl);
}

template <>
std::shared_ptr<MetricHistogram> MetricFamily<MetricHistogram>::addMetric(const Metric::Labels& labels, const Metric::BucketBoundaries& bucketBoundaries) {
    prometheus::Family<prometheus::Histogram>& familyImpl = prometheus::BuildHistogram()
                                                                .Name(this->getName())
                                                                .Help(this->getDesc())
                                                                .Register(this->registryImplRef);
    this->familyImplRef = static_cast<void*>(&familyImpl);
    prometheus::Histogram& histogramImpl = familyImpl.Add(labels, bucketBoundaries);
    return std::make_shared<MetricHistogram>(labels, bucketBoundaries, histogramImpl);
}

template <>
void MetricFamily<MetricCounter>::remove(std::shared_ptr<MetricCounter> metric) {
    auto family = static_cast<prometheus::Family<prometheus::Counter>*>(this->familyImplRef);
    family->Remove(&metric->counterImpl);
}

template <>
void MetricFamily<MetricGauge>::remove(std::shared_ptr<MetricGauge> metric) {
    auto family = static_cast<prometheus::Family<prometheus::Gauge>*>(this->familyImplRef);
    family->Remove(&metric->gaugeImpl);
}

template <>
void MetricFamily<MetricHistogram>::remove(std::shared_ptr<MetricHistogram> metric) {
    auto family = static_cast<prometheus::Family<prometheus::Histogram>*>(this->familyImplRef);
    family->Remove(&metric->histogramImpl);
}

}  // namespace ovms
