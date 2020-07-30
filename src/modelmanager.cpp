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
#include "modelmanager.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>

#include "config.hpp"
#include "directoryversionreader.hpp"
#include "filesystem.hpp"
#include "gcsfilesystem.hpp"
#include "localfilesystem.hpp"
#include "s3filesystem.hpp"
#include "schema.hpp"

namespace ovms {

const uint ModelManager::WATCHER_INTERVAL_SEC = 1;

Status ModelManager::start() {
    auto& config = ovms::Config::instance();
    // start manager using config file
    if (config.configPath() != "") {
        return start(config.configPath());
    }

    // start manager using commandline parameters
    ModelConfig modelConfig{
        config.modelName(),
        config.modelPath(),
        config.targetDevice(),
        config.batchSize(),
        config.nireq()};

    auto status = modelConfig.parsePluginConfig(config.pluginConfig());
    if (!status.ok()) {
        spdlog::error("Couldn't parse plugin config");
        return status;
    }

    status = modelConfig.parseModelVersionPolicy(config.modelVersionPolicy());
    if (!status.ok()) {
        spdlog::error("Couldn't parse model version policy");
        return status;
    }

    status = modelConfig.parseShapeParameter(config.shape());
    if (!status.ok()) {
        spdlog::error("Couldn't parse shape parameter");
        return status;
    }

    bool batchSizeSet = (modelConfig.getBatchingMode() != FIXED || modelConfig.getBatchSize() != 0);
    bool shapeSet = (modelConfig.getShapes().size() > 0);

    spdlog::debug("Batch size set: {}, shape set: {}", batchSizeSet, shapeSet);
    if (batchSizeSet && shapeSet) {
        spdlog::warn("Both shape and batch size have been defined. Batch size parameter will be ignored.");
        modelConfig.setBatchingMode(FIXED);
        modelConfig.setBatchSize(0);
    }
    return reloadModelWithVersions(modelConfig);
}

Status ModelManager::start(const std::string& jsonFilename) {
    Status status = loadConfig(jsonFilename);
    if (!status.ok()) {
        return status;
    }
    if (!monitor.joinable()) {
        std::future<void> exitSignal = exit.get_future();
        std::thread t(std::thread(&ModelManager::watcher, this, std::move(exitSignal)));
        monitor = std::move(t);
        monitor.detach();
    }
    return StatusCode::OK;
}

Status ModelManager::loadConfig(const std::string& jsonFilename) {
    spdlog::info("Loading configuration from {}", jsonFilename);

    std::ifstream ifs(jsonFilename.c_str());
    if (!ifs.good()) {
        spdlog::error("File is invalid {}", jsonFilename);
        return StatusCode::FILE_INVALID;
    }

    rapidjson::Document configJson;
    rapidjson::IStreamWrapper isw(ifs);
    if (configJson.ParseStream(isw).HasParseError()) {
        spdlog::error("Configuration file is not a valid JSON file.");
        return StatusCode::JSON_INVALID;
    }

    if (validateJsonAgainstSchema(configJson, MODELS_CONFIG_SCHEMA) != StatusCode::OK) {
        return StatusCode::JSON_INVALID;
    }

    const auto itr = configJson.FindMember("model_config_list");
    if (itr == configJson.MemberEnd() || !itr->value.IsArray()) {
        spdlog::error("Configuration file doesn't have models property.");
        return StatusCode::JSON_INVALID;
    }

    // TODO reload model if no version change, just config change eg. CPU_STREAMS_THROUGHPUT
    configFilename = jsonFilename;
    std::set<std::string> modelsInConfigFile;
    for (const auto& configs : itr->value.GetArray()) {
        ModelConfig modelConfig;
        auto status = modelConfig.parseNode(configs["config"]);
        if (status != StatusCode::OK) {
            return status;
        }
        reloadModelWithVersions(modelConfig);
        modelsInConfigFile.emplace(modelConfig.getName());
    }
    retireModelsRemovedFromConfigFile(modelsInConfigFile);
    return StatusCode::OK;
}

void ModelManager::retireModelsRemovedFromConfigFile(const std::set<std::string>& modelsExistingInConfigFile) {
    std::set<std::string> modelsCurrentlyLoaded;
    for (auto& nameModelPair : getModels()) {
        modelsCurrentlyLoaded.insert(nameModelPair.first);
    }
    std::vector<std::string> modelsToUnloadAllVersions(getModels().size());
    auto it = std::set_difference(
        modelsCurrentlyLoaded.begin(), modelsCurrentlyLoaded.end(),
        modelsExistingInConfigFile.begin(), modelsExistingInConfigFile.end(),
        modelsToUnloadAllVersions.begin());
    modelsToUnloadAllVersions.resize(it - modelsToUnloadAllVersions.begin());
    for (auto& modelName : modelsToUnloadAllVersions) {
        try {
            models.at(modelName)->retireAllVersions();
        } catch (const std::out_of_range& e) {
            SPDLOG_ERROR("Unknown error occured when tried to retire all versions of model:{}", modelName);
        }
    }
}

void ModelManager::watcher(std::future<void> exit) {
    spdlog::info("Started config watcher thread");
    int64_t lastTime;
    struct stat statTime;

    stat(configFilename.c_str(), &statTime);
    lastTime = statTime.st_ctime;
    while (exit.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout) {
        std::this_thread::sleep_for(std::chrono::seconds(WATCHER_INTERVAL_SEC));
        stat(configFilename.c_str(), &statTime);
        if (lastTime != statTime.st_ctime) {
            lastTime = statTime.st_ctime;
            loadConfig(configFilename);
            spdlog::info("Model configuration changed");
        }
    }
    spdlog::info("Exited config watcher thread");
}

void ModelManager::join() {
    exit.set_value();
    if (monitor.joinable()) {
        monitor.join();
    }
}

std::shared_ptr<IVersionReader> ModelManager::getVersionReader(const std::string& path) {
    return std::make_shared<DirectoryVersionReader>(path);
}

void ModelManager::getVersionsToChange(
    const std::map<model_version_t, std::shared_ptr<ModelInstance>>& modelVersionsInstances,
    model_versions_t requestedVersions,
    std::shared_ptr<model_versions_t>& versionsToStartIn,
    std::shared_ptr<model_versions_t>& versionsToReloadIn,
    std::shared_ptr<model_versions_t>& versionsToRetireIn) {
    std::sort(requestedVersions.begin(), requestedVersions.end());
    model_versions_t registeredModelVersions;
    spdlog::info("Currently registered versions count:{}", modelVersionsInstances.size());
    for (const auto& [version, versionInstance] : modelVersionsInstances) {
        spdlog::info("version:{} state:{}", version, ovms::ModelVersionStateToString(versionInstance->getStatus().getState()));
        registeredModelVersions.push_back(version);
    }

    model_versions_t alreadyRegisteredVersionsWhichAreRequested(requestedVersions.size());
    model_versions_t::iterator it = std::set_intersection(
        requestedVersions.begin(), requestedVersions.end(),
        registeredModelVersions.begin(), registeredModelVersions.end(),
        alreadyRegisteredVersionsWhichAreRequested.begin());
    alreadyRegisteredVersionsWhichAreRequested.resize(it - alreadyRegisteredVersionsWhichAreRequested.begin());

    std::shared_ptr<model_versions_t> versionsToReload = std::make_shared<model_versions_t>();
    for (const auto& version : alreadyRegisteredVersionsWhichAreRequested) {
        try {
            if (modelVersionsInstances.at(version)->getStatus().willEndUnloaded()) {
                versionsToReload->push_back(version);
            }
        } catch (std::out_of_range& e) {
            spdlog::error("Data race occured during versions update. Could not found version. Details:{}", e.what());
        }
    }

    std::shared_ptr<model_versions_t> versionsToRetire = std::make_shared<model_versions_t>(registeredModelVersions.size());
    it = std::set_difference(
        registeredModelVersions.begin(), registeredModelVersions.end(),
        requestedVersions.begin(), requestedVersions.end(),
        versionsToRetire->begin());
    versionsToRetire->resize(it - versionsToRetire->begin());
    try {
        it = std::remove_if(versionsToRetire->begin(),
            versionsToRetire->end(),
            [&modelVersionsInstances](model_version_t version) {
                return modelVersionsInstances.at(version)->getStatus().willEndUnloaded();
            });
    } catch (std::out_of_range& e) {
        spdlog::error("Data race occured during versions update. Could not found version. Details:{}", e.what());
    }
    versionsToRetire->resize(it - versionsToRetire->begin());

    std::shared_ptr<model_versions_t> versionsToStart = std::make_shared<model_versions_t>(requestedVersions.size());
    it = std::set_difference(
        requestedVersions.begin(), requestedVersions.end(),
        registeredModelVersions.begin(), registeredModelVersions.end(),
        versionsToStart->begin());
    versionsToStart->resize(it - versionsToStart->begin());

    versionsToStartIn = std::move(versionsToStart);
    versionsToReloadIn = std::move(versionsToReload);
    versionsToRetireIn = std::move(versionsToRetire);
}

std::shared_ptr<ovms::Model> ModelManager::getModelIfExistCreateElse(const std::string& modelName) {
    auto modelIt = models.find(modelName);
    if (models.end() == modelIt) {
        models[modelName] = modelFactory(modelName);
    }
    return models[modelName];
}

std::shared_ptr<FileSystem> getFilesystem(const std::string& basePath) {
    if (basePath.rfind("s3://", 0) == 0) {
        Aws::SDKOptions options;
        Aws::InitAPI(options);
        return std::make_shared<S3FileSystem>(options, basePath);
    }
    if (basePath.rfind("gs://", 0) == 0) {
        return std::make_shared<ovms::GCSFileSystem>();
    }
    return std::make_shared<LocalFileSystem>();
}

Status ModelManager::reloadModelWithVersions(ModelConfig& config) {
    auto fs = getFilesystem(config.getBasePath());
    std::string localPath;
    spdlog::info("Getting model from {}", config.getBasePath());
    auto sc = fs->downloadFileFolder(config.getBasePath(), &localPath);
    if (sc != StatusCode::OK) {
        spdlog::error("Couldn't download model from {}", config.getBasePath());
        return sc;
    }
    config.setBasePath(localPath);
    std::vector<model_version_t> requestedVersions;
    std::shared_ptr<IVersionReader> versionReader = getVersionReader(localPath);
    auto status = versionReader->readAvailableVersions(requestedVersions);
    if (!status.ok()) {
        return status;
    }
    requestedVersions = config.getModelVersionPolicy()->filter(requestedVersions);
    // TODO check if reload whole model when part of config changes (eg. CPU_THROUGHPUT_STREAMS)
    // right now assumes no need to reload model
    std::shared_ptr<model_versions_t> versionsToStart;
    std::shared_ptr<model_versions_t> versionsToReload;
    std::shared_ptr<model_versions_t> versionsToRetire;

    auto model = getModelIfExistCreateElse(config.getName());
    getVersionsToChange(model->getModelVersions(), requestedVersions, versionsToStart, versionsToReload, versionsToRetire);

    status = model->addVersions(versionsToStart, config);
    if (!status.ok()) {
        spdlog::error("Error occurred while loading model: {} versions; error: {}",
            config.getName(),
            status.string());
        return status;
    }
    status = model->reloadVersions(versionsToReload, config);
    if (!status.ok()) {
        spdlog::error("Error occurred while reloading model: {}; versions; error: {}",
            config.getName(),
            status.string());
        return status;
    }
    status = model->retireVersions(versionsToRetire);
    if (!status.ok()) {
        spdlog::error("Error occurred while unloading model: {}; versions; error: {}",
            config.getName(),
            status.string());
        return status;
    }
    return StatusCode::OK;
}

}  // namespace ovms