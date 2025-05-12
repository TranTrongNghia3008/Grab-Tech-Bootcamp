import apiClient from "./apiClient";

export function autoMLSession(datasetId, targetColumn, featureColumns) {
    const payload = {
        dataset_id: datasetId,
        target_column: targetColumn,
        feature_columns: featureColumns,  // <-- đảm bảo là array
        name: "string",
    };
    console.log("autoMLSession", payload);
    return apiClient(`/v1/automl_sessions`, {
        method: "POST",
        body: payload,
    });
}

export function tuningSession(sessionId, modelType) {
    const payload = {
        model_id_to_tune: modelType,
    };
    return apiClient(`/v1/${sessionId}/step2-tune`, {
        method: "POST",
        body: payload,
    });
}

export function finalizeModel(sessionId, modelNameInput) {
    console.log("finalizeModel", sessionId, modelNameInput);
    const payload = {
        model_name_override: modelNameInput,
    };
    return apiClient(`/v1/${sessionId}/step3-finalize`, {
        method: "POST",
        body: payload,
    });
}

export function predictModel(finalizedModelId, file) {
    console.log("predictModel", finalizedModelId, file);
    const formData = new FormData();
    formData.append("file", file);
    return apiClient(`/v1/finalized_models/${finalizedModelId}/predict`, {
        method: "POST",
        body: formData,
    });
}

export function getAutoMLResults(datasetId) {
    return apiClient(`/v1/datasets/${datasetId}/automl-session-results`, {
        method: "GET"
    })
}

export function getListFinalizedModels(datasetId) {
    return apiClient(`/v1/datasets/${datasetId}/finalized-models`, {
        method: "GET"
    })
}

export function downLoadFinalizedModel(finalizedModelId) {
    return apiClient(`/v1/finalized-models/${finalizedModelId}/download`, {
        method: "GET",
        raw: true, 
    })
}