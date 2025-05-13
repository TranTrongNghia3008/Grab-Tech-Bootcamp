import apiClient from "./apiClient";

export function getSummaryStatistics(datasetId) {
    return apiClient(`/v1/datasets/${datasetId}/eda/stats`, {
        method: "GET",
    });
}

export function getCorrelation(datasetId) {
    return apiClient(`/v1/datasets/${datasetId}/eda/corr`, {
        method: "GET",
    });
}


export function getDataProfile(datasetId) {
    return apiClient(`/v1/datasets/${datasetId}/profile/download`, {
        method: "GET",
    });
}