import apiClient from "./apiClient";

export function getPreviewIssues(datasetId) {
    const formData = new FormData();
    formData.append("dataset_id", datasetId);
    return apiClient(`/v1/datasets/${datasetId}/cleaning/preview`, {
        method: "GET",
        body: formData,
    });
}

export function cleaningDataset(datasetId, cleaningOptions) {
    return apiClient(`/v1/datasets/${datasetId}/cleaning`, {
        method: "POST",
        body: cleaningOptions,
    });
}

export function getCleaningStatus(job_id) {
    return apiClient(`/v1/cleaning/${job_id}/status`, {
        method: "GET",
    });
}