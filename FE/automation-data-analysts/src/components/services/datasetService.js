import apiClient from "./apiClient";

export function upLoadDataset(projectName, file) {
  const formData = new FormData();
  formData.append("project_name", projectName);
  formData.append("file", file);

  return apiClient("/v1/datasets", {
    method: "POST",
    body: formData,
  });
}

export function getPreviewDataset(datasetId) {
  return apiClient(`/v1/datasets/${datasetId}/preview`, {
    method: "GET"
  })
}

export function getAllByCreation() {
  return apiClient("/v1/datasets/all-by-creation/", {
    method: "GET"
  })
}

export function getAnalysisReport(datasetId) {
  return apiClient(`/v1/datasets/${datasetId}/analysis-report`, {
    method: "GET"
  })
}

export function updateProjectName(datasetId, projectName) {
  const payload = {
    project_name: projectName
  }
  return apiClient(`/v1/datasets/${datasetId}/project_name/`, {
    method: "PATCH",
    body: payload
  })
}

export function deleteDataset(datasetId) {
  return apiClient(`/v1/datasets/${datasetId}`, {
    method: "DELETE"
  })
}