import apiClient from "./apiClient";

export function upLoadDataset(file) {
  const formData = new FormData();
  formData.append("file", file);

  return apiClient("/v1/datasets", {
    method: "POST",
    body: formData,
  });
}