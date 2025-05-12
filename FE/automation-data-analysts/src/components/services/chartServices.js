import apiClient from "./apiClient";

export function getChartColumns({ datasetId, chartType }) {
    return apiClient(`/v1/get_chart_columns?dataset_id=${datasetId}&chart_type=${chartType}`, {
        method: "GET",
    })
}

export function getChartSummary({ datasetId, xColumn, yColumn = "", bins = 10 }) {
  const params = new URLSearchParams();
  params.append("dataset_id", datasetId);
  params.append("x_column", xColumn);
  if (yColumn) params.append("y_column", yColumn);
  if (bins !== undefined) params.append("bins", bins);
  console.log(params.toString())

  return apiClient(`/v1/get_summary?${params.toString()}`, {
    method: "GET",
  });
}