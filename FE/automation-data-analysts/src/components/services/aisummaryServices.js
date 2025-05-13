import apiClient from "./apiClient";

export function getAISummaryStatistics(data) {
    const payload = {
        data: data
    }
    return apiClient("/v1/summary-stats", {
        method: "POST",
        body: payload
    }) 
}

export function getAICorrelationMatrix(data) {
    const payload = {
        data: data
    }
    return apiClient("/v1/correlation-matrix", {
        method: "POST",
        body: payload
    }) 
}

export function getModelPerformanceAnalysis(data) {
    const columns = Object.keys(data[0]);

    const dataList = data.map(item => columns.map(col => item[col]));

    const payload = {
        "columns": columns,
        "data": dataList
    };
    console.log("getModelPerformanceAnalysis", payload)
    return apiClient("/v1/model-performance", {
        method: "POST",
        body: payload
    }) 
}

export function getTunedModelEvaluation( tunedResultsForEvaluation, imageFile = null, imageUrl = null ) {
    const formData = new FormData();
    const fixedTuningData = JSON.parse(JSON.stringify(tunedResultsForEvaluation).replace(/\\\\|\\/g, "/"));
    console.log("fixedTuningData", fixedTuningData)

    formData.append("tuning_data_json", JSON.stringify(fixedTuningData)); 

    if (imageFile) {
        formData.append("feature_importance_image", imageFile);
    }

    if (imageUrl) {
        formData.append("image_url", imageUrl);
    }

    return apiClient("/v1/tuned-model-evaluation", {
        method: "POST",
        body: formData,
    });
}
  
export function getBaselineModelEvaluation( {metrics, featureImageUrl = null}) {
    const formData = new FormData();
    formData.append("metrics_data_json", JSON.stringify(metrics)); 
    console.log(metrics)

    console.log(featureImageUrl)

    if (featureImageUrl) {
        formData.append("feature_importance_image_path", `../FE/automation-data-analysts/public/${featureImageUrl}`);
    }

    return apiClient("/v1/baseline-model-evaluation", {
        method: "POST",
        body: formData,
    });
}
  