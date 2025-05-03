import { useState } from "react";
import { TbAnalyzeFilled } from "react-icons/tb";
import { Loader2 } from "lucide-react";
import { Card } from "../../components/ui";

export default function AnalyzeModel({ availableModels = [], sessionId = 1 }) {
  console.log("Available models:", availableModels);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [modelDetails, setModelDetails] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSelectModel = async (modelId) => {
    setSelectedModelId(modelId);
    setLoading(true);
    setModelDetails(null);

    const selectedModel = availableModels.find((m) => m.index === modelId);
    if (!selectedModel) {
      setLoading(false);
      return;
    }

    // Dựng đường dẫn ảnh
    const imagePath = `/automl_outputs/automl_${sessionId}/plots/baseline_${modelId}_Feature_Importance.png`

    // Giả lập lấy metrics (có thể map lại tùy ý)
    const metrics = Object.entries(selectedModel)
    .filter(([key]) => key !== "index" && key !== "Model")
    .reduce((acc, [key, val]) => {
      acc[key] = val;
      return acc;
    }, {});


    setModelDetails({
      metrics,
      featureImageUrl: imagePath
    });

    setLoading(false);
  };

  return (
    <Card className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
        <h3 className="text-gray-800 text-xl flex items-center gap-2">
          <TbAnalyzeFilled />
          Analyze a Model
        </h3>

        <select
          value={selectedModelId}
          onChange={(e) => handleSelectModel(e.target.value)}
          className="border border-gray-300 rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 w-full sm:w-1/2"
        >
          <option value="">Select a model to analyze...</option>
          {availableModels.map((m) => (
            <option key={m.index} value={m.index}>
              {m.Model}
            </option>
          ))}
        </select>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-gray-700">
          <Loader2 size={16} className="animate-spin text-green-500" />
          <span className="capitalize">Analyzing selected model...</span>
        </div>
      )}

      {!loading && modelDetails && (
        <div
          className={`grid gap-8 ${
            modelDetails.featureImageUrl ? "grid-cols-3" : "grid-cols-1"
          }`}
        >
          {/* Metrics */}
          <div className="col-span-1 sm:col-span-1">
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(modelDetails.metrics).map(([metric, val]) => (
                <div
                  key={metric}
                  className="border border-gray-200 rounded-lg px-3 py-2 bg-green-50 hover:shadow transition text-center"
                >
                  <p className="text-xs text-gray-500 uppercase tracking-wide">{metric}</p>
                  <p className="text-base text-green-700 mt-1">
                    {typeof val === "number" ? val.toFixed(3) : val}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Feature Importance Image */}
          {modelDetails.featureImageUrl && (
            <div className="col-span-2 sm:col-span-2 flex justify-center">
              <img
                src={modelDetails.featureImageUrl}
                alt="Feature Importances"
                className="max-w-[500px] w-full object-contain rounded-md shadow"
              />
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
