import { useState } from "react";
import { TbAnalyzeFilled } from "react-icons/tb";
import { Loader2 } from "lucide-react";
import { Button, Card } from "../../components/ui";
import { getBaselineModelEvaluation } from "../../components/services/aisummaryServices";
import { parseAISummary } from "../../utils/parseHtml";

export default function AnalyzeModel({ availableModels = [], sessionId = 1, imgPath = "" }) {
  console.log("Available models:", availableModels);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [modelDetails, setModelDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [baselineModelEvaluation, setBaselineModelEvaluation] = useState(null);
  const [loadingBaselineModelEvaluation, setLoadingBaselineModelEvaluation] = useState(false);

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
    let imagePath = `/automl_outputs/automl_${sessionId}/plots/baseline_${modelId}_Feature_Importance.png`
    if (imgPath) {
      imagePath = imgPath
      .replace(/\\/g, "/") // convert Windows \ to /
      .replace("../FE/automation-data-analysts/public", "");
    }

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

  const handleFetchBaselineModelEvaluation = async () => {
    setLoadingBaselineModelEvaluation(true);
    try {
      const res = await getBaselineModelEvaluation(modelDetails);
      
      setBaselineModelEvaluation(parseAISummary(res.summary_html)); 
    } catch (err) {
      console.error("Failed to fetch Baseline Model Evaluation:", err);
    } finally {
      setLoadingBaselineModelEvaluation(false);
    }
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

      {!imgPath && modelDetails && (
        <>
          <div className="flex justify-between bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm">
          <p className="me-5 my-auto">
          <strong>Curious about how your tuned model truly performs - and what’s driving its decisions?</strong> <br/>
          We've analyzed the model’s consistency, key contributing features, and provided actionable insights to help you confidently move toward deployment or further refinement.          </p>
          <Button
            onClick={handleFetchBaselineModelEvaluation}
            disabled={loadingBaselineModelEvaluation}
          >
            {loadingBaselineModelEvaluation ? "Analyzing..." : "Explore"}
          </Button>
        </div>
        {baselineModelEvaluation && (
          <div
            className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm"
            dangerouslySetInnerHTML={{ __html: baselineModelEvaluation }}
          />
        )}
        </>
      )}
    </Card>
  );
}
