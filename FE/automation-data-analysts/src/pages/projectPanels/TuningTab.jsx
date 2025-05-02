import { useState } from "react";
import { FaCogs, FaCalculator, FaDownload } from "react-icons/fa";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { Card, Button } from "../../components/ui";
import DataTable from "../../components/DataTable";
import AnalyzeModel from "./AnalyzeModel";

export default function TuningTab({ bestModelId }) {
  const [modelType, setModelType] = useState("");
  const [customGrid, setCustomGrid] = useState("");
  const [loading, setLoading] = useState(false);
  const [jobStatus, setJobStatus] = useState(null);

  const [bestParams, setBestParams] = useState({});
  const [cvMetrics, setCvMetrics] = useState([]);
  const [showErrors, setShowErrors] = useState(false);
  
  const modelOptions = [
    { id: "lr", name: "Logistic Regression" },
    { id: "ridge", name: "Ridge Classifier" },
    { id: "et", name: "Extra Trees" },
    { id: "nb", name: "Naive Bayes" },
    { id: "knn", name: "K Neighbors" },
  ];

  const handleTrainModel = () => {
    if (!modelType) {
      setShowErrors(true)
      return;
    }

    if (customGrid) {
      try {
        JSON.parse(customGrid);
      } catch (err) {
        alert("Invalid Custom Grid JSON: " + err.message);
        return;
      }
    }

    setLoading(true);
    setJobStatus("pending");

    setTimeout(() => {
      setJobStatus("running");

      setTimeout(() => {
        setBestParams({
          C: 2.1,
          penalty: "l2",
          solver: "lbfgs",
          random_state: 42
        });

        setCvMetrics([
          { Fold: 0, Accuracy: 0.8420, AUC: 0.8620, Recall: 0.7552, Precision: 0.8250, F1: 0.7885, Kappa: 0.6680, MCC: 0.6702 },
          { Fold: 1, Accuracy: 0.8650, AUC: 0.9150, Recall: 0.8652, Precision: 0.8200, F1: 0.8420, Kappa: 0.7259, MCC: 0.7268 },
          { Fold: "Mean", Accuracy: 0.8535, AUC: 0.8885, Recall: 0.8102, Precision: 0.8225, F1: 0.8152, Kappa: 0.6969, MCC: 0.6985 }
        ]);
        setLoading(false);
        setJobStatus("done");
      }, 2000);
    }, 1000);
  };

  const handleDownloadModel = () => {
    const blob = new Blob(["This is your tuned model (.pkl)"], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "tuned_model.pkl";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-8">
      <Card className="space-y-6">
        {/* Select Model & Custom Grid */}
        <div className="flex flex-col sm:flex-row gap-6">
          {/* Model Type */}
          <div className="flex-1 space-y-4">
            <label className="text-sm font-medium text-gray-700 mb-1">
              Model Type
            </label>
            <p className="text-xs text-green-600 my-1">
                Suggested model: {modelOptions.find((model) => model.id === bestModelId)?.name} (based on previous best results)
            </p>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="w-full border border-gray-300 rounded-md px-3 py-2 my-0 focus:outline-none focus:ring-2 focus:ring-green-500"
            >
              <option value="">Select Model</option>
              {modelOptions.map((m) => (
                <option key={m.id} value={m.name}>
                  {m.name}
                </option>
              ))}
            </select>
            
            {showErrors && !modelType && (
                <p className="text-xs text-red-600 mt-1">Please select a model type.</p>
            )}
          </div>

          {/* Custom Grid */}
          <div className="flex-1">
            <label className="text-sm font-medium text-gray-700 mb-1">
              Custom Grid (Optional)
            </label>
            <textarea
              rows={4}
              value={customGrid}
              onChange={(e) => setCustomGrid(e.target.value)}
              placeholder='e.g., {"n_estimators": [100, 200], "max_depth": [10, 20]}'
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 text-sm"
            />
          </div>
        </div>

        {/* Train Button */}
        <div className="text-right">
          <Button onClick={handleTrainModel} disabled={loading || jobStatus === "running"}>
            {loading || jobStatus === "running" ? "Training..." : "Train Model"}
          </Button>
        </div>

        {/* Status */}
        {jobStatus && (
          <div className="flex items-center gap-2 text-sm text-gray-700">
            {jobStatus === "done" && <CheckCircle size={16} className="text-green-600" />}
            {jobStatus === "error" && <AlertCircle size={16} className="text-red-600" />}
            {jobStatus === "running" && <Loader2 size={16} className="animate-spin text-green-500" />}
            {jobStatus === "pending" && <Loader2 size={16} className="animate-pulse text-yellow-500" />}
            <span className="capitalize">Status: {jobStatus}</span>
          </div>
        )}
      </Card>

      {/* Best Params */}
      {Object.keys(bestParams).length > 0 && (
        <Card className="space-y-4">
          <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
            <FaCogs /> Best Parameters
          </h3>
          <DataTable data={Object.entries(bestParams).map(([key, val]) => ({ Parameter: key, Value: String(val) }))} />
        </Card>
      )}

      {/* Cross-Validation Metrics */}
      {cvMetrics.length > 0 && (
        <Card className="space-y-4">
          <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
            <FaCalculator /> Cross-Validation Metrics
          </h3>
          <DataTable data={cvMetrics} />
        </Card>
      )}

      {/* Analyze Model */}
      {jobStatus === "done" && (
        <AnalyzeModel availableModels={[{ modelId: "tuned_model", modelName: "Your Tuned Model" }]} />
      )}

      {/* Download Model */}
      {jobStatus === "done" && (
        <div className="text-right">
          <Button onClick={handleDownloadModel}>
            <FaDownload className="mr-2" /> Download Model (.pkl)
          </Button>
        </div>
      )}
    </div>
  );
}
