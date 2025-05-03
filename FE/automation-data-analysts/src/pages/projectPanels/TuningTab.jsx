import { useState } from "react";
import { FaCogs, FaCalculator, FaDownload } from "react-icons/fa";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { Card, Button } from "../../components/ui";
import DataTable from "../../components/DataTable";
import AnalyzeModel from "./AnalyzeModel";
import { finalizeModel, tuningSession } from "../../components/services/modelingServices";

export default function TuningTab({ sessionId, bestModelId, comparisonResults = [], setIsFinalized }) {
  const [modelType, setModelType] = useState("");
  const [customGrid, setCustomGrid] = useState("");
  const [loading, setLoading] = useState(false);
  const [jobStatus, setJobStatus] = useState(null);

  const [bestParams, setBestParams] = useState({});
  const [cvMetrics, setCvMetrics] = useState([]);
  const [showErrors, setShowErrors] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [isloadingFinalized, setIsLoadingFinalized] = useState(false);
  
  const modelOptions = Array.isArray(comparisonResults)
  ? comparisonResults.map((model) => ({
      id: model.index,
      name: model.Model,
    }))
  : [];

  const handleTrainModel = async () => {
    if (!modelType) {
      setShowErrors(true);
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
  
    try {
      setJobStatus("running");
      const tuneResults = await tuningSession(sessionId, modelType);
      console.log("Tuning results:", tuneResults);
  
      // Set best parameters
      setBestParams(tuneResults.best_params || {});
  
      // Format CV metrics
      const columns = tuneResults.cv_metrics_table.columns;
      const rows = tuneResults.cv_metrics_table.data;
  
      const formattedMetrics = rows.map((row) => {
        const rowObj = {};
        columns.forEach((col, i) => {
          rowObj[col] = typeof row[i] === "number" ? Number(row[i].toFixed(4)) : row[i];
        });
        return rowObj;
      });
  
      setCvMetrics(formattedMetrics);

      const meanRow = formattedMetrics.find((item) => item.Fold === "Mean");

      if (meanRow) {
        const dynamicMetrics = Object.entries(meanRow)
          .filter(([key]) => key !== "Fold")
          .map(([key, value]) => ({ [key]: value }));

        // Chuyển về 1 object thay vì mảng các object đơn
        const mergedMetrics = Object.assign({}, ...dynamicMetrics);

        setAvailableModels([
          {
            index: tuneResults.tuned_model_id,
            Model: "Your Tuned Model",
            ...mergedMetrics,
          },
        ]);
      }
  
  
      setJobStatus("done");
    } catch (error) {
      console.error("Tuning failed:", error);
      setJobStatus("error");
    } finally {
      setLoading(false);
    }
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
  
  const handleFinalizeModel = async () => {
    try {
      setIsLoadingFinalized(true);
      await finalizeModel(sessionId, modelType);
      setIsFinalized(true);
    } catch (error) {
      console.error("Failed to finalize model:", error);
    } finally {
      setIsLoadingFinalized(false);
    }
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
                <option key={m.id} value={m.id}>
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
        <AnalyzeModel availableModels={availableModels} sessionId={sessionId}/>
      )}

      {/* Download Model */}
      {jobStatus === "done" && (
        <div className="flex justify-end gap-4">
          <Button onClick={handleDownloadModel}>
            <FaDownload className="mr-2" /> Download Model (.pkl)
          </Button>
          <Button variant="outline" onClick={handleFinalizeModel} disabled={isloadingFinalized}>
            {isloadingFinalized ? "Finalizing..." : "Finalize Model"}
          </Button>
        </div>
      )}
    </div>
  );
}
