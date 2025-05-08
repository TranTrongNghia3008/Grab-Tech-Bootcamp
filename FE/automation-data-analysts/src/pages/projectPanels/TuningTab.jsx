import { useState, useEffect } from "react";
import { FaCogs, FaCalculator, FaDownload } from "react-icons/fa";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { Card, Button } from "../../components/ui";
import DataTable from "../../components/DataTable";
import AnalyzeModel from "./AnalyzeModel";
import { finalizeModel, tuningSession } from "../../components/services/modelingServices";
import { useAppContext } from "../../contexts/AppContext";
import { getTunedModelEvaluation } from "../../components/services/aisummaryServices";
import { parseAISummary } from "../../utils/parseHtml";

export default function TuningTab({ sessionId, bestModelId, comparisonResults = [], setIsFinalized, setFinalizedModelId }) {
  const { state, updateState } = useAppContext();
  const { tuningResults } = state;
  const [modelType, setModelType] = useState("");
  const [customGrid, setCustomGrid] = useState("");
  const [loading, setLoading] = useState(false);
  const [jobStatus, setJobStatus] = useState(null);

  const [bestParams, setBestParams] = useState({});
  const [cvMetrics, setCvMetrics] = useState([]);
  const [showErrors, setShowErrors] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [isloadingFinalized, setIsLoadingFinalized] = useState(false);
  const [tunedModelPath, setTunedModelPath] = useState(null);
  const [tunedFeatureImportancePlotPath, setTunedFeatureImportancePlotPath] = useState(null);
  const [tunedResultsForEvaluation, setTunedResultsForEvaluation] = useState(null);
  const [tunedModelEvaluation, setTunedModelEvaluation] = useState(null);
  const [loadingTunedModelEvaluation, setLoadingTunedModelEvaluation] = useState(false);

  useEffect(() => {
    if (tuningResults) {
      console.log("Tuning Results from state:", tuningResults);
      processTuningResults(tuningResults)
    }
  }, [tuningResults]);

  const processTuningResults = (tuneResults) => {
    setTunedModelPath(tuneResults.tuned_model_save_path_base);
    setTunedFeatureImportancePlotPath(tuneResults.feature_importance_plot_path);
  
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
  };
  
  
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
      updateState({tuningResults: tuneResults})
      setTunedResultsForEvaluation(tuneResults)
      
      // const tunedModelEvaluation = await getTunedModelEvaluation(tuneResults)
      // console.log("Tuned Model Evaluation: ", tunedModelEvaluation)

      processTuningResults(tuneResults)
    } catch (error) {
      console.error("Tuning failed:", error);
      setJobStatus("error");
    } finally {
      setLoading(false);
    }
  };
  
  const handleDownloadModel = () => {
    const formattedPath = tunedModelPath
      .replace(/\\/g, "/") // convert Windows \ to /
      .replace("../FE/automation-data-analysts/public", "");
  
    // Assume file is model.pkl inside that directory
    const downloadUrl = `${formattedPath}.pkl`;
  
    // Trigger download
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = "tuned_model.pkl";
    a.click();
  };
  
  
  const handleFinalizeModel = async () => {
    try {
      setIsLoadingFinalized(true);
      const finalizeResults = await finalizeModel(sessionId, modelType);
      console.log("Finalize results:", finalizeResults);
      setFinalizedModelId(finalizeResults.finalized_model_db_id);
      setIsFinalized(true);
    } catch (error) {
      console.error("Failed to finalize model:", error);
    } finally {
      setIsLoadingFinalized(false);
    }
  };
  
    const handleFetchTunedModelEvaluation = async () => {
      setLoadingTunedModelEvaluation(true);
      try {
        const res = await getTunedModelEvaluation(tunedResultsForEvaluation);
        
        setTunedModelEvaluation(parseAISummary(res.summary_html)); 
      } catch (err) {
        console.error("Failed to fetch Tuned Model Evaluation:", err);
      } finally {
        setLoadingTunedModelEvaluation(false);
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
              <option value="">Select model for tuning...</option>
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
        <AnalyzeModel availableModels={availableModels} sessionId={sessionId} imgPath={tunedFeatureImportancePlotPath}/>
      )}

      {jobStatus === "done" && (
        <>
          <div className="flex justify-between bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm">
          <p className="me-5 my-auto">
          <strong>Curious about how your tuned model truly performs - and what’s driving its decisions?</strong> <br/>
          We've analyzed the model’s consistency, key contributing features, and provided actionable insights to help you confidently move toward deployment or further refinement.          </p>
          <Button
            onClick={handleFetchTunedModelEvaluation}
            disabled={loadingTunedModelEvaluation}
          >
            {loadingTunedModelEvaluation ? "Analyzing..." : "Explore"}
          </Button>
        </div>
        {tunedModelEvaluation && (
          <div
            className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm"
            dangerouslySetInnerHTML={{ __html: tunedModelEvaluation }}
          />
        )}
        </>
      )}

      

      {/* Download Model */}
      {jobStatus === "done" && (
        <div className="flex justify-end gap-4">
          <Button onClick={handleDownloadModel}>
            <FaDownload className="mr-2" /> Download tuned model
          </Button>
          <Button variant="outline" onClick={handleFinalizeModel} disabled={isloadingFinalized}>
            {isloadingFinalized ? "Finalizing..." : "Finalize Model"}
          </Button>
        </div>
      )}
    </div>
  );
}
