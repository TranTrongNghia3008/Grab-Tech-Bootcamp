import { useState } from "react";
import { FaBullseye, FaWrench } from "react-icons/fa";
import { Tooltip } from 'react-tooltip';
import { Card } from "../../components/ui";
import BaselineTab from "./BaselineTab";
import TuningTab from "./TuningTab";
import PredictionTab from "./PredictionTab";
import { useAppContext } from "../../contexts/AppContext";
import { getModelPerformanceAnalysis } from "../../components/services/aisummaryServices";
import { parseAISummary } from "../../utils/parseHtml";

export default function ModelingPanel() {
  const { state } = useAppContext();
  const { datasetId, sessionId, comparisonResults, selectedTarget, selectedFeatures } = state;
  // const datasetId = 4; 
  const [activeTab, setActiveTab] = useState("Baseline");
  const [isFinalized, setIsFinalized] = useState(false);
  const [finalizedModelId, setFinalizedModelId] = useState(null);
  const [modelPerformanceAnalysis, setModelPerformanceAnalysis] = useState(null);
  const [loadingModelPerformanceAnalysis, setLoadingModelPerformanceAnalysis] = useState(false);

  console.log("ModelingPanel - datasetId:", datasetId);
  console.log("ModelingPanel - sessionId:", sessionId);
  console.log("ModelingPanel - comparisonResults:", comparisonResults);

  const formattedComparisonResults= comparisonResults.data.map(row => {
    const obj = {};
    comparisonResults.columns.forEach((col, idx) => {
      obj[col] = row[idx];
    });
    return obj;
  });

  const bestModel = formattedComparisonResults.reduce((best, current) => {
    const score = current["RMSE"] - current["R2"]; // Giảm RMSE, tăng R2
    const bestScore = best["RMSE"] - best["R2"];
    return score < bestScore ? current : best;
  });

  const handleFetchModelPerformanceAnalysis = async () => {
    setLoadingModelPerformanceAnalysis(true);
    try {
      const res = await getModelPerformanceAnalysis(comparisonResults);
      
      setModelPerformanceAnalysis(parseAISummary(res.summary_html)); 
    } catch (err) {
      console.error("Failed to fetch Model Performance Analysis:", err);
    } finally {
      setLoadingModelPerformanceAnalysis(false);
    }
  };

  const bestModelId = bestModel["index"];
  return (
    <div className="space-y-8">
      <h2 className="text-xl font-bold">Modeling</h2>

      {/* Target & Features */}
      <div className="space-y-4">
        <div className="flex">
          {/* Target */}
          <div className="space-y-1 text-gray-700 me-4 pe-4 border-r border-gray-300">
            <div className="flex items-center gap-2">
              <FaBullseye className="text-green-600" />
              <h4 className="text-sm font-semibold">Target Column</h4>
            </div>
            <p className="text-sm ml-6">
              {selectedTarget ? (
                <span className="inline-block bg-green-100 text-green-800 px-2 py-1 rounded text-xs">
                  {selectedTarget}
                </span>
              ) : (
                <span className="text-gray-400">No target selected.</span>
              )}
            </p>
          </div>

          {/* Features */}
          <div className="space-y-1 text-gray-700">
            <div className="flex items-center gap-2">
              <FaWrench className="text-green-600" />
              <h4 className="text-sm font-semibold">Feature Columns</h4>
            </div>
            {selectedFeatures.length > 0 ? (
              <div className="flex flex-wrap gap-2 ml-6">
                {selectedFeatures.map((feature) => (
                  <span
                    key={feature}
                    className="inline-block bg-green-50 border border-green-300 text-green-700 px-3 py-1 rounded-full text-xs"
                  >
                    {feature}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-sm ml-6 text-gray-400">No features selected.</p>
            )}
          </div>
        </div>

        <p className="text-xs text-gray-400 italic ml-6">
          The selected target and features will be used for model training and evaluation.
        </p>
      </div>

      {/* Tabs */}
      <div>
        <div className="flex space-x-4 mb-6 border-b border-gray-300">
          {["Baseline", "Tuning", "Prediction"].map((tab) => {
            const isDisabled = tab === "Prediction" && !isFinalized;

            const tooltipText = {
              Baseline: "View baseline models and evaluation metrics.",
              Tuning: "Adjust hyperparameters and optimize the best model.",
              Prediction: "Use the finalized model to make predictions."
            }[tab];

            return (
              <button
                key={tab}
                onClick={() => {
                  if (!isDisabled) setActiveTab(tab);
                }}
                disabled={isDisabled}
                className={`px-4 py-2 text-sm font-medium transition rounded-t-md 
                  ${activeTab === tab
                    ? "text-green-700 border-b-2 border-green-600"
                    : "hover:bg-green-100"
                  }
                  ${isDisabled ? "text-gray-400 cursor-not-allowed" : ""}
                `}
                data-tooltip-id="tab-tooltip"
                data-tooltip-content={tooltipText}
              >
                {tab}
              </button>
            );
          })}
          <Tooltip id="tab-tooltip" place="top-end" />
        </div>


        {/* Tab content */}
        {activeTab === "Baseline" && 
          <BaselineTab 
            comparisonResults={formattedComparisonResults} 
            sessionId={sessionId} bestModel={bestModel}
            modelPerformanceAnalysis={modelPerformanceAnalysis}
            loadingModelPerformanceAnalysis={loadingModelPerformanceAnalysis}
            handleFetchModelPerformanceAnalysis={handleFetchModelPerformanceAnalysis}
          />
        }
        {activeTab === "Tuning" && 
          <TuningTab 
            sessionId={sessionId} 
            bestModelId={bestModelId} 
            comparisonResults={formattedComparisonResults} 
            setIsFinalized={setIsFinalized} 
            setFinalizedModelId={setFinalizedModelId}
          />
        }
        {activeTab === "Prediction" && 
          <PredictionTab 
            finalizedModelId={finalizedModelId} 
          />
        }
      </div>

    </div>
  );
}
