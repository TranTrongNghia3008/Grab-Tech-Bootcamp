import { useEffect, useState } from "react";
import { FaBullseye, FaWrench } from "react-icons/fa";
import { Tooltip } from 'react-tooltip';
import { Loader2 } from "lucide-react";
import { Card, Modal, Toast } from "../../components/ui";
import BaselineTab from "./BaselineTab";
import TuningTab from "./TuningTab";
import PredictionTab from "./PredictionTab";
import { useAppContext } from "../../contexts/AppContext";
import { getModelPerformanceAnalysis } from "../../components/services/aisummaryServices";
import { parseAISummary } from "../../utils/parseHtml";
import SetupModelModal from "./SetupModelModal";
import { autoMLSession } from "../../components/services/modelingServices";

export default function ModelingPanel() {
  const { state, updateState } = useAppContext();
  const { datasetId, sessionId, autoMLResults, columns} = state;
  const [activeTab, setActiveTab] = useState("Baseline");
  const [isFinalized, setIsFinalized] = useState(false);
  const [modelPerformanceAnalysis, setModelPerformanceAnalysis] = useState(null);
  const [loadingModelPerformanceAnalysis, setLoadingModelPerformanceAnalysis] = useState(false);
  const [setupModelModal, setSetupModelModal] = useState(0)
  const [selectedTarget, setSelectedTarget] = useState(null)
  const [selectedFeatures, setSelectedFeatures] = useState([])
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzingStatus, setAnalyzingStatus] = useState(null);
  const [showToastAnalyzing, setShowToastAnalyzing] = useState(false);
  const [comparisonResults, setComparisonResults] = useState(null)
  const [bestModel, setBestModel] = useState(null)

  console.log("ModelingPanel - datasetId:", datasetId);
  console.log("ModelingPanel - sessionId:", sessionId);
  console.log("ModelingPanel - autoMLResults:", autoMLResults);

  useEffect(() => {
    if (autoMLResults) 
      {
        setSelectedTarget(autoMLResults.step1_results.target_column)
        setSelectedFeatures(autoMLResults.step1_results.feature_columns)
        setComparisonResults(autoMLResults.step1_results.comparison_results)
        const formattedComparisonResults= autoMLResults.step1_results.comparison_results.data.map(row => {
          const obj = {};
          autoMLResults.step1_results.comparison_results.columns.forEach((col, idx) => {
            obj[col] = row[idx];
          });
          return obj;
        });
        setComparisonResults(formattedComparisonResults)

        const best = formattedComparisonResults.reduce((best, current) => {
          const score = current["RMSE"] - current["R2"]; // Giảm RMSE, tăng R2
          const bestScore = best["RMSE"] - best["R2"];
          return score < bestScore ? current : best;
        });
        setBestModel(best)

        console.log("ModelingPanel - comparisonResults:", formattedComparisonResults);

        updateState({ turningResults: autoMLResults.step2_results })

      }
  }, [])

  // useEffect(() => {
  //   if (target) setSelectedTarget(target)
  //   if (features) setSelectedFeatures(features)
  // }, [target, features])

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

  useEffect(() => {
    if (analyzingStatus === "completed" || analyzingStatus === "failed") {
      setShowToastAnalyzing(true);
      setTimeout(() => {
        setShowToastAnalyzing(false);
      }, 3000);
    }
  }, [analyzingStatus]);

  const handleFinishSetupModelModal = async () => {
    updateState({ target: selectedTarget, features: selectedFeatures })
    setSetupModelModal(0);
    setAnalyzing(true);

    try { 
      const results = await autoMLSession(datasetId, selectedTarget, selectedFeatures);
      setAnalyzing(false);
      console.log("Finished analyzing and training models!");
      console.log("Results:", results);
      updateState({sessionId: results.session_id, comparisonResults: results.comparison_results});
      setAnalyzingStatus("completed");
    } catch (error) {
      console.error("Error during analyzing and training models:", error);
      setAnalyzing(false);
      setAnalyzingStatus("failed");
    } 
  };

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
          <div className="space-y-1 text-gray-700 flex-1">
            <div className="flex items-center gap-2">
              <FaWrench className="text-green-600" />
              <h4 className="text-sm font-semibold">Feature Columns</h4>
            </div>
            {selectedFeatures.length > 0 ? (
              <div className="ml-6 overflow-x-auto" style={{ maxWidth: "calc(100vw - 300px)" }}>
                <div className="flex gap-2 w-max">
                  {selectedFeatures.map((feature) => (
                    <span
                      key={feature}
                      className="inline-block bg-green-50 border border-green-300 text-green-700 px-3 py-1 rounded-full text-xs whitespace-nowrap"
                    >
                      {feature}
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-sm ml-6 text-gray-400">No features selected.</p>
            )}
          </div>
        </div>

        <p className="text-xs text-gray-400 italic ml-6">
          The selected target and features will be used for model training and evaluation.
        </p>
        <p className="text-xs text-gray-400 italic ml-6">
          Do you want to change target and feature?
          <button
            onClick={() => setSetupModelModal(1)}
            className="text-green-600 underline hover:text-green-700 ml-1 hover:cursor-pointer"
          >
            Click here
          </button>
        </p>
        {setupModelModal === 1 && (
          <Modal
            title="Select Target and Features"
            onClose={() => {
              setSetupModelModal(0);
              setSelectedTarget(autoMLResults.step1_results.target_column);
              setSelectedFeatures(autoMLResults.step1_results.feature_columns);
            }}
          >
            <SetupModelModal
              availableColumns={columns}
              selectedTarget={selectedTarget}
              setSelectedTarget={setSelectedTarget}
              selectedFeatures={selectedFeatures}
              setSelectedFeatures={setSelectedFeatures}
              onCancel={() => setSetupModelModal(0)}
              onFinish={handleFinishSetupModelModal}
            />
          </Modal>
        )}
      </div>


      {/* Tabs */}
      <div>
        <div className="flex space-x-4 mb-6 border-b border-gray-300">
          {["Baseline", "Tuning", "Prediction"].map((tab) => {
            const isDisabled = tab === "Prediction" && (!isFinalized || !selectedTarget);

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
            comparisonResults={comparisonResults} 
            sessionId={sessionId} bestModel={bestModel}
            modelPerformanceAnalysis={modelPerformanceAnalysis}
            loadingModelPerformanceAnalysis={loadingModelPerformanceAnalysis}
            handleFetchModelPerformanceAnalysis={handleFetchModelPerformanceAnalysis}
          />
        }
        {activeTab === "Tuning" && 
          <TuningTab 
            sessionId={sessionId} 
            bestModelId={bestModel.index} 
            comparisonResults={comparisonResults} 
            setIsFinalized={setIsFinalized} 
          />
        }
        {activeTab === "Prediction" && 
          <PredictionTab 
            datasetId={datasetId}
          />
        }
      </div>
      {showToastAnalyzing && analyzingStatus === "completed" && (        
        <Toast type="success" message="Model training completed successfully!" />
      )}
      {showToastAnalyzing && analyzingStatus === "failed" && (
        <Toast type="error" message="An error occurred during model training." />
      )}
      {/* Analyzing Overlay */}
        {analyzing && (
          <div className="fixed inset-0 bg-white/70 bg-opacity-70 flex flex-col items-center justify-center z-50">
            <Loader2 className="animate-spin text-green-600" size={48} />
            <p className="mt-4 text-gray-700 font-semibold">Analyzing and training models...</p>
          </div>
        )}
    </div>
  );
}
