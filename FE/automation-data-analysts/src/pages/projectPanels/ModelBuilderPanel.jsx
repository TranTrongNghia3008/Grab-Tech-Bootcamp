import { useState, useEffect } from "react";
import { FaBrain, FaMagic, FaDownload } from "react-icons/fa";
import { Card, Button } from "../../components/ui";
import DataTable from "../../components/DataTable";
import SelectTargetFeaturesModel from "./SelectTargetFeatures";
import AnalyzeModel from "./AnalyzeModel";
import { Loader2 } from "lucide-react";
import PredictOnNewData from "./PredictOnNewData";

export default function ModelBuilderPanel() {
  const [target, setTarget] = useState("");
  const [features, setFeatures] = useState([]);
  const [modelType, setModelType] = useState("");
  const [customGrid, setCustomGrid] = useState(""); // NEW
  const [jobStatus, setJobStatus] = useState(null);
  const [bestParams, setBestParams] = useState({});
  const [cvMetrics, setCvMetrics] = useState([]);
  const [loading, setLoading] = useState(false);

  const [modelOptions, setModelOptions] = useState([]);
  const [bestModelId, setBestModelId] = useState(null);

  // For Predict on New Data
  const [selectedDataset, setSelectedDataset] = useState("");
  const [uploading, setUploading] = useState(false);
  const [predictedData, setPredictedData] = useState([]);

  const availableColumns = ["age", "income", "gender", "education", "experience"];

  const dataset = [
    { id: 1, name: 'Dataset A', createdAt: '2021-10-01', updatedAt: '2023-01-15' },
    { id: 2, name: 'Dataset B', createdAt: '2022-02-20', updatedAt: '2023-04-10' },
    { id: 3, name: 'Dataset C', createdAt: '2020-11-05', updatedAt: '2021-12-15' },
    { id: 4, name: 'Dataset D', createdAt: '2023-03-12', updatedAt: '2023-05-01' },
  ];

  useEffect(() => {
    const storedComparisonResults = localStorage.getItem("comparisonResults");
    if (storedComparisonResults) {
      const parsedResults = JSON.parse(storedComparisonResults);
      const options = parsedResults.map(item => ({
        id: item.modelId,
        name: item.modelName,
      }));
      setModelOptions(options);
    }

    const storedBestModelId = localStorage.getItem("best_model_id");
    if (storedBestModelId) {
      setBestModelId(storedBestModelId);
    }
  }, []);

  const handleTrainModel = () => {
    if (!target || features.length === 0 || !modelType) {
      alert("Please fill all fields!");
      return;
    }

    if (customGrid) {
      try {
        JSON.parse(customGrid);
      } catch (err) {
        alert(err);
        return;
      }
    }

    setLoading(true);
    setJobStatus("pending");

    setTimeout(() => {
      setJobStatus("running");

      setTimeout(() => {
        setBestParams({
          C: 3.243,
          class_weight: "balanced",
          solver: "lbfgs",
          max_iter: 1000,
          random_state: 123
        });

        setCvMetrics([
          { Fold: 0, Accuracy: 0.8320, AUC: 0.8444, Recall: 0.7292, Precision: 0.8140, F1: 0.7692, Kappa: 0.6378, MCC: 0.6402 },
          { Fold: 1, Accuracy: 0.8640, AUC: 0.9221, Recall: 0.8542, Precision: 0.8039, F1: 0.8283, Kappa: 0.7159, MCC: 0.7168 },
          { Fold: "Mean", Accuracy: 0.8234, AUC: 0.8772, Recall: 0.7406, Precision: 0.7862, F1: 0.7618, Kappa: 0.6219, MCC: 0.6235 }
        ]);
        setJobStatus("done");
        setLoading(false);
      }, 2000);
    }, 1000);
  };

  const handleSelectDataset = (datasetId) => {
    setSelectedDataset(datasetId);

    if (!datasetId) {
      setPredictedData([]);
      return;
    }

    setUploading(true);

    setTimeout(() => {
      setPredictedData([
        { age: 22, income: 29000, gender: "F", prediction_label: 1, prediction_score: 0.78 },
        { age: 30, income: 41000, gender: "M", prediction_label: 0, prediction_score: 0.45 },
      ]);
      setUploading(false);
    }, 1500);
  };

  const handleDownloadModel = () => {
    const blob = new Blob(["This is your trained model (.pkl)"], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "trained_model.pkl";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-8">
      <h2 className="text-xl font-bold">Custom Model Builder</h2>

      <div className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900">
        Select target and features, choose a model, define (optional) custom hyperparameter grid, and train a custom machine learning model.
      </div>

      {/* Select Target, Features, Model */}
      <SelectTargetFeaturesModel
        availableColumns={availableColumns}
        target={target}
        setTarget={setTarget}
        features={features}
        setFeatures={setFeatures}
        modelType={modelType}
        setModelType={setModelType}
        modelOptions={modelOptions}
        bestModelId={bestModelId}
        handleTrain={handleTrainModel}
        loading={loading}
        jobStatus={jobStatus}
        trainLabel="Train Model"
        showModelSelection={true}
        customGrid={customGrid}
        setCustomGrid={setCustomGrid}
      />

      {/* Best Params */}
      {Object.keys(bestParams).length > 0 && (
        <Card className="space-y-4">
          <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
            <FaBrain /> Best Parameters
          </h3>
          <DataTable data={Object.entries(bestParams).map(([key, val]) => ({ Parameter: key, Value: String(val) }))} />
        </Card>
      )}

      {/* Cross-validation Metrics */}
      {cvMetrics.length > 0 && (
        <Card className="space-y-4">
          <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
            <FaBrain /> Cross-Validation Metrics
          </h3>
          <DataTable data={cvMetrics} />
        </Card>
      )}

      {/* Analyze Trained Model */}
      {jobStatus === "done" && (
        <AnalyzeModel availableModels={[{ modelId: "trained_model", modelName: "Your Trained Model" }]} />
      )}

      {/* Predict on New Data */}
      {jobStatus === "done" && (
        <PredictOnNewData
          datasetOptions={dataset}
          selectedDataset={selectedDataset}
          handleSelectDataset={handleSelectDataset}
          uploading={uploading}
          predictedData={predictedData}
        />
      )}

      {/* Download Trained Model */}
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
