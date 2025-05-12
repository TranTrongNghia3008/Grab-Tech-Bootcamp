import { useState, useEffect } from "react";
import { FaMagic, FaDownload, FaChartLine } from "react-icons/fa"; 
import { Button, Card } from "../../components/ui";
import DataTable from "../../components/DataTable";
import UploadDropzone from "../../components/UploadDropzone"; 
import { getListFinalizedModels, predictModel } from "../../components/services/modelingServices";
import { useAppContext } from "../../contexts/AppContext";

export default function PredictionTab({ datasetId }) {
  const { state, updateState } = useAppContext();
  const { predictedResults } = state;
  const [uploadedFile, setUploadedFile] = useState(null);
  const [predictedData, setPredictedData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [predictResults, setPredictResults] = useState(null); 
  const [finalizedModels, setFinalizedModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState(null);

  useEffect(() => {
      if (predictedResults) {
        console.log("Predicted Results from state:", predictedResults);
        processPredictResults(predictedResults)
      }
    }, [predictedResults]);

  useEffect(() => {
    async function fetchModels() {
      try {
        const models = await getListFinalizedModels(datasetId); // datasetId bạn truyền vào props
        setFinalizedModels(models);
        if (models.length > 0) setSelectedModelId(models[0].id);
      } catch (err) {
        console.error("Failed to load finalized models", err);
      }
    }
    fetchModels();
  }, [datasetId]);


  const processPredictResults = (predictResults) => {
    setPredictResults(predictResults); 
    const { columns, data: rows } = predictResults.preview_predictions;
    const formattedData = rows.map((row) => {
      const rowObj = {};
      columns.forEach((col, i) => {
        rowObj[col] = row[i];
      });
      return rowObj;
    });

    setPredictedData(formattedData);
  }

  const handleFileAccepted = (file) => {
    setUploadedFile(file);
    setPredictedData([]); 
  };

  const handlePredict = async () => {
    if (!uploadedFile) {
      alert("Please upload a CSV file first!");
      return;
    }
    if (!selectedModelId) {
      alert("Please select a model first!");
      return;
    }

    setLoading(true);

    try {
      const predictResults = await predictModel(selectedModelId, uploadedFile);
      updateState({ predictedResults: predictResults });
      setUploadedFile(false);
    } catch (error) {
      console.error("Prediction failed:", error);
      alert("Prediction failed. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  

  const handleDownloadPredicted = () => {
    if (!predictResults?.full_csv_base64) {
      alert("No prediction result available to download.");
      return;
    }
  
    const byteCharacters = atob(predictResults.full_csv_base64);
    const byteNumbers = new Array(byteCharacters.length).fill().map((_, i) => byteCharacters.charCodeAt(i));
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: "text/csv;charset=utf-8;" });
  
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "predicted_full.csv";
    a.click();
    URL.revokeObjectURL(url);
  };
  
  const handleViewDriftReport = () => {
    // Giả lập mở file drift_report.html, bạn chỉnh path thật đúng sau này
    const reportUrl = "/drift_reports/prediction_drift_report.html"; 
    window.open(reportUrl, "_blank");
  };

  return (
    <div className="space-y-8">
      <h2 className="text-xl font-bold">Predict New Data</h2>

      <div className="flex flex-col sm:flex-row sm:items-end gap-4 w-full">
        {/* Select model */}
        <div className="sm:w-1/3 w-full">
          <label htmlFor="model-select" className="block text-sm font-medium text-gray-700 mb-1">
            Select a finalized model
          </label>
          <select
            id="model-select"
            className="block w-full border border-green-300 focus:ring-green-500 focus:border-green-500 rounded-md shadow-sm px-3 py-2 text-sm"
            value={selectedModelId || ""}
            onChange={(e) => setSelectedModelId(parseInt(e.target.value))}
          >
            {finalizedModels.map((model) => (
              <option key={model.id} value={model.id}>
                {model.model_name}
              </option>
            ))}
          </select>
        </div>

        {/* Upload file */}
        <div className="sm:w-2/3 w-full">
          {!uploadedFile ? (
            <UploadDropzone onFileAccepted={handleFileAccepted} />
          ) : (
            <div className="flex items-center justify-between px-4 py-3 bg-green-100 border border-green-200 rounded-md text-sm text-green-800">
              <span>📄 File uploaded: {uploadedFile.name}</span>
              <button
                onClick={() => {
                  setUploadedFile(null);
                  setPredictedData([]);
                  setPredictResults(null);
                }}
                className="text-xs text-red-500 hover:underline ml-4"
              >
                Remove
              </button>
            </div>
          )}
        </div>
      </div>


      {/* Predict Button */}
      {uploadedFile && (
        <div className="text-right">
          <Button onClick={handlePredict} disabled={loading}>
            {loading ? (
              <div className="flex items-center gap-2">
                <FaMagic className="animate-spin" /> Predicting...
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <FaMagic /> Predict
              </div>
            )}
          </Button>
        </div>
      )}

      {/* Prediction Results */}
      {predictedData.length > 0 && (
        <Card className="space-y-6">
          <h3 className="text-gray-800 text-xl flex items-center gap-2">
            Preview Prediction Results
          </h3>
          <DataTable data={predictedData} />

          <div className="flex justify-end gap-4">
            <Button onClick={handleDownloadPredicted}>
              <FaDownload className="mr-2" /> Download Predicted Data
            </Button>

            <Button variant="outline" onClick={handleViewDriftReport}>
              <FaChartLine className="mr-2" /> View Prediction Drift Report
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
}
