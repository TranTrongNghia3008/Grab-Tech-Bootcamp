import { useState } from "react";
import { FaMagic, FaDownload, FaChartLine } from "react-icons/fa"; // Import thêm FaChartLine
import { Button, Card } from "../../components/ui";
import DataTable from "../../components/DataTable";
import UploadDropzone from "../../components/UploadDropzone"; 
import { predictModel } from "../../components/services/modelingServices";

export default function PredictionTab({ finalizedModelId }) {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [predictedData, setPredictedData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [predictResults, setPredictResults] = useState(null); 
  const handleFileAccepted = (file) => {
    setUploadedFile(file);
    setPredictedData([]); 
  };

  const handlePredict = async () => {
    if (!uploadedFile) {
      alert("Please upload a CSV file first!");
      return;
    }
  
    setLoading(true);
  
    try {
      const predictResults = await predictModel(finalizedModelId, uploadedFile);
      setPredictResults(predictResults); 
      console.log("Predict results:", predictResults);
  
      // Chuyển từ columns + data => array of objects
      const { columns, data: rows } = predictResults.preview_predictions;
      const formattedData = rows.map((row) => {
        const rowObj = {};
        columns.forEach((col, i) => {
          rowObj[col] = row[i];
        });
        return rowObj;
      });
  
      setPredictedData(formattedData);
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

      {!uploadedFile ? (
        <UploadDropzone onFileAccepted={handleFileAccepted} />
      ) : (
        <div className="flex items-center justify-between px-4 py-3 bg-green-100 border border-green-200 rounded-md text-sm text-green-800">
          <span>📄 File uploaded: {uploadedFile.name}</span>
          <button
            onClick={() => { setUploadedFile(null); setPredictedData([]); setPredictResults(null); }}
            className="text-xs text-red-500 hover:underline ml-4"
          >
            Remove
          </button>
        </div>
      )}

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
              <FaDownload className="mr-2" /> Download Predicted CSV
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
