import { useState } from "react";
import { FaMagic, FaDownload, FaChartLine } from "react-icons/fa"; // Import thêm FaChartLine
import { Button, Card } from "../../components/ui";
import DataTable from "../../components/DataTable";
import UploadDropzone from "../../components/UploadDropzone"; 

export default function PredictionTab() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [data, setData] = useState([]);
  const [predictedData, setPredictedData] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileAccepted = (file) => {
    setUploadedFile(file);
    // Giả lập đọc file CSV
    setTimeout(() => {
      setData([
        { age: 25, income: 40000, gender: "F" },
        { age: 30, income: 55000, gender: "M" },
        { age: 22, income: 30000, gender: "F" }
      ]);
    }, 500);
  };

  const handlePredict = () => {
    if (!data.length) {
      alert("Please upload a CSV file first!");
      return;
    }

    setLoading(true);

    setTimeout(() => {
      const predicted = data.map((row) => ({
        ...row,
        prediction_label: Math.random() > 0.5 ? 1 : 0,
        prediction_score: Math.random().toFixed(2),
      }));

      setPredictedData(predicted);
      setLoading(false);
    }, 1500);
  };

  const handleDownloadPredicted = () => {
    const headers = Object.keys(predictedData[0]).join(",");
    const rows = predictedData.map(row => Object.values(row).join(","));
    const csvContent = [headers, ...rows].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "predicted_data.csv";
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
            onClick={() => { setUploadedFile(null); setData([]); setPredictedData([]); }}
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
