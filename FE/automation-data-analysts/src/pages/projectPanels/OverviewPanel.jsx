import { useEffect, useState } from "react";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { FaTableCells } from "react-icons/fa6";
import { FaSearch, FaRocket, FaSync, FaExclamationTriangle, FaArrowRight } from "react-icons/fa";
import Modal from "../../components/ui/Modal";
import mockData from "../../components/mock/sampleData.json"; 
import DataTable from "../../components/DataTable";
import Button from "../../components/ui/Button";
import Card from "../../components/ui/Card";
import Toast from "../../components/ui/Toast";
import NextModal from "./NextModal";

export default function OverviewPanel({ setIsTargetFeatureSelected }) {
  const [showCleanModal, setShowCleanModal] = useState(false);
  const [cleanOptions, setCleanOptions] = useState({
    remove_duplicates: true,
    handle_missing_values: true,
    smooth_noisy_data: true,
    handle_outliers: true,
    reduce_cardinality: true,
    encode_categorical_values: true,
    feature_scaling: true,
  });

  const [previewIssues, setPreviewIssues] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [showPreviewIssues, setShowPreviewIssues] = useState(false);

  const [cleanStatus, setCleanStatus] = useState(null);
  const [showToast, setShowToast] = useState(false);
  const [data, setData] = useState([]);
  const [csvFileName, setCsvFileName] = useState('data.csv');
  const [numRows, setNumRows] = useState(0);
  const [numColumns, setNumColumns] = useState(0);

  const [showNextModal, setShowNextModal] = useState(false);
  const [selectedTarget, setSelectedTarget] = useState("");
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzingStatus, setAnalyzingStatus] = useState(null);

  useEffect(() => {
    setData(mockData);
    setCsvFileName("file_name.csv");
    setNumRows(mockData.length);
    setNumColumns(mockData[0] ? Object.keys(mockData[0]).length : 0);

    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "auto";
    };
  }, []);

  useEffect(() => {
    if (cleanStatus === "done" || cleanStatus === "error") {
      setShowToast(true);
      setTimeout(() => {
        setShowToast(false);
      }, 3000);
    }
    if (analyzingStatus === "done" || analyzingStatus === "error") {
      setShowToast(true);
      setTimeout(() => {
        setShowToast(false);
      }, 3000);
    }
  }, [cleanStatus, analyzingStatus]);

  const handlePreviewIssues = () => {
    if (showPreviewIssues) {
      setShowPreviewIssues(false);
      return;
    }
    setPreviewLoading(true);
    setShowPreviewIssues(true);
    setTimeout(() => {
      setPreviewIssues({
        missing: { age: 5, income: 2 },
        outliers: { income: 3 },
        duplicates: 1
      });
      setPreviewLoading(false);
    }, 800);
  };

  const handleCleanData = () => {
    setShowCleanModal(false);
    setCleanStatus("pending");

    setTimeout(() => {
      setCleanStatus("running");

      setTimeout(() => {
        setCleanStatus("done");
      }, 2000);
    }, 1000);
  };

  const handleFinishNextModal = () => {
    localStorage.setItem("selectedTarget", selectedTarget);
    localStorage.setItem("selectedFeatures", JSON.stringify(selectedFeatures));


    setShowNextModal(false);
    setAnalyzing(true);

    setTimeout(() => {
      setAnalyzing(false);
      console.log("Finished analyzing and training models!");
      setIsTargetFeatureSelected(true); 
      setAnalyzingStatus("done");
    }, 3000);
  };

  const columns = data.length > 0 ? Object.keys(data[0]) : [];

  return (
    <div className="space-y-6 relative pb-32">
      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center">
          <h2 className="text-xl font-bold">Data Preparation</h2>
          <div className="bg-[#FFFDF3] shadow-sm border border-[#E4F3E9] rounded p-2 ms-5">
            <div className="flex items-center gap-2">
              <FaTableCells className="text-green-600 text-3xl" />
              <div className="text-sm">
                <div className="font-bold text-gray-800">{csvFileName}</div>
                <div className="text-gray-600">{numRows} rows, {numColumns} columns</div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex gap-4">
          <Button onClick={handlePreviewIssues} variant="primary">
            <div className="flex items-center gap-2">
              <FaSearch />
              Preview Detected Issues
            </div>
          </Button>
          <Button onClick={() => setShowCleanModal(true)} variant="outline">
            <div className="flex items-center gap-2">
              <FaSync />
              Clean Data
            </div>
          </Button>
        </div>
      </div>

      {/* Cleaning Status */}
      {cleanStatus && (
        <div className="flex items-center gap-2 text-sm text-gray-700 mb-3">
          {cleanStatus === "done" && <CheckCircle size={16} className="text-green-600" />}
          {cleanStatus === "error" && <AlertCircle size={16} className="text-red-600" />}
          {cleanStatus === "running" && <Loader2 size={16} className="animate-spin text-green-500" />}
          {cleanStatus === "pending" && <Loader2 size={16} className="animate-pulse text-yellow-500" />}
          <span className="capitalize">Cleaning status: {cleanStatus}</span>
        </div>
      )}

      {/* Preview Issues */}
      {previewLoading && showPreviewIssues ? (
        <p className="text-gray-500">Detecting issues in your dataset...</p>
      ) : showPreviewIssues && previewIssues ? (
        <Card className="relative bg-white border border-yellow-400 rounded-md shadow-sm p-6 mb-6">
          <button
            onClick={() => setShowPreviewIssues(false)}
            className="absolute top-3 right-3 text-gray-400 hover:text-gray-700 transition"
            aria-label="Close preview"
          >
            &times;
          </button>
          <div className="flex items-center mb-4">
            <FaExclamationTriangle className="text-yellow-500 text-xl mr-2" />
            <h3 className="text-lg font-semibold text-yellow-700">Data Quality Issues Detected</h3>
          </div>
          <ul className="text-sm text-gray-700 space-y-2 pl-6 list-disc">
            <li>
              <span className="font-medium text-gray-800">Missing values:</span>{" "}
              {Object.entries(previewIssues.missing)
                .map(([col, count]) => `${col}: ${count}`)
                .join(", ")}
            </li>
            <li>
              <span className="font-medium text-gray-800">Outliers:</span>{" "}
              {Object.entries(previewIssues.outliers)
                .map(([col, count]) => `${col}: ${count}`)
                .join(", ")}
            </li>
            <li>
              <span className="font-medium text-gray-800">Duplicates:</span> {previewIssues.duplicates}
            </li>
          </ul>
        </Card>
      ) : null}

      {/* Table */}
      <DataTable data={data} />

      {/* Clean Modal */}
      {showCleanModal && (
        <Modal onClose={() => setShowCleanModal(false)} title="Cleaning Options">
          <div className="space-y-5 text-sm text-gray-800">
            <div className="space-y-3">
              {Object.keys(cleanOptions).map((key) => (
                <label key={key} className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    className="accent-green-600 h-4 w-4"
                    checked={cleanOptions[key]}
                    onChange={(e) =>
                      setCleanOptions({ ...cleanOptions, [key]: e.target.checked })
                    }
                  />
                  <span>{key.replace(/_/g, " ")}</span>
                </label>
              ))}
            </div>

            <div className="pt-4 text-right">
              <Button onClick={handleCleanData} variant="primary">
                <div className="flex items-center gap-2">
                  <FaRocket />
                  Start Cleaning
                </div>
              </Button>
            </div>
          </div>
        </Modal>
      )}

      {/* Toast */}
      {showToast && cleanStatus === "done" && (
        <Toast type="success" message="Data cleaning completed successfully!" />
      )}
      {showToast && cleanStatus === "error" && (
        <Toast type="error" message="An error occurred during cleaning." />
      )}
      {showToast && analyzingStatus === "done" && (        
        <Toast type="success" message="Model training completed successfully!" />
      )}
      {showToast && analyzingStatus === "error" && (
        <Toast type="error" message="An error occurred during model training." />
      )}

      {/* Next Button */}
      <div className="fixed bottom-6 right-6">
        <Button onClick={() => setShowNextModal(true)} variant="primary">
          <div className="flex items-center gap-2">
            <FaArrowRight />
            Next
          </div>
        </Button>
      </div>

      {/* Next Modal */}
      {showNextModal && (
        <Modal title="Select Target and Features" onClose={() => setShowNextModal(false)}>
          <NextModal
            availableColumns={columns}
            selectedTarget={selectedTarget}
            setSelectedTarget={setSelectedTarget}
            selectedFeatures={selectedFeatures}
            setSelectedFeatures={setSelectedFeatures}
            onCancel={() => setShowNextModal(false)}
            onFinish={handleFinishNextModal}
          />
        </Modal>
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
