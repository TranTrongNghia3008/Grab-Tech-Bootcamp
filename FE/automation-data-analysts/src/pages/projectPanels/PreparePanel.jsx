import { useEffect, useState } from "react";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { FaTableCells } from "react-icons/fa6";

import Modal from "../../components/ui/Modal";
import mockData from "../../components/mock/sampleData.json"; 
import DataTable from "../../components/DataTable";
import Button from "../../components/ui/Button"; // Đưa Button vào
import Card from "../../components/ui/Card"; // Đưa Card vào
import Toast from "../../components/ui/Toast"; // Đưa Toast vào

export default function PreparePanel() {
  const [showCleanModal, setShowCleanModal] = useState(false);
  const [cleanOptions, setCleanOptions] = useState({
    missing: true,
    outliers: true,
    duplicates: true
  });

  const [previewIssues, setPreviewIssues] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [showPreviewIssues, setShowPreviewIssues] = useState(false);

  const [cleanStatus, setCleanStatus] = useState(null); // ← trạng thái chính
  const [showToast, setShowToast] = useState(false);
  const [data, setData] = useState([]);
  const [csvFileName, setCsvFileName] = useState('data.csv');
  const [numRows, setNumRows] = useState(0);
  const [numColumns, setNumColumns] = useState(0);

  useEffect(() => {
    setData(mockData);
    setCsvFileName("file_name.csv"); // Tên file CSV
    setNumRows(mockData.length);
    setNumColumns(mockData[0] ? Object.keys(mockData[0]).length : 0);
    // Vô hiệu hóa cuộn toàn trang khi vào trang này
    document.body.style.overflow = "hidden";

    // Reset cuộn khi component unmount
    return () => {
      document.body.style.overflow = "auto";
    };
  }, []);

  useEffect(() => {
    // Khi trạng thái cleaning thành công hoặc thất bại, hiển thị toast trong 3 giây
    if (cleanStatus === "done" || cleanStatus === "error") {
      setShowToast(true);

      // Ẩn Toast sau 3 giây (3000ms)
      setTimeout(() => {
        setShowToast(false);
      }, 3000);
    }
  }, [cleanStatus]);

  const handlePreviewIssues = async () => {
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
    console.log("Cleaning with options:", cleanOptions);
    setShowCleanModal(false);

    setCleanStatus("pending");

    setTimeout(() => {
      setCleanStatus("running");

      setTimeout(() => {
        setCleanStatus("done");
      }, 2000);
    }, 1000);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center">
          <h2 className="text-xl font-bold">Data Preparation</h2>
          <div className="bg-[#FFFDF3] shadow-sm border border-[#E4F3E9] rounded p-2 ms-5">
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2">
                <FaTableCells className="text-green-600 text-3xl" />
              </div>
              <div className="text-sm">
                <div className="font-bold text-gray-800">{csvFileName}</div> {/* Tên file CSV */}
                <div className="text-gray-600">{numRows} rows, {numColumns} columns</div> {/* Số hàng và số cột */}
              </div>
            </div>
          </div>
        </div>

        <div className="flex gap-4">
          <Button onClick={handlePreviewIssues} variant="primary">
            🔍 Preview Detected Issues
          </Button>
          <Button onClick={() => setShowCleanModal(true)} variant="outline">
            🧹 Clean Data
          </Button>
        </div>
      </div>

      {/* Dòng trạng thái Cleaning */}
      {cleanStatus && (
        <div className="flex items-center gap-2 text-sm text-gray-700 mb-3">
          {cleanStatus === "done" && <CheckCircle size={16} className="text-green-600" />}
          {cleanStatus === "error" && <AlertCircle size={16} className="text-red-600" />}
          {cleanStatus === "running" && <Loader2 size={16} className="animate-spin text-green-500" />}
          {cleanStatus === "pending" && <Loader2 size={16} className="animate-pulse text-yellow-500" />}
          <span className="capitalize">Cleaning status: {cleanStatus}</span>
        </div>
      )}

      {/* Detected Issues */}
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
            <span className="text-yellow-500 text-xl mr-2">⚠️</span>
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

      {/* Data Table */}
      <DataTable data={data} />

      {/* Cleaning Options Modal */}
      {showCleanModal && (
        <Modal onClose={() => setShowCleanModal(false)} title="Cleaning Options">
          <div className="space-y-5 text-sm text-gray-800">
            <div className="space-y-3">
              <label className="flex items-center gap-3">
                <input
                  type="checkbox"
                  className="accent-green-600 h-4 w-4"
                  checked={cleanOptions.missing}
                  onChange={(e) =>
                    setCleanOptions({ ...cleanOptions, missing: e.target.checked })
                  }
                />
                <span className="flex items-center gap-2">
                  <span className="text-blue-600">🧩</span>
                  Handle missing values
                </span>
              </label>

              <label className="flex items-center gap-3">
                <input
                  type="checkbox"
                  className="accent-green-600 h-4 w-4"
                  checked={cleanOptions.outliers}
                  onChange={(e) =>
                    setCleanOptions({ ...cleanOptions, outliers: e.target.checked })
                  }
                />
                <span className="flex items-center gap-2">
                  <span className="text-orange-500">📈</span>
                  Remove outliers
                </span>
              </label>

              <label className="flex items-center gap-3">
                <input
                  type="checkbox"
                  className="accent-green-600 h-4 w-4"
                  checked={cleanOptions.duplicates}
                  onChange={(e) =>
                    setCleanOptions({ ...cleanOptions, duplicates: e.target.checked })
                  }
                />
                <span className="flex items-center gap-2">
                  <span className="text-red-500">🔁</span>
                  Remove duplicates
                </span>
              </label>
            </div>

            <div className="pt-4 text-right">
              <Button onClick={handleCleanData} variant="primary">
                🚀 Start Cleaning
              </Button>
            </div>
          </div>
        </Modal>
      )}

      {/* Toast for Cleaning Complete */}
      {showToast && cleanStatus === "done" && (
        <Toast type="success" message="Data cleaning completed successfully!" />
      )}
      {showToast && cleanStatus === "error" && (
        <Toast type="error" message="An error occurred during cleaning." />
      )}

      <div className="absolute bottom-0 left-0 right-0 h-20 bg-gradient-to-t from-white to-transparent"></div>
    </div>
  );
}
