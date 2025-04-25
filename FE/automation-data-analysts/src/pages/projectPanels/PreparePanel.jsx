import { useEffect, useState } from "react";
import Modal from "../../components/ui/Modal";
import mockData from "../../components/mock/sampleData.json"; 
import DataTable from "../../components/DataTable";


export default function PreparePanel() {
  // const datasetId = 123; // This should come from props/context later
  // const cleaningId = 456; // Simulated cleaning job ID

  const [showCleanModal, setShowCleanModal] = useState(false);
  const [cleanOptions, setCleanOptions] = useState({
    missing: true,
    outliers: true,
    duplicates: true
  });

  const [previewIssues, setPreviewIssues] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [showPreviewIssues, setShowPreviewIssues] = useState(false);

  const [toast, setToast] = useState(null);


  const [data, setData] = useState([]);

  useEffect(() => {
    setData(mockData);
  }, []);

  const handlePreviewIssues = async () => {
    // Toggle display
    if (showPreviewIssues) {
      setShowPreviewIssues(false);
      return;
    }
  
    setPreviewLoading(true);
    setShowPreviewIssues(true);
  
    // Simulate API call
    setTimeout(() => {
      setPreviewIssues({
        missing: { age: 5, income: 2 },
        outliers: { income: 3 },
        duplicates: 1
      });
      setPreviewLoading(false);
    }, 800);
  };

  const handleCheckStatus = async () => {
    setTimeout(() => {
      const simulatedStatus = "running"; // change to "pending", "done", "error"
  
      // Show toast
      setToast({
        message: `Cleaning status: ${simulatedStatus}`,
        type: simulatedStatus
      });
  
      // Auto-hide after 3 seconds
      setTimeout(() => setToast(null), 3000);
    }, 800);
  };
  

  const handleCleanData = () => {
    console.log("Cleaning with options:", cleanOptions);
    setShowCleanModal(false);
    alert("Cleaning job has been submitted!");
  };

  return (
    
    <div>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">Data Preparation</h2>
        <div className="flex gap-4">
          <button
            onClick={handlePreviewIssues}
            className="bg-yellow-500 text-white px-4 py-2 rounded hover:bg-yellow-600"
          >
            🔍 Preview Detected Issues
          </button>
          <button
            onClick={handleCheckStatus}
            className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600"
          >
            📊 Check Cleaning Status
          </button>
          <button
            onClick={() => setShowCleanModal(true)}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            🧹 Clean Data
          </button>

        </div>
      </div>

      {/* Detected Issues */}
      {previewLoading && showPreviewIssues ? (
        <p className="text-gray-500">Detecting issues in your dataset...</p>
      ) : showPreviewIssues && previewIssues ? (
        <div className="relative bg-white border border-yellow-400 rounded-md shadow-sm p-6 mb-6">
          {/* Close button */}
          <button
            onClick={() => setShowPreviewIssues(false)}
            className="absolute top-3 right-3 text-gray-400 hover:text-gray-700 transition"
            aria-label="Close preview"
          >
            &times;
          </button>

          {/* Header */}
          <div className="flex items-center mb-4">
            <span className="text-yellow-500 text-xl mr-2">⚠️</span>
            <h3 className="text-lg font-semibold text-yellow-700">Data Quality Issues Detected</h3>
          </div>

          {/* Issues List */}
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
        </div>
      ) : null}



      {/* Data Table */}
      <DataTable data={data} />

      {/* <div className="overflow-auto border rounded">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-200 text-gray-700">
            <tr>
              {data[0] &&
                Object.keys(data[0]).map((col) => (
                  <th key={col} className="px-4 py-2 text-left">
                    {col}
                  </th>
                ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => (
              <tr key={idx} className="even:bg-gray-50">
                {Object.values(row).map((val, i) => (
                  <td key={i} className="px-4 py-2">{val}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div> */}

      {/* Cleaning Options Modal */}
      {showCleanModal && (
        <Modal onClose={() => setShowCleanModal(false)} title="Cleaning Options">
          <div className="space-y-5 text-sm text-gray-800">
            <div className="space-y-3">
              <label className="flex items-center gap-3">
                <input
                  type="checkbox"
                  className="accent-blue-500 h-4 w-4"
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
                  className="accent-blue-500 h-4 w-4"
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
                  className="accent-blue-500 h-4 w-4"
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
              <button
                onClick={handleCleanData}
                className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-md shadow-sm transition"
              >
                🚀 Start Cleaning
              </button>
            </div>
          </div>
        </Modal>

      )}

      {toast && (
        <div className={`fixed bottom-4 right-4 z-50 px-4 py-3 rounded shadow-lg text-sm
          ${
            toast.type === "done"
              ? "bg-green-100 text-green-800 border border-green-300"
              : toast.type === "running"
              ? "bg-blue-100 text-blue-800 border border-blue-300"
              : toast.type === "error"
              ? "bg-red-100 text-red-800 border border-red-300"
              : "bg-gray-100 text-gray-800 border border-gray-300"
          }
        `}>
          <span className="font-medium capitalize">{toast.message}</span>
        </div>
      )}

    </div>
  );
}
