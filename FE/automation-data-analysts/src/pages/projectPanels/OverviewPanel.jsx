import { useEffect, useState,  } from "react";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { FaTableCells } from "react-icons/fa6";
import { FaSearch, FaRocket, FaSync, FaExclamationTriangle, FaArrowRight } from "react-icons/fa";
import { Tooltip } from 'react-tooltip';
import Modal from "../../components/ui/Modal";
import mockData from "../../components/mock/sampleData.json"; 
import DataTable from "../../components/DataTable";
import Button from "../../components/ui/Button";
import Card from "../../components/ui/Card";
import Toast from "../../components/ui/Toast";
import SetupModelModal from "./SetupModelModal";
import { useAppContext } from "../../contexts/AppContext";
import { getPreviewIssues, cleaningDataset, getCleaningStatus } from "../../components/services/cleaningServices";
import { autoMLSession } from "../../components/services/modelingServices";



export default function OverviewPanel({ setIsTargetFeatureSelected }) {
  // const { state } = useAppContext(); 
  const { state, updateState } = useAppContext();
  const { datasetId } = state;
  // const datasetId = 13; // Thay thế bằng datasetId thực tế từ context hoặc props
  // const [showCleanModal, setShowCleanModal] = useState(false);
  const [cleanOptions, setCleanOptions] = useState({
    "remove_duplicates": true,
    "handle_missing_values": true,
    "handle_outliers": true,
    "smooth_noisy_data": false,
    "reduce_cardinality": false,
    "encode_categorical_values": false,
    "feature_scaling": false
  });
  const [isDefaultChecked, setIsDefaultChecked] = useState(true);

  const [previewIssues, setPreviewIssues] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [showPreviewIssues, setShowPreviewIssues] = useState(false);

  const [cleanStatus, setCleanStatus] = useState(null);
  const [showToast, setShowToast] = useState(false);
  const [data, setData] = useState([]);
  const [csvFileName, setCsvFileName] = useState('data.csv');
  const [numRows, setNumRows] = useState(0);
  const [numColumns, setNumColumns] = useState(0);

  const [selectedTarget, setSelectedTarget] = useState("");
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzingStatus, setAnalyzingStatus] = useState(null);
  const [stepModal, setStepModal] = useState(0); 
  const [note, setNote] = useState(<>Confirm that this is your dataset. Click <strong>Next</strong> to review and prepare it for training.</>)

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
    const fetchPreviewIssues = async () => {
      try {
          if (showPreviewIssues) {
            setShowPreviewIssues(false);
            return;
          }
          setPreviewLoading(true);
          setShowPreviewIssues(true);
          const results = await getPreviewIssues(datasetId);
          setPreviewLoading(false);
          setPreviewIssues(results);
      } catch (error) {
        console.error("Error API:", error);
      }
    }
    fetchPreviewIssues();
  }, [datasetId])

  useEffect(() => {
    if (cleanStatus === "completed" || cleanStatus === "failed") {
      setShowToast(true);
      setTimeout(() => {
        setShowToast(false);
      }, 3000);
    }
    if (analyzingStatus === "completed" || analyzingStatus === "failed") {
      setShowToast(true);
      setTimeout(() => {
        setShowToast(false);
      }, 3000);
    }
  }, [cleanStatus, analyzingStatus]);

  // const handlePreviewIssues = async () => {
  //   if (showPreviewIssues) {
  //     setShowPreviewIssues(false);
  //     return;
  //   }
  //   setPreviewLoading(true);
  //   setShowPreviewIssues(true);
  //   const results = await getPreviewIssues(datasetId);
  //   setPreviewLoading(false);
  //   setPreviewIssues(results);
  // };

  const handleCleanData = async () => {
    // setShowCleanModal(false);
    setCleanStatus("pending");
  
    try {
      const results = await cleaningDataset(datasetId, cleanOptions);
      setCleanStatus("running");
      const jobId = results.id;
  
      const intervalId = setInterval(async () => {
        try {
          const res = await getCleaningStatus(jobId);
          const status = res.status;

          if (status === "completed") {
            setPreviewIssues(null)
          }
  
          if (status === "completed" || status === "failed") {
            setCleanStatus(status);
            clearInterval(intervalId); // Dừng kiểm tra khi job kết thúc
          }
        } catch (error) {
          console.error("Failed to check cleaning status:", error);
          setCleanStatus("failed");
          clearInterval(intervalId);
        }
      }, 2000); 

      // const resultsPreviewIssue = await getPreviewIssues(datasetId);
      // setPreviewIssues(resultsPreviewIssue);
  
    } catch (error) {
      console.error("Cleaning failed:", error);
      setCleanStatus("failed");
    }
  };

  const handleFinishSetupModelModal = async () => {
    localStorage.setItem("selectedTarget", selectedTarget);
    localStorage.setItem("selectedFeatures", JSON.stringify(selectedFeatures));


    setStepModal(0);
    setAnalyzing(true);

    try { 
      const results = await autoMLSession(datasetId, selectedTarget, selectedFeatures);
      setAnalyzing(false);
      console.log("Finished analyzing and training models!");
      console.log("Results:", results);
      updateState({sessionId: results.session_id, comparisonResults: results.comparison_results});
      setIsTargetFeatureSelected(true); 
      setAnalyzingStatus("completed");
      setNote(<>Now explore the <strong>Data Insight</strong>, <strong>Modeling</strong>, and <strong>Export</strong> tabs.</>)
    } catch (error) {
      console.error("Error during analyzing and training models:", error);
      setAnalyzing(false);
      setAnalyzingStatus("failed");
    } 
  };

  // const columns = data.length > 0 ? Object.keys(data[0]) : [];
  const columns = ["Store ID","Employee Number" ,"Area" ,"Date" ,"Sales" ,"Marketing Spend" ,"Electronics Sales" ,"Home Sales" ,"Clothes Sales"]

  const cleanOptionDescriptions = {
    remove_duplicates: "Remove duplicate records from the dataset.",
    handle_missing_values: "Handle missing values by filling, removing, or replacing them.",
    handle_outliers: "Detect and process outliers in the dataset.",
    smooth_noisy_data: "Smooth out noisy data to improve analysis quality.",
    reduce_cardinality: "Reduce the number of unique values in categorical columns.",
    encode_categorical_values: "Convert categorical data into numerical format.",
    feature_scaling: "Normalize features so they share the same scale."
  };  
  
  return (
    <div className="space-y-6 relative pb-32">
      {/* Header */}
      <div className="flex items-center mb-4">
        <div className="flex items-center me-7">
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

        <div className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-xl text-green-900 shadow-sm">
          {note}
        </div>


        {/* <div className="flex gap-4">
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
        </div> */}
      </div>

      

      {/* Preview Issues */}
      

      {/* Table */}
      <DataTable data={data} />

      {/* Clean Modal */}
      {/* {showCleanModal && (
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
      )} */}

      {/* Toast */}
      {showToast && cleanStatus === "completed" && (
        <Toast type="success" message="Data cleaning completed successfully!" />
      )}
      {showToast && cleanStatus === "failed" && (
        <Toast type="error" message="An error occurred during cleaning." />
      )}
      {showToast && analyzingStatus === "completed" && (        
        <Toast type="success" message="Model training completed successfully!" />
      )}
      {showToast && analyzingStatus === "failed" && (
        <Toast type="error" message="An error occurred during model training." />
      )}

      {/* Next Button */}
      <div className="fixed bottom-6 right-6">
        <Button onClick={() => setStepModal(1)} variant="primary">
          <div className="flex items-center gap-2">
            <FaArrowRight />
            Next
          </div>
        </Button>
      </div>

      {stepModal === 1 && (
        <Modal title="Data Issues Overview" onClose={() => setStepModal(0)}>
          <div className="text-sm space-y-4 min-h-[400px]">
            <div className="flex justify-between items-center gap-2 mb-4">
              <div className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm w-full">
                Oops.. We found some issues in your uploaded dataset.
              </div>
              {/* <Button onClick={handlePreviewIssues} variant="primary">
                <div className="flex items-center gap-2">
                  <FaSearch />
                  
                </div>
              </Button> */}
            </div>
            {previewLoading && showPreviewIssues ? (
              <p className="text-gray-500">Detecting issues in your dataset...</p>
            ) : showPreviewIssues && previewIssues ? (
              <Card className="relative bg-white border border-yellow-400 rounded-md shadow-sm p-6 mb-6">
                {/* <button
                  onClick={() => setShowPreviewIssues(false)}
                  className="absolute top-3 right-3 text-gray-400 hover:text-gray-700 transition"
                  aria-label="Close preview"
                >
                  &times;
                </button> */}
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

            <div className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm">
              Click <strong>Next</strong> to clean data.
            </div>
          </div>
          <div className="text-right pt-4">
              <Button variant="primary" onClick={() => setStepModal(2)}>
                Next
              </Button>
            </div>
        </Modal>
      )}

      {stepModal === 2 && (
        <Modal title="Cleaning Options" onClose={() => setStepModal(0)}>
          <div className="flex flex-col space-y-5 text-sm text-gray-800 min-h-[400px]">
            <div className="space-y-3">
              <div className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm">
                Do you want to change the clean data options?
              </div>

              <div className="ms-7 space-y-4">
                {/* ✅ Checkbox Default */}
                <label className="flex items-center gap-3 font-semibold">
                  <input
                    type="checkbox"
                    className="accent-green-600 h-4 w-4"
                    checked={isDefaultChecked}
                    onChange={(e) => {
                      const checked = e.target.checked;
                      setIsDefaultChecked(checked);

                      if (checked) {
                        setCleanOptions((prev) => {
                          const updated = { ...prev };
                          // ✅ Tick mặc định 3 mục
                          updated.remove_duplicates = true;
                          updated.handle_missing_values = true;
                          updated.handle_outliers = true;

                          // ❌ Bỏ tick các mục khác
                          Object.keys(updated).forEach((key) => {
                            if (!["remove_duplicates", "handle_missing_values", "handle_outliers"].includes(key)) {
                              updated[key] = false;
                            }
                          });

                          return updated;
                        });
                      }
                    }}
                  />
                  Use default clean options
                </label>


                {/* ✅ Render từng option */}
                {Object.keys(cleanOptions).map((key) => {
                  const isDefault = ["remove_duplicates", "handle_missing_values", "handle_outliers"].includes(key);
                  const isDisabled = !isDefault && isDefaultChecked;
                  const tooltipId = `tooltip-${key}`;

                  return (
                    <>
                    <label
                      key={key}
                      className={`flex items-center gap-3 ${isDisabled ? "opacity-50" : ""}`}
                      data-tooltip-id={tooltipId}
                      data-tooltip-content={cleanOptionDescriptions[key]}
                    >
                      <input
                        type="checkbox"
                        className="accent-green-600 h-4 w-4"
                        checked={cleanOptions[key]}
                        disabled={isDisabled}
                        onChange={(e) =>
                          setCleanOptions({ ...cleanOptions, [key]: e.target.checked })
                        }
                      />
                      <span>{key.replace(/_/g, " ")}</span>
                      
                    </label>
                    <Tooltip id={tooltipId} place="top-start" />
                    </>
                  );
                })}
              </div>

            </div>

            {/* Cleaning Status */}
            {cleanStatus && (
              <div className="flex items-center gap-2 text-sm text-gray-700 mb-3 mt-auto">
                {cleanStatus === "completed" && <CheckCircle size={16} className="text-green-600" />}
                {cleanStatus === "failed" && <AlertCircle size={16} className="text-red-600" />}
                {cleanStatus === "running" && <Loader2 size={16} className="animate-spin text-green-500" />}
                {cleanStatus === "pending" && <Loader2 size={16} className="animate-pulse text-yellow-500" />}
                <span className="capitalize">Cleaning status: {cleanStatus}</span>
              </div>
            )}
          </div>

          <div className="pt-4 text-right space-x-3">
            <Button variant="outline" onClick={() => setStepModal(1)}>
              Back
            </Button>

            <Button
              onClick={async () => {
                await handleCleanData();
              }}
              variant="primary"
            >
              <div className="flex items-center gap-2">
                <FaRocket />
                Start Cleaning
              </div>
            </Button>

            <Button
              variant="primary"
              onClick={() => setStepModal(3)}
              disabled={cleanStatus !== "completed"}
              className={cleanStatus !== "completed" ? "opacity-50 cursor-not-allowed" : ""}
            >
              Next
            </Button>
          </div>
        </Modal>
      )}


      {stepModal === 3 && (
        <Modal title="Select Target and Features" onClose={() => setStepModal(0)}>
          <SetupModelModal
            availableColumns={columns}
            selectedTarget={selectedTarget}
            setSelectedTarget={setSelectedTarget}
            selectedFeatures={selectedFeatures}
            setSelectedFeatures={setSelectedFeatures}
            onCancel={() => setStepModal(2)}
            onFinish={handleFinishSetupModelModal}
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
