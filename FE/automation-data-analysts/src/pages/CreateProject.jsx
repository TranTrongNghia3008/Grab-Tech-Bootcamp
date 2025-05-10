import { useState } from "react";
import { useNavigate } from "react-router-dom";
import UploadDropzone from "../components/UploadDropzone";
import MainLayout from "../layout/MainLayout";
import { Button, Card, Modal } from "../components/ui";
import { FiFolder } from "react-icons/fi";
import { upLoadDataset } from "../components/services/datasetService";
import { useAppContext } from "../contexts/AppContext";


export default function CreateProject() {
  const [projectName, setProjectName] = useState("");
  const [file, setFile] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dbConnectionStatus, setDbConnectionStatus] = useState(null); // Để theo dõi kết quả kết nối DB
  const [isModalOpen, setIsModalOpen] = useState(false); // Trạng thái mở modal
  const [datasetList, setDatasetList] = useState([]); // Trạng thái lưu trữ danh sách datasets có sẵn
  const [selectedDataset, setSelectedDataset] = useState(""); // Trạng thái để lưu dataset đã chọn
  const [activeTab, setActiveTab] = useState(1); // Trạng thái tab đang hoạt động
  const navigate = useNavigate();
  const { updateState } = useAppContext();

  // Giả lập tải danh sách datasets có sẵn
  const fetchDatasetList = () => {
    setDatasetList([
      { id: "1", name: "Sales Forecasting Dataset" },
      { id: "2", name: "Customer Data" },
      { id: "3", name: "Marketing Campaign Dataset" },
    ]);
  };

  // Giả lập gọi hàm khi component mount
  useState(() => {
    fetchDatasetList();
  }, []);

  const handleUpload = (selectedFile) => {
    setFile(selectedFile);
    setError("");
  };

  const handleCreateWithUpload = async () => {
    if (!projectName.trim()) {
      setError("Please enter a project name.");
      return;
    }
    if (!file) {
      setError("Please upload a dataset file.");
      return;
    }

    setError("");
    setLoading(true);

    try {
      const result = await upLoadDataset(projectName, file); // Gọi API tải lên dataset
      console.log("Dataset ID:", result.id); // Log kết quả tải lên cho việc gỡ lỗi
      updateState({ datasetId: result.id, projectName: projectName }); // Cập nhật datasetId vào context nếu cần
      const newProject = {
        id: result.projectId || Date.now(),
        name: projectName.trim(),
        updatedAt: new Date().toISOString(),
      };
  
      localStorage.setItem("currentProject", JSON.stringify(newProject));
      localStorage.setItem("dataset", JSON.stringify(result.dataset || {})); // Tuỳ theo response của backend
  
      navigate("/project/" + newProject.id);
    } catch (error) {
      console.error("Upload failed:", error);
      setError("Failed to upload dataset. Please try again.");
    } finally {
      setLoading(false);
    }

    // const reader = new FileReader();
    // reader.onload = (event) => {
    //   const csvData = event.target.result;
    //   console.log("CSV Data:", csvData); // Log CSV data for debugging

    //   // Giả lập gọi API tạo project
    //   setTimeout(() => {
    //     const newProject = {
    //       id: Date.now(),
    //       name: projectName.trim(),
    //       updatedAt: new Date().toISOString(),
    //     };

    //     localStorage.setItem("currentProject", JSON.stringify(newProject));
    //     localStorage.setItem("dataset", csvData);

    //     setLoading(false);
    //     navigate("/project/" + newProject.id);
    //   }, 1000);
    // };

    // reader.readAsText(file);
  };

  const handleCreateWithDbConnection = () => {
    if (!projectName.trim()) {
      setError("Please enter a project name.");
      return;
    }
    if (dbConnectionStatus !== "success") {
      setError("Database connection failed. Please try again.");
      return;
    }

    setError("");
    setLoading(true);

    // Giả lập gọi API tạo project từ DB
    setTimeout(() => {
      const newProject = {
        id: Date.now(),
        name: projectName.trim(),
        updatedAt: new Date().toISOString(),
      };

      localStorage.setItem("currentProject", JSON.stringify(newProject));
      setLoading(false);
      navigate("/project/" + newProject.id);
    }, 1000);
  };

  const handleCreateWithDatasetSelection = () => {
    if (!projectName.trim()) {
      setError("Please enter a project name.");
      return;
    }
    if (!selectedDataset) {
      setError("Please select an existing dataset.");
      return;
    }

    setError("");
    setLoading(true);

    // Giả lập gọi API tạo project từ dataset đã chọn
    setTimeout(() => {
      const newProject = {
        id: Date.now(),
        name: projectName.trim(),
        updatedAt: new Date().toISOString(),
      };

      localStorage.setItem("currentProject", JSON.stringify(newProject));
      setLoading(false);
      navigate("/project/" + newProject.id);
    }, 1000);
  };

  const handleTabChange = (tabIndex) => {
    setActiveTab(tabIndex); // Cập nhật tab đang hoạt động
  };

  const handleResetFile = () => {
    setFile(null);
    setSelectedDataset(""); // Đặt lại dataset đã chọn
  };

  const handleDbConnection = () => {
    setIsModalOpen(true); // Mở modal khi bấm vào nút kết nối DB
  };

  const handleModalClose = () => {
    setIsModalOpen(false); // Đóng modal khi người dùng nhấn "Close"
  };

  const handleSubmitDbConnection = () => {
    setIsModalOpen(false);
    setDbConnectionStatus("loading"); // Giả lập trạng thái đang kết nối DB
    setTimeout(() => {
      const isSuccess = 1; // Giả lập kết quả kết nối DB
      setDbConnectionStatus(isSuccess ? "success" : "error");
    }, 1500);
  };

  return (
    <MainLayout>
      <Card>
        <h1 className="text-2xl font-bold text-[#1B1F1D] flex items-center gap-2">
          <FiFolder className="text-green-600" size={24} />
          Create New Project
        </h1>

        {/* Project Name */}
        <div className="mb-5">
          <label htmlFor="projectName" className="block text-sm font-medium text-gray-700 mb-1">
            Project Name
          </label>
          <input
            id="projectName"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            placeholder="e.g. Sales Forecasting"
            className="w-full border border-gray-300 rounded-md px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
          />
        </div>

        <div className="flex space-x-6 border-b border-green-800/30 mb-6">
          <button
            onClick={() => handleTabChange(1)}
            className={`pb-2 text-sm font-semibold ${
              activeTab === 1
                ? "text-green-800 border-b-2 border-green-800"
                : "text-green-800 hover:text-green-900"
            }`}
          >
            Upload Dataset
          </button>
          <button
            onClick={() => handleTabChange(2)}
            className={`pb-2 text-sm font-semibold ${
              activeTab === 2
                ? "text-green-800 border-b-2 border-green-800"
                : "text-green-800 hover:text-green-900"
            }`}
          >
            Connect to DB
          </button>
          <button
            onClick={() => handleTabChange(3)}
            className={`pb-2 text-sm font-semibold ${
              activeTab === 3
                ? "text-green-800 border-b-2 border-green-800"
                : "text-green-800 hover:text-green-900"
            }`}
          >
            Choose Dataset
          </button>
        </div>


        {/* Tab content */}
        {activeTab === 1 && (
          <div>
            <h3 className="text-xl font-medium text-gray-800 mb-4">Upload Dataset</h3>
            {!file ? (
              <UploadDropzone onFileAccepted={handleUpload} />
            ) : (
              <div className="flex items-center justify-between px-4 py-3 bg-green-100 border border-green-200 rounded-md text-sm text-green-800">
                <span>📄 File uploaded: {file.name}</span>
                <button
                  onClick={handleResetFile}
                  className="text-xs text-red-500 hover:underline ml-4"
                >
                  Remove
                </button>
              </div>
            )}
            <div className="text-right mt-6">
              <Button
                onClick={handleCreateWithUpload}
                disabled={loading}
                className={`px-5 py-2 rounded-md text-white transition ${
                  loading ? "bg-green-400 cursor-wait" : "bg-green-600 hover:bg-green-700 hover:cursor-pointer"
                }`}
              >
                {loading ? "Creating..." : "Create Project"}
              </Button>
            </div>
          </div>
        )}

        {activeTab === 2 && (
          <div>
            <h3 className="text-xl font-medium text-gray-800 mb-4">Connect to Database</h3>
            <Button
              onClick={handleDbConnection}
            >
              {dbConnectionStatus === "loading" ? "Connecting..." : "Connect"}
            </Button>

            {dbConnectionStatus === "success" && (
              <p className="mt-4 text-green-600">Successfully connected to the database!</p>
            )}
            {dbConnectionStatus === "error" && (
              <p className="mt-4 text-red-600">Failed to connect to the database. Please try again.</p>
            )}

            <div className="text-right mt-6">
              <Button
                onClick={handleCreateWithDbConnection}
                disabled={loading}
                className={`px-5 py-2 rounded-md text-white transition ${
                  loading ? "bg-green-400 cursor-wait" : "bg-green-600 hover:bg-green-700 hover:cursor-pointer"
                }`}
              >
                {loading ? "Creating..." : "Create Project"}
              </Button>
            </div>
          </div>
        )}

        {activeTab === 3 && (
          <div>
            <h3 className="text-xl font-medium text-gray-800 mb-4">Choose Existing Dataset</h3>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="w-full border border-gray-300 rounded-md px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
            >
              <option value="">Select Dataset</option>
              {datasetList.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </option>
              ))}
            </select>
            <div className="text-right mt-6">
              <Button
                onClick={handleCreateWithDatasetSelection}
                disabled={loading}
                className={`px-5 py-2 rounded-md text-white transition ${
                  loading ? "bg-green-400 cursor-wait" : "bg-green-600 hover:bg-green-700 hover:cursor-pointer"
                }`}
              >
                {loading ? "Creating..." : "Create Project"}
              </Button>
            </div>
          </div>
        )}

        {/* Error */}
        {error && <p className="text-red-600 text-sm mb-4">{error}</p>}
      </Card>


      {/* Modal for DB Connection */}
      {isModalOpen && (
        <Modal title="Database Connection" onClose={handleModalClose}>
        <div className="p-4">
          <p className="text-gray-600 mb-6">Please enter your database details below to establish a secure connection.</p>

          <form onSubmit={(e) => e.preventDefault()} className="space-y-5">
            {/* Database URL */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1">Database URL</label>
              <input
                type="text"
                placeholder="e.g., mongodb+srv://..."
                className="w-full border border-gray-300 focus:border-green-600 focus:ring-green-600 rounded-lg px-4 py-2 text-sm shadow-sm outline-none transition"
              />
            </div>

            {/* Username */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1">Username</label>
              <input
                type="text"
                placeholder="Your DB username"
                className="w-full border border-gray-300 focus:border-green-600 focus:ring-green-600 rounded-lg px-4 py-2 text-sm shadow-sm outline-none transition"
              />
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-1">Password</label>
              <input
                type="password"
                placeholder="••••••••"
                className="w-full border border-gray-300 focus:border-green-600 focus:ring-green-600 rounded-lg px-4 py-2 text-sm shadow-sm outline-none transition"
              />
            </div>

            {/* Submit */}
            <div className="pt-2 text-right">
              <Button
                onClick={handleSubmitDbConnection}
                className="bg-green-600 hover:bg-green-700 text-white font-medium px-6 py-2 rounded-lg transition"
              >
                Connect
              </Button>
            </div>
          </form>
        </div>
      </Modal>

      )}
    </MainLayout>
  );
}
