import { useState } from "react";
import { useNavigate } from "react-router-dom";
import UploadDropzone from "../components/UploadDropzone";
import MainLayout from "../layout/MainLayout";

export default function CreateProject() {
  const [projectName, setProjectName] = useState("");
  const [file, setFile] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleUpload = (selectedFile) => {
    setFile(selectedFile);
    setError("");
  };

  const handleCreate = () => {
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

    const reader = new FileReader();
    reader.onload = (event) => {
      const csvData = event.target.result;

      // 👉 Giả lập gọi API tạo project (thay thế bằng fetch/axios sau)
      setTimeout(() => {
        const newProject = {
          id: Date.now(),
          name: projectName.trim(),
          updatedAt: new Date().toISOString()
        };

        localStorage.setItem("currentProject", JSON.stringify(newProject));
        localStorage.setItem("dataset", csvData);

        setLoading(false);
        navigate("/project/" + newProject.id);
      }, 1000);
    };

    reader.readAsText(file);
  };

  const handleResetFile = () => {
    setFile(null);
  };

  return (
    <MainLayout>
      <div className="min-h-[calc(100vh-64px)] flex items-center justify-center">
        <div className="w-full max-w-xl bg-green-50 p-8 rounded-xl shadow-md">
          <h2 className="text-2xl font-bold mb-6 text-green-700 text-center">📁 Create New Project</h2>
  
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
  
          {/* Upload */}
          <div className="mb-5">
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
          </div>
  
          {/* Error */}
          {error && <p className="text-red-600 text-sm mb-4">{error}</p>}
  
          {/* Submit Button */}
          <div className="text-right">
            <button
              onClick={handleCreate}
              disabled={loading}
              className={`px-5 py-2 rounded-md text-white transition ${
                loading ? "bg-green-400 cursor-wait" : "bg-green-600 hover:bg-green-700 hover:cursor-pointer"
              }`}
            >
              {loading ? "Creating..." : "Create Project"}
            </button>
          </div>
        </div>
      </div>
    </MainLayout>
  );
  
}
