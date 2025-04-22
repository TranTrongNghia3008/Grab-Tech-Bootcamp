import { useState } from "react";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";

export default function ModelBuilderPanel() {
  const [target, setTarget] = useState("");
  const [features, setFeatures] = useState([]);
  const [modelType, setModelType] = useState("logistic_regression");
  const [split, setSplit] = useState(0.8);
  const [metrics, setMetrics] = useState(null);
  const [importances, setImportances] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});

  const columns = ["age", "income", "gender", "education", "experience"];
  const modelTypes = ["logistic_regression", "decision_tree", "random_forest"];

  const handleTrain = () => {
    const newErrors = {};
    if (!target) newErrors.target = "Please select a target variable.";
    if (features.length === 0) newErrors.features = "Please select at least one feature.";
    setErrors(newErrors);

    if (Object.keys(newErrors).length > 0) return;

    setLoading(true);
    setJobStatus("pending");

    setTimeout(() => {
      setJobStatus("running");
      setTimeout(() => {
        setMetrics({ accuracy: 0.93, precision: 0.91, recall: 0.89 });
        setImportances([
          { feature: "income", importance: 0.38 },
          { feature: "age", importance: 0.32 },
          { feature: "experience", importance: 0.30 }
        ]);
        setJobStatus("done");
        setLoading(false);
      }, 2000);
    }, 1000);
  };

  const handleFeatureToggle = (col) => {
    if (features.includes(col)) {
      setFeatures(features.filter((f) => f !== col));
    } else {
      setFeatures([...features, col]);
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Custom Model Builder</h2>

      <div className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm">
        🛠️ <span className="font-medium">Custom Model Builder</span> lets you pick target, features, model type, and hyperparameters for training a custom ML model.
      </div>

      <div className="space-y-4 bg-white p-6 rounded-xl shadow-md">
        {/* Target */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">🎯 Target (Y)</label>
          <select
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
          >
            <option value="">Select target</option>
            {columns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
          {errors.target && <p className="text-sm text-red-600 mt-1">{errors.target}</p>}
        </div>

        {/* Features */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">📊 Features (X)</label>
          <div className="flex flex-wrap gap-2">
            {columns.map((col) => (
              <label
                key={col}
                className="flex items-center gap-1 text-sm text-gray-700 bg-gray-100 px-3 py-1 rounded-md cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={features.includes(col)}
                  onChange={() => handleFeatureToggle(col)}
                />
                {col}
              </label>
            ))}
          </div>
          {errors.features && <p className="text-sm text-red-600 mt-1">{errors.features}</p>}
        </div>

        {/* Model Type */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">🧠 Model Type</label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
          >
            {modelTypes.map((type) => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>

        {/* Train/Test Split */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">🔀 Train/Test Split</label>
          <div className="flex items-center space-x-4">
            <div className="relative">
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={split}
                onChange={(e) => setSplit(parseFloat(e.target.value))}
                className="w-24 border-2 border-gray-300 rounded-lg px-4 py-2 text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition duration-200"
              />
            </div>
            <div className="flex-1">
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={split}
                onChange={(e) => setSplit(parseFloat(e.target.value))}
                className="w-full h-2 bg-blue-200 rounded-lg cursor-pointer"
              />
            </div>
            <span className="text-gray-700">{(split * 100).toFixed(0)}%</span>
          </div>
        </div>

        {/* Train Button */}
        <div className="text-right">
          <button
            onClick={handleTrain}
            disabled={loading}
            className={`px-5 py-2 rounded-md text-white transition ${
              loading ? "bg-green-400 cursor-wait" : "bg-green-600 hover:bg-green-700"
            }`}
          >
            {loading ? "Training..." : "Train Custom Model"}
          </button>
        </div>

        {/* Job Status */}
        {jobStatus && (
          <div className="flex items-center gap-2 text-sm text-gray-700 mt-2">
            {jobStatus === "done" && <CheckCircle size={16} className="text-green-600" />}
            {jobStatus === "error" && <AlertCircle size={16} className="text-red-600" />}
            {jobStatus === "running" && <Loader2 size={16} className="animate-spin text-green-500" />}
            <span className="capitalize">Status: {jobStatus}</span>
          </div>
        )}
      </div>

      {/* Metrics */}
      {metrics && (
        <div className="bg-white p-6 rounded-xl shadow-md">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">📈 Model Metrics</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            {Object.entries(metrics).map(([key, val]) => (
              <div
                key={key}
                className="border border-gray-200 rounded-lg px-4 py-3 bg-green-50 hover:shadow transition"
              >
                <p className="text-xs text-gray-500 uppercase tracking-wide">{key}</p>
                <p className="text-lg font-semibold text-green-700">
                  {typeof val === "number" ? val.toFixed(3) : val}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Importances */}
      {importances && (
        <div className="space-y-4 bg-white p-6 rounded-xl shadow-md overflow-auto">
          <h3 className="font-semibold text-gray-800 text-xl mb-4">🧠 Feature Importances</h3>
          <div className="space-y-2">
            {importances.map((item, idx) => (
              <div key={idx} className="text-sm">
                <div className="flex justify-between mb-1">
                  <span className="text-gray-700">{item.feature}</span>
                  <span className="text-gray-600 font-medium">{(item.importance * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-2 bg-gray-200 rounded">
                  <div
                    className="h-2 bg-green-500 rounded"
                    style={{ width: `${item.importance * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
