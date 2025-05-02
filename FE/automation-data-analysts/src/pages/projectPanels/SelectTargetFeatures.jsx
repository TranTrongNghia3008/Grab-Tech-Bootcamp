import { useState } from "react";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { FaBullseye, FaWrench, FaRobot, FaCogs } from "react-icons/fa";
import { Button, Card } from "../../components/ui";

export default function SelectTargetFeaturesModel({
  availableColumns,
  target,
  setTarget,
  features,
  setFeatures,
  handleTrain,
  loading,
  jobStatus,
  trainLabel = "Train",
  showModelSelection = false,
  modelType,
  setModelType,
  modelOptions = [],
  bestModelId = "",
  customGrid,
  setCustomGrid,
}) {
  const [showErrors, setShowErrors] = useState(false);
  const [gridError, setGridError] = useState(null);

  const validateAndTrain = () => {
    if (!target || features.length === 0 || (showModelSelection && !modelType)) {
      setShowErrors(true);
      return;
    }
    if (customGrid) {
      try {
        JSON.parse(customGrid);
        setGridError(null);
      } catch (err) {
        setGridError(err);
        return;
      }
    }
    handleTrain();
  };

  const handleFeatureChange = (col) => {
    if (features.includes(col)) {
      setFeatures(features.filter((f) => f !== col));
    } else {
      setFeatures([...features, col]);
    }
  };

  return (
    <Card className="space-y-6">
      <div className="flex flex-col sm:flex-row gap-6">
        <div className="flex-1 space-y-4">
          {/* Target Column */}
          <div>
            <label className="text-sm font-medium text-gray-700 mb-1 flex items-center gap-2">
              <FaBullseye className="text-green-600" />
              Target Column
            </label>
            <select
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
            >
              <option value="">Select Target</option>
              {availableColumns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
            {showErrors && !target && (
              <p className="text-xs text-red-600 mt-1">Please select a target column.</p>
            )}
          </div>

          {/* Model Selection */}
          {showModelSelection && (
            <div>
              <label className="text-sm font-medium text-gray-700 mb-1 flex items-center gap-2">
                <FaRobot className="text-green-600" />
                Model Type
              </label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
              >
                <option value="">Select Model Type</option>
                {modelOptions.map((model) => (
                  <option key={model.id} value={model.name}>{model.name}</option>
                ))}
              </select>

              {bestModelId && modelOptions.some((model) => model.id === bestModelId) && (
                <p className="text-xs text-green-600 mt-1">
                  Suggested model: {modelOptions.find((model) => model.id === bestModelId)?.name} (based on previous best results)
                </p>
              )}
              {showErrors && !modelType && (
                <p className="text-xs text-red-600 mt-1">Please select a model type.</p>
              )}
            </div>
          )}
        </div>

        {/* Feature Columns */}
        <div className="flex-1">
          <label className="text-sm font-medium text-gray-700 mb-1 flex items-center gap-2">
            <FaWrench className="text-green-600" />
            Feature Columns
          </label>
          <div className="border border-gray-300 rounded-md px-3 py-2 max-h-[150px] overflow-y-auto space-y-2">
            {availableColumns.map((col) => (
              <label key={col} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  value={col}
                  checked={features.includes(col)}
                  onChange={() => handleFeatureChange(col)}
                  className="accent-green-600"
                />
                {col}
              </label>
            ))}
          </div>
          {showErrors && features.length === 0 && (
            <p className="text-xs text-red-600 mt-1">Please select at least one feature column.</p>
          )}
        </div>
      </div>

      {/* Custom Grid Search */}
      {showModelSelection && (
        <div>
          <label className="text-sm font-medium text-gray-700 mb-1 flex items-center gap-2">
            <FaCogs className="text-green-600" />
            Custom Grid Search (Optional)
          </label>
          <textarea
            rows={4}
            value={customGrid}
            onChange={(e) => setCustomGrid(e.target.value)}
            placeholder='Example: {"n_estimators": [100, 200], "max_depth": [10, 20]}'
            className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 text-sm"
          />
          {gridError && (
            <p className="text-xs text-red-600 mt-1">{gridError}</p>
          )}
        </div>
      )}

      {/* Train Button */}
      <div className="text-right">
        <Button
          onClick={validateAndTrain}
          disabled={loading || jobStatus === "running"}
        >
          {loading || jobStatus === "running" ? "Training..." : trainLabel}
        </Button>
      </div>

      {/* Status */}
      {jobStatus && (
        <div className="flex items-center gap-2 text-sm text-gray-700">
          {jobStatus === "done" && <CheckCircle size={16} className="text-green-600" />}
          {jobStatus === "error" && <AlertCircle size={16} className="text-red-600" />}
          {jobStatus === "running" && <Loader2 size={16} className="animate-spin text-green-500" />}
          {jobStatus === "pending" && <Loader2 size={16} className="animate-pulse text-yellow-500" />}
          <span className="capitalize">Status: {jobStatus}</span>
        </div>
      )}
    </Card>
  );
}
