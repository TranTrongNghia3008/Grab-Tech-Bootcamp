import { useState, useEffect } from "react";
import { FaBullseye, FaWrench } from "react-icons/fa";
import { Button, Card } from "../../../components/ui";

export default function SetupModelModal({
  availableColumns,
  selectedTarget,
  setSelectedTarget,
  selectedFeatures,
  setSelectedFeatures,
  onCancel,
  onFinish,
}) {
  const [showErrors, setShowErrors] = useState(false);

  useEffect(() => {
    const newFeatures = availableColumns.filter((col) => col !== selectedTarget);
    if (
      selectedFeatures.length === 0 ||
      selectedFeatures.includes(selectedTarget)
    ) {
      setSelectedFeatures(newFeatures);
    }
  }, [availableColumns, selectedTarget]);
  
  

  const handleFeatureChange = (col) => {
    if (selectedFeatures.includes(col)) {
      setSelectedFeatures(selectedFeatures.filter((f) => f !== col));
    } else {
      if (col !== selectedTarget) { 
        setSelectedFeatures([...selectedFeatures, col]);
      }
    }
  };

  const handleFinish = () => {
    if (!selectedTarget || selectedFeatures.length === 0) {
      setShowErrors(true);
      return;
    }
    onFinish();
  };

  return (
      <div className="space-y-4 min-h-[400px]">
        <p className="text-sm text-gray-700">
          Select the <strong>Target</strong> and <strong>Features</strong> you want to use to build a predictive model.
        </p>

        {/* Target Column */}
        <div>
          <label className="text-sm font-medium text-gray-700 mb-1 flex items-center gap-2">
            <FaBullseye className="text-green-600" />
            Target Column
          </label>
          <select
            value={selectedTarget}
            onChange={(e) => setSelectedTarget(e.target.value)}
            className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
          >
            <option value="">Choose the column you want to predict</option>
            {availableColumns.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
          {showErrors && !selectedTarget && (
            <p className="text-xs text-red-600 mt-1">Please select a target column.</p>
          )}
        </div>

        {/* Feature Columns */}
        <div>
          <label className="text-sm font-medium text-gray-700 mb-1 flex items-center gap-2">
            <FaWrench className="text-green-600" />
            Feature Columns
          </label>
          <div className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm mb-1">
            Do you want to change <strong>Feature Columns</strong>? <br />
            (Default selected all, except <strong>Target Column</strong>)
          </div>
          <div className="border border-gray-300 rounded-md px-3 py-2 max-h-[150px] overflow-y-auto space-y-2">
          {availableColumns.map((col) => (
            <label key={col} className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              value={col}
              checked={selectedFeatures.includes(col)}
              onChange={() => handleFeatureChange(col)}
              disabled={col === selectedTarget}
              className="accent-green-600"
            />

              <span className={col === selectedTarget ? "text-gray-400" : ""}>{col}</span>
            </label>
          ))}

          </div>
          {showErrors && selectedFeatures.length === 0 && (
            <p className="text-xs text-red-600 mt-1">Please select at least one feature column.</p>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end gap-4 pt-4">
          <Button variant="outline" onClick={onCancel}>
            Back
          </Button>
          <Button variant="primary" onClick={handleFinish}>
            Finish
          </Button>
        </div>
      </div>

  );
}
