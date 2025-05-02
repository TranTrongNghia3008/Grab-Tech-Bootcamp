import { useState, useEffect } from "react";
import { FaBullseye, FaWrench } from "react-icons/fa";
import { Button, Card } from "../../components/ui";

export default function NextModal({
  availableColumns,
  selectedTarget,
  setSelectedTarget,
  selectedFeatures,
  setSelectedFeatures,
  onCancel,
  onFinish,
}) {
  const [showErrors, setShowErrors] = useState(false);

  // Khi chọn lại target, tự động remove nó khỏi feature nếu trùng
  useEffect(() => {
    if (selectedTarget && selectedFeatures.includes(selectedTarget)) {
      setSelectedFeatures(selectedFeatures.filter((f) => f !== selectedTarget));
    }
  }, [selectedTarget, selectedFeatures, setSelectedFeatures]);

  const handleFeatureChange = (col) => {
    if (selectedFeatures.includes(col)) {
      setSelectedFeatures(selectedFeatures.filter((f) => f !== col));
    } else {
      if (col !== selectedTarget) { // Không cho chọn target làm feature
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
    <Card className="space-y-6">
      <div className="space-y-6">
        {/* Mô tả ngắn */}
        <p className="text-sm text-gray-700">
          Select the <strong>target column</strong> and <strong>features</strong> you want to use to build a predictive model.
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
            <option value="">Select Target</option>
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
          <div className="border border-gray-300 rounded-md px-3 py-2 max-h-[150px] overflow-y-auto space-y-2">
            {availableColumns.map((col) => (
              <label key={col} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  value={col}
                  checked={selectedFeatures.includes(col)}
                  onChange={() => handleFeatureChange(col)}
                  disabled={col === selectedTarget} // Không cho chọn chính target
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
            Cancel
          </Button>
          <Button variant="primary" onClick={handleFinish}>
            Finish
          </Button>
        </div>
      </div>
    </Card>
  );
}
