import { Loader2 } from "lucide-react";
import { FaMagic } from "react-icons/fa";
import { Card } from "../../components/ui";
import DataTable from "../../components/DataTable";

export default function PredictOnNewData({
  datasetOptions,
  selectedDataset,
  handleSelectDataset,
  uploading,
  predictedData
}) {
  return (
    <Card className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
        <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
          <FaMagic /> Predict on New Data
        </h3>

        {/* Dataset Selection */}
        <select
          value={selectedDataset}
          onChange={(e) => handleSelectDataset(e.target.value)}
          className="border border-gray-300 rounded-md px-4 py-2 w-full sm:w-1/2 focus:outline-none focus:ring-2 focus:ring-green-500"
        >
          <option value="">Select a dataset to predict...</option>
          {datasetOptions.map((ds) => (
            <option key={ds.id} value={ds.id}>
              {ds.name}
            </option>
          ))}
        </select>
      </div>

      {/* Status */}
      {uploading && (
        <div className="flex items-center gap-2 text-sm text-gray-700">
          <Loader2 size={16} className="animate-spin text-green-500" />
          <span className="capitalize">Predicting on selected dataset...</span>
        </div>
      )}

      {/* Prediction Results */}
      {predictedData.length > 0 && (
        <DataTable data={predictedData} />
      )}
    </Card>
  );
}
