import { useState } from "react";
import { LuGitCompare } from "react-icons/lu"; 
import { Card } from "../../components/ui";
import DataTable from "../../components/DataTable";

export default function CompareModels({ models = [] }) {
    const [selectedModels, setSelectedModels] = useState([]);

    const handleToggleModel = (modelId) => {
        setSelectedModels((prev) =>
        prev.includes(modelId)
            ? prev.filter((id) => id !== modelId)
            : [...prev, modelId]
        );
    };

    const selectedModelsData = selectedModels.map((modelId) => {
        const model = models.find((m) => m.modelId === modelId);
        return {
        Model: model?.modelName,
        Accuracy: model?.Accuracy,
        AUC: model?.AUC,
        Recall: model?.Recall,
        Precision: model?.Precision,
        F1: model?.F1,
        Kappa: model?.Kappa,
        MCC: model?.MCC,
        };
    });

    return (
        <Card className="space-y-6">
  {/* Header */}
  <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
    <LuGitCompare/>
    Compare Multiple Models
  </h3>

  {/* Flex container with 2 columns */}
  <div className="flex flex-col md:flex-row gap-6">
    {/* Left Column: Checkboxes */}
    <div className="w-full md:w-1/6 pt-4">
      <div className="">
        {models.map((model) => (
          <label
            key={model.modelId}
            className="flex items-center gap-2 text-sm cursor-pointer"
          >
            <input
              type="checkbox"
              value={model.modelId}
              checked={selectedModels.includes(model.modelId)}
              onChange={() => handleToggleModel(model.modelId)}
              className="accent-green-600"
            />
            {model.modelName}
          </label>
        ))}
      </div>
    </div>

    {/* Right Column: Comparison Table */}
    <div className="w-full md:w-5/6">
      {selectedModels.length >= 2 && (
        <div className="pt-4">
          <DataTable data={selectedModelsData} />
        </div>
      )}
    </div>
  </div>
</Card>

    );
}
