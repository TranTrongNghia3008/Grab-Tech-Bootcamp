import { useState } from "react";
import { LuGitCompare } from "react-icons/lu"; 
import { Card } from "../../../components/ui";
import DataTable from "../../../components/DataTable";

export default function CompareModels({ models = [] }) {
  const [selectedModels, setSelectedModels] = useState([]);

  const handleToggleModel = (modelIndex) => {
    setSelectedModels((prev) =>
      prev.includes(modelIndex)
        ? prev.filter((id) => id !== modelIndex)
        : [...prev, modelIndex]
    );
  };

  const selectedModelsData = selectedModels.map((modelIndex) => {
    const model = models.find((m) => m.index === modelIndex);
    if (!model) return {};

    const { ...rest } = model; // Bỏ cột index
    return rest;
  });

  return (
    <Card className="space-y-6">
      {/* Header */}
      <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
        <LuGitCompare />
        Compare Multiple Models
      </h3>

      {/* Flex container with 2 columns */}
      <div className="flex flex-col md:flex-row gap-6">
        {/* Left Column: Checkboxes */}
        <div className="w-full md:w-1/6 pt-4 space-y-1">
          {models.map((model) => (
            <label
              key={model.index}
              className="flex items-center gap-2 text-sm cursor-pointer"
            >
              <input
                type="checkbox"
                value={model.index}
                checked={selectedModels.includes(model.index)}
                onChange={() => handleToggleModel(model.index)}
                className="accent-green-600"
              />
              {model.Model}
            </label>
          ))}
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
