import { useState } from "react";
import VariableCard from "./VariableCard";

const dummyColumns = [
  { name: "PassengerId", type: "Real number", isReal: true, isUnique: true },
  { name: "Survived", type: "Real number", isReal: true, isUnique: false },
  { name: "Pclass", type: "Real number", isReal: true, isUnique: false },
  { name: "Age", type: "Real number", isReal: true, isUnique: false },
  { name: "Fare", type: "Real number", isReal: true, isUnique: false },
  { name: "Name", type: "Text", isReal: false, isUnique: true },
];

export default function VariablesTab() {
  const [selectedColumns, setSelectedColumns] = useState([]);

  const columnsToShow = selectedColumns.length > 0
    ? dummyColumns.filter(col => selectedColumns.includes(col.name))
    : dummyColumns;

  return (
    <div className="space-y-8">
      {/* Select Columns */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Select Columns</label>
        <select
          multiple
          value={selectedColumns}
          onChange={(e) =>
            setSelectedColumns(Array.from(e.target.selectedOptions, (option) => option.value))
          }
          className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-green-500"
        >
          {dummyColumns.map((col) => (
            <option key={col.name} value={col.name}>
              {col.name}
            </option>
          ))}
        </select>
        <p className="text-xs text-gray-400 mt-1">Hold Ctrl (Cmd) to select multiple columns.</p>
      </div>

      {/* Render Variable Cards */}
      <div className="space-y-6">
        {columnsToShow.map((col) => (
          <VariableCard
            key={col.name}
            variable={{
              name: col.name,
              type: col.type,
              isReal: col.isReal,
              isUnique: col.isUnique,
              metrics: [
                { label: "Distinct", value: 891, highlight: col.isUnique },
                { label: "Distinct (%)", value: "100.0%", highlight: col.isUnique },
                { label: "Missing", value: 0 },
                { label: "Missing (%)", value: "0.0%" },
                ...(col.isReal
                  ? [
                      { label: "Infinite", value: 0 },
                      { label: "Infinite (%)", value: "0.0%" },
                      { label: "Mean", value: 446 },
                    ]
                  : []
                ),
              ],
              extraMetrics: col.isReal
                ? [
                    { label: "Minimum", value: 1 },
                    { label: "Maximum", value: 891 },
                    { label: "Zeros", value: 0 },
                    { label: "Zeros (%)", value: "0.0%" },
                    { label: "Negative", value: 0 },
                    { label: "Negative (%)", value: "0.0%" },
                    { label: "Memory size", value: "7.1 KiB" },
                  ]
                : [{ label: "Memory size", value: "7.1 KiB" }],
              previewImage: col.isReal
                ? "/preview_histogram.png"
                : "/preview_wordcloud.png",
              detailImage: col.isReal
                ? "/detail_histogram.png"
                : "/detail_wordcloud.png",
            }}
          />
        ))}
      </div>
    </div>
  );
}
