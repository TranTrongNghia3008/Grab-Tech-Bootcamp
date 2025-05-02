export default function OverviewContent() {
    const datasetStats = [
      { label: "Number of variables", value: 12 },
      { label: "Number of observations", value: 891 },
      { label: "Missing cells", value: 866 },
      { label: "Missing cells (%)", value: "8.1%" },
      { label: "Total size in memory", value: "83.7 KiB" },
      { label: "Average record size in memory", value: "96.1 B" }
    ];
  
    const variableTypes = [
      { type: "Numeric", count: 7 },
      { type: "Text", count: 5 }
    ];
  
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Dataset Statistics */}
        <div className="space-y-4">
          <h3 className="text-xl font-semibold text-gray-800">Dataset statistics</h3>
          <div className="space-y-2">
            {datasetStats.map((item, idx) => (
              <div
                key={idx}
                className="flex justify-between bg-gray-50 p-2 rounded-md"
              >
                <span className="font-medium text-gray-700">{item.label}</span>
                <span className="text-gray-900">{item.value}</span>
              </div>
            ))}
          </div>
        </div>
  
        {/* Variable Types */}
        <div className="space-y-4">
          <h3 className="text-xl font-semibold text-gray-800">Variable types</h3>
          <div className="space-y-2">
            {variableTypes.map((item, idx) => (
              <div
                key={idx}
                className="flex justify-between bg-gray-50 p-2 rounded-md"
              >
                <span className="font-medium text-gray-700">{item.type}</span>
                <span className="text-gray-900">{item.count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }
  