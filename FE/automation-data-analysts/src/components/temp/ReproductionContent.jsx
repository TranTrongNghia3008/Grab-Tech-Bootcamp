export default function ReproductionContent() {
    const reproductionInfo = [
      { label: "Analysis started", value: "2025-04-26 04:52:21.370679" },
      { label: "Analysis finished", value: "2025-04-26 04:52:21.508188" },
      { label: "Duration", value: "0.14 seconds" },
      { label: "Software version", value: <a href="https://pypi.org/project/ydata-profiling/" target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">ydata-profiling v4.16.1</a> },
      { label: "Download configuration", value: <a href="/config.json" download className="text-blue-600 underline">config.json</a> }
    ];
  
    return (
      <div className="space-y-4">
        <h3 className="text-xl font-semibold text-gray-800">Reproduction</h3>
        <div className="space-y-2">
          {reproductionInfo.map((item, idx) => (
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
    );
  }
  