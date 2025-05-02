import { useState } from "react";
import { FaTag } from "react-icons/fa";
import { Card } from "../../components/ui";

export default function VariableCard({ variable }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <Card className="p-6">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-xl font-bold text-blue-600">{variable.name}</h2>
          <p className="text-gray-600 text-sm flex items-center gap-1">
            {variable.type} 
            {variable.isReal && <span className="text-gray-400">(R)</span>}
          </p>

          {variable.isUnique && (
            <div className="mt-2">
              <span className="text-xs bg-red-500 text-white px-2 py-1 rounded">Unique</span>
            </div>
          )}
        </div>

        {!expanded && (
          <button
            className="text-sm bg-gray-100 hover:bg-gray-200 px-3 py-2 rounded text-gray-700"
            onClick={() => setExpanded(true)}
          >
            More details
          </button>
        )}
      </div>

      {/* Main content */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Left - Metrics */}
        <div className="space-y-2 md:col-span-2">
          <div className="grid grid-cols-2 gap-4 text-sm">
            {variable.metrics.map((item, idx) => (
              <div
                key={idx}
                className={`flex justify-between items-center px-3 py-2 ${
                  item.highlight ? "bg-red-50 text-red-600 font-semibold" : "bg-gray-100"
                } rounded`}
              >
                <span>{item.label}</span>
                <span>{item.value}</span>
              </div>
            ))}
          </div>

          {/* Extra metrics when expanded */}
          {expanded && variable.extraMetrics && (
            <div className="grid grid-cols-2 gap-4 mt-4 text-sm">
              {variable.extraMetrics.map((item, idx) => (
                <div
                  key={idx}
                  className="flex justify-between items-center px-3 py-2 bg-gray-100 rounded"
                >
                  <span>{item.label}</span>
                  <span>{item.value}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right - Chart */}
        <div className="flex justify-center items-center">
          <img
            src={expanded ? variable.detailImage : variable.previewImage}
            alt="Preview"
            className="max-h-[220px] w-full object-contain"
          />
        </div>
      </div>
    </Card>
  );
}
