import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

export default function DataTable({ data }) {
  const [sortConfig, setSortConfig] = useState({ key: null, direction: "asc" });

  if (!data || data.length === 0) return <p className="text-gray-500">No data available</p>;

  const columns = Object.keys(data[0]);

  const sortedData = [...data].sort((a, b) => {
    const { key, direction } = sortConfig;
    if (!key) return 0;
    const valA = a[key] ?? "";
    const valB = b[key] ?? "";

    if (typeof valA === "number" && typeof valB === "number") {
      return direction === "asc" ? valA - valB : valB - valA;
    }

    return direction === "asc"
      ? String(valA).localeCompare(String(valB))
      : String(valB).localeCompare(String(valA));
  });

  const handleSort = (col) => {
    if (sortConfig.key === col) {
      setSortConfig({ key: col, direction: sortConfig.direction === "asc" ? "desc" : "asc" });
    } else {
      setSortConfig({ key: col, direction: "asc" });
    }
  };

  return (
      <div className="min-w-[900px] overflow-x-auto border rounded max-h-[500px]">
        <table className="text-sm border-collapse">
            <div className="top-0 sticky z-10 border-b border-gray-300">
                <thead className="bg-gray-100 text-gray-700">
                    <tr>
                    {columns.map((col) => (
                        <th
                        key={col}
                        className="px-4 py-3 text-left font-medium cursor-pointer select-none whitespace-nowrap hover:bg-gray-200 transition w-[150px] min-w-[150px] border-r border-gray-300"
                        onClick={() => handleSort(col)}
                        >
                        <div className="flex items-center gap-1">
                            {col}
                            {sortConfig.key === col ? (
                            sortConfig.direction === "asc" ? (
                                <ChevronUp size={16} className="text-gray-500" />
                            ) : (
                                <ChevronDown size={16} className="text-gray-500" />
                            )
                            ) : (
                            <ChevronDown size={16} className="text-gray-300" />
                            )}
                        </div>
                        </th>
                    ))}
                    </tr>
                </thead>
            </div>
            <div className="">
                <tbody>
                    {sortedData.map((row, idx) => (
                    <tr
                        key={idx}
                        className="even:bg-gray-50 hover:bg-blue-50 transition"
                    >
                        {columns.map((col) => (
                        <td key={col} className="px-4 py-2 whitespace-nowrap w-[150px] min-w-[150px] border-r border-gray-300">
                            {row[col]}
                        </td>
                        ))}
                    </tr>
                    ))}
                </tbody>
            </div>
        </table>
      </div>
  );
}
