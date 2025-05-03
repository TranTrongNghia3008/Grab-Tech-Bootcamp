import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

export default function DataTable({ data }) {
  const [sortConfig, setSortConfig] = useState({ key: null, direction: "asc" });

  if (!data || data.length === 0) return <p className="text-[#888888]">No data available</p>;

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
    <div className="min-w-[900px] overflow-x-auto border rounded max-h-[600px] bg-[#FFFDF3] font-[Poppins]">
      <table className="text-sm border-collapse w-full">
        <thead className="sticky top-0 z-10 bg-[#E4F3E9] text-[#1B1F1D] border-b border-[#CDEBD5]">
          <tr>
            {columns.map((col) => (
              <th
                key={col}
                className="px-4 py-3 text-left font-medium cursor-pointer select-none whitespace-nowrap hover:bg-[#CDEBD5] transition w-[200px] min-w-[200px] border-r border-[#CDEBD5]"
                onClick={() => handleSort(col)}
              >
                <div className="flex items-center gap-1">
                  {col}
                  {sortConfig.key === col ? (
                    sortConfig.direction === "asc" ? (
                      <ChevronUp size={16} className="text-[#00843D]" />
                    ) : (
                      <ChevronDown size={16} className="text-[#00843D]" />
                    )
                  ) : (
                    <ChevronDown size={16} className="text-[#CDEBD5]" />
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedData.map((row, idx) => (
            <tr
              key={idx}
              className="even:bg-[#FFF9E5] hover:bg-[#CDEBD5] transition"
            >
              {columns.map((col) => (
                <td
                  key={col}
                  className="px-4 py-2 whitespace-nowrap w-[200px] min-w-[200px] border-r border-[#CDEBD5] text-[#1B1F1D]"
                >
                  {typeof row[col] === "number"
                    ? row[col].toFixed(3)
                    : row[col]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
