import { useEffect, useState } from "react";
import DataTable from "../../components/DataTable";

export default function EDAPanel() {
//   const datasetId = 123; 

  const [stats, setStats] = useState(null);
  const [corr, setCorr] = useState(null);
  const [chart, setChart] = useState(null);
  const [loading, setLoading] = useState(false);
  const [reports, setReports] = useState([]);
  const [loadingReports, setLoadingReports] = useState(false);
  const [fetched, setFetched] = useState(false); 

  const [chartForm, setChartForm] = useState({
    type: "bar",
    x: "",
    y: ""
  });

  const columns = ["age", "income", "gender", "score"]; // giả lập cột

  // Mock fetch stats
  useEffect(() => {
    // Gọi API thực tế: GET /v1/datasets/{id}/eda/stats
    setTimeout(() => {
      setStats([
        { column: "age", mean: 30.5, std: 5.2, min: 20, max: 40, mode: 28 },
        { column: "income", mean: 50000, std: 10000, min: 20000, max: 100000, mode: 48000 },
        { column: "education", mean: 16.3, std: 2.5, min: 12, max: 20, mode: 16 },
        { column: "years_of_experience", mean: 7.5, std: 3.1, min: 1, max: 20, mode: 5 },
        { column: "hours_per_week", mean: 42.3, std: 8.2, min: 30, max: 60, mode: 40 },
        { column: "age11", mean: 30.5, std: 5.2, min: 20, max: 40, mode: 28 },
        { column: "income11", mean: 50000, std: 10000, min: 20000, max: 100000, mode: 47000 },
        { column: "education11", mean: 16.3, std: 2.5, min: 12, max: 20, mode: 14 },
        { column: "years_of_experience11", mean: 7.5, std: 3.1, min: 1, max: 20, mode: 3 },
        { column: "hours_per_week11", mean: 42.3, std: 8.2, min: 30, max: 60, mode: 40 }
      ]);
      
      
      
    }, 800);

    // Gọi correlation
    setTimeout(() => {
      setCorr([
        ["", "age", "income", "score", "experience"],
        ["age", 1.0, 0.65, -0.2, 0.8],
        ["income", 0.65, 1.0, 0.3, 0.5],
        ["score", -0.2, 0.3, 1.0, -0.6],
        ["experience", 0.8, 0.5, -0.6, 1.0]
      ]);
    }, 1000);

    // Gọi danh sách báo cáo
    setReports([
      { name: "eda_report_01.pdf", url: "#" },
      { name: "eda_sales_2024.pdf", url: "#" }
    ]);
  }, []);

  const handleGenerateChart = async () => {
    setLoading(true);

    // Gọi API POST /v1/datasets/{id}/eda/charts
    // Dữ liệu giả lập:
    setTimeout(() => {
      setChart("https://images.squarespace-cdn.com/content/v1/55b6a6dce4b089e11621d3ed/62a2d66b-8435-4e41-8df9-262db165ed79/NPL+and+Reserves+combo+chart.png");
      setLoading(false);
    }, 1200);
  };

  const handleGetReport = () => {
    setLoadingReports(true);

    setTimeout(() => {
      const mockReports = [
        { name: 'EDA_Report_Q1.pdf', url: '/mock-reports/EDA_Report_Q1.pdf' },
        { name: 'EDA_Report_Q2.pdf', url: '/mock-reports/EDA_Report_Q2.pdf' },
        { name: 'EDA_Report_Summary.pdf', url: '/mock-reports/EDA_Report_Summary.pdf' }
      ];
      setReports(mockReports);
      setLoadingReports(false);
      setFetched(true);
    }, 1000);
  };

  const handleDownloadAll = () => {
    reports.forEach((report, idx) => {
      setTimeout(() => {
        const link = document.createElement('a');
        link.href = report.url;
        link.download = report.name;
        link.click();
      }, idx * 300);
    });
  };


  return (
    <div className="space-y-6">
      <div className="">
        <h2 className="text-xl font-bold">Exploratory Data Analysis</h2>

        {/* Summary Stats */}
        {stats && (
          <div className="overflow-x-auto space-y-4 bg-white p-6 rounded-xl shadow-md">
            <h3 className="font-semibold text-gray-800 text-xl mb-4">📊 Summary Statistics</h3>
            <DataTable data={stats} />
            {/* <div className="overflow-y-auto max-h-[400px] border rounded-lg shadow-md">
              <table className="min-w-full text-sm">
                <thead className="bg-gray-100 text-left">
                  <tr>
                    <th className="p-4 text-gray-700 font-medium border-b border-r border-gray-300">Column</th>
                    <th className="p-4 text-gray-700 font-medium border-b border-r border-gray-300">Mean</th>
                    <th className="p-4 text-gray-700 font-medium border-b border-r border-gray-300">Std</th>
                    <th className="p-4 text-gray-700 font-medium border-b border-r border-gray-300">Min</th>
                    <th className="p-4 text-gray-700 font-medium border-b border-r border-gray-300">Max</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(stats).map(([col, s]) => (
                    <tr key={col} className="even:bg-gray-50">
                      <td className="p-4 text-gray-600 border-r border-gray-300">{col}</td>
                      <td className="p-4 text-gray-600 border-r border-gray-300">{s.mean}</td>
                      <td className="p-4 text-gray-600 border-r border-gray-300">{s.std}</td>
                      <td className="p-4 text-gray-600 border-r border-gray-300">{s.min}</td>
                      <td className="p-4 text-gray-600 border-r border-gray-300">{s.max}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div> */}
          </div>
        )}
      </div>


      {/* Correlation Matrix */}
      {corr && (
        <div className="space-y-4 bg-white p-6 rounded-xl shadow-md overflow-auto">
          <h3 className="font-semibold text-gray-800 text-xl mb-4">📈 Correlation Matrix</h3>
          <table className="border border-gray-200 text-sm table-auto">
            <tbody>
              {corr.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  {row.map((cell, colIdx) => {
                    const value = typeof cell === "number" ? cell : null;
                    const bgColor = value !== null ? `rgba(0, 177, 64, ${Math.abs(value)})` : "#f9fafb";


                    return (
                      <td
                        key={colIdx}
                        className="w-12 h-12 border border-gray-200 text-center align-middle text-xs font-medium truncate"
                        style={{
                          backgroundColor: bgColor,
                          color: Math.abs(value) > 0.5 ? "white" : "black",
                          maxWidth: "100px", 
                          overflow: "hidden", 
                          textOverflow: "ellipsis" 
                        }}
                        title={value !== null ? value.toFixed(2) : ""}
                      >
                        {value !== null ? value.toFixed(2) : cell}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}



      {/* Chart Generation */}
      <div className="space-y-4 bg-white p-6 rounded-xl shadow-md">
        <h3 className="font-semibold text-gray-800 text-xl mb-4">
          📉 Generate Chart
        </h3>

        <div className="flex flex-wrap items-center gap-4">
          {/* Chart Type */}
          <div className="flex flex-col text-sm">
            <label htmlFor="chartType" className="mb-1 text-gray-600 font-medium">Chart Type</label>
            <select
              id="chartType"
              value={chartForm.type}
              onChange={(e) => setChartForm({ ...chartForm, type: e.target.value })}
              className="border border-green-600 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-600"
            >
              <option value="bar">Bar</option>
              <option value="line">Line</option>
              <option value="scatter">Scatter</option>
              <option value="histogram">Histogram</option>
            </select>
          </div>

          {/* X Axis */}
          <div className="flex flex-col text-sm">
            <label htmlFor="xAxis" className="mb-1 text-gray-600 font-medium">X Axis</label>
            <select
              id="xAxis"
              value={chartForm.x}
              onChange={(e) => setChartForm({ ...chartForm, x: e.target.value })}
              className="border border-green-600 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-600"
            >
              <option value="">Select X</option>
              {columns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* Y Axis (if not histogram) */}
          {chartForm.type !== "histogram" && (
            <div className="flex flex-col text-sm">
              <label htmlFor="yAxis" className="mb-1 text-gray-600 font-medium">Y Axis</label>
              <select
                id="yAxis"
                value={chartForm.y}
                onChange={(e) => setChartForm({ ...chartForm, y: e.target.value })}
                className="border border-green-600 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-600"
              >
                <option value="">Select Y</option>
                {columns.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
          )}

          {/* Generate Button */}
          <div className="pt-5">
            <button
              onClick={handleGenerateChart}
              className="bg-green-600 text-white px-5 py-2 rounded-lg hover:bg-green-700 transition shadow-sm hover:cursor-pointer"
            >
              Generate
            </button>
          </div>
        </div>

        {/* Chart Result */}
        <div className="pt-6">
          {loading ? (
            <p className="text-sm text-gray-500">🔄 Generating chart...</p>
          ) : chart && (
            <div className="border rounded-lg p-4 bg-gray-50 shadow-inner">
              <img src={chart} alt="EDA chart" className="w-full max-h-[500px] object-contain" />
            </div>
          )}
        </div>
      </div>


      {/* EDA Reports Section */}
      <div className="space-y-4 bg-white p-6 rounded-xl shadow-md">
      <h3 className="font-semibold text-gray-800 text-xl mb-4">📁 EDA Reports</h3>

      {/* Nút Get Reports */}
      {!fetched && (
        <button
          onClick={handleGetReport}
          className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 hover:cursor-pointer"
          disabled={loadingReports}
        >
          {loadingReports ? 'Loading...' : 'Get Reports'}
        </button>
      )}

      {/* Hiển thị khi đã fetch */}
      {fetched && (
        <>
          <button
            onClick={handleDownloadAll}
            className="bg-white text-green-600 px-4 py-2 rounded hover:bg-green-600 hover:text-white border border-green-600 transition hover:cursor-pointer"
          >
            Download All
          </button>

          <ul className="divide-y divide-gray-200 border rounded-md mt-2">
            {reports.map((r, idx) => (
              <li key={idx} className="flex items-center justify-between p-3 hover:bg-gray-50">
                <span className="text-green-600 font-medium">{r.name}</span>
                <a
                  href={r.url}
                  download
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700"
                >
                  Download
                </a>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>

      
    </div>
  );
}
