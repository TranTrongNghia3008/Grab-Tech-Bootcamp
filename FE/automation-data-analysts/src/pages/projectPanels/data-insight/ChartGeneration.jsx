import { useState, useEffect, useRef } from "react";
import { FiBarChart2 } from "react-icons/fi";
import { Loader2 } from "lucide-react";
import { Line, Bar, Scatter } from "react-chartjs-2";
import { Chart, registerables } from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";
import { Button, Card } from "../../../components/ui";
import { getChartColumns, getChartSummary } from "../../../components/services/chartServices";

Chart.register(...registerables, zoomPlugin);

const ChartGeneration = ({ datasetId, columns }) => {
  const chartRef = useRef(null);
  const [chartForm, setChartForm] = useState({
    type: "bar",
    x: "",
    y: "",
    bins: 10,
  });

  const [chartOptions, setChartOptions] = useState({
    borderColor: "#4BC0C0",
    backgroundColor: "rgba(75, 192, 192, 0.5)",
    pointRadius: 4,
    lineTension: 0.3,
  });

  const [xOptions, setXOptions] = useState([]);
  const [yOptions, setYOptions] = useState([]);
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState("");
  const [hasChanged, setHasChanged] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    const fetchColumns = async () => {
      setLoading("Loading available columns...");
      try {
        const res = await getChartColumns({ datasetId, chartType: chartForm.type });

        // Đồng bộ tên cột từ res với cột thật (columns gốc)
        const normalize = (str) => str.replace(/[_\s]/g, "").toLowerCase();
        const mapToOriginal = (colList) =>
          colList.map((col) => {
            const matched = columns.find((orig) => normalize(orig) === normalize(col));
            return matched || col;
          });

        const xMapped = mapToOriginal(res.x_columns || []);
        const yMapped = mapToOriginal(res.y_columns || []);

        setXOptions(xMapped);
        setYOptions(yMapped);
        setChartForm((prev) => ({ ...prev, x: "", y: "" }));
      } catch (err) {
        console.error("Error fetching columns:", err);
      } finally {
        setLoading("");
      }
    };
    fetchColumns();
  }, [chartForm.type, columns]);


  const handleGenerateChart = async () => {
    if (!chartForm.x || (chartForm.type !== "histogram" && !chartForm.y)) {
      setErrorMsg("Please select valid X and Y columns before generating the chart.");
      return;
    }

    setLoading("Generating chart...");
    setErrorMsg(""); // clear previous error

    try {
      const params = {
        datasetId,
        xColumn: chartForm.x,
      };
      if (chartForm.type === "histogram") params.bins = chartForm.bins;
      else params.yColumn = chartForm.y;

      const res = await getChartSummary(params);
      const labels = res.x;
      const data = res.y;

      const dataset = {
        label:
          chartForm.type === "histogram"
            ? `Frequency of ${chartForm.x}`
            : `${chartForm.y} vs ${chartForm.x}`,
        data:
          chartForm.type === "scatter"
            ? labels.map((x, i) => ({ x, y: data[i] }))
            : data,
        borderColor: chartOptions.borderColor,
        backgroundColor: chartOptions.backgroundColor,
        fill: chartForm.type === "line",
        tension: chartOptions.lineTension,
        pointRadius: chartForm.type === "scatter" ? chartOptions.pointRadius : undefined,
      };

      setChartData({
        labels: chartForm.type === "scatter" ? undefined : labels,
        datasets: [dataset],
      });
      setHasChanged(false); // mark chart as synced
    } catch (err) {
      console.error("Error generating chart:", err);
      setErrorMsg("Something went wrong while generating the chart.");
    } finally {
      setLoading("");
    }
  };

  const handleDownloadChart = () => {
    const chart = chartRef.current;
    if (chart) {
      const base64 = chart.toBase64Image();
      const link = document.createElement("a");
      link.href = base64;
      link.download = "chart.png";
      link.click();
    }
  };

  const ChartComponent = {
    bar: Bar,
    line: Line,
    scatter: Scatter,
    histogram: Bar,
  }[chartForm.type];

  return (
    <Card>
      <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
        <FiBarChart2 /> Generate Chart
      </h3>

      <div className="flex flex-wrap gap-4">
        {/* Chart Type */}
        <div className="flex flex-col text-sm">
          <label className="text-gray-600 font-medium mb-1">Chart Type</label>
          <select
            value={chartForm.type}
            onChange={(e) => {
              setChartForm({ type: e.target.value, x: "", y: "", bins: 10 });
              setChartData(null);
              setHasChanged(true);
            }}
            className="border border-green-600 rounded-md px-3 py-2 focus:ring-2 focus:ring-green-600"
          >
            <option value="bar">Bar</option>
            <option value="line">Line</option>
            <option value="scatter">Scatter</option>
            <option value="histogram">Histogram</option>
          </select>
        </div>

        {/* X Axis */}
        <div className="flex flex-col text-sm">
          <label className="text-gray-600 font-medium mb-1">X Axis</label>
          <select
            value={chartForm.x}
            onChange={(e) => {
              setChartForm({ ...chartForm, x: e.target.value });
              setChartData(null);
              setHasChanged(true);
            }}
            className="border border-green-600 rounded-md px-3 py-2"
          >
            <option value="">Select X</option>
            {xOptions
              .filter((col) => col !== chartForm.y)
              .map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
          </select>
        </div>

        {/* Y Axis */}
        {chartForm.type !== "histogram" && (
          <div className="flex flex-col text-sm">
            <label className="text-gray-600 font-medium mb-1">Y Axis</label>
            <select
              value={chartForm.y}
              onChange={(e) => {
                setChartForm({ ...chartForm, y: e.target.value });
                setChartData(null);
                setHasChanged(true);
              }}
              className="border border-green-600 rounded-md px-3 py-2"
            >
              <option value="">Select Y</option>
              {yOptions
                .filter((col) => col !== chartForm.x)
                .map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
            </select>
          </div>
        )}

        {/* Bins for Histogram */}
        {chartForm.type === "histogram" && (
          <div className="flex flex-col text-sm">
            <label className="text-gray-600 font-medium mb-1">Bins</label>
            <input
              type="number"
              min={2}
              max={50}
              value={chartForm.bins}
              onChange={(e) => {
                setChartForm({ ...chartForm, bins: Number(e.target.value) });
                setChartData(null);
                setHasChanged(true);
              }}
              className="border border-green-600 rounded-md px-3 py-2"
            />
          </div>
        )}

        {/* Chart Options */}
        <div className="flex flex-wrap gap-4 text-sm">
          <div className="flex flex-col">
            <label className="text-gray-600 font-medium mb-1">Color</label>
            <input
              type="color"
              value={chartOptions.borderColor}
              onChange={(e) => {
                setChartOptions({
                  ...chartOptions,
                  borderColor: e.target.value,
                  backgroundColor: `${e.target.value}80`,
                });
                setChartData(null);
                setHasChanged(true);
              }}
            />
          </div>

          {chartForm.type === "scatter" && (
            <div className="flex flex-col">
              <label className="text-gray-600 font-medium mb-1">Point Size</label>
              <input
                type="number"
                min={1}
                max={10}
                value={chartOptions.pointRadius}
                onChange={(e) => {
                  setChartOptions({ ...chartOptions, pointRadius: Number(e.target.value) });
                  setChartData(null);
                  setHasChanged(true);
                }}
                className="border border-green-600 rounded-md px-3 py-2"
              />
            </div>
          )}

          {chartForm.type === "line" && (
            <div className="flex flex-col">
              <label className="text-gray-600 font-medium mb-1">Line Tension</label>
              <input
                type="number"
                step="0.1"
                min={0}
                max={1}
                value={chartOptions.lineTension}
                onChange={(e) => {
                  setChartOptions({ ...chartOptions, lineTension: Number(e.target.value) });
                  setChartData(null);
                  setHasChanged(true);
                }}
                className="border border-green-600 rounded-md px-3 py-2"
              />
            </div>
          )}
        </div>

        {/* Generate + Download */}
        <div className="pt-6">
          <Button onClick={handleGenerateChart} disabled={!!loading}>
            {loading ? (
              <div className="flex items-center gap-2">
                <Loader2 className="animate-spin" /> Working...
              </div>
            ) : (
              "Generate"
            )}
          </Button>
        </div>

        {chartData && !hasChanged && (
          <div className="pt-6">
            <Button variant="outline" onClick={handleDownloadChart}>
              Download Chart
            </Button>
          </div>
        )}
      </div>
      {errorMsg && (
        <div className="text-red-600 text-sm mt-2">{errorMsg}</div>
      )}

      {/* Chart Result */}
      <div className="pt-6">
        {loading && (
          <div className="flex items-center gap-2 text-sm text-gray-700">
            <Loader2 size={16} className="animate-spin text-green-500" />
            <span>{loading}</span>
          </div>
        )}

        {hasChanged && !loading && (
          <p className="text-gray-400 italic ml-6">
            Click <strong>Generate Chart</strong> to create your chart.
          </p>
        )}

        {chartData && !loading && !hasChanged && (
          <div className="bg-gray-50 p-4 border rounded-lg shadow-inner">
            <ChartComponent
              ref={chartRef}
              data={chartData}
              options={{
                responsive: true,
                plugins: {
                  zoom: {
                    pan: { enabled: true, mode: "xy" },
                    zoom: {
                      wheel: { enabled: true },
                      pinch: { enabled: true },
                      mode: "xy"
                    }
                  }
                },
                scales:
                  chartForm.type === "scatter"
                    ? { x: { type: "linear", position: "bottom" } }
                    : {}
              }}
            />
          </div>
        )}
      </div>
    </Card>
  );
};

export default ChartGeneration;
