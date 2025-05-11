import { useEffect, useState } from "react";
import DataTable from "../../components/DataTable";
import { Button, Card } from "../../components/ui";
import { FaChartBar, FaProjectDiagram, FaListAlt, FaEye } from "react-icons/fa";
import { FiBarChart2, FiFileText, FiDatabase } from "react-icons/fi";
import { ImDatabase } from "react-icons/im";
import { FaChartLine } from "react-icons/fa6";
import { IoIosWarning } from "react-icons/io";
import ChartGeneration from "./ChartGeneration";
import { useAppContext } from "../../contexts/AppContext";
import { getCorrelation, getSummaryStatistics } from "../../components/services/EDAServices";
import { getAICorrelationMatrix, getAISummaryStatistics } from "../../components/services/aisummaryServices";
import { parseAISummary } from "../../utils/parseHtml";
import { getAnalysisReport } from "../../components/services/datasetService";

export default function DataInsightPanel() {
  const { state } = useAppContext(); 
  const { datasetId } = state;

  const [stats, setStats] = useState(null);
  const [corr, setCorr] = useState(null);
  const [corrForSummary, setCorrForSummary] = useState(null)
  const [aiSummaryStatsRes, setAISummaryStatsRes] = useState(null);
  const [loadingAISummaryStats, setLoadingAISummaryStats] = useState(false);
  const [aiCorrelationMatrix, setAICorrelationMatrix] = useState(null);
  const [loadingAICorrelationMatrix, setLoadingCorrelationMatrix] = useState(false);
  const [featureData, setFeatureData] = useState([]);
  const [overview, setOverview] = useState(null); // cho tổng quan


  useEffect(() => {
    const fetchData = async () => {
      try {
        // Gọi API thống kê tổng quan
        const statsRes = await getSummaryStatistics(datasetId);
        console.log("Summary Statistics: ", statsRes)
        const transformedData = Object.entries(statsRes).map(([key, value]) => ({
            column: key,
            ...value
        }));
        console.log(transformedData)
        setStats(transformedData);
  
        // Gọi API correlation
        const corrRes = await getCorrelation(datasetId);
        setCorrForSummary(corrRes);
        console.log("Correlation data:", corrRes);

        const labels = Object.keys(corrRes);
        const transformedCorr = [];

        // Thêm dòng đầu tiên (tiêu đề cột)
        transformedCorr.push(["", ...labels]);

        // Thêm từng dòng dữ liệu
        for (const rowLabel of labels) {
            const row = [rowLabel];
            for (const colLabel of labels) {
                row.push(corrRes[rowLabel][colLabel]);
            }
            transformedCorr.push(row);
        }
        setCorr(transformedCorr);  
        console.log(transformedCorr)
      } catch (error) {
        console.error("Error API:", error);
      }
    };
  
    fetchData();
  }, [datasetId]);

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        const res = await getAnalysisReport(datasetId);
        setOverview({
          totalRecords: res.total_records,
          dataQuality: res.data_quality_score,
          missingPercentage: res.overall_missing_percentage,
        });

        const features = res.features.map((f) => ({
          "Feature Name": f.feature_name,
          "Type": f.dtype.charAt(0).toUpperCase() + f.dtype.slice(1),
          "Missing Values": `${f.missing_percentage}%`,
          "Unique Values": `${f.unique_percentage}%`,
          "Actions": (
            <FaEye
              className="text-gray-600 hover:text-green-600 cursor-pointer mx-auto"
              onClick={() => handleViewFeature(f.feature_name)}
            />
          ),
        }));

        setFeatureData(features);
      } catch (err) {
        console.error("Failed to fetch analysis report:", err);
      }
    };

    fetchAnalysis();
  }, [datasetId]);


  const handleViewFeature = (featureName) => {
    console.log("Viewing details for:", featureName);
    // hoặc mở modal, fetch data,...
  };

  
  const handleFetchAISummaryStatistics = async () => {
    setLoadingAISummaryStats(true);
    try {
      const res = await getAISummaryStatistics(stats);
      
      setAISummaryStatsRes(parseAISummary(res.summary_html)); 
    } catch (err) {
      console.error("Failed to fetch summary stats:", err);
    } finally {
      setLoadingAISummaryStats(false);
    }
  };
  
  const handleFetchAICorrelationMatrix = async () => {
    setLoadingCorrelationMatrix(true);
    try {
      const res = await getAICorrelationMatrix(corrForSummary);
      
      setAICorrelationMatrix(parseAISummary(res.summary_html)); 
    } catch (err) {
      console.error("Failed to fetch summary stats:", err);
    } finally {
      setLoadingCorrelationMatrix(false);
    }
  };

  // Hàm tính toán giá trị tương quan cao nhất
  const getMaxCorrelation = () => {
    let max = -Infinity;
    corr.forEach(row => {
      row.forEach(cell => {
        if (typeof cell === "number" && cell > max && cell !== 1.0) { // Loại bỏ giá trị 1 (tương quan với chính nó)
          max = cell;
        }
      });
    });
    return max.toFixed(3);
  };

  // Hàm tính toán giá trị tương quan thấp nhất
  const getMinCorrelation = () => {
    let min = Infinity;
    corr.forEach(row => {
      row.forEach(cell => {
        if (typeof cell === "number" && cell < min && cell !== 1.0) { // Loại bỏ giá trị 1 (tương quan với chính nó)
          min = cell;
        }
      });
    });
    return min.toFixed(3);
  };

  // Hàm tính toán giá trị trung bình của ma trận tương quan
  const getAvgCorrelation = () => {
    let sum = 0;
    let count = 0;
    corr.forEach(row => {
      row.forEach(cell => {
        if (typeof cell === "number" && cell !== 1.0) { // Loại bỏ giá trị 1 (tương quan với chính nó)
          sum += cell;
          count++;
        }
      });
    });
    return (sum / count).toFixed(3);
  };


  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Exploratory Data Analysis</h2>

      {overview && (
        <Card>
          <h3 className="text-gray-800 text-xl flex items-center gap-2">
            <FiDatabase />
            Dataset Overview
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Total Records */}
            <Card className="bg-gray-100">
              <div className="flex justify-between text-gray-500 mb-3">
                <p className="text-sm font-bold mb-1">Total Records</p>
                <ImDatabase />
              </div>
              <p className="text-2xl font-bold text-gray-800 mb-1">
                {overview.totalRecords.toLocaleString()}
              </p>
              <p className="text-xs text-gray-600 mt-1">Latest updated</p>
            </Card>

            {/* Data Quality Score */}
            <Card className="bg-gray-100">
              <div className="flex justify-between text-gray-500 mb-3">
                <p className="text-sm font-bold mb-1">Data Quality Score</p>
                <FaChartLine />
              </div>
              <p className="text-2xl font-bold text-gray-800 mb-1">
                {overview.dataQuality}%
              </p>
              <p className="text-xs text-gray-600 mt-1">Overall score</p>
            </Card>

            {/* Missing Values */}
            <Card className="bg-gray-100">
              <div className="flex justify-between text-gray-500 mb-3">
                <p className="text-sm font-bold mb-1">Missing Values</p>
                <IoIosWarning />
              </div>
              <p className="text-2xl font-bold text-gray-800 mb-1">
                {overview.missingPercentage}%
              </p>
              <p className="text-xs text-gray-600 mt-1">Missing rate</p>
            </Card>
          </div>
        </Card>
      )}


      {featureData && (
        <Card>
          <h3 className="text-gray-800 text-xl flex items-center gap-2">
            <FaListAlt />
            Feature Analysis</h3>
          <DataTable data={featureData} />
        </Card>
      )}

      {/* Summary Stats */}
      {stats && (
        <Card>
          <h3 className="text-gray-800 text-xl flex items-center gap-2">
            <FaChartBar />
            Summary Statistics
          </h3>
          <DataTable data={stats} />
          <div className="flex justify-between bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm">
            <p className="me-5 my-auto">
              We've compiled a concise statistical summary from your data - uncover the key insights hidden beneath the surface.
            </p>
            <Button
              onClick={handleFetchAISummaryStatistics}
              disabled={loadingAISummaryStats}
            >
              {loadingAISummaryStats ? "Analyzing..." : "Explore"}
            </Button>
          </div>
          {aiSummaryStatsRes && (
            <div
              className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm"
              dangerouslySetInnerHTML={{ __html: aiSummaryStatsRes }}
            />
          )}
        </Card>

      )}

      {/* Correlation Matrix */}
      {corr && (
        <Card>
          <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
            <FaProjectDiagram/>
            Correlation Matrix
          </h3>
          
          <div className="flex flex-col md:flex-row gap-6">
            {/* Left side: Description */}
            <div className="w-full md:w-1/4 text-gray-700 text-sm leading-relaxed space-y-3 flex flex-col justify-center">
              <p><strong>Highest correlation:</strong> {getMaxCorrelation()}</p>
              <p><strong>Lowest correlation:</strong> {getMinCorrelation()}</p>
              <p><strong>Average correlation:</strong> {getAvgCorrelation()}</p>

              {/* Legend */}
              <div className="flex items-center gap-4 mt-4">
                <div className="flex items-center gap-1">
                  <span className="inline-block w-4 h-4 rounded" style={{ backgroundColor: "rgba(0,132,61,1)" }}></span>
                  <span>High correlation</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="inline-block w-4 h-4 rounded" style={{ backgroundColor: "rgba(0,132,61,0.05)" }}></span>
                  <span>Low correlation</span>
                </div>
              </div>
            </div>

            {/* Right side: Correlation Matrix */}
            <div className="flex-1">
              <table className="border border-gray-200 text-sm table-auto w-full">
                <tbody>
                  {corr.map((row, rowIdx) => (
                    <tr key={rowIdx}>
                    {row.map((cell, colIdx) => {
                      const value = typeof cell === "number" ? cell : null;

                      // Tăng sắc độ: chuẩn hóa alpha trong khoảng [0.05, 1]
                      let bgColor = "#f9fafb"; // màu header
                      if (value !== null) {
                        const absVal = Math.abs(value);
                        const minAlpha = 0.05; // tránh màu quá nhạt
                        const maxAlpha = 1;
                        const adjustedAlpha = minAlpha + (maxAlpha - minAlpha) * absVal;
                        bgColor = `rgba(0, 132, 61, ${adjustedAlpha})`;
                      }

                      return (
                        <td
                          key={colIdx}
                          className="w-12 h-12 border border-gray-200 text-center align-middle text-xs font-medium truncate"
                          style={{
                            backgroundColor: bgColor,
                            color: value !== null && Math.abs(value) > 0.5 ? "white" : "black",
                            maxWidth: "100px",
                            overflow: "hidden",
                            textOverflow: "ellipsis"
                          }}
                          title={value !== null ? value.toFixed(3) : ""}
                        >
                          {value !== null ? value.toFixed(3) : cell}
                        </td>
                      );
                    })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="flex justify-between bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm">
            <p className="me-5 my-auto">
            Explore how your variables move together - some relationships align with expectations, while others may surprise you.
            </p>
            <Button
              onClick={handleFetchAICorrelationMatrix}
              disabled={loadingAICorrelationMatrix}
            >
              {loadingAICorrelationMatrix ? "Analyzing..." : "Explore"}
            </Button>
          </div>
          {aiCorrelationMatrix && (
            <div
              className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm"
              dangerouslySetInnerHTML={{ __html: aiCorrelationMatrix }}
            />
          )}
        </Card>
      )}

      
      {/* Chart Generation */}
      <ChartGeneration/>
    </div>
  );
}