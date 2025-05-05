import { useEffect, useState } from "react";
import DataTable from "../../components/DataTable";
import { Button, Card } from "../../components/ui";
import { FaChartBar, FaProjectDiagram } from "react-icons/fa";
import { FiBarChart2, FiFileText } from "react-icons/fi";
import { Loader2 } from "lucide-react";
import ChartGeneration from "./ChartGeneration";
import { useAppContext } from "../../contexts/AppContext";
import { getCorrelation, getSummaryStatistics } from "../../components/services/EDAServices";

export default function DataInsightPanel() {
  const { state } = useAppContext(); 
  const { datasetId } = state;
  // const datasetId = 13; // Thay thế bằng datasetId thực tế từ context hoặc props

  const [stats, setStats] = useState(null);
  const [corr, setCorr] = useState(null);
  const [reports, setReports] = useState([]);
  const [loadingReports, setLoadingReports] = useState(false);
  const [fetched, setFetched] = useState(false); 

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Gọi API thống kê tổng quan
        const statsRes = await getSummaryStatistics(datasetId);
        const transformedData = Object.entries(statsRes).map(([key, value]) => ({
          column: key,
          ...value
      }));
        setStats(transformedData);
  
        // Gọi API correlation
        const corrRes = await getCorrelation(datasetId);
        setCorr(corrRes);
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
      } catch (error) {
        console.error("Error API:", error);
      }
    };
  
    fetchData();
  }, [datasetId]);
  


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

      {/* Summary Stats */}
      {stats && (
        <Card>
          <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
            <FaChartBar/>
            Summary Statistics
          </h3>
          <DataTable data={stats} />
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
        </Card>
      )}

      
      {/* Chart Generation */}
      <ChartGeneration/>

      {/* EDA Reports Section */}
      <Card>
        <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
          <FiFileText /> EDA Reports
        </h3>

        {/* Nút Get Reports */}
        {!fetched && (
          <Button
            onClick={handleGetReport}
            disabled={loadingReports}
          >
            {loadingReports ? 'Loading...' : 'Get Reports'}
          </Button>
        )}

        {/* Hiển thị khi đã fetch */}
        {fetched && (
          <>
            <Button
              onClick={handleDownloadAll}           
            >
              Download All
            </Button>

            <ul className="divide-y divide-gray-200 border rounded-md mt-2">
              {reports.map((r, idx) => (
                <li key={idx} className="flex items-center justify-between p-3 hover:bg-gray-50">
                  <span className="text-green-600 font-medium">{r.name}</span>
                  <Button
                    href={r.url}
                    download
                    target="_blank"
                    rel="noopener noreferrer"                  >
                    Download
                  </Button>
                </li>
              ))}
            </ul>
          </>
        )}
      </Card>

      
    </div>
  );
}