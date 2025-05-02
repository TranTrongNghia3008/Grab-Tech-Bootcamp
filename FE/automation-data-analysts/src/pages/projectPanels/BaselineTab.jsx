import { useEffect, useState } from "react";
import { FaBalanceScale } from "react-icons/fa";
import { Card } from "../../components/ui";
import DataTable from "../../components/DataTable";
import AnalyzeModel from "./AnalyzeModel";
import CompareModels from "./CompareModels";

export default function BaselineTab() {
  const [comparisonResults, setComparisonResults] = useState([]);

  useEffect(() => {
    const temp = localStorage.getItem("comparisonResults");
    if (temp) {
      setComparisonResults(JSON.parse(temp));
    }
  }, []);

  return (
    <div className="space-y-8">
      {comparisonResults.length > 0 && (
        <>
          <Card className="space-y-4">
            <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
              <FaBalanceScale />
              Model Comparison
            </h3>
            <DataTable data={comparisonResults} />
          </Card>

          <AnalyzeModel availableModels={comparisonResults} />
          <CompareModels models={comparisonResults} />
        </>
      )}
    </div>
  );
}
