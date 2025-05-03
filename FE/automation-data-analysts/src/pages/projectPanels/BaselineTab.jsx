import { FaBalanceScale } from "react-icons/fa";
import { Card } from "../../components/ui";
import DataTable from "../../components/DataTable";
import AnalyzeModel from "./AnalyzeModel";
import CompareModels from "./CompareModels";

export default function BaselineTab({ comparisonResults = [], sessionId = 1 }) {
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

          <AnalyzeModel availableModels={comparisonResults} sessionId={sessionId}/>
          <CompareModels models={comparisonResults} />
        </>
      )}
    </div>
  );
}
