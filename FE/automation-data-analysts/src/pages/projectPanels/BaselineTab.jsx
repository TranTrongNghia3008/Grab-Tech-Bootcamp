import { useState } from "react";
import { FaBalanceScale } from "react-icons/fa";
import { Button, Card } from "../../components/ui";
import DataTable from "../../components/DataTable";
import AnalyzeModel from "./AnalyzeModel";
import CompareModels from "./CompareModels";

export default function BaselineTab({ comparisonResults = null, sessionId = 1, bestModel, modelPerformanceAnalysis, loadingModelPerformanceAnalysis, handleFetchModelPerformanceAnalysis }) {
  const [showComparison, setShowComparison] = useState(false);

  return (
    <div className="space-y-8">
      {comparisonResults && (
        <>
          <div className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-xl text-green-900 shadow-sm">
            <p>
              We ran a series of powerful algorithms behind the scenes — and <strong>{bestModel?.Model}</strong> outperformed them all. Want to uncover why it rose above the rest? 
              <button
                className="text-blue-600 underline hover:cursor-pointer ms-1"
                onClick={() => setShowComparison(true)}
              >
                Click here to find out.
              </button>
            </p>

          </div>

          {comparisonResults && (
            <>
              {showComparison && (
                <Card className="space-y-4">
                  <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
                    <FaBalanceScale />
                    Model Comparison
                  </h3>
                  <DataTable data={comparisonResults} />
                  <div className="flex justify-between bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm">
                    <p className="me-5 my-auto">
                      Not all models are created equal - discover which ones deliver both speed and accuracy, and which might be slowing you down                    </p>
                    <Button
                      onClick={handleFetchModelPerformanceAnalysis}
                      disabled={loadingModelPerformanceAnalysis}
                    >
                      {loadingModelPerformanceAnalysis ? "Analyzing..." : "Explore"}
                    </Button>
                  </div>
                  {modelPerformanceAnalysis && (
                    <div
                      className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm"
                      dangerouslySetInnerHTML={{ __html: modelPerformanceAnalysis }}
                    />
                  )}
                </Card>
              )}

              <AnalyzeModel availableModels={comparisonResults} sessionId={sessionId} />
              <CompareModels models={comparisonResults} />
            </>
          )}
        </>
      )}
    </div>
  );
}
