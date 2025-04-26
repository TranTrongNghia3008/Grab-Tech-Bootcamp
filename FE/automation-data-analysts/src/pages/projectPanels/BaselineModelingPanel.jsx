import { useState } from "react";
import { FaBalanceScale } from "react-icons/fa";
import { Button, Card } from "../../components/ui";
import DataTable from "../../components/DataTable";
import SelectTargetFeaturesModel from "./SelectTargetFeatures";
import AnalyzeModel from "./AnalyzeModel";
import CompareModels from "./CompareModels";

export default function BaselineModelingPanel() {
    const [target, setTarget] = useState("");
    const [features, setFeatures] = useState([]);
    const [jobStatus, setJobStatus] = useState(null);
    const [comparisonResults, setComparisonResults] = useState([]);
    const [loading, setLoading] = useState(false);

    const availableColumns = ["age", "income", "gender", "education", "experience"];

    const handleTrainBaseline = () => {
    if (!target || features.length === 0) {
        alert("Please select target and feature columns.");
        return;
    }

    setLoading(true);
    setJobStatus("pending");

    setTimeout(() => {
        setJobStatus("running");

        setTimeout(() => {
        const tempComparisonResults = [
            { modelId: "lr", modelName: "Logistic Regression", Accuracy: 0.8209, AUC: 0.855, Recall: 0.6699, Precision: 0.8313, F1: 0.7419, Kappa: 0.6072, MCC: 0.6155 },
            { modelId: "ridge", modelName: "Ridge Classifier", Accuracy: 0.7528, AUC: 0.8647, Recall: 0.6023, Precision: 0.7521, F1: 0.6755, Kappa: 0.4273, MCC: 0.4679 },
            { modelId: "et", modelName: "Extra Trees", Accuracy: 0.7400, AUC: 0.7837, Recall: 0.5555, Precision: 0.7233, F1: 0.6244, Kappa: 0.4088, MCC: 0.4356 },
            { modelId: "nb", modelName: "Naive Bayes", Accuracy: 0.6709, AUC: 0.7925, Recall: 0.5011, Precision: 0.6322, F1: 0.5599, Kappa: 0.1808, MCC: 0.2747 },
            { modelId: "knn", modelName: "K Neighbors", Accuracy: 0.6275, AUC: 0.5906, Recall: 0.4122, Precision: 0.5433, F1: 0.4671, Kappa: 0.1654, MCC: 0.1713 }
        ]
        setComparisonResults(tempComparisonResults);
        
        setJobStatus("done");
        setLoading(false);
        localStorage.setItem("comparisonResults", JSON.stringify(tempComparisonResults));
        localStorage.setItem("best_model_id", "ridge");
        console.log("Comparison results saved to localStorage:", tempComparisonResults);
        }, 2000);
    }, 1000);
    
    };



    return (
        <div className="space-y-8">
            <h2 className="text-xl font-bold">Baseline Modeling</h2>

            <div className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900">
            Train default baseline models (Random Forest, Logistic Regression...), compare their performance and explain feature importance.
            </div>

            {/* Select Target & Features */}
            <SelectTargetFeaturesModel
            availableColumns={availableColumns}
            target={target}
            setTarget={setTarget}
            features={features}
            setFeatures={setFeatures}
            handleTrain={handleTrainBaseline}
            loading={loading}
            jobStatus={jobStatus}
            trainLabel="Train Baseline"
            showModelSelection={false}
        />


            
            {/* Model Comparison */}
            {comparisonResults.length > 0 && (
            <Card className="space-y-4">
                <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
                    <FaBalanceScale/>
                    Model Comparison
                </h3>
                <DataTable data={comparisonResults} />
            </Card>
            )}

            {/* Analyze Model */}
            {comparisonResults.length > 0 && (
                <AnalyzeModel
                availableModels={comparisonResults}
                />
            )}


            {/* Compare Multiple Models */}
            {comparisonResults.length > 0 && (
            <CompareModels models={comparisonResults} />
            )}
        </div>
    );
}
