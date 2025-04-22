import { useState } from "react";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";

export default function BaselineModelingPanel() {
    const [jobStatus, setJobStatus] = useState(null); // pending, running, done, error
    //   const [jobId, setJobId] = useState(null);
    const [metrics, setMetrics] = useState(null);
    const [importances, setImportances] = useState(null);
    const [loading, setLoading] = useState(false);

    //   const datasetId = 123; 

    const handleTrainBaseline = () => {
        setLoading(true);
        setJobStatus("pending");

        // 🔁 Giả lập API gọi baseline training
        setTimeout(() => {
        //   const fakeJobId = "job_" + Date.now();
        //   setJobId(fakeJobId);
        setJobStatus("running");

        // 🔁 Giả lập xử lý job và nhận kết quả
        setTimeout(() => {
            setJobStatus("done");
            setMetrics({
            mse: 0.23,
            r2: 0.81,
            accuracy: 0.92
            });
            setImportances([
            { feature: "age", importance: 0.45 },
            { feature: "income", importance: 0.33 },
            { feature: "gender", importance: 0.22 }
            ]);
            setLoading(false);
        }, 2000);
        }, 1000);
    };

    return (
        <div className="space-y-6">
            <h2 className="text-xl font-bold">Baseline Modeling</h2>
            <div className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm">
            📊 <span className="font-medium">Baseline Modeling</span> will automatically train a default model (e.g. Random Forest), evaluate it, and explain which features are most important.
            </div>
            
            {/* Train Button */}
            <div>
                <button
                onClick={handleTrainBaseline}
                disabled={loading || jobStatus === "running"}
                className={`px-5 py-2 rounded-md text-white transition ${
                    loading || jobStatus === "running"
                    ? "bg-green-400 cursor-wait"
                    : "bg-green-600 hover:bg-green-700"
                }`}
                >
                {loading || jobStatus === "running" ? "Training..." : "Train Baseline"}
                </button>
            </div>

            {/* Status */}
            {jobStatus && (
                <div className="flex items-center gap-2 text-sm text-gray-700">
                {jobStatus === "done" && <CheckCircle size={16} className="text-green-600" />}
                {jobStatus === "error" && <AlertCircle size={16} className="text-red-600" />}
                {jobStatus === "running" && <Loader2 size={16} className="animate-spin text-green-500" />}
                <span className="capitalize">Status: {jobStatus}</span>
                </div>
            )}

            {/* Metrics */}
            {metrics && (
                <div className="bg-white p-6 rounded-xl shadow-md">
                    <h3 className="text-xl font-semibold text-gray-800 mb-4">📈 Model Metrics</h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                    {Object.entries(metrics).map(([key, val]) => (
                        <div
                        key={key}
                        className="border border-gray-200 rounded-lg px-4 py-3 bg-green-50 hover:shadow transition"
                        >
                        <p className="text-xs text-gray-500 uppercase tracking-wide">{key}</p>
                        <p className="text-lg font-semibold text-green-700">
                            {typeof val === "number" ? val.toFixed(3) : val}
                        </p>
                        </div>
                    ))}
                    </div>
                </div>
                )}


            {/* Feature Importances */}
            {importances && (
                <div className="space-y-4 bg-white p-6 rounded-xl shadow-md overflow-auto">
                <h3 className="font-semibold text-gray-800 text-xl mb-4">🧠 Feature Importances</h3>
                <div className="space-y-2">
                    {importances.map((item, idx) => (
                    <div key={idx} className="text-sm">
                        <div className="flex justify-between mb-1">
                        <span className="text-gray-700">{item.feature}</span>
                        <span className="text-gray-600 font-medium">{(item.importance * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-2 bg-gray-200 rounded">
                        <div
                            className="h-2 bg-green-500 rounded"
                            style={{ width: `${item.importance * 100}%` }}
                        ></div>
                        </div>
                    </div>
                    ))}
                </div>
                </div>
            )}
        </div>
    );
}
