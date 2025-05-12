import { useEffect, useState } from "react";
import { FileDown, FileCode, FileArchive, Download } from "lucide-react";
import { MdOutlineModelTraining } from "react-icons/md";
import { Button, Card } from "../../components/ui";
import { downLoadFinalizedModel, getListFinalizedModels } from "../../components/services/modelingServices";
import { useAppContext } from "../../contexts/AppContext";

export default function ExportPanel() {
  const { state } = useAppContext();
  const { datasetId } = state;
  const jobId = "job_123"; // Replace with actual job ID from context/state
  const [format, setFormat] = useState("csv");
  const [language, setLanguage] = useState("python");

  // Loading states
  const [loadingPred, setLoadingPred] = useState(false);
  const [loadingArtifacts, setLoadingArtifacts] = useState(false);
  const [loadingCode, setLoadingCode] = useState(false);
  const [finalizedModels, setFinalizedModels] = useState([]);
  const [loadingModels, setLoadingModels] = useState(false);

  useEffect(() => {
    async function fetchModels() {
      setLoadingModels(true);
      try {
        const res = await getListFinalizedModels(datasetId); 
        console.log(res)
        setFinalizedModels(res);
      } catch (err) {
        console.error("Error fetching finalized models:", err);
      } finally {
        setLoadingModels(false);
      }
    }

    fetchModels();
  }, [datasetId]);

  const handleDownloadFinalizedModel = async (id, model_name) => {
    try {
      const blob = await downLoadFinalizedModel(id);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `finalized_model_${model_name}.pkl`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Download failed:", err);
    }
  };


  const handleDownloadPredictions = () => {
    setLoadingPred(true);
    setTimeout(() => {
      const link = document.createElement("a");
      link.href = `/mock/models/${jobId}/predictions.${format}`;
      link.download = `predictions.${format}`;
      link.click();
      setLoadingPred(false);
    }, 1000);
  };

  const handleDownloadArtifacts = () => {
    setLoadingArtifacts(true);
    setTimeout(() => {
      const link = document.createElement("a");
      link.href = `/mock/models/${jobId}/model_artifacts.zip`;
      link.download = "model_artifacts.zip";
      link.click();
      setLoadingArtifacts(false);
    }, 1000);
  };

  const handleDownloadCode = () => {
    setLoadingCode(true);
    setTimeout(() => {
      const link = document.createElement("a");
      link.href = `/mock/models/${jobId}/code-snippet.${language}`;
      link.download = `code-snippet.${language}`;
      link.click();
      setLoadingCode(false);
    }, 1000);
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Export Results</h2>

      {/* Intro */}
      <div className="bg-green-50 border border-green-200 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm">
        Export your model's <strong>predictions</strong>, <strong>trained artifacts</strong>, and <strong>code snippet</strong> for reuse or deployment.
      </div>

      {/* Panel */}
      <Card>
        {/* Predictions */}
        <div>
          <h3 className="font-semibold mb-2 text-gray-800">📈 Download Predictions</h3>
          <div className="flex flex-wrap items-center gap-3">
            <select
              value={format}
              onChange={(e) => setFormat(e.target.value)}
              className="border border-gray-300 rounded px-3 py-2 text-sm"
            >
              <option value="csv">CSV</option>
              <option value="json">JSON</option>
            </select>
            <Button
              onClick={handleDownloadPredictions}
              disabled={loadingPred}
            >
              <FileDown size={16} />
              {loadingPred ? "Preparing..." : "Download"}
            </Button>
          </div>
        </div>

        {/* Artifacts */}
        <div>
          <h3 className="font-semibold mb-2 text-gray-800">📦 Download Model Artifacts</h3>
          <Button
            onClick={handleDownloadArtifacts}
            disabled={loadingArtifacts}
          >
            <FileArchive size={16} />
            {loadingArtifacts ? "Preparing..." : "Download Artifacts"}
          </Button>
        </div>

        {/* Code Snippet */}
        <div>
          <h3 className="font-semibold mb-2 text-gray-800">💻 Download Code Snippet</h3>
          <div className="flex flex-wrap items-center gap-3">
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="border border-gray-300 rounded px-3 py-2 text-sm"
            >
              <option value="python">Python</option>
              {/* <option value="r">R</option> */}
            </select>
            <Button
              onClick={handleDownloadCode}
              disabled={loadingCode}
            >
              <FileCode size={16} />
              {loadingCode ? "Preparing..." : "Download Code"}
            </Button>
          </div>
        </div>
      </Card>
      {/* Finalized Models List */}
      <Card>
        <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
          <MdOutlineModelTraining />
          Finalized Models
        </h3>

        {loadingModels ? (
          <p className="text-sm text-gray-500">Loading models...</p>
        ) : finalizedModels.length === 0 ? (
          <p className="text-sm text-gray-500">No finalized models available.</p>
        ) : (
          <div className="border rounded-md overflow-hidden">
            <table className="min-w-full text-sm text-gray-700">
              <thead className="bg-gray-100 border-b">
                <tr>
                  <th className="text-left px-4 py-2 font-medium">Model Name</th>
                  <th className="text-left px-4 py-2 font-medium">Created At</th>
                  <th className="text-left px-4 py-2 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {finalizedModels.map((model) => (
                  <tr key={model.id} className="border-t">
                    <td className="px-4 py-2">{model.model_name}</td>
                    <td className="px-4 py-2">
                      {new Date(model.created_at).toLocaleString("en-GB", {
                        hour: "2-digit",
                        minute: "2-digit",
                        day: "2-digit",
                        month: "2-digit",
                        year: "numeric",
                        timeZone: "Asia/Ho_Chi_Minh"
                      })}
                    </td>
                    <td className="px-4 py-2">
                      <button
                        onClick={() => handleDownloadFinalizedModel(model.id, model.model_name)}
                        className="text-green-600 hover:text-green-800 flex items-center gap-1 text-sm"
                      >
                        <Download size={16} /> Download
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

    </div>
  );
}
