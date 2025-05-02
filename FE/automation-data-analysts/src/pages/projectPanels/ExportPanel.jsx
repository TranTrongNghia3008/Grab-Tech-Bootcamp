import { useState } from "react";
import { FileDown, FileCode, FileArchive } from "lucide-react";
import { Button, Card } from "../../components/ui";

export default function ExportPanel() {
  const jobId = "job_123"; // Replace with actual job ID from context/state
  const [format, setFormat] = useState("csv");
  const [language, setLanguage] = useState("python");

  // Loading states
  const [loadingPred, setLoadingPred] = useState(false);
  const [loadingArtifacts, setLoadingArtifacts] = useState(false);
  const [loadingCode, setLoadingCode] = useState(false);

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
    </div>
  );
}
