import { useState } from "react";
import { Loader2, Download } from "lucide-react";
import { SiConvertio } from "react-icons/si";
import { Button, Card } from "../components/ui";
import UploadZipDropzone from "../components/UploadZipDropzone";
import MainLayout from "../layout/MainLayout";

export default function ExtractTable() {
  const [zipFile, setZipFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [csvBlobUrl, setCsvBlobUrl] = useState(null);
  const [error, setError] = useState("");

  const handleUpload = async () => {
    if (!zipFile) return;

    const formData = new FormData();
    formData.append("zip_file", zipFile);

    try {
      setLoading(true);
      setCsvBlobUrl(null);
      setError("");

      const res = await fetch("http://localhost:5000/extract-table", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || "Failed to extract ZIP");
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setCsvBlobUrl(url);
    } catch (err) {
      console.error(err);
      setError("❌ Extraction failed. Please try again or check your file.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setZipFile(null);
    setCsvBlobUrl(null);
    setError("");
  };

  return (
    <MainLayout>
      <h1 className="text-2xl font-bold text-[#1B1F1D] flex items-center gap-2 mb-4">
        <SiConvertio className="text-green-600" size={24} />
        Extract Table
      </h1>

      <div className="bg-green-50 border border-green-200 mb-4 px-4 py-3 rounded-md text-sm text-green-900 shadow-sm">
        Upload a ZIP file containing image snippets of the same table. The system will automatically extract the table data using AI, merge the results, and provide a downloadable CSV file.
      </div>

      <Card>
        <h2 className="text-xl font-bold text-gray-800 mb-4">Upload your images data</h2>

        {!zipFile ? (
          <UploadZipDropzone
            onFileAccepted={(file) => {
              setZipFile(file);
              setError("");
            }}
          />
        ) : (
          <div className="flex items-center justify-between px-4 py-3 bg-green-100 border border-green-200 rounded-md text-sm text-green-800">
            <span>📦 ZIP uploaded: {zipFile.name}</span>
            <button
              onClick={handleReset}
              className="text-xs text-red-500 hover:underline ml-4"
            >
              Remove
            </button>
          </div>
        )}

        {error && (
          <p className="text-red-600 text-sm mt-2 font-medium">{error}</p>
        )}

        <div className="mt-4 flex items-center gap-4">
          <Button onClick={handleUpload} disabled={loading || !zipFile}>
            {loading ? (
              <div className="flex items-center gap-2">
                <Loader2 className="animate-spin" />
                Extracting...
              </div>
            ) : (
              "Extract ZIP"
            )}
          </Button>

          {csvBlobUrl && (
            <a
              href={csvBlobUrl}
              download="extracted_table.csv"
              className="inline-flex items-center gap-2 text-sm text-blue-600 hover:underline"
            >
              <Download size={16} />
              Download Extracted CSV
            </a>
          )}
        </div>
      </Card>
    </MainLayout>
  );
}
