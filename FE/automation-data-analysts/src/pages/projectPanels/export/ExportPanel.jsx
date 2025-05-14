import { useEffect, useState } from "react";
import { FileDown, FileCode, FileArchive, Download } from "lucide-react";
import { MdOutlineModelTraining } from "react-icons/md";
import { Button, Card } from "../../../components/ui";
import { downLoadFinalizedModel, getListFinalizedModels } from "../../../components/services/modelingServices";
import { useAppContext } from "../../../contexts/AppContext";
import { getDataProfile } from "../../../components/services/EDAServices";

export default function ExportPanel() {
  const { state } = useAppContext();
  const { datasetId } = state;
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

  const handleDownloadProfile = async () => {
    try {
      const blob = await getDataProfile(datasetId);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `data_profile_${datasetId}.html`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Download profile failed:", err);
    }
  };


  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Export Results</h2>

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

      {/* Data Profile Export */}
      <Card>
        <h3 className="text-gray-800 text-xl mb-4 flex items-center gap-2">
          <FileCode />
          Export Data Profile
        </h3>
        <p className="text-sm text-gray-500 mb-3">
          Download the dataset profile summary.
        </p>
        <Button
          onClick={handleDownloadProfile}
          className="flex items-center gap-2"
        >
          <FileDown size={16} />
          Download Profile
        </Button>
      </Card>

    </div>
  );
}
