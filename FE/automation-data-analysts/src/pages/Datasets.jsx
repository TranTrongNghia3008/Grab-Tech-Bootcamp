import { useEffect, useState } from "react";
import DataTable from "../components/DataTable";
import MainLayout from "../layout/MainLayout";
import { FiDatabase } from "react-icons/fi";
import { getAllByCreation } from "../components/services/datasetService";

export default function DatasetsPage() {
  const [search, setSearch] = useState("");
  const [projects, setProjects] = useState([]);

  useEffect(() => {
      document.title = "DataMate - Dataset";
    }, []);

  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const data = await getAllByCreation();
        console.log("Fetched projects:", data);
        setProjects(data.datasets); 
      } catch (error) {
        console.error("Failed to fetch projects:", error);
      }
    };

    fetchProjects();
  }, []);

  // Filter datasets based on search
  const filteredData = projects.filter((row) =>
    row.project_name.toLowerCase().includes(search.toLowerCase()) ||
    row.id.toString().includes(search) ||
    row.createdAt.includes(search)
  );

  const displayData = filteredData.map((item) => ({
    ID: item.id,
    Dataset: item.project_name,
    Created: new Date(item.created_at).toLocaleString("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
    }),
  }));

  return (
    <MainLayout>
        <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center mb-4 gap-3">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold text-[#1B1F1D] flex items-center gap-2">
            <FiDatabase className="text-green-600" size={24} />
            Datasets
          </h1>
          <div className="w-8 h-8 rounded-md bg-green-100 text-green-700 flex items-center justify-center text-sm font-semibold">
            {filteredData.length}
          </div>
        </div>
            
            {/* Search input */}
            <input
                type="text"
                className="w-full sm:max-w-sm p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-600 text-sm mb-4"
                placeholder="Search dataset..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
            />
        </div>
        
        {/* Data Table */}
        <DataTable data={displayData} />

    </MainLayout>
  );
}
