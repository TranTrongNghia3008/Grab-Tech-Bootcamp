import { useState } from "react";
import DataTable from "../components/DataTable";
import MainLayout from "../layout/MainLayout";


const dataset = [
  { id: 1, name: 'Dataset A', createdAt: '2021-10-01', updatedAt: '2023-01-15' },
  { id: 2, name: 'Dataset B', createdAt: '2022-02-20', updatedAt: '2023-04-10' },
  { id: 3, name: 'Dataset C', createdAt: '2020-11-05', updatedAt: '2021-12-15' },
  { id: 4, name: 'Dataset D', createdAt: '2023-03-12', updatedAt: '2023-05-01' },
];

export default function DatasetsPage() {
  const [search, setSearch] = useState("");

  // Filter datasets based on search
  const filteredData = dataset.filter((row) =>
    row.name.toLowerCase().includes(search.toLowerCase()) ||
    row.id.toString().includes(search) ||
    row.createdAt.includes(search) ||
    row.updatedAt.includes(search)
  );

  return (
    <MainLayout>
        <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center mb-4 gap-3">
            <h1 className="text-2xl font-bold text-[#1B1F1D]">Datasets</h1>
            
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
        <DataTable data={filteredData} />

    </MainLayout>
  );
}
