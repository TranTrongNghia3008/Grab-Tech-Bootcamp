import { useState } from "react";
import { useNavigate } from "react-router-dom";
import MainLayout from "../layout/MainLayout";
import { Button, Card, Modal } from "../components/ui";
import { ArrowDownAZ, ArrowUpAZ, Calendar, Pencil, Trash2 } from "lucide-react";

export default function Projects() {
  const navigate = useNavigate();

  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState("updatedAt");
  const [sortDir, setSortDir] = useState("desc");
  const [projects, setProjects] = useState([
    { id: 1, name: "Customer Segmentation", updatedAt: "2024-04-20T10:00:00Z" },
    { id: 2, name: "Sales Forecasting", updatedAt: "2024-04-15T14:30:00Z" },
    { id: 3, name: "Churn Prediction", updatedAt: "2024-04-18T09:20:00Z" },
    { id: 4, name: "Product Recommendation", updatedAt: "2024-04-17T11:45:00Z" },
    { id: 5, name: "Market Basket Analysis", updatedAt: "2024-04-14T08:10:00Z" },
    { id: 6, name: "Ad Click Prediction", updatedAt: "2024-04-13T16:00:00Z" },
    { id: 7, name: "Social Media Sentiment", updatedAt: "2024-04-16T13:35:00Z" },
    { id: 8, name: "Loan Default Risk", updatedAt: "2024-04-12T10:30:00Z" },
    { id: 9, name: "Energy Consumption Forecast", updatedAt: "2024-04-11T07:15:00Z" },
    { id: 10, name: "Fraud Detection", updatedAt: "2024-04-10T17:25:00Z" }
  ]);

  const [showRenameModal, setShowRenameModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [selectedProject, setSelectedProject] = useState(null);
  const [newName, setNewName] = useState("");

  const handleCreateProject = () => navigate("/projects/create");

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const filteredProjects = projects
    .filter((p) => p.name.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      const valA = a[sortKey];
      const valB = b[sortKey];
      if (sortKey === "updatedAt") {
        return sortDir === "asc"
          ? new Date(valA) - new Date(valB)
          : new Date(valB) - new Date(valA);
      }
      return sortDir === "asc"
        ? String(valA).localeCompare(String(valB))
        : String(valB).localeCompare(String(valA));
    });

  return (
    <MainLayout>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center mb-4 gap-3">
        <h1 className="text-2xl font-bold text-[#1B1F1D]">📁 My Projects</h1>
        <Button onClick={handleCreateProject}>+ New Project</Button>
      </div>

      {/* Search & Sort */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <input
          type="text"
          className="w-full sm:max-w-sm p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-600 text-sm"
          placeholder="Search project..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <div className="flex items-center gap-3 text-sm text-gray-600">
          <span className="font-medium">Sort by:</span>

          <button
            className={`flex items-center gap-1 px-3 py-1.5 rounded-md border transition
              ${sortKey === "name"
                ? "bg-green-100 text-green-700 border-green-300"
                : "border-transparent hover:bg-green-50"} 
              hover:cursor-pointer`}
            onClick={() => handleSort("name")}
          >
            Name
            {sortKey === "name" &&
              (sortDir === "asc" ? (
                <ArrowDownAZ size={14} className="text-green-600" />
              ) : (
                <ArrowUpAZ size={14} className="text-green-600" />
              ))}
          </button>

          <button
            className={`flex items-center gap-1 px-3 py-1.5 rounded-md border transition
              ${sortKey === "updatedAt"
                ? "bg-green-100 text-green-700 border-green-300"
                : "border-transparent hover:bg-green-50"} 
              hover:cursor-pointer`}
            onClick={() => handleSort("updatedAt")}
          >
            Updated <Calendar size={14} className="text-green-600" />
          </button>
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4">
        {filteredProjects.map((project) => (
          <Card
          key={project.id}
          className="relative cursor-pointer hover:shadow-md transition group"
          onClick={() => {
            localStorage.setItem("currentProject", JSON.stringify(project));
            navigate("/project/" + project.id);
          }}
        >
          <h2 className="font-semibold text-lg text-green-700">{project.name}</h2>
          <p className="text-xs text-gray-500 mt-1">
            Last updated: {new Date(project.updatedAt).toLocaleString()}
          </p>

          {/* Actions */}
          <div className="absolute top-2 right-2 flex gap-2 opacity-0 group-hover:opacity-100 transition">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setSelectedProject(project);
                setNewName(project.name);
                setShowRenameModal(true);
              }}
              className="text-gray-500 hover:text-yellow-600"
              title="Rename"
            >
              <Pencil size={16} />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setSelectedProject(project);
                setShowDeleteModal(true);
              }}
              className="text-gray-400 hover:text-red-600"
              title="Delete"
            >
              <Trash2 size={16} />
            </button>
          </div>
        </Card>

        ))}
      </div>

      {/* Rename Modal */}
      {showRenameModal && selectedProject && (
        <Modal title="Rename Project" onClose={() => setShowRenameModal(false)}>
          <div className="space-y-4 text-sm">
            <label className="block">
              New name:
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className="mt-1 w-full border border-gray-300 px-3 py-2 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </label>
            <div className="text-right space-x-2">
              <Button variant="muted" onClick={() => setShowRenameModal(false)}>Cancel</Button>
              <Button
                onClick={() => {
                  setProjects((prev) =>
                    prev.map((p) =>
                      p.id === selectedProject.id
                        ? { ...p, name: newName, updatedAt: new Date().toISOString() }
                        : p
                    )
                  );
                  setShowRenameModal(false);
                }}
              >
                Save
              </Button>
            </div>
          </div>
        </Modal>
      )}

      {/* Delete Modal */}
      {showDeleteModal && selectedProject && (
        <Modal title="Delete Project" onClose={() => setShowDeleteModal(false)}>
          <div className="space-y-4 text-sm">
            <p>
              Are you sure you want to delete <strong>{selectedProject.name}</strong>?
            </p>
            <div className="text-right space-x-2">
              <Button variant="muted" onClick={() => setShowDeleteModal(false)}>Cancel</Button>
              <Button
                variant="danger"
                className="bg-red-600 hover:bg-red-700 text-white"
                onClick={() => {
                  setProjects((prev) => prev.filter((p) => p.id !== selectedProject.id));
                  setShowDeleteModal(false);
                }}
              >
                Delete
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </MainLayout>
  );
}
