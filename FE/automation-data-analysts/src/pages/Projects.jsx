import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import MainLayout from "../layout/MainLayout";
import { Button, Card, Modal, Toast } from "../components/ui";
import { ArrowDownAZ, ArrowUpAZ, Calendar, Pencil, Trash2 } from "lucide-react";
import { FiFolder } from "react-icons/fi";
import { useAppContext } from "../contexts/AppContext";
import { deleteDataset, getAllByCreation, updateProjectName } from "../components/services/datasetService";

export default function Projects() {
  const { updateState } = useAppContext();
  const navigate = useNavigate();

  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState("created_at");
  const [sortDir, setSortDir] = useState("desc");
  const [projects, setProjects] = useState([]);
  const [showRenameModal, setShowRenameModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [selectedProject, setSelectedProject] = useState(null);
  const [newName, setNewName] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [deleteError, setDeleteError] = useState("");
  const [toast, setToast] = useState({ show: false, type: "", message: "" });

  useEffect(() => {
    document.title = "DataMate - Projects";
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

  useEffect(() => {
    if (toast.show) {
      const timer = setTimeout(() => setToast({ show: false, type: "", message: "" }), 3000);
      return () => clearTimeout(timer);
    }
  }, [toast]);

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
    .filter((p) => p.project_name.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      const valA = a[sortKey];
      const valB = b[sortKey];
      if (sortKey === "created_at") {
        return sortDir === "asc"
          ? new Date(valA) - new Date(valB)
          : new Date(valB) - new Date(valA);
      }
      return sortDir === "asc"
        ? String(valA).localeCompare(String(valB))
        : String(valB).localeCompare(String(valA));
    });

  function getRandomColor() {
    const tailwindColors = [
      "bg-blue-100 text-blue-800",
      "bg-red-100 text-red-800",
      "bg-green-100 text-green-800",
      "bg-yellow-100 text-yellow-800",
      "bg-purple-100 text-purple-800",
      "bg-pink-100 text-pink-800",
      "bg-indigo-100 text-indigo-800",
      "bg-orange-100 text-orange-800"
    ];
    return tailwindColors[Math.floor(Math.random() * tailwindColors.length)];
  }

  const handleRenameProject = async () => {
    if (!newName) {
      setErrorMessage("Project name cannot be empty.");
      return;
    }

    try {
      await updateProjectName(selectedProject.id, newName);

      setProjects((prev) =>
        prev.map((p) =>
          p.id === selectedProject.id
            ? {
                ...p,
                project_name: newName,
                created_at: new Date().toISOString(),
              }
            : p
        )
      );
      setShowRenameModal(false);
      setToast({ show: true, type: "success", message: "Project renamed successfully!" });
    } catch (error) {
      console.error("Failed to rename project:", error);
    }
  };

const handleDeleteProject = async () => {
  try {
    await deleteDataset(selectedProject.id); // gọi API xóa
    setProjects((prev) => prev.filter((p) => p.id !== selectedProject.id));
    setShowDeleteModal(false);
    setDeleteError(""); 
    setToast({ show: true, type: "success", message: "Project deleted successfully!" });
  } catch (error) {
    console.error("Failed to delete project:", error);
    setDeleteError("Failed to delete project. Please try again.");
  }
};


  return (
    <MainLayout>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center mb-4 gap-3">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold text-[#1B1F1D] flex items-center gap-2">
            <FiFolder className="text-green-600" size={24} /> My Projects
          </h1>
          <div className="w-8 h-8 rounded-md bg-green-100 text-green-700 flex items-center justify-center text-sm font-semibold">
            {filteredProjects.length}
          </div>
        </div>
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
              ${sortKey === "project_name"
                ? "bg-green-100 text-green-700 border-green-300"
                : "border-transparent hover:bg-green-50"} 
              hover:cursor-pointer`}
            onClick={() => handleSort("project_name")}
          >
            Name
            {sortKey === "project_name" &&
              (sortDir === "asc" ? (
                <ArrowDownAZ size={14} className="text-green-600" />
              ) : (
                <ArrowUpAZ size={14} className="text-green-600" />
              ))}
          </button>

          <button
            className={`flex items-center gap-1 px-3 py-1.5 rounded-md border transition
              ${sortKey === "created_at"
                ? "bg-green-100 text-green-700 border-green-300"
                : "border-transparent hover:bg-green-50"} 
              hover:cursor-pointer`}
            onClick={() => handleSort("created_at")}
          >
            Created <Calendar size={14} className="text-green-600" />
          </button>
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-6">
  {filteredProjects.map((project) => {
    const { columns, data: rows } = project.data_preview;
    const columnColors = getRandomColor();

    return (
      <div
        key={project.id}
        className="relative group border border-gray-200 shadow-sm rounded-lg p-0 hover:shadow-lg transition-transform duration-300 hover:scale-101 cursor-pointer bg-white"
        onClick={() => {
          updateState({
            datasetId: project.id,
            projectName: project.project_name || "Untitled Project",
            isClean: project.is_clean,
            isModel: project.is_model,
          });
          localStorage.setItem(
            "currentProject",
            JSON.stringify({
              datasetId: project.id,
              projectName: project.project_name || "Untitled Project",
              isClean: project.is_clean,
              isModel: project.is_model,
            })
          );
          navigate("/project/" + project.id);
        }}
      >
        <div className="text-[11px] border border-gray-200 rounded overflow-hidden mb-2">
          <div className="grid grid-cols-4">
            {columns.map((col) => (
              <div
                key={col}
                className={`p-1 text-center ${columnColors} border-r border-gray-200 truncate whitespace-nowrap overflow-hidden`}
                title={col}
              >
                {col}
              </div>
            ))}
          </div>
          {rows.map((row, rowIdx) => (
            <div key={rowIdx} className="grid grid-cols-4 even:bg-gray-50">
              {row.map((cell, cellIdx) => (
                <div
                  key={cellIdx}
                  className="p-1 text-center text-gray-700 border-r border-gray-200 truncate whitespace-nowrap overflow-hidden"
                  title={cell}
                >
                  {cell}
                </div>
              ))}
            </div>
          ))}
          <div className="absolute bottom-15 left-0 right-0 h-15 bg-gradient-to-t from-white to-transparent pointer-events-none" />
        </div>

        <div className="px-4">
          {/* Project Name */}
          <h2 className="font-semibold text-lg text-green-700 group-hover:text-green-800 line-clamp-1">
            {project.project_name || "Untitled Project"}
          </h2>

          {/* Time */}
          <p className="text-xs text-gray-500 mt-1 mb-2">
            Created at:&nbsp;
            <span className="font-medium text-gray-600">
              {new Date(project.created_at).toLocaleString("en-GB", {
                hour: "2-digit",
                minute: "2-digit",
                day: "2-digit",
                month: "2-digit",
                year: "numeric",
              })}
            </span>
          </p>
        </div>

        {/* Hover Actions */}
        <div className="absolute top-2 right-2 flex gap-2 opacity-0 group-hover:opacity-100 transition">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setSelectedProject(project);
              setNewName(project.project_name);
              setErrorMessage("");
              setShowRenameModal(true);
            }}
            className="text-gray-500 hover:text-yellow-600 bg-white rounded-full p-1 shadow-sm hover:shadow-md transition"
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
            className="text-gray-400 hover:text-red-600 bg-white rounded-full p-1 shadow-sm hover:shadow-md transition"
            title="Delete"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>
    );
  })}
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

            {errorMessage && (
              <p className="text-red-500 text-xs">{errorMessage}</p>
            )}

            <div className="text-right space-x-2">
              <Button variant="muted" onClick={() => setShowRenameModal(false)}>Cancel</Button>
              <Button
                onClick={handleRenameProject}
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
            Are you sure you want to delete <strong>{selectedProject.project_name}</strong>?
          </p>

          {deleteError && (
            <p className="text-red-500 text-xs">{deleteError}</p>
          )}

          <div className="text-right space-x-2">
            <Button variant="muted" onClick={() => setShowDeleteModal(false)}>Cancel</Button>
            <Button
              variant="danger"
              className="bg-red-600 hover:bg-red-700 text-white"
              onClick={handleDeleteProject}
            >
              Delete
            </Button>
          </div>
        </div>
      </Modal>
    )}
    {toast.show && <Toast type={toast.type} message={toast.message} />}
    </MainLayout>
  );
}
