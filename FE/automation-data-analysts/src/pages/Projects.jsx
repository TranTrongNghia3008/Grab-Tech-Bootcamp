import MainLayout from "../layout/MainLayout";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Projects() {
  const [search, setSearch] = useState("");
  const [projects, setProjects] = useState([
    { id: 1, name: "Customer Segmentation" },
    { id: 2, name: "Sales Forecasting" }
  ]);

  const navigate = useNavigate();

  const handleCreateProject = () => {
    // const projectName = prompt("Nhập tên project mới:");
    const projectName = "New Project"; // Thay thế bằng prompt hoặc modal thực tế
    if (projectName) {
      const newProject = {
        id: Date.now(), // hoặc UUID nếu muốn
        name: projectName
      };

      setProjects([...projects, newProject]);

      // 👉 Lưu tạm project vào localStorage hoặc navigate kèm state
      localStorage.setItem("currentProject", JSON.stringify(newProject));

      // 👉 Chuyển sang trang upload
      navigate("/projects/upload");
    }
  };

  const filteredProjects = projects.filter(p =>
    p.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <MainLayout>
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold">Projects</h1>
        <button
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          onClick={handleCreateProject}
        >
          + New Project
        </button>
      </div>

      <input
        type="text"
        className="w-full mb-4 p-2 border border-gray-300 rounded"
        placeholder="Search project..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      <div className="space-y-2">
        {filteredProjects.map((project) => (
          <div
            key={project.id}
            className="p-4 bg-white shadow rounded hover:bg-gray-100 cursor-pointer"
            onClick={() => {
              localStorage.setItem("currentProject", JSON.stringify(project));
              const projectId = project.id;
              navigate("/project/" + projectId);
            }}
          >
            {project.name}
          </div>
        ))}
      </div>
    </MainLayout>
  );
}
