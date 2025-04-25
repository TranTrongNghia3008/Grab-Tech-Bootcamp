import { useParams } from "react-router-dom";
import MainLayout from "../layout/MainLayout";
import { useState } from "react";
import PreparePanel from "./projectPanels/PreparePanel";

const tabs = ["Prepare", "EDA", "Model Builder", "Chatbot", "Export"];

export default function ProjectDetail() {
  const { id } = useParams();
  const [activeTab, setActiveTab] = useState("Cleaning");

  return (
    <MainLayout>
      <h1 className="text-2xl font-bold mb-2">Project: {id}</h1>
      <div className="flex space-x-4 mb-6 border-b pb-2">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`py-2 px-4 rounded-t font-medium ${
              activeTab === tab ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-700"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      <div>
        {activeTab === "Prepare" && <PreparePanel />}
        {activeTab === "EDA" && <EDAPanel />}
        {activeTab === "Model Builder" && <ModelBuilderPanel />}
        {activeTab === "Chatbot" && <ChatbotPanel />}
        {activeTab === "Export" && <ExportPanel />}
      </div>
    </MainLayout>
  );
}
