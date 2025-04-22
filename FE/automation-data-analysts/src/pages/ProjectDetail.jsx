import { useParams } from "react-router-dom";
import MainLayout from "../layout/MainLayout";
import { useState } from "react";
import PreparePanel from "./projectPanels/PreparePanel";
import EDAPanel from "./projectPanels/EDAPanel";
import ChatbotPanel from "./projectPanels/ChatbotPanel";
import BaselineModelingPanel from "./projectPanels/BaselineModelingPanel";
import ModelBuilderPanel from "./projectPanels/ModelBuilderPanel";

const tabs = ["Prepare", "EDA", "Baseline Modeling", "Model Builder", "Chatbot", "Export"];

export default function ProjectDetail() {
  const { id } = useParams();
  const [activeTab, setActiveTab] = useState("Prepare");

  return (
    <MainLayout>
      <h1 className="text-2xl font-bold mb-2">Project: {id}</h1>
      <div className="flex space-x-4 mb-6 border-b pb-2">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`py-2 px-4 rounded-t font-medium ${
              activeTab === tab ? "bg-green-600 text-white" : "bg-gray-200 text-gray-700"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      <div>
        {activeTab === "Prepare" && <PreparePanel />}
        {activeTab === "EDA" && <EDAPanel />}
        {activeTab === "Baseline Modeling" && <BaselineModelingPanel />}
        {activeTab === "Model Builder" && <ModelBuilderPanel />}
        {activeTab === "Chatbot" && <ChatbotPanel />}
        {/* {activeTab === "Export" && <ExportPanel />} */}
      </div>
    </MainLayout>
  );
}
