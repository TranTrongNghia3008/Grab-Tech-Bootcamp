import { useParams } from "react-router-dom";
import MainLayout from "../layout/MainLayout";
import { useState } from "react";
import PreparePanel from "./projectPanels/PreparePanel";
import EDAPanel from "./projectPanels/EDAPanel";
import ChatbotPanel from "./projectPanels/ChatbotPanel";
import BaselineModelingPanel from "./projectPanels/BaselineModelingPanel";
import ModelBuilderPanel from "./projectPanels/ModelBuilderPanel";
import ExportPanel from "./projectPanels/ExportPanel";

const tabs = [
  "Prepare",
  "EDA",
  "Baseline Modeling",
  "Model Builder",
  "Chatbot",
  "Export"
];

export default function ProjectDetail() {
  const { id } = useParams();
  const [activeTab, setActiveTab] = useState("Prepare");

  return (
    <MainLayout>
      <h1 className="text-2xl font-bold mb-2 text-green-700">📊 Project: {id}</h1>

      {/* Tabs */}
      <div className="flex justify-center mb-6 border-b">
        <div className="inline-flex bg-gray-100 rounded-xl shadow-inner px-2 py-1 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 mx-1 rounded-lg text-sm font-medium whitespace-nowrap transition
                ${
                  activeTab === tab
                    ? "bg-green-600 text-white shadow"
                    : "text-gray-700 hover:bg-green-50"
                }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div>
        {activeTab === "Prepare" && <PreparePanel />}
        {activeTab === "EDA" && <EDAPanel />}
        {activeTab === "Baseline Modeling" && <BaselineModelingPanel />}
        {activeTab === "Model Builder" && <ModelBuilderPanel />}
        {activeTab === "Chatbot" && <ChatbotPanel />}
        {activeTab === "Export" && <ExportPanel />}
      </div>
    </MainLayout>
  );
}
