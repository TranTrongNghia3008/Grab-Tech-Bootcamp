import { useParams, useNavigate } from "react-router-dom";
import { IoChevronBack } from "react-icons/io5";
import {
  FaDatabase,
  FaRobot,
  FaChartBar,
  FaMagic,
  FaFileExport,
} from "react-icons/fa";
import { MdOutlineCleaningServices } from "react-icons/md";
import { useState } from "react";
import PreparePanel from "./projectPanels/PreparePanel";
import EDAPanel from "./projectPanels/EDAPanel";
import ChatbotPanel from "./projectPanels/ChatbotPanel";
import BaselineModelingPanel from "./projectPanels/BaselineModelingPanel";
import ModelBuilderPanel from "./projectPanels/ModelBuilderPanel";
import ExportPanel from "./projectPanels/ExportPanel";

const tabs = [
  { label: "Prepare", icon: <MdOutlineCleaningServices className="mr-2" /> },
  { label: "EDA", icon: <FaChartBar className="mr-2" /> },
  { label: "Baseline Modeling", icon: <FaMagic className="mr-2" /> },
  { label: "Model Builder", icon: <FaDatabase className="mr-2" /> },
  { label: "Chatbot", icon: <FaRobot className="mr-2" /> },
  { label: "Export", icon: <FaFileExport className="mr-2" /> },
];

export default function ProjectDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("Prepare");

  return (
    <div className="bg-[#FFFDF3] min-h-screen">
      <div className="flex flex-wrap items-center justify-between gap-y-2 mb-2 border-b border-[#CDEBD5] p-4 bg-[#00843D] sticky top-0 z-100">
        {/* Title with back button */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate("/projects")}
            className="flex items-center text-white text-lg font-medium hover:underline hover:opacity-90 transition hover:bg-[#006C35] duration-200 border border-[#00843D] rounded-full p-1 hover:cursor-pointer"
          >
            <IoChevronBack />
          </button>
          <h1 className="text-2xl font-bold text-white whitespace-nowrap">
            Project: {id}
          </h1>
        </div>

        {/* Tabs */}
        <div className="inline-flex bg-[#E4F3E9] rounded-xl shadow-inner px-2 py-1 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.label}
              onClick={() => setActiveTab(tab.label)}
              className={`flex items-center px-4 py-2 mx-1 rounded-lg text-sm font-medium whitespace-nowrap transition hover:cursor-pointer
                ${activeTab === tab.label ? "bg-[#00843D] text-white shadow" : "text-[#1B1F1D] hover:bg-[#CDEBD5]"}
              `}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="px-10 space-y-6">
        {activeTab === "Prepare" && <PreparePanel />}
        {activeTab === "EDA" && <EDAPanel />}
        {activeTab === "Baseline Modeling" && <BaselineModelingPanel />}
        {activeTab === "Model Builder" && <ModelBuilderPanel />}
        {activeTab === "Chatbot" && <ChatbotPanel />}
        {activeTab === "Export" && <ExportPanel />}
      </div>
    </div>
  );
}
