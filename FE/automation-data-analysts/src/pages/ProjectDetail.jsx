import { useParams, useNavigate } from "react-router-dom";
import { IoChevronBack } from "react-icons/io5";
import { FaRobot, FaChartBar, FaFileExport, FaBrain } from "react-icons/fa";
import { FiTool } from "react-icons/fi";
import { MdOutlineCleaningServices } from "react-icons/md";
import { FaQuestionCircle } from "react-icons/fa";
import { useState } from "react";

import OverviewPanel from "./projectPanels/OverviewPanel";
import DataInsightPanel from "./projectPanels/DataInsightPanel";
import ChatbotPanel from "./projectPanels/ChatbotPanel";
import ModelingPanel from "./projectPanels/ModelingPanel";
import ExportPanel from "./projectPanels/ExportPanel";
import { useAppContext } from "../contexts/AppContext";

const tabs = [
  { label: "Overview", icon: <MdOutlineCleaningServices className="mr-2" /> },
  { label: "Data Insight", icon: <FaChartBar className="mr-2" /> },
  { label: "Modeling", icon: <FaBrain className="mr-2" /> },
  { label: "Virtual Assistant", icon: <FaRobot className="mr-2" /> },
  { label: "Export", icon: <FaFileExport className="mr-2" /> },
];

export default function ProjectDetail() {
  const { state } = useAppContext(); // Lấy state từ context
  const { id } = useParams();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("Overview");

  // Dummy trạng thái kiểm tra xem đã chọn target và feature chưa
  const [isTargetFeatureSelected, setIsTargetFeatureSelected] = useState(false);

  const handleTabClick = (tabLabel) => {
    if (!isTargetFeatureSelected && ["Modeling", "Export"].includes(tabLabel)) {
      return; // Không cho chuyển nếu chưa chọn target/feature
    }
    setActiveTab(tabLabel);
  };

  console.log("Current datasetId:", state.datasetId);

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

        {/* Tabs + Help icon */}
        <div className="flex items-center gap-4">
          <div className="inline-flex bg-[#E4F3E9] rounded-xl shadow-inner px-2 py-1 overflow-x-auto">
            {tabs.map((tab) => {
              const isDisabled = !isTargetFeatureSelected && ["Modeling", "Export"].includes(tab.label);
              return (
                <button
                  key={tab.label}
                  onClick={() => handleTabClick(tab.label)}
                  disabled={isDisabled}
                  className={`flex items-center px-4 py-2 mx-1 rounded-lg text-sm font-medium whitespace-nowrap transition
                    ${activeTab === tab.label ? "bg-[#00843D] text-white shadow hover:cursor-pointer" : ""}
                    ${isDisabled ? "opacity-50 cursor-not-allowed" : "text-[#1B1F1D] hover:bg-[#CDEBD5] hover:cursor-pointer"}
                  `}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Help Icon */}
          <button
            title="Click here for help"
            className="text-white text-xl hover:opacity-80 transition"
            onClick={() => alert("This is a guide about how to work with the project panels.")}
          >
            <FaQuestionCircle />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="px-10 pb-10 space-y-6">
        {activeTab === "Overview" && (
          <OverviewPanel setIsTargetFeatureSelected={setIsTargetFeatureSelected} />
        )}
        {activeTab === "Data Insight" && <DataInsightPanel />}
        {activeTab === "Modeling" && <ModelingPanel />}
        {activeTab === "Virtual Assistant" && <ChatbotPanel />}
        {activeTab === "Export" && <ExportPanel />}
      </div>
    </div>
  );
}
