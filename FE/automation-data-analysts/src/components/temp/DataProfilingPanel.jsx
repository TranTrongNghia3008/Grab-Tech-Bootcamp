import { useState } from "react";
import OverviewTab from "./OverviewTab";
import VariablesTab from "../../pages/projectPanels/VariablesTab";

const tabs = ["Overview", "Variables"];

export default function DataProfilingPanel() {
  const [activeTab, setActiveTab] = useState("Overview");

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-[#00843D]">Data Profiling</h1>

      {/* Tabs */}
      <div className="flex gap-4 border-b pb-2">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`py-2 px-4 font-medium ${
              activeTab === tab ? "border-b-2 border-[#00843D] text-[#00843D]" : "text-gray-600"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div>
        {activeTab === "Overview" && <OverviewTab />}
        {activeTab === "Variables" && <VariablesTab />}
      </div>
    </div>
  );
}
