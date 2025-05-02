import { useState } from "react";
import { Card } from "../../components/ui";
import OverviewContent from "./OverviewContent";
import AlertsContent from "./AlertsContent";
import ReproductionContent from "./ReproductionContent";

export default function OverviewTab() {
  const [activeTab, setActiveTab] = useState("Overview");

  const tabClasses = (tab) =>
    `px-4 py-2 text-sm font-medium rounded-t-md ${
      activeTab === tab
        ? "bg-white text-green-700 border-b-2 border-green-700"
        : "text-gray-500 hover:text-green-600"
    }`;

  return (
    <Card className="p-0">
      {/* Tabs */}
      <div className="flex border-b bg-gray-100 px-4">
        {["Overview", "Alerts", "Reproduction"].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={tabClasses(tab)}
          >
            {tab}
            {tab === "Alerts" && <span className="ml-1 bg-gray-400 text-white rounded-full px-2 text-xs">8</span>}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="p-6">
        {activeTab === "Overview" && <OverviewContent />}
        {activeTab === "Alerts" && <AlertsContent />}
        {activeTab === "Reproduction" && <ReproductionContent />}
      </div>
    </Card>
  );
}
