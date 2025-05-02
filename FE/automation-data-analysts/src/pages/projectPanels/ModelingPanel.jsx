import { useState, useEffect } from "react";
import { FaBullseye, FaWrench } from "react-icons/fa";
import { Card } from "../../components/ui";
import BaselineTab from "./BaselineTab";
import TuningTab from "./TuningTab";
import PredictionTab from "./PredictionTab";

export default function ModelingPanel() {
  const [selectedTarget, setSelectedTarget] = useState("");
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [activeTab, setActiveTab] = useState("Baseline");

  useEffect(() => {
    const target = localStorage.getItem("selectedTarget");
    const features = JSON.parse(localStorage.getItem("selectedFeatures") || "[]");
    if (target) setSelectedTarget(target);
    if (features.length > 0) setSelectedFeatures(features);
  }, []);

  const bestModelId = "lr"
  return (
    <div className="space-y-8">
      <h2 className="text-xl font-bold">Modeling</h2>

      {/* Target & Features */}
      <div className="space-y-4">
        <div className="flex">
          {/* Target */}
          <div className="space-y-1 text-gray-700 me-4 pe-4 border-r border-gray-300">
            <div className="flex items-center gap-2">
              <FaBullseye className="text-green-600" />
              <h4 className="text-sm font-semibold">Target Column</h4>
            </div>
            <p className="text-sm ml-6">
              {selectedTarget ? (
                <span className="inline-block bg-green-100 text-green-800 px-2 py-1 rounded text-xs">
                  {selectedTarget}
                </span>
              ) : (
                <span className="text-gray-400">No target selected.</span>
              )}
            </p>
          </div>

          {/* Features */}
          <div className="space-y-1 text-gray-700">
            <div className="flex items-center gap-2">
              <FaWrench className="text-green-600" />
              <h4 className="text-sm font-semibold">Feature Columns</h4>
            </div>
            {selectedFeatures.length > 0 ? (
              <div className="flex flex-wrap gap-2 ml-6">
                {selectedFeatures.map((feature) => (
                  <span
                    key={feature}
                    className="inline-block bg-green-50 border border-green-300 text-green-700 px-3 py-1 rounded-full text-xs"
                  >
                    {feature}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-sm ml-6 text-gray-400">No features selected.</p>
            )}
          </div>
        </div>

        <p className="text-xs text-gray-400 italic ml-6">
          The selected target and features will be used for model training and evaluation.
        </p>
      </div>

      {/* Tabs */}
      <div>
        <div className="flex space-x-4 mb-6 border-b border-gray-300">
          {["Baseline", "Tuning", "Prediction"].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-sm font-medium transition rounded-t-md 
                ${
                  activeTab === tab
                    ? "text-green-700 border-b-2 border-green-600"
                    : "hover:bg-green-100"
                }
              `}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {activeTab === "Baseline" && <BaselineTab />}
        {activeTab === "Tuning" && <TuningTab bestModelId={bestModelId} />}
        {activeTab === "Prediction" && <PredictionTab />}
      </div>

    </div>
  );
}
