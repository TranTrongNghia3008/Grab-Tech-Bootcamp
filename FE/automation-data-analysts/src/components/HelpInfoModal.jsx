import { FiGrid, FiBarChart2, FiCpu, FiMessageCircle, FiDownload, FiX } from "react-icons/fi";
import React from "react";
import { Modal } from "./ui";

export default function HelpInfoModal({ onClose }) {
  return (
    <Modal title="Getting Started Guide" onClose={onClose}>
        <div className="text-sm space-y-6 text-gray-700 max-h-[70vh] overflow-y-auto">
            <div className="p-6 text-sm overflow-y-auto space-y-6 text-gray-800 leading-relaxed">
                <p>
                    This application is designed to help you explore, clean, model, and
                    analyze your dataset effortlessly using automation and AI — no code
                    needed!
                </p>

                <div className="space-y-4">
                    {/* Overview */}
                    <section>
                    <h3 className="flex items-center gap-2 text-green-700 font-semibold text-base">
                        <FiGrid /> Overview
                    </h3>
                    <ul className="list-disc ml-6 mt-1 space-y-1">
                        <li>Preview dataset structure (rows and columns).</li>
                        <li>Detect issues like missing values or duplicates.</li>
                        <li>Use the cleaning assistant (remove duplicates, scale, etc.).</li>
                        <li>Select a target feature for model training.</li>
                    </ul>
                    </section>

                    {/* Data Insight */}
                    <section>
                    <h3 className="flex items-center gap-2 text-green-700 font-semibold text-base">
                        <FiBarChart2 /> Data Insight
                    </h3>
                    <ul className="list-disc ml-6 mt-1 space-y-1">
                        <li>Overview of total records, data quality, and missing values.</li>
                        <li>Analyze each column's type, uniqueness, and null rate.</li>
                        <li>Explore summary statistics: mean, median, min/max, std.</li>
                        <li>Visualize correlation between numeric features.</li>
                        <li>Auto-generate charts: histograms, bar plots, etc.</li>
                    </ul>
                    </section>

                    {/* Modeling */}
                    <section>
                    <h3 className="flex items-center gap-2 text-green-700 font-semibold text-base">
                        <FiCpu /> Modeling
                    </h3>
                    <ul className="list-disc ml-6 mt-1 space-y-1">
                        <li><strong>Baseline:</strong> Compare multiple models and metrics (MAE, RMSE, R²).</li>
                        <li><strong>Tuning:</strong> Adjust hyperparameters and analyze feature importance.</li>
                        <li><strong>Prediction:</strong> Apply the model to predict new data.</li>
                    </ul>
                    </section>

                    {/* Virtual Assistant */}
                    <section>
                    <h3 className="flex items-center gap-2 text-green-700 font-semibold text-base">
                        <FiMessageCircle /> Virtual Assistant
                    </h3>
                    <ul className="list-disc ml-6 mt-1 space-y-1">
                        <li>Ask in natural language: "Average sales by area?"</li>
                        <li>Get auto-generated insights, code, plots, and suggestions.</li>
                        <li>Conversations are saved and can be renamed or deleted.</li>
                    </ul>
                    </section>

                    {/* Export */}
                    <section>
                    <h3 className="flex items-center gap-2 text-green-700 font-semibold text-base">
                        <FiDownload /> Export
                    </h3>
                    <ul className="list-disc ml-6 mt-1 space-y-1">
                        <li>Download related documents.</li>
                    </ul>
                    </section>
                </div>

                <div className="border-t pt-4 text-xs text-gray-600">
                    <strong>Tips:</strong> Start from the Overview tab, analyze your data, train models, ask questions with the assistant, and export your results.
                    <br />
                    Still need help? Contact us at <strong>support@yourapp.com</strong>.
                </div>
            </div>
        </div>
    </Modal>
  );
}
