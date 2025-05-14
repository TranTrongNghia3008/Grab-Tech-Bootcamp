import { Card } from "../ui";

export default function AlertsContent() {
  const alerts = [
    { column: "Age", message: "has 177 (19.9%) missing values", type: "Missing" },
    { column: "Cabin", message: "has 687 (77.1%) missing values", type: "Missing" },
    { column: "PassengerId", message: "has unique values", type: "Unique" },
    { column: "Name", message: "has unique values", type: "Unique" },
    { column: "Survived", message: "has 549 (61.6%) zeros", type: "Zeros" },
    { column: "SibSp", message: "has 608 (68.2%) zeros", type: "Zeros" },
    { column: "Parch", message: "has 678 (76.1%) zeros", type: "Zeros" },
    { column: "Fare", message: "has 15 (1.7%) zeros", type: "Zeros" }
  ];

  const badgeColor = (type) => {
    switch (type) {
      case "Missing":
        return "bg-cyan-400 text-white";
      case "Unique":
        return "bg-red-400 text-white";
      case "Zeros":
        return "bg-cyan-400 text-white";
      default:
        return "bg-gray-300 text-gray-700";
    }
  };

  return (
    <Card className="space-y-6">
      <h3 className="text-2xl font-bold text-gray-800">Alerts</h3>

      <div className="space-y-2">
        {alerts.map((alert, idx) => (
          <div key={idx} className="flex justify-between items-center bg-gray-100 px-4 py-2 rounded-md">
            <div className="text-sm text-gray-700">
              <span className="text-blue-600 font-medium">{alert.column}</span> {alert.message}
            </div>
            <span className={`px-2 py-1 text-xs rounded-full font-semibold ${badgeColor(alert.type)}`}>
              {alert.type}
            </span>
          </div>
        ))}
      </div>
    </Card>
  );
}
