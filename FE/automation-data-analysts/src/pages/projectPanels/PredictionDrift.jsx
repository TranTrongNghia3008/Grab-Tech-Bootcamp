import { LineChart, Line, XAxis, YAxis, Tooltip, BarChart, Bar, ResponsiveContainer } from 'recharts';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../../components/ui';


// Mocked Data
const mockData = [
  { column: 'Age', type: 'num', driftScore: 0.12, test: 'KS Test', driftDetected: false },
  { column: 'Income', type: 'num', driftScore: 0.34, test: 'KS Test', driftDetected: true },
  { column: 'Gender', type: 'cat', driftScore: 0.05, test: 'Chi-Squared', driftDetected: false },
  { column: 'Device Type', type: 'cat', driftScore: 0.29, test: 'Chi-Squared', driftDetected: true },
];

// Drift Data cho LineChart
const driftLineData = [
  { date: '2024-01', Reference: 0.1, Current: 0.15 },
  { date: '2024-02', Reference: 0.11, Current: 0.16 },
  { date: '2024-03', Reference: 0.13, Current: 0.19 },
  { date: '2024-04', Reference: 0.12, Current: 0.22 },
];

// Data distribution cho num
const numDistributionData = [
  { name: '0-10', Reference: 30, Current: 25 },
  { name: '10-20', Reference: 40, Current: 35 },
  { name: '20-30', Reference: 20, Current: 30 },
];

// Data distribution cho cat
const catDistributionData = [
  { name: 'Male', Reference: 60, Current: 55 },
  { name: 'Female', Reference: 40, Current: 45 },
];

// Main Table
export default function PredictionDriftTable() {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Prediction Drift Report</h1>
      <div className="overflow-x-auto">
        <table className="w-full table-auto border-collapse">
          <thead>
            <tr className="bg-gray-100">
              <th className="border px-4 py-2 text-left">Column</th>
              <th className="border px-4 py-2 text-left">Type</th>
              <th className="border px-4 py-2 text-left">Visualization</th>
              <th className="border px-4 py-2 text-left">Drift Score</th>
              <th className="border px-4 py-2 text-left">Test</th>
              <th className="border px-4 py-2 text-left">Drift Detected</th>
            </tr>
          </thead>
          <tbody>
            {mockData.map((item, index) => (
              <tr key={index} className="border-b">
                <td className="border px-4 py-4 font-medium">{item.column}</td>
                <td className="border px-4 py-4 capitalize">{item.type}</td>
                <td className="border px-4 py-4">
                  <Tabs defaultValue="drift">
                    <TabsList>
                      <TabsTrigger value="drift">DATA DRIFT</TabsTrigger>
                      <TabsTrigger value="distribution">DATA DISTRIBUTION</TabsTrigger>
                    </TabsList>

                    {/* DATA DRIFT tab */}
                    <TabsContent value="drift">
                      <div className="h-32">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={driftLineData}>
                            <XAxis dataKey="date" hide />
                            <YAxis hide />
                            <Tooltip />
                            <Line type="monotone" dataKey="Reference" stroke="#3b82f6" strokeWidth={2} />
                            <Line type="monotone" dataKey="Current" stroke="#f87171" strokeWidth={2} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </TabsContent>

                    {/* DATA DISTRIBUTION tab */}
                    <TabsContent value="distribution">
                      <div className="h-32">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={item.type === 'num' ? numDistributionData : catDistributionData}>
                            <XAxis dataKey="name" hide />
                            <YAxis hide />
                            <Tooltip />
                            <Bar dataKey="Reference" fill="#60a5fa" />
                            <Bar dataKey="Current" fill="#f87171" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </TabsContent>

                  </Tabs>
                </td>

                <td className="border px-4 py-4">{item.driftScore}</td>
                <td className="border px-4 py-4">{item.test}</td>
                <td className="border px-4 py-4">
                  {item.driftDetected ? (
                    <span className="px-2 py-1 text-xs rounded-full bg-red-100 text-red-800">Drift</span>
                  ) : (
                    <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">No Drift</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
