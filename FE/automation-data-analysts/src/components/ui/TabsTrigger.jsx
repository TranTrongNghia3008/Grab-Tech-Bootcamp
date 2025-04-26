import { useContext } from 'react';
import { TabsContext } from './Tabs';

export function TabsTrigger({ value, children }) {
  const { activeTab, setActiveTab } = useContext(TabsContext);

  const isActive = activeTab === value;

  return (
    <button
      onClick={() => setActiveTab(value)}
      className={`flex-1 px-4 py-2 font-semibold border-b-2 ${
        isActive ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500'
      }`}
    >
      {children}
    </button>
  );
}
