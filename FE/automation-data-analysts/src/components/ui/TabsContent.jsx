import { useContext } from 'react';
import { TabsContext } from './Tabs';

export function TabsContent({ value, children }) {
  const { activeTab } = useContext(TabsContext);

  if (activeTab !== value) return null;

  return <div className="mt-2">{children}</div>;
}
