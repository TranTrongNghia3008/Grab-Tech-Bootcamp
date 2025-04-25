import Sidebar from "../components/Sidebar";

export default function MainLayout({ children }) {
  return (
    <div className="flex m-0 p-0 h-screen">
      <Sidebar />
      <main className="flex-1 p-6 bg-gray-100 w-0.5 h-screen">
        {children}
      </main>
    </div>
  );
}
