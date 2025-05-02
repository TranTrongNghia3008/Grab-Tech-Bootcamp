import Sidebar from "../components/Sidebar";

export default function MainLayout({ children }) {
  return (
    <div className="flex m-0 p-0 overflow-y-auto h-screen bg-[#00843D]">
      <Sidebar />
      <main className="flex-1 overflow-y-auto h-screen px-10 py-6 rounded bg-[#FFFDF3]">
        {children}
      </main>
    </div>
  );
}
