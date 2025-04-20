import Sidebar from "../components/Sidebar";

export default function MainLayout({ children }) {
  return (
    <div className="flex m-0 p-0 overflow-y-auto h-screen">
      <Sidebar />
      <main className=" pl-5 flex-1 overflow-y-auto h-screen p-5">
        {children}
      </main>
    </div>
  );
}
