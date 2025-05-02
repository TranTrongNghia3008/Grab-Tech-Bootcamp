import { Link, useLocation } from "react-router-dom";
import { FaProjectDiagram, FaTachometerAlt, FaDatabase, FaCog, FaUserCircle } from "react-icons/fa"; 
import { LuArrowLeftToLine, LuArrowRightToLine } from "react-icons/lu";
import { useState, useEffect } from "react";

const navItems = [
  { name: "Projects", path: "/projects", icon: <FaProjectDiagram /> },
  { name: "Dashboard", path: "/dashboard", icon: <FaTachometerAlt /> },
  { name: "Datasets", path: "/datasets", icon: <FaDatabase /> },
  { name: "Settings", path: "/settings", icon: <FaCog /> }
];

export default function Sidebar() {
  const location = useLocation();
  const currentUser = "Trong-Nghia"; // Sẽ lấy từ context hoặc props sau

  // Lấy trạng thái thu gọn từ localStorage, mặc định là false nếu chưa có
  const [isCollapsed, setIsCollapsed] = useState(() => {
    const savedState = localStorage.getItem("sidebar-collapsed");
    return savedState === "true";
  });

  useEffect(() => {
    // Lưu trạng thái thu gọn vào localStorage khi thay đổi
    localStorage.setItem("sidebar-collapsed", isCollapsed);
  }, [isCollapsed]);

  return (
    <aside className={`h-screen ${isCollapsed ? "w-20" : "w-64"} bg-[#00843D] text-white flex flex-col justify-between p-6 transition-all`}>
      {/* Top section */}
      <div>
        {/* User greeting và nút đóng */}
        <div className="flex justify-between items-center mb-6">
          {!isCollapsed && (
            <Link
            to="/"
            className="text-2xl text-white"
            style={{ fontFamily: "GrabCommunityInline" }}
            >
                DataMate
            </Link>
          )}
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="text-white text-2xl hover:bg-[#006C35] transition duration-200 border border-[#00843D] rounded-full p-1 hover:cursor-pointer"
            aria-label="Toggle Sidebar"
          >
            {isCollapsed ? <LuArrowRightToLine /> : <LuArrowLeftToLine />} {/* Thay đổi biểu tượng khi thu gọn */}
          </button>
        </div>

        {/* Navigation */}
        <nav className="space-y-2 text-sm">
          {navItems.map((item) => (
            <Link
              key={item.name}
              to={item.path}
              className={`px-4 py-2 rounded-md transition flex items-center gap-2 ${isCollapsed ? "justify-center" : ""} ${
                location.pathname === item.path
                  ? "bg-white text-[#00843D] font-semibold"
                  : "hover:bg-[#006C35]"
              }`}
            >
              <span>{item.icon}</span> {/* Hiển thị icon */}
              {!isCollapsed && item.name} {/* Chỉ hiển thị tên nếu sidebar không thu gọn */}
            </Link>
          ))}
        </nav>
      </div>

      {/* Footer */}
      <div className="mt-6">
          <div className={`flex items-center gap-2 hover:cursor-pointer ${isCollapsed ? "justify-center" : ""}`}>
            <FaUserCircle size={24} className="text-white" />
            {!isCollapsed && (
              <span className="text-sm">
                {currentUser}
              </span>
            )}
          </div>
      </div>
    </aside>
  );
}
