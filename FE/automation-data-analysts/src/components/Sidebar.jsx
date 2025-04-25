import { Link, useLocation } from 'react-router-dom';

const navItems = [
  { name: "Projects", path: "/projects" },
  { name: "Dashboard", path: "/dashboard" },
  { name: "Settings", path: "/settings" }
];

export default function Sidebar() {
  const location = useLocation();
  const currentUser = "Nghia"; // Thay bằng data thực sau này

  return (
    <div className="h-screen w-64 bg-gray-800 text-white flex flex-col justify-between p-4">
      <div>
        <h2 className="text-xl font-bold text-blue-400 mb-6">👋 Xin chào, {currentUser}</h2>

        <nav className="space-y-2">
          {navItems.map((item) => (
            <Link
              key={item.name}
              to={item.path}
              className={`block px-4 py-2 rounded hover:bg-gray-700 transition ${
                location.pathname === item.path ? "bg-gray-700 font-semibold" : ""
              }`}
            >
              {item.name}
            </Link>
          ))}
        </nav>
      </div>

      <div className="text-sm text-gray-400 mt-6">
        © 2025 Automation Tool for Data Analysts
      </div>
    </div>
  );
}
