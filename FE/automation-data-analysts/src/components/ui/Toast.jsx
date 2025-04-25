export default function Toast({ type = "info", message }) {
    const base =
      "fixed bottom-4 right-4 z-50 px-4 py-3 rounded shadow-lg text-sm font-medium";
  
    const colors = {
      success: "bg-green-100 text-green-800 border border-green-300",
      error: "bg-red-100 text-red-800 border border-red-300",
      info: "bg-blue-100 text-blue-800 border border-blue-300",
      warning: "bg-yellow-100 text-yellow-800 border border-yellow-300"
    };
  
    return <div className={`${base} ${colors[type]}`}>{message}</div>;
  }
  