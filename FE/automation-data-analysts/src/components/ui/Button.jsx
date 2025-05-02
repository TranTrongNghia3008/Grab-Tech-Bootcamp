export default function Button({ children, variant = "primary", className = "", ...props }) {
  const base =
    "inline-flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition focus:outline-none hover:cursor-pointer";

  const variants = {
    primary: "bg-[#00843D] text-white hover:bg-[#006C35]",
    outline: "bg-white text-[#00843D] border border-[#00843D] hover:bg-[#E4F3E9]",
    muted: "bg-gray-100 text-gray-600 hover:bg-gray-200"
  };

  return (
    <button
      className={`${base} ${variants[variant] || ""} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
