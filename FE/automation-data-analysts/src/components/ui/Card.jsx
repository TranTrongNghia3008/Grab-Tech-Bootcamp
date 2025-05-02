export default function Card({ children, className = "", ...props }) {
    return (
      <div
        className={`space-y-6 bg-[#FFFDF3] shadow-sm border border-[#E4F3E9] rounded p-6 transition-transform duration-300 hover:scale-101 ${className}`}
        {...props}
      >
        {children}
      </div>
    );
  }
  