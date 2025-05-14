import { Link } from "react-router-dom";
import { Sparkles, BarChart2, Bot, Settings2 } from "lucide-react";
import { Button, Card } from "../components/ui";
import { useEffect } from "react";

export default function Home() {
    useEffect(() => {
        document.title = "DataMate";
    }, []);
    return (
        <div className="min-h-screen bg-[#FFFDF3] text-[#1B1F1D] font-poppins">
            {/* Header */}
            <header className="bg-[#E4F3E9] shadow-sm sticky top-0 z-50">
                <div className="max-w-7xl mx-auto p-4 flex items-center justify-between">
                {/* Logo */}
                <Link
                to="/"
                className="text-2xl text-[#00843D]"
                style={{ fontFamily: "GrabCommunityInline" }}
                >
                    DataMate
                </Link>

                {/* Nav */}
                <nav className="flex gap-6 text-sm text-gray-700 items-center">
                    <Link
                    to="/"
                    className="hover:text-[#00843D] transition font-medium"
                    >
                    Home
                    </Link>
                    <a href="#features" className="hover:text-[#00843D] transition font-medium">
                    Features
                    </a>
                    <a href="#features" className="hover:text-[#00843D] transition font-medium">
                    Pricing
                    </a>
                    <a href="#features" className="hover:text-[#00843D] transition font-medium">
                    About
                    </a>
                    <Link
                    
                    to="/login"
                    className="text-sm bg-[#00843D] text-white px-4 py-2 rounded hover:bg-green-800 transition"
                    >
                    Login
                    </Link>
                </nav>
                </div>
            </header>
            {/* Hero Section */}
            <section className="max-w-5xl mx-auto px-6 py-20 text-center">
                <h1 className="text-4xl sm:text-5xl font-extrabold leading-tight text-[#00843D] mb-6">
                Automate Your Data Analysis
                </h1>
                <p className="text-lg text-gray-700 max-w-2xl mx-auto mb-8">
                Clean, visualize, model and interact with your data faster than ever before – no code required.
                </p>
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link to="/login">
                    <Button className="text-lg px-6 py-3">Sign In</Button>
                </Link>
                <Link to="/signup">
                    <Button variant="outline" className="text-lg px-6 py-3">Create Account</Button>
                </Link>
                </div>
            </section>

            {/* Features */}
            <section className="bg-[#E4F3E9] py-16">
                <div className="max-w-5xl mx-auto px-6 text-center">
                <h2 className="text-3xl font-bold mb-10 text-[#00843D]">What You Can Do</h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-8 text-sm text-gray-800">
                    <Feature icon={<Sparkles size={32} />} title="Data Cleaning" desc="Detect and fix missing values, outliers, and duplicates easily." />
                    <Feature icon={<BarChart2 size={32} />} title="Data Insight" desc="Generate summary stats and beautiful charts with one click." />
                    <Feature icon={<Settings2 size={32} />} title="Modeling" desc="Train baseline or custom ML models with intuitive UI." />
                    <Feature icon={<Bot size={32} />} title="Virtual Assistant" desc="Ask questions in natural language and get data-driven answers." />
                </div>
                </div>
            </section>

            {/* CTA */}
            <section className="py-20 text-center bg-[#FFF9E5] px-6">
                <h2 className="text-3xl font-bold text-[#00843D] mb-4">Ready to accelerate your workflow?</h2>
                <p className="text-gray-700 mb-6">Join thousands of data professionals simplifying their day-to-day analysis.</p>
                <Link to="/projects">
                <Button className="text-lg px-6 py-3">Get Started Free</Button>
                </Link>
            </section>

            {/* Footer */}
            <footer className="text-center text-sm text-gray-500 py-6">
                © 2025 Automation Tool for Data Analysts
            </footer>
        </div>
    );
}

function Feature({ icon, title, desc }) {
  return (
    <Card className="hover:cursor-pointer  hover:scale-105">
      <div className="text-[#00843D] mb-3">{icon}</div>
      <h3 className="font-semibold text-lg mb-2">{title}</h3>
      <p className="text-gray-600 text-sm">{desc}</p>
    </Card>
  );
}
