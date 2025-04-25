import './App.css'
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Projects from './pages/Projects';
import Upload from './pages/Upload';
import ProjectDetail from './pages/ProjectDetail';

function App() {


  return (
    <BrowserRouter>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/projects" element={<Projects />} />
        <Route path="/projects/upload" element={<Upload />} />
        <Route path="/project/:id" element={<ProjectDetail />} />
        {/* thêm các route khác */}
      </Routes>
    </BrowserRouter>
  )
}

export default App
