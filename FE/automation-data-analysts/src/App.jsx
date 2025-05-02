import './App.css'
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Projects from './pages/Projects';
import ProjectDetail from './pages/ProjectDetail';
import CreateProject from './pages/CreateProject';
import DatasetsPage from './pages/Datasets';
import Home from './pages/Home';
import PredictionDriftTable from './pages/projectPanels/PredictionDrift';

function App() {


  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/projects" element={<Projects />} />
        <Route path="/datasets" element={<DatasetsPage />} />
        <Route path="/projects/create" element={<CreateProject />} />
        <Route path="/project/:id" element={<ProjectDetail />} />
        <Route path="/predict" element={<PredictionDriftTable/>}/>
        {/* thêm các route khác */}
      </Routes>
    </BrowserRouter>
  )
}

export default App
