import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import TrainerPage from './pages/TrainerPage';
import DatasetPage from './pages/DatasetPage';
import TrainPage from './pages/TrainPage';
import TestPage from './pages/TestPage';
import CropPage from './pages/CropPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/trainer" replace />} />
          <Route path="trainer" element={<TrainerPage />} />
          <Route path="dataset" element={<DatasetPage />} />
          <Route path="train" element={<TrainPage />} />
          <Route path="test" element={<TestPage />} />
          <Route path="crop" element={<CropPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
