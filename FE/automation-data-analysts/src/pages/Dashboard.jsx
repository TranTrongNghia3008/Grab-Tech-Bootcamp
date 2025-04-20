import React from 'react';
import UploadDropzone from '../components/UploadDropzone';
import MainLayout from "../layout/MainLayout";

export default function Dashboard() {
    // const handleUpload = (file) => {
    //     console.log("Đã chọn file:", file);

    //   };

  return (
    <MainLayout>
        <h1 className="text-2xl font-bold mb-4">Dashboard</h1>
        {/* <p>Chào mừng bạn đến với công cụ phân tích dữ liệu tự động 🎉</p>
        <div className="p-6">
            <h2 className="text-2xl font-bold mb-4">Tải lên dữ liệu</h2>
            <UploadDropzone onFileAccepted={handleUpload} />
        </div> */}
    </MainLayout>
  );
}




