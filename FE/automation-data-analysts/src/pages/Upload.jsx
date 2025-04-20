import UploadDropzone from "../components/UploadDropzone";
import MainLayout from "../layout/MainLayout";


export default function Upload() {
  const handleUpload = (file) => {
    console.log("Đã chọn file:", file);
    // Lưu file vào localStorage
    const reader = new FileReader();
    reader.onload = (event) => {
      const csvData = event.target.result;
      localStorage.setItem("dataset", csvData);
      // console.log("Dữ liệu đã được lưu vào localStorage:", csvData);
    };
    reader.readAsText(file);
  };

  return (
    <MainLayout>
        <div className="p-6">
            <h2 className="text-2xl font-bold mb-4">Upload data</h2>
            <UploadDropzone onFileAccepted={handleUpload} />
        </div>
    </MainLayout>
  );
}
