import { useDropzone } from "react-dropzone";

export default function UploadZipDropzone({ onFileAccepted }) {
  const { getRootProps, getInputProps } = useDropzone({
    accept: { "application/zip": [".zip"] },
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        onFileAccepted(acceptedFiles[0]);
      }
    },
  });

  return (
    <div
      {...getRootProps()}
      className="border-dashed border-2 border-gray-300 p-10 rounded-lg text-center cursor-pointer bg-gray-50 hover:bg-gray-100"
    >
      <input {...getInputProps()} />
      <p>📦 Drag and drop ZIP file here, or click to select</p>
    </div>
  );
}
