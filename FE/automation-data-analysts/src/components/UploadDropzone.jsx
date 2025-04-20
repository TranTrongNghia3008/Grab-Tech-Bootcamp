import { useDropzone } from 'react-dropzone';

export default function UploadDropzone({ onFileAccepted }) {
  const { getRootProps, getInputProps } = useDropzone({
    accept: { 'text/csv': ['.csv'] },
    onDrop: (acceptedFiles) => {
      onFileAccepted(acceptedFiles[0]);
    }
  });

  return (
    <div
      {...getRootProps()}
      className="border-dashed border-2 border-gray-300 p-10 rounded-lg text-center cursor-pointer bg-gray-50 hover:bg-gray-100"
    >
      <input {...getInputProps()} />
      <p>Drag and drop CSV file here, or click to select file</p>
    </div>
  );
}
