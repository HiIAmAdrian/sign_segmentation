
import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface FileUploaderProps {
  label: string;
  icon: React.ReactNode;
  accept: string;
  onFileSelected: (file: File) => void;
  fileName?: string;
}

const FileUploader = ({
  label,
  icon,
  accept,
  onFileSelected,
  fileName,
}: FileUploaderProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.match(accept.replace('*', '.*')) || accept === file.type) {
        onFileSelected(file);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelected(e.target.files[0]);
    }
  };

  return (
    <div
      className={cn(
        "border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors",
        isDragging ? "border-primary bg-primary/10" : "border-gray-300 hover:border-primary/50",
        fileName ? "bg-secondary/50" : ""
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        accept={accept}
        onChange={handleFileChange}
      />
      <div className="flex flex-col items-center space-y-2">
        <div className="p-2 rounded-full bg-primary/10 text-primary">
          {icon}
        </div>
        <h3 className="font-medium">{label}</h3>
        {fileName ? (
          <p className="text-sm text-muted-foreground break-all">
            {fileName}
          </p>
        ) : (
          <p className="text-sm text-muted-foreground">
            Drag and drop or click to upload
          </p>
        )}
      </div>
    </div>
  );
};

export default FileUploader;
