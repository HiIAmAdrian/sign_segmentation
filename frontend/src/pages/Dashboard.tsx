
import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { processFiles, SegmentResponse } from '../services/apiService';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useToast } from '@/components/ui/use-toast';
import { Upload, Video, RefreshCcw, FileVideo, Play, Pause } from 'lucide-react';
import FileUploader from '../components/FileUploader';
import VideoPlayer from '../components/VideoPlayer';

// ... keep existing code (Dashboard) the same except VideoPlayer usage

const Dashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [suitFile, setSuitFile] = useState<File | null>(null);
  const [gloveRightFile, setGloveRightFile] = useState<File | null>(null);
  const [gloveLeftFile, setGloveLeftFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [segmentData, setSegmentData] = useState<SegmentResponse | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  const videoRef1 = useRef<HTMLVideoElement>(null);
  const videoRef2 = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (!user) {
      navigate('/login');
    }
  }, [user, navigate]);

  useEffect(() => {
    if (videoFile) {
      const url = URL.createObjectURL(videoFile);
      setVideoUrl(url);

      return () => {
        URL.revokeObjectURL(url);
      };
    }
  }, [videoFile]);

  const handleFileUpload = (file: File, type: 'suit' | 'gloveRight' | 'gloveLeft' | 'video') => {
    switch (type) {
      case 'suit':
        setSuitFile(file);
        break;
      case 'gloveRight':
        setGloveRightFile(file);
        break;
      case 'gloveLeft':
        setGloveLeftFile(file);
        break;
      case 'video':
        setVideoFile(file);
        break;
    }
  };

  const handleProcessFiles = async () => {
    if (!suitFile || !gloveRightFile || !gloveLeftFile) {
      toast({
        variant: 'destructive',
        title: 'Missing files',
        description: 'Please upload all required CSV files',
      });
      return;
    }

    if (!videoFile) {
      toast({
        variant: 'destructive',
        title: 'Missing video',
        description: 'Please upload a video file',
      });
      return;
    }

    setIsProcessing(true);

    try {
      const data = await processFiles(suitFile, gloveRightFile, gloveLeftFile);
      setSegmentData(data);

      toast({
        title: 'Processing complete',
        description: 'Sign language segments identified successfully',
      });
    } catch (error) {
      console.error('Error processing files:', error);
      toast({
        variant: 'destructive',
        title: 'Processing error',
        description: 'Failed to process files. Please try again.',
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setSuitFile(null);
    setGloveRightFile(null);
    setGloveLeftFile(null);
    setVideoFile(null);
    setVideoUrl(null);
    setSegmentData(null);
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const togglePlayPause = () => {
    if (videoRef1.current && videoRef2.current) {
      if (isPlaying) {
        videoRef1.current.pause();
        videoRef2.current.pause();
      } else {
        videoRef1.current.play();
        videoRef2.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleTimeUpdate = (time: number) => {
    setCurrentTime(time);

    if (videoRef1.current && videoRef2.current) {
      // Keep videos in sync
      const tolerance = 0.1;
      if (Math.abs(videoRef1.current.currentTime - videoRef2.current.currentTime) > tolerance) {
        videoRef2.current.currentTime = videoRef1.current.currentTime;
      }
    }
  };

  const allFilesUploaded = suitFile && gloveRightFile && gloveLeftFile && videoFile;

  return (
      <div className="min-h-screen bg-background">
        <header className="bg-white shadow">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
            <h1 className="text-xl font-semibold text-gray-900">Sign Sync Visualizer</h1>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600">Welcome, {user?.username}</span>
              <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    logout();
                    navigate('/login');
                  }}
              >
                Logout
              </Button>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {!segmentData ? (
              <Card>
                <CardContent className="p-6">
                  <h2 className="text-lg font-medium mb-4">Upload Files</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <FileUploader
                        label="Suit File (CSV)"
                        icon={<Upload size={24} />}
                        accept=".csv"
                        onFileSelected={(file) => handleFileUpload(file, 'suit')}
                        fileName={suitFile?.name}
                    />
                    <FileUploader
                        label="Right Glove File (CSV)"
                        icon={<Upload size={24} />}
                        accept=".csv"
                        onFileSelected={(file) => handleFileUpload(file, 'gloveRight')}
                        fileName={gloveRightFile?.name}
                    />
                    <FileUploader
                        label="Left Glove File (CSV)"
                        icon={<Upload size={24} />}
                        accept=".csv"
                        onFileSelected={(file) => handleFileUpload(file, 'gloveLeft')}
                        fileName={gloveLeftFile?.name}
                    />
                    <FileUploader
                        label="Video File"
                        icon={<FileVideo size={24} />}
                        accept="video/*"
                        onFileSelected={(file) => handleFileUpload(file, 'video')}
                        fileName={videoFile?.name}
                    />
                  </div>
                  <div className="mt-6 flex justify-end">
                    <Button
                        onClick={handleProcessFiles}
                        disabled={!allFilesUploaded || isProcessing}
                        className="w-full md:w-auto"
                    >
                      {isProcessing ? 'Processing...' : 'Process Files'}
                    </Button>
                  </div>
                </CardContent>
              </Card>
          ) : (
              <div className="space-y-6">
                <div className="flex justify-between items-center">
                  <h2 className="text-2xl font-bold">Sign Language Segments</h2>
                  <div className="flex gap-2">
                    <Button
                        variant="outline"
                        size="icon"
                        onClick={togglePlayPause}
                        className="rounded-full"
                    >
                      {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                    </Button>
                    <Button
                        variant="outline"
                        onClick={handleReset}
                        className="flex gap-2 items-center"
                    >
                      <RefreshCcw size={16} />
                      Reset
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-medium mb-2 flex items-center">
                      <div className="w-3 h-3 rounded-full bg-segment-bilstm mr-2"></div>
                      BiLSTM Model
                    </h3>
                    <VideoPlayer
                        videoRef={videoRef1}
                        videoUrl={videoUrl}
                        segments={segmentData.bilstm_segments}
                        currentTime={currentTime}
                        onTimeUpdate={handleTimeUpdate}
                        isPlaying={isPlaying}
                        setIsPlaying={setIsPlaying}
                        type="bilstm"
                        videoRef2={videoRef2}
                        muted={false}  // Left video with sound
                    />
                  </div>

                  <div>
                    <h3 className="text-lg font-medium mb-2 flex items-center">
                      <div className="w-3 h-3 rounded-full bg-segment-bigru mr-2"></div>
                      BiGRU Model
                    </h3>
                    <VideoPlayer
                        videoRef={videoRef2}
                        videoUrl={videoUrl}
                        segments={segmentData.bigru_segments}
                        currentTime={currentTime}
                        onTimeUpdate={handleTimeUpdate}
                        isPlaying={isPlaying}
                        setIsPlaying={setIsPlaying}
                        type="bigru"
                        videoRef2={videoRef1}
                        muted={true}  // Right video muted
                    />
                  </div>
                </div>
              </div>
          )}
        </main>
      </div>
  );
};

export default Dashboard;

