// Dashboard.tsx
import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { processFiles, SegmentResponse } from '../services/apiService';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useToast } from '@/components/ui/use-toast';
import { Upload, Video, RefreshCcw, FileVideo, Play, Pause, Archive } from 'lucide-react'; // Added Archive
import FileUploader from '../components/FileUploader';
import VideoPlayer from '../components/VideoPlayer';

const Dashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [suitFile, setSuitFile] = useState<File | null>(null);
  const [gloveRightFile, setGloveRightFile] = useState<File | null>(null);
  const [gloveLeftFile, setGloveLeftFile] = useState<File | null>(null);
  const [bagFile, setBagFile] = useState<File | null>(null); // New state for BAG file
  const [videoFile, setVideoFile] = useState<File | null>(null); // This is for the player
  const [videoUrl, setVideoUrl] = useState<string | null>(null); // For the player

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

  // This useEffect is for the VISUALIZATION video file
  useEffect(() => {
    if (videoFile) {
      const url = URL.createObjectURL(videoFile);
      setVideoUrl(url);
      return () => {
        URL.revokeObjectURL(url);
      };
    } else {
      setVideoUrl(null); // Clear video URL if no video file
    }
  }, [videoFile]);

  const handleFileUpload = (file: File, type: 'suit' | 'gloveRight' | 'gloveLeft' | 'video' | 'bag') => {
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
      case 'video': // For the player
        setVideoFile(file);
        break;
      case 'bag': // For backend processing
        setBagFile(file);
        break;
    }
  };

  const handleProcessFiles = async () => {
    if (!suitFile || !gloveRightFile || !gloveLeftFile) {
      toast({
        variant: 'destructive',
        title: 'Missing CSV files',
        description: 'Please upload all required CSV files',
      });
      return;
    }

    if (!bagFile) { // Check for BAG file
      toast({
        variant: 'destructive',
        title: 'Missing BAG file',
        description: 'Please upload a .bag file for processing',
      });
      return;
    }

    // Note: videoFile (for player) is not required for processing

    setIsProcessing(true);

    try {
      const data = await processFiles(suitFile, gloveRightFile, gloveLeftFile, bagFile); // Pass bagFile
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
    setBagFile(null); // Reset BAG file
    setVideoFile(null); // Reset video file for player
    setVideoUrl(null);
    setSegmentData(null);
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const togglePlayPause = () => {
    if (!videoUrl) { // Cannot play if no video is loaded for visualization
      toast({ title: "No video loaded", description: "Please upload a video file for visualization.", variant: "default" });
      return;
    }
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
    // Sync logic remains the same
    if (videoRef1.current && videoRef2.current && videoUrl) {
      const tolerance = 0.1;
      if (Math.abs(videoRef1.current.currentTime - videoRef2.current.currentTime) > tolerance) {
        videoRef2.current.currentTime = videoRef1.current.currentTime;
      }
    }
  };

  // Files required for backend processing
  const allFilesForProcessingUploaded = suitFile && gloveRightFile && gloveLeftFile && bagFile;

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
                  <h2 className="text-lg font-medium mb-4">Upload Files for Processing</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <FileUploader
                        label="Suit File (CSV)"
                        icon={<Upload size={24} />} // Consider using FileText or FileCode from lucide-react
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
                    <FileUploader // Uploader for BAG file (sent to backend)
                        label="BAG File (.bag)"
                        icon={<Archive size={24} />}
                        accept=".bag"
                        onFileSelected={(file) => handleFileUpload(file, 'bag')}
                        fileName={bagFile?.name}
                    />
                  </div>
                  <h2 className="text-lg font-medium mb-4 pt-4 border-t">Upload Video for Visualization (Optional)</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <FileUploader // Uploader for Video file (for frontend player only)
                        label="Video File (e.g., .mp4, .webm)"
                        icon={<FileVideo size={24} />}
                        accept="video/*"
                        onFileSelected={(file) => handleFileUpload(file, 'video')}
                        fileName={videoFile?.name}
                    />
                    <div /> {/* Placeholder for grid layout if only one item in this row */}
                  </div>
                  <div className="mt-8 flex justify-end">
                    <Button
                        onClick={handleProcessFiles}
                        disabled={!allFilesForProcessingUploaded || isProcessing}
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
                        disabled={!videoUrl} // Disable if no visualization video is loaded
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

                {!videoUrl && (
                    <div className="p-4 text-center text-muted-foreground bg-secondary rounded-md">
                      No video uploaded for visualization. Segments are processed, but playback is unavailable.
                      You can upload a video on the previous screen by clicking "Reset".
                    </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-medium mb-2 flex items-center">
                      <div className="w-3 h-3 rounded-full bg-segment-bilstm mr-2"></div>
                      BiLSTM Model
                    </h3>
                    <VideoPlayer
                        videoRef={videoRef1}
                        videoUrl={videoUrl} // Uses the separate videoUrl for playback
                        segments={segmentData.bilstm_segments}
                        currentTime={currentTime}
                        onTimeUpdate={handleTimeUpdate}
                        isPlaying={isPlaying}
                        setIsPlaying={setIsPlaying}
                        type="bilstm"
                        videoRef2={videoRef2}
                        muted={false}
                    />
                  </div>

                  <div>
                    <h3 className="text-lg font-medium mb-2 flex items-center">
                      <div className="w-3 h-3 rounded-full bg-segment-bigru mr-2"></div>
                      BiGRU Model
                    </h3>
                    <VideoPlayer
                        videoRef={videoRef2}
                        videoUrl={videoUrl} // Uses the separate videoUrl for playback
                        segments={segmentData.bigru_segments}
                        currentTime={currentTime}
                        onTimeUpdate={handleTimeUpdate}
                        isPlaying={isPlaying}
                        setIsPlaying={setIsPlaying}
                        type="bigru"
                        videoRef2={videoRef1}
                        muted={true}
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