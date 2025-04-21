
import { useState, useEffect, RefObject } from 'react';
import { cn } from '@/lib/utils';

interface Segment {
  start_ms: number;
  end_ms: number;
}

interface VideoPlayerProps {
  videoRef: RefObject<HTMLVideoElement>;
  videoUrl: string | null;
  segments: Segment[];
  currentTime: number;
  onTimeUpdate: (time: number) => void;
  isPlaying: boolean;
  setIsPlaying: (playing: boolean) => void;
  type: 'bilstm' | 'bigru';
  videoRef2: RefObject<HTMLVideoElement>;
  muted?: boolean; // Added muted prop to control audio
}

const overlayColors: Record<'bilstm' | 'bigru', string> = {
  bilstm: 'rgba(155, 135, 245, 0.16)', // softer purple
  bigru: 'rgba(249, 115, 22, 0.12)',   // softer orange
};

const VideoPlayer = ({
                       videoRef,
                       videoUrl,
                       segments,
                       currentTime,
                       onTimeUpdate,
                       isPlaying,
                       setIsPlaying,
                       type,
                       videoRef2,
                       muted = false,
                     }: VideoPlayerProps) => {
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      setDuration(video.duration);
    };

    const handleTimeUpdate = () => {
      if (video) {
        onTimeUpdate(video.currentTime * 1000); // Convert to ms
      }
    };

    const handlePlay = () => {
      setIsPlaying(true);
      if (videoRef2.current) {
        videoRef2.current.currentTime = video.currentTime;
        videoRef2.current.play();
      }
    };

    const handlePause = () => {
      setIsPlaying(false);
      if (videoRef2.current) {
        videoRef2.current.pause();
      }
    };

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
    };
  }, [videoRef, onTimeUpdate, setIsPlaying, videoRef2]);

  // Only color overlay if current frame is inside any segment
  const isInsideSegment = segments.some(
      (segment) => currentTime >= segment.start_ms && currentTime <= segment.end_ms
  );

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!videoRef.current || !duration) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const offsetX = e.clientX - rect.left;
    const percent = offsetX / rect.width;
    const newTime = percent * duration;

    videoRef.current.currentTime = newTime;
    if (videoRef2.current) {
      videoRef2.current.currentTime = newTime;
    }
  };

  return (
      <div className="rounded-lg overflow-hidden bg-black">
        <div className="relative">
          {videoUrl && (
              <video
                  ref={videoRef}
                  src={videoUrl}
                  className="w-full h-auto"
                  controls={false}
                  muted={muted}
              />
          )}

          {/* Full-frame color overlay only when inside a segment */}
          <div
              className="absolute top-0 left-0 w-full h-full pointer-events-none transition-all"
              style={{
                backgroundColor: isInsideSegment ? overlayColors[type] : 'transparent',
                zIndex: 2,
                transition: "background-color 0.3s"
              }}
          />
          {/* No segment bars in the video overlay anymore */}
        </div>

        {/* Custom timeline */}
        <div className="px-4 py-2 bg-gray-900">
          <div
              className="h-6 w-full bg-gray-800 rounded relative cursor-pointer"
              onClick={handleTimelineClick}
              style={{ position: 'relative' }}
          >
            {/* Progress bar */}
            <div
                className="h-full bg-primary/50 rounded-l"
                style={{
                  width: `${(currentTime / 1000) / (duration || 1) * 100}%`,
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  zIndex: 1,
                }}
            />

            {/* Segment timeline bars (keep only here, NOT in video area) */}
            {segments.map((segment, index) => {
              const startPercent = (segment.start_ms / 1000) / (duration || 1) * 100;
              const endPercent = (segment.end_ms / 1000) / (duration || 1) * 100;
              const widthPercent = endPercent - startPercent;

              return (
                  <div
                      key={index}
                      className={cn(
                          "absolute h-full timeline-segment",
                          type
                      )}
                      style={{
                        left: `${startPercent}%`,
                        width: `${widthPercent}%`,
                        zIndex: 2,
                      }}
                  />
              );
            })}

            {/* Time indicator */}
            <div
                className="absolute top-0 w-0.5 h-full bg-white"
                style={{
                  left: `${(currentTime / 1000) / (duration || 1) * 100}%`,
                  zIndex: 3,
                }}
            />
          </div>
        </div>
      </div>
  );
};

export default VideoPlayer;

