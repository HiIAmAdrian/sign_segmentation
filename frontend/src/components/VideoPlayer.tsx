import { useState, useEffect, RefObject } from 'react';
import { cn } from '@/lib/utils';
import { useToast } from '@/components/ui/use-toast';

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
    muted?: boolean;
}

const overlayColors: Record<'bilstm' | 'bigru', string> = {
    bilstm: 'rgba(155, 135, 245, 0.16)',
    bigru: 'rgba(249, 115, 22, 0.12)',
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
    const { toast } = useToast();

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const handleError = (e: Event) => {
            console.error('Video Error:', e);
            const videoElement = e.target as HTMLVideoElement;
            if (videoElement.error) {
                console.error('Video Error Code:', videoElement.error.code);
                console.error('Video Error Message:', videoElement.error.message);
            }
            toast({
                variant: 'destructive',
                title: "Video Playback Error",
                description: `Could not load or play video. Code: ${videoElement.error?.code}`,
            });
        };

        const handleLoadedMetadata = () => {
            setDuration(video.duration);
        };

        const handleTimeUpdate = () => {
            if (video) {
                onTimeUpdate(video.currentTime * 1000); // Convert to ms
            }
        };

        const handlePlay = (event: Event) => {
            setIsPlaying(true);
            if (videoRef2.current && videoRef2.current.paused) {
                videoRef2.current.currentTime = video.currentTime;
                videoRef2.current.play().catch(e => console.error("Error playing videoRef2 on play sync:", e));
            }
        };

        const handlePause = (event: Event) => {
            setIsPlaying(false);
            if (videoRef2.current && !videoRef2.current.paused) {
                videoRef2.current.pause();
            }
        };

        video.addEventListener('loadedmetadata', handleLoadedMetadata);
        video.addEventListener('timeupdate', handleTimeUpdate);
        video.addEventListener('play', handlePlay);
        video.addEventListener('pause', handlePause);
        video.addEventListener('error', handleError);

        return () => {
            video.removeEventListener('loadedmetadata', handleLoadedMetadata);
            video.removeEventListener('timeupdate', handleTimeUpdate);
            video.removeEventListener('play', handlePlay);
            video.removeEventListener('pause', handlePause);
            video.removeEventListener('error', handleError);
        };
           }, [videoRef, videoUrl, onTimeUpdate, setIsPlaying, videoRef2, type, toast]);

    const isInsideSegment = segments.some(
        (segment) => currentTime >= segment.start_ms && currentTime <= segment.end_ms
    );

    const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (!videoRef.current || !duration) { // Use the state variable 'duration'
            console.warn(`VideoPlayer (${type}): Timeline click ignored. VideoRef or duration not ready. Duration: ${duration}`);
            return;
        }

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
                        playsInline
                    />
                )}
                <div
                    className="absolute top-0 left-0 w-full h-full pointer-events-none transition-all"
                    style={{
                        backgroundColor: isInsideSegment ? overlayColors[type] : 'transparent',
                        zIndex: 2,
                        transition: "background-color 0.3s"
                    }}
                />
            </div>

            {videoUrl && duration > 0 && (
                <div className="px-4 py-2 bg-gray-900">
                    <div
                        data-testid="video-timeline"
                        className="h-6 w-full bg-gray-800 rounded relative cursor-pointer"
                        onClick={handleTimelineClick}
                        style={{ position: 'relative' }}
                    >
                        <div
                            className="h-full bg-primary/50 rounded-l"
                            style={{
                                width: `${(currentTime / 1000) / duration * 100}%`,
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                zIndex: 1,
                            }}
                        />

                        {segments.map((segment, index) => {
                            const startPercent = (segment.start_ms / 1000) / duration * 100;
                            const endPercent = (segment.end_ms / 1000) / duration * 100;
                            const widthPercent = Math.max(0, endPercent - startPercent);

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

                        <div
                            className="absolute top-0 w-0.5 h-full bg-white"
                            style={{
                                left: `${(currentTime / 1000) / duration * 100}%`,
                                zIndex: 3,
                            }}
                        />
                    </div>
                </div>
            )}
        </div>
    );
};

export default VideoPlayer;