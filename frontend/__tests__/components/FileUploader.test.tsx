import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, act } from '@testing-library/react';
import React, { createRef } from 'react';
import VideoPlayer from '@/components/VideoPlayer';
import { Toaster } from '@/components/ui/toaster';

const mockToastFn = vi.fn();
vi.mock('@/components/ui/use-toast', () => ({
    useToast: () => ({
        toast: mockToastFn,
    }),
}));

describe('VideoPlayer', () => {
    const mockVideoRef = createRef<HTMLVideoElement>();
    const mockVideoRef2 = createRef<HTMLVideoElement>();
    const mockOnTimeUpdate = vi.fn();
    const mockSetIsPlaying = vi.fn();

    const defaultProps = {
        videoRef: mockVideoRef,
        videoUrl: 'mock-video.mp4',
        segments: [{ start_ms: 1000, end_ms: 3000 }],
        currentTime: 0,
        onTimeUpdate: mockOnTimeUpdate,
        isPlaying: false,
        setIsPlaying: mockSetIsPlaying,
        type: 'bilstm' as 'bilstm' | 'bigru',
        videoRef2: mockVideoRef2,
        muted: false,
    };

    const defineMediaProperties = (element: HTMLVideoElement | null, duration: number | undefined = undefined) => {
        if (element) {
            Object.defineProperty(element, 'duration', {
                writable: true,
                configurable: true,
                value: duration === undefined ? NaN : duration,
            });
            Object.defineProperty(element, 'networkState', {
                writable: true,
                configurable: true,
                value: HTMLMediaElement.NETWORK_EMPTY,
            });
            Object.defineProperty(element, 'readyState', {
                writable: true,
                configurable: true,
                value: HTMLMediaElement.HAVE_NOTHING,
            });
            Object.defineProperty(element, 'videoWidth', {
                writable: true,
                configurable: true,
                value: 640,
            });
            Object.defineProperty(element, 'videoHeight', {
                writable: true,
                configurable: true,
                value: 360,
            });
        }
    };


    beforeEach(() => {
        vi.clearAllMocks();

        const mockMediaElementBase = {
            play: vi.fn(() => Promise.resolve()),
            pause: vi.fn(),
            load: vi.fn(),
            addEventListener: vi.fn(),
            removeEventListener: vi.fn(),
            currentTime: 0,
            dispatchEvent: vi.fn(),
            error: null as MediaError | null,
        };

        // @ts-ignore
        mockVideoRef.current = { ...mockMediaElementBase } as HTMLVideoElement;
        // @ts-ignore
        mockVideoRef2.current = { ...mockMediaElementBase, play: vi.fn(), pause: vi.fn() } as HTMLVideoElement;

        defineMediaProperties(mockVideoRef.current);
        defineMediaProperties(mockVideoRef2.current);
    });

    const renderPlayer = (props = defaultProps) => {
        if (props.videoUrl && mockVideoRef.current) {
            defineMediaProperties(mockVideoRef.current, 10);
        } else if (mockVideoRef.current) {
            defineMediaProperties(mockVideoRef.current, undefined);
        }
        if (props.videoUrl && mockVideoRef2.current) {
            defineMediaProperties(mockVideoRef2.current, 10);
        } else if (mockVideoRef2.current){
            defineMediaProperties(mockVideoRef2.current, undefined);
        }

        return render(
            <>
                <VideoPlayer {...props} />
                <Toaster />
            </>
        );
    };

    it('renders video element if videoUrl is provided', () => {
        const { container } = renderPlayer();
        const videoElement = container.querySelector('video');
        expect(videoElement).toBeInTheDocument();
        expect(videoElement).toHaveAttribute('src', 'mock-video.mp4');
    });

    it('does not render video element if videoUrl is null', () => {
        const { container } = renderPlayer({ ...defaultProps, videoUrl: null });
        const videoElement = container.querySelector('video');
        expect(videoElement).not.toBeInTheDocument();
    });

    it('calls onTimeUpdate when video time updates (simulated)', () => {
        renderPlayer();
        const videoElement = mockVideoRef.current!;
        act(() => {
            const loadedMetadataEvent = new Event('loadedmetadata');
            videoElement.dispatchEvent(loadedMetadataEvent);
        });
        act(() => {
            mockVideoRef.current!.currentTime = 5;
            const timeUpdateEvent = new Event('timeupdate');
            videoElement.dispatchEvent(timeUpdateEvent);
        });
        expect(mockOnTimeUpdate).toHaveBeenCalledWith(5000);
    });

    it('applies overlay color when currentTime is within a segment', () => {
        const { container } = renderPlayer({ ...defaultProps, currentTime: 1500 });
        const overlayDiv = container.querySelector('video + div[style*="background-color"]');
        expect(overlayDiv).toHaveStyle('background-color: rgba(155, 135, 245, 0.16)');
    });

    it('does not apply overlay color when currentTime is outside a segment', () => {
        const { container } = renderPlayer({ ...defaultProps, currentTime: 500 });
        const overlayDiv = container.querySelector('video + div[style*="background-color"]');
        expect(overlayDiv).toHaveStyle({ backgroundColor: 'rgba(0, 0, 0, 0)' });
    });

    it('mutes video if muted prop is true', () => {
        const { container } = renderPlayer({ ...defaultProps, muted: true });
        const videoElement = container.querySelector('video');
        expect(videoElement).toHaveProperty('muted', true);
    });
});