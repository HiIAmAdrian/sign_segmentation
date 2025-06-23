import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Dashboard from '@/pages/Dashboard';
import { AuthProvider } from '@/contexts/AuthContext';
import { Toaster } from '@/components/ui/toaster';
import * as apiService from '@/services/apiService';

vi.mock('@/components/FileUploader', () => ({
    default: ({ label, onFileSelected, fileName, accept }: any) => (
        <div data-testid={`uploader-${label.split(' ')[0].toLowerCase()}`}>
            <input
                type="file"
                data-testid={`input-${label.split(' ')[0].toLowerCase()}`}
                onChange={(e) => onFileSelected(e.target.files?.[0])}
                accept={accept}
            />
            <span>{label}</span>
            {fileName && <span data-testid={`filename-${label.split(' ')[0].toLowerCase()}`}>{fileName}</span>}
        </div>
    ),
}));
vi.mock('@/components/VideoPlayer', () => ({
    default: ({ type }: any) => <div data-testid={`videoplayer-${type}`}>VideoPlayer Mock ({type})</div>,
}));

const mockLogout = vi.fn();
const mockNavigateDash = vi.fn();
vi.mock('@/contexts/AuthContext', async (importOriginal) => {
    const actual = await importOriginal<typeof import('@/contexts/AuthContext')>();
    return {
        ...actual,
        useAuth: () => ({
            user: { username: 'testuser' },
            isAuthenticated: true,
            logout: mockLogout,
            login: vi.fn(),
        }),
    };
});

vi.mock('react-router-dom', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router-dom')>();
    return {
        ...actual,
        useNavigate: () => mockNavigateDash,
    };
});

const mockToastFnDashboard = vi.fn();
vi.mock('@/components/ui/use-toast', () => ({
    useToast: () => ({
        toast: mockToastFnDashboard,
    }),
}));

const mockProcessFiles = vi.spyOn(apiService, 'processFiles');

describe('Dashboard Page', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    const renderDashboard = () => {
        return render(
            <BrowserRouter>
                <AuthProvider>
                    <Dashboard />
                    <Toaster />
                </AuthProvider>
            </BrowserRouter>
        );
    };

    it('renders file uploaders when no segment data exists', async () => {
        renderDashboard();
        expect(await screen.findByTestId('uploader-suit')).toBeInTheDocument();
        expect(screen.getByTestId('uploader-right')).toBeInTheDocument();
        expect(screen.getByTestId('uploader-left')).toBeInTheDocument();
        expect(screen.getByTestId('uploader-bag')).toBeInTheDocument();
        expect(screen.getByTestId('uploader-video')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Process Files/i })).toBeInTheDocument();
    });

    it('enables Process Files button when all required CSVs and BAG are uploaded', async () => {
        renderDashboard();
        const suitInput = screen.getByTestId('input-suit');
        const gloveRInput = screen.getByTestId('input-right');
        const gloveLInput = screen.getByTestId('input-left');
        const bagInput = screen.getByTestId('input-bag');
        const processButton = screen.getByRole('button', { name: /Process Files/i });

        const fakeFile = new File(['dummy'], 'dummy.csv', { type: 'text/csv' });
        const fakeBagFile = new File(['dummybag'], 'dummy.bag', { type: 'application/octet-stream' });

        expect(processButton).toBeDisabled();

        await act(async () => {
            fireEvent.change(suitInput, { target: { files: [fakeFile] } });
            fireEvent.change(gloveRInput, { target: { files: [fakeFile] } });
            fireEvent.change(gloveLInput, { target: { files: [fakeFile] } });
            fireEvent.change(bagInput, { target: { files: [fakeBagFile] } });
        });

        await waitFor(() => {
            expect(processButton).not.toBeDisabled();
        });
    });


    it('calls processFiles and shows video players on successful processing', async () => {
        const mockSegmentData = {
            bilstm_segments: [{ start_ms: 10, end_ms: 20 }],
            bigru_segments: [{ start_ms: 10, end_ms: 20 }],
            message: "Success", num_frames_processed: 100, num_features_final: 50, trim_applied_input_ms: 1000
        };
        mockProcessFiles.mockResolvedValue(mockSegmentData);
        renderDashboard();

        const suitInput = screen.getByTestId('input-suit');
        const gloveRInput = screen.getByTestId('input-right');
        const gloveLInput = screen.getByTestId('input-left');
        const bagInput = screen.getByTestId('input-bag');
        const fakeFile = new File(['dummy'], 'dummy.csv', { type: 'text/csv' });
        const fakeBagFile = new File(['dummybag'], 'dummy.bag', { type: 'application/octet-stream' });

        await act(async () => {
            fireEvent.change(suitInput, { target: { files: [fakeFile] } });
            fireEvent.change(gloveRInput, { target: { files: [fakeFile] } });
            fireEvent.change(gloveLInput, { target: { files: [fakeFile] } });
            fireEvent.change(bagInput, { target: { files: [fakeBagFile] } });
        });

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /Process Files/i }));
        });

        expect(mockProcessFiles).toHaveBeenCalledTimes(1);
        await waitFor(() => {
            expect(screen.getByTestId('videoplayer-bilstm')).toBeInTheDocument();
            expect(screen.getByTestId('videoplayer-bigru')).toBeInTheDocument();
            expect(mockToastFnDashboard).toHaveBeenCalledWith(expect.objectContaining({ title: 'Processing complete' }));
        });
    });

    it('handles reset button correctly', async () => {
        mockProcessFiles.mockResolvedValue({ bilstm_segments: [], bigru_segments: [] } as any);
        renderDashboard();
        const fakeFile = new File(['dummy'], 'dummy.csv', { type: 'text/csv' });
        const fakeBagFile = new File(['dummybag'], 'dummy.bag', { type: 'application/octet-stream' });
        await act(async () => {
            fireEvent.change(screen.getByTestId('input-suit'), { target: { files: [fakeFile] } });
            fireEvent.change(screen.getByTestId('input-right'), { target: { files: [fakeFile] } });
            fireEvent.change(screen.getByTestId('input-left'), { target: { files: [fakeFile] } });
            fireEvent.change(screen.getByTestId('input-bag'), { target: { files: [fakeBagFile] } });
        });
        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /Process Files/i }));
        });
        await waitFor(() => expect(screen.getByTestId('videoplayer-bilstm')).toBeInTheDocument());

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /Reset/i }));
        });

        await waitFor(() => {
            expect(screen.getByTestId('uploader-suit')).toBeInTheDocument();
            expect(screen.queryByTestId('videoplayer-bilstm')).not.toBeInTheDocument();
        });
    });

    it('calls logout and navigates on logout button click', async () => {
        renderDashboard();
        await act(async () => {
            fireEvent.click(screen.getByRole('button', {name: /Logout/i}));
        });
        expect(mockLogout).toHaveBeenCalled();
        expect(mockNavigateDash).toHaveBeenCalledWith('/login');
    });

    it('creates object URL when a video file for visualization is selected', async () => {
        renderDashboard();
        const videoInput = screen.getByTestId('input-video');
        const fakeVideoFile = new File(['videocontent'], 'video.mp4', { type: 'video/mp4' });

        await act(async () => {
            fireEvent.change(videoInput, { target: { files: [fakeVideoFile] } });
        });

        await waitFor(() => {
            expect(global.URL.createObjectURL).toHaveBeenCalledWith(fakeVideoFile);
        });
    });
});