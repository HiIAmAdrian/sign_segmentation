import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { processFiles, SegmentResponse } from '@/services/apiService';

describe('apiService', () => {
    const mockSuitFile = new File(['suit_data'], 'suit.csv', { type: 'text/csv' });
    const mockGloveRFile = new File(['gloveR_data'], 'gloveR.csv', { type: 'text/csv' });
    const mockGloveLFile = new File(['gloveL_data'], 'gloveL.csv', { type: 'text/csv' });
    const mockBagFile = new File(['bag_data'], 'test.bag', { type: 'application/octet-stream' });

    const mockSuccessResponse: SegmentResponse = {
        bilstm_segments: [{ start_ms: 100, end_ms: 200 }],
        bigru_segments: [{ start_ms: 150, end_ms: 250 }],
    };

    beforeEach(() => {
        global.fetch = vi.fn();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('processFiles should successfully call the API and return data', async () => {
        (fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
            ok: true,
            json: async () => mockSuccessResponse,
        } as Response);

        const result = await processFiles(mockSuitFile, mockGloveRFile, mockGloveLFile, mockBagFile);

        expect(fetch).toHaveBeenCalledTimes(1);
        expect(fetch).toHaveBeenCalledWith('http://localhost:5000/segment_pipeline', {
            method: 'POST',
            body: expect.any(FormData),
        });

        const fetchCall = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
        const formData = fetchCall[1].body as FormData;
        expect(formData.get('suit_file')).toEqual(mockSuitFile);
        expect(formData.get('glove_right_file')).toEqual(mockGloveRFile);
        expect(formData.get('glove_left_file')).toEqual(mockGloveLFile);
        expect(formData.get('bag_file')).toEqual(mockBagFile);

        expect(result).toEqual(mockSuccessResponse);
    });

    it('processFiles should throw an error if API response is not ok', async () => {
        const errorText = 'Internal Server Error';
        (fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
            ok: false,
            status: 500,
            text: async () => errorText,
        } as Response);

        await expect(
            processFiles(mockSuitFile, mockGloveRFile, mockGloveLFile, mockBagFile)
        ).rejects.toThrow(`API error: 500 - ${errorText}`);
    });

    it('processFiles should throw an error on network failure', async () => {
        const networkError = new Error('Network failed');
        (fetch as ReturnType<typeof vi.fn>).mockRejectedValue(networkError);

        await expect(
            processFiles(mockSuitFile, mockGloveRFile, mockGloveLFile, mockBagFile)
        ).rejects.toThrow(networkError.message);
    });
});