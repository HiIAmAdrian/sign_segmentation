// apiService.ts
interface SegmentData {
  start_ms: number;
  end_ms: number;
}

export interface SegmentResponse {
  bilstm_segments: SegmentData[];
  bigru_segments: SegmentData[];
}

export const processFiles = async (
    suitFile: File,
    gloveRightFile: File,
    gloveLeftFile: File,
    bagFile: File // Changed from videoFile to bagFile
): Promise<SegmentResponse> => {
  const formData = new FormData();
  formData.append('suit_file', suitFile);
  formData.append('glove_right_file', gloveRightFile);
  formData.append('glove_left_file', gloveLeftFile);
  formData.append('bag_file', bagFile); // Changed key to 'bag_file' and using bagFile

  try {
    const response = await fetch('http://localhost:5000/segment', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.text(); // Get more error details
      throw new Error(`API error: ${response.status} - ${errorBody}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error calling the API:', error);
    throw error;
  }
};