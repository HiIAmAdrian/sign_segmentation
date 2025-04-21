
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
  gloveLeftFile: File
): Promise<SegmentResponse> => {
  const formData = new FormData();
  formData.append('suit_file', suitFile);
  formData.append('glove_right_file', gloveRightFile);
  formData.append('glove_left_file', gloveLeftFile);

  try {
    const response = await fetch('http://localhost:5000/segment', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error calling the API:', error);
    throw error;
  }
};
