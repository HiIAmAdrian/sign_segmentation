#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <stdbool.h>
#include <stdint.h>
#include <librealsense2/h/rs_types.h> // For rs2_intrinsics

#ifdef __cplusplus
extern "C" {
#endif

 // Struct for a 3D landmark point (in camera coordinates)
 typedef struct {
  float x;
  float y;
  float z;
 } C_LandmarkPoint3D;

#define DLIB_LANDMARK_COUNT 68 // Dlib's 68-point model

 // Dlib Landmark Detection Functions
 void* initialize_dlib_landmark_detector(const char* shape_predictor_model_path);
 int detect_dlib_face_landmarks_3d(void* detector_handle,
                                   const unsigned char* bgr_image_data,
                                   int width, int height, int stride,
                                   bool input_is_rgb,
                                   const uint16_t* depth_frame_data,
                                   int depth_width, int depth_height,
                                   float depth_scale,
                                   const rs2_intrinsics* depth_intrinsics,
                                   C_LandmarkPoint3D detected_landmarks_arr[DLIB_LANDMARK_COUNT]);
 void release_dlib_landmark_detector(void* detector_handle);

 // NEW: C-Callable Wrapper for Bag Processing
 void process_bag_for_landmarks_c_interface(
     int sentence_num,
     const char* bag_filename_c,
     void* dlib_handler_c, // This is your DlibHandler*
     const char* output_csv_dir_c,
     double trim_duration_sec);

#ifdef __cplusplus
}
#endif

#endif // FACE_DETECTOR_H