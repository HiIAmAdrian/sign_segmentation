// This would be in a new .cpp file or within your existing face_landmark_detector_dlib.cpp
// It would NOT be called directly from your C main(), but is for illustration.
// To integrate, you'd need a C-callable wrapper around this C++ logic.

#include <librealsense2/rs.hpp> // C++ API
#include "face_detector.h"     // For C_LandmarkPoint3D and DLIB_LANDMARK_COUNT
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h> // If still using cv_image for dlib
#include <opencv2/imgproc.hpp>  // For cvtColor if needed by dlib::cv_image

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

// If rs_frame_image is a custom class from the example, you'd need its definition.
// For now, let's assume dlib::cv_image or dlib::array2d<rgb_pixel> directly from cv::Mat
// which is a common way.

// Forward declare the C-callable Dlib functions if they are in this file
// (or include the header that declares them if they are in a separate Dlib wrapper .cpp)
// extern "C" void* initialize_dlib_landmark_detector(const char* model_path);
// extern "C" int detect_dlib_face_landmarks_3d(...);
// extern "C" void release_dlib_landmark_detector(void* handle);

// Assuming DlibHandler struct is defined as before
struct DlibHandler {
    dlib::frontal_face_detector face_detector;
    dlib::shape_predictor shape_predictor;
};


void process_sentence_bag_cpp_version(
    int sentence_num,
    const std::string& bag_filename,
    DlibHandler* dlib_handler, // Pass the DlibHandler directly
    const std::string& output_csv_dir,
    double trim_duration_sec)
{
    std::cout << "Processing sentence " << sentence_num << " from " << bag_filename
              << " (skipping first " << std::fixed << std::setprecision(2) << trim_duration_sec << "s) [Dlib C++]..." << std::endl;

    try {
        rs2::config cfg;
        cfg.enable_device_from_file(bag_filename, false); // false = do not repeat playback

        rs2::pipeline pipe;
        rs2::pipeline_profile profile = pipe.start(cfg);

        rs2::device playback_dev = profile.get_device();
        if (!playback_dev.is<rs2::playback>()) {
            std::cerr << "  Error: Device from profile is not a playback device!" << std::endl;
            return;
        }
        rs2::playback playback = playback_dev.as<rs2::playback>();

        if (trim_duration_sec > 0.0) {
            std::chrono::duration<double, std::micro> seek_time_us(trim_duration_sec * 1000000.0);
            std::cout << "  Seeking playback to " << seek_time_us.count() << " us (" << trim_duration_sec << " s)..." << std::endl;
            playback.set_real_time(false); // Important for seek and frame-by-frame
            playback.seek(std::chrono::duration_cast<std::chrono::nanoseconds>(seek_time_us));
            std::cout << "  Playback seek complete." << std::endl;
            // rs_utils::sleep(200); // Optional small delay
        }

        rs2::align align_to_color(RS2_STREAM_COLOR);

        // Get Depth Scale and Intrinsics
        float depth_scale = 0.0f;
        rs2_intrinsics depth_intrinsics;
        bool intrinsics_found = false;
        for (auto&& sensor : playback_dev.query_sensors()) {
            if (sensor.is<rs2::depth_sensor>()) {
                depth_scale = sensor.as<rs2::depth_sensor>().get_depth_scale();
                 std::cout << "  DEBUG: Depth scale = " << depth_scale << std::endl;
                for (auto&& sprofile : sensor.get_stream_profiles()) {
                    if (sprofile.is<rs2::video_stream_profile>()) {
                        depth_intrinsics = sprofile.as<rs2::video_stream_profile>().get_intrinsics();
                        intrinsics_found = true;
                        std::cout << "  DEBUG: Depth intrinsics found (fx=" << depth_intrinsics.fx << ", fy=" << depth_intrinsics.fy << ")." << std::endl;
                        break;
                    }
                }
            }
            if (depth_scale != 0.0f && intrinsics_found) break;
        }
        if (!intrinsics_found) { std::cerr << "Error: Depth intrinsics not found!" << std::endl; return; }
        if (depth_scale == 0.0f) { std::cerr << "Warning: Depth scale is 0. Using 1.0." << std::endl; depth_scale = 1.0f; }


        std::string csv_filepath = output_csv_dir + "/sentence_" + std::to_string(sentence_num) + "_dlib_landmarks_cpp.csv";
        std::ofstream csv_file(csv_filepath);
        if (!csv_file.is_open()) { std::cerr << "Error: Cannot open CSV " << csv_filepath << std::endl; return; }
        csv_file << "sentence_id,frame_id,landmark_id,x_cam,y_cam,z_cam\n";

        int frames_processed_for_csv = 0;
        C_LandmarkPoint3D landmark_buffer[DLIB_LANDMARK_COUNT]; // From your C header

        std::cout << "  DEBUG: Entering C++ processing loop..." << std::endl;

        while (true) {
            rs2::frameset frames;
            if (!pipe.poll_for_frames(&frames)) { // Non-blocking poll
                if (playback.get_current_status() == RS2_PLAYBACK_STATUS_STOPPED) {
                    std::cout << "  Playback status STOPPED. Exiting C++ loop." << std::endl;
                    break;
                }
                // rs_utils::sleep(1); // Small sleep if polling
                continue; // No new frame yet
            }

            rs2::frameset aligned_frames = align_to_color.process(frames);

            rs2::video_frame color_frame = aligned_frames.get_color_frame();
            rs2::depth_frame depth_frame = aligned_frames.get_depth_frame();

            if (!color_frame || !depth_frame) {
                // std::cerr << "  Warning: Missing color or depth frame in aligned set for CSV frame " << frames_processed_for_csv << std::endl;
                continue;
            }

            // Get data for Dlib
            int color_w = color_frame.get_width();
            int color_h = color_frame.get_height();
            int color_stride = color_frame.get_stride_in_bytes();
            const unsigned char* color_data = (const unsigned char*)color_frame.get_data();

            int depth_w = depth_frame.get_width();
            int depth_h = depth_frame.get_height();
            const uint16_t* depth_data_ptr = (const uint16_t*)depth_frame.get_data();


            // --- Call Dlib C-Interface function (or integrate Dlib C++ directly here) ---
            // Assuming your C++ Dlib wrapper is still used as an extern "C" block
            int num_faces = detect_dlib_face_landmarks_3d(
                static_cast<void*>(dlib_handler),
                color_data, color_w, color_h, color_stride,
                depth_data_ptr, depth_w, depth_h, depth_scale,
                &depth_intrinsics,
                landmark_buffer
            );

            if (num_faces > 0) {
                for (int i = 0; i < DLIB_LANDMARK_COUNT; ++i) {
                    csv_file << sentence_num << "," << frames_processed_for_csv << "," << i << ","
                             << std::fixed << std::setprecision(6) << landmark_buffer[i].x << ","
                             << std::fixed << std::setprecision(6) << landmark_buffer[i].y << ","
                             << std::fixed << std::setprecision(6) << landmark_buffer[i].z << "\n";
                }
            }
            frames_processed_for_csv++;
            if (frames_processed_for_csv % 30 == 0) { // Print progress
                std::cout << "  Processed " << frames_processed_for_csv << " frames for CSV..." << std::endl;
            }
        } // End while loop

        if (csv_file.is_open()) csv_file.close();
        std::cout << "  Saved Dlib landmark data to " << csv_filepath << " (" << frames_processed_for_csv << " frames written)." << std::endl;
        pipe.stop(); // Stop this instance of the pipeline
        std::cout << "  Finished processing " << bag_filename << std::endl;

    } catch (const rs2::error & e) {
        std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }
}