#include "face_detector.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <opencv2/imgproc.hpp> // For cvtColor

#include <librealsense2/rs.hpp>    // C++ RealSense API for bag processing
#include <librealsense2/rsutil.h>  // For C API rs2_deproject_pixel_to_point

#include <vector>
#include <cstdio>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <memory>
#include <thread> // For std::this_thread::sleep_for

struct DlibHandler {
    dlib::frontal_face_detector face_detector;
    dlib::shape_predictor shape_predictor;
};

void deproject_dlib_cpp(float point3d[3], const rs2_intrinsics* intrin, const float pixel[2], float depth_in_meters) {
    rs2_deproject_pixel_to_point(point3d, intrin, pixel, depth_in_meters);
}

void process_sentence_bag_cpp_logic(
    int sentence_num,
    const std::string& bag_filename,
    DlibHandler* dlib_handler_ptr,
    const std::string& output_csv_dir,
    double trim_duration_sec)
{
    std::cout << "C++: Processing sentence " << sentence_num << " from " << bag_filename
              << " (skipping first " << std::fixed << std::setprecision(2) << trim_duration_sec << "s) [Dlib C++]..." << std::endl;

    try {
        rs2::config cfg;
        cfg.enable_device_from_file(bag_filename, false); // false = do not repeat playback

        rs2::pipeline pipe;
        rs2::pipeline_profile profile = pipe.start(cfg);

        rs2::device device_from_profile = profile.get_device();
        if (!device_from_profile || !device_from_profile.is<rs2::playback>()) {
            std::cerr << "  C++ Error: Device from profile is not a valid playback device for file: " << bag_filename << std::endl;
            return;
        }
        rs2::playback playback = device_from_profile.as<rs2::playback>();

        if (trim_duration_sec > 0.001) {
            std::chrono::duration<double, std::micro> seek_time_us_double(trim_duration_sec * 1000000.0);
            std::chrono::microseconds seek_time_us(static_cast<long long>(seek_time_us_double.count()));
            std::cout << "  C++: Seeking playback to " << seek_time_us.count() << " us (" << trim_duration_sec << " s)..." << std::endl;
            playback.set_real_time(false); // Set before seek
            playback.seek(std::chrono::duration_cast<std::chrono::nanoseconds>(seek_time_us));
            std::cout << "  C++: Playback seek complete." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        rs2::align align_to_color(RS2_STREAM_COLOR);

        float depth_scale = 0.0f;
        rs2_intrinsics depth_intrinsics_cpp = {0}; // C++ SDK rs2_intrinsics
        bool intrinsics_found = false;
        bool scale_found = false;

        for (rs2::stream_profile sp : profile.get_streams()) {
            if (sp.stream_type() == RS2_STREAM_DEPTH && sp.is<rs2::video_stream_profile>()) {
                if (!intrinsics_found) {
                    depth_intrinsics_cpp = sp.as<rs2::video_stream_profile>().get_intrinsics();
                    intrinsics_found = true;
                    std::cout << "  DEBUG C++: Depth intrinsics found (w=" << depth_intrinsics_cpp.width
                              << ",h=" << depth_intrinsics_cpp.height << ",fx=" << depth_intrinsics_cpp.fx
                              << " from " << sp.stream_name() << ")." << std::endl;
                }
                // Find the sensor that provides this depth stream to get its scale
                for (rs2::sensor sensor : device_from_profile.query_sensors()) {
                    for (rs2::stream_profile sensor_sp : sensor.get_stream_profiles()) {
                        if (sensor_sp.unique_id() == sp.unique_id()) {
                            if (auto depth_s = sensor.as<rs2::depth_sensor>()) {
                                if (!scale_found) {
                                    depth_scale = depth_s.get_depth_scale();
                                    std::cout << "  DEBUG C++: Depth scale = " << depth_scale << " from sensor " << sensor.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
                                    if (depth_scale > 0.000001f) scale_found = true;
                                }
                                break;
                            }
                        }
                    }
                    if (scale_found) break;
                }
            }
            if (intrinsics_found && scale_found) break;
        }

        if (!intrinsics_found) { std::cerr << "C++ Error: Depth intrinsics not found!" << std::endl; pipe.stop(); return; }
        if (!scale_found || std::abs(depth_scale) < 0.000001f ) { // Check against a small epsilon
            std::cerr << "C++ Warning: Depth scale is effectively 0 or not found. Using default 0.001." << std::endl;
            depth_scale = 0.001f;
        }

        std::string csv_filepath = output_csv_dir + "/sentence_" + std::to_string(sentence_num) + "_dlib_landmarks_cpp.csv";
        std::ofstream csv_file(csv_filepath);
        if (!csv_file.is_open()) { std::cerr << "C++ Error: Cannot open CSV " << csv_filepath << std::endl; pipe.stop(); return; }
        csv_file << "sentence_id,frame_id,landmark_id,x_cam,y_cam,z_cam\n";

        int frames_processed_for_csv = 0;
        C_LandmarkPoint3D landmark_buffer[DLIB_LANDMARK_COUNT];
        std::cout << "  DEBUG C++: Entering processing loop..." << std::endl;
        int consecutive_poll_timeouts = 0;
        const int MAX_CONSECUTIVE_POLL_TIMEOUTS = 200;

        while (true) {
            rs2::frameset frames;
            if (pipe.poll_for_frames(&frames)) {
                consecutive_poll_timeouts = 0;

                rs2::frameset aligned_frames = align_to_color.process(frames);
                rs2::video_frame color_frame = aligned_frames.get_color_frame();
                rs2::depth_frame depth_frame = aligned_frames.get_depth_frame();

                if (!color_frame || !depth_frame) {
                    continue;
                }

                int color_w = color_frame.get_width();
                int color_h = color_frame.get_height();
                int color_stride = color_frame.get_stride_in_bytes();
                const unsigned char* color_data = (const unsigned char*)color_frame.get_data();
                rs2_format actual_recorded_format = color_frame.get_profile().format();
                bool color_data_is_rgb_from_bag = (actual_recorded_format == RS2_FORMAT_RGB8);

                int depth_w = depth_frame.get_width();
                int depth_h = depth_frame.get_height();
                const uint16_t* depth_data_ptr = (const uint16_t*)depth_frame.get_data();

                rs2_intrinsics c_depth_intrinsics; // C-style struct from rs_types.h
                c_depth_intrinsics.width = depth_intrinsics_cpp.width;
                c_depth_intrinsics.height = depth_intrinsics_cpp.height;
                c_depth_intrinsics.ppx = depth_intrinsics_cpp.ppx;
                c_depth_intrinsics.ppy = depth_intrinsics_cpp.ppy;
                c_depth_intrinsics.fx = depth_intrinsics_cpp.fx;
                c_depth_intrinsics.fy = depth_intrinsics_cpp.fy;
                c_depth_intrinsics.model = static_cast<rs2_distortion>(depth_intrinsics_cpp.model);
                for(int i=0; i<5; ++i) c_depth_intrinsics.coeffs[i] = depth_intrinsics_cpp.coeffs[i];

                int num_faces = detect_dlib_face_landmarks_3d(
                    static_cast<void*>(dlib_handler_ptr),
                    color_data, color_w, color_h, color_stride,
                    color_data_is_rgb_from_bag, // Pass the flag
                    depth_data_ptr, depth_w, depth_h, depth_scale,
                    &c_depth_intrinsics,
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
                if (frames_processed_for_csv > 0 && frames_processed_for_csv % 30 == 0) {
                    std::cout << "  C++ Processed " << frames_processed_for_csv << " frames for CSV..." << std::endl;
                }

            } else {
                uint64_t current_pos_ns = 0;
                try { current_pos_ns = playback.get_position(); } catch(const rs2::error&){}
                uint64_t duration_ns = 0;
                try { duration_ns = playback.get_duration().count(); } catch(const rs2::error&){}

                if (duration_ns > 0 && current_pos_ns >= duration_ns) {
                     std::cout << "  C++ Playback reached end of stream (pos >= dur). Exiting loop." << std::endl;
                     break;
                }
                consecutive_poll_timeouts++;
                if (consecutive_poll_timeouts > MAX_CONSECUTIVE_POLL_TIMEOUTS) {
                    std::cout << "  C++ Warning: Polled for frames " << MAX_CONSECUTIVE_POLL_TIMEOUTS
                              << " times without success. Assuming end or issue." << std::endl;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        if (csv_file.is_open()) csv_file.close();
        std::cout << "  C++ Saved Dlib landmark data to " << csv_filepath << " (" << frames_processed_for_csv << " frames written)." << std::endl;
        pipe.stop();
        std::cout << "  C++ Finished processing " << bag_filename << std::endl;

    } catch (const rs2::error & e) {
        std::cerr << "C++ RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "C++ Standard exception: " << e.what() << std::endl;
    }
}


extern "C" {
    void* initialize_dlib_landmark_detector(const char* shape_predictor_model_path) {
        DlibHandler* handler = new DlibHandler();
        try {
            handler->face_detector = dlib::get_frontal_face_detector();
            dlib::deserialize(shape_predictor_model_path) >> handler->shape_predictor;
            fprintf(stdout, "C++ Dlib: Landmark detector initialized successfully with '%s'.\n", shape_predictor_model_path);
        } catch (const std::exception& e) {
            fprintf(stderr, "C++ Dlib: ERROR: Exception during Dlib initialization: %s\n", e.what());
            delete handler;
            return nullptr;
        }
        return static_cast<void*>(handler);
    }

    int detect_dlib_face_landmarks_3d(void* detector_handle,
                                      const unsigned char* image_data, // Renamed bgr_image_data to image_data
                                      int width, int height, int stride,
                                      bool input_is_rgb, // NEW parameter
                                      const uint16_t* depth_frame_data,
                                      int depth_width, int depth_height,
                                      float depth_scale,
                                      const rs2_intrinsics* depth_intrinsics_c,
                                      C_LandmarkPoint3D detected_landmarks_arr[DLIB_LANDMARK_COUNT])
    {
        if (!detector_handle || !image_data || !depth_frame_data || !depth_intrinsics_c) {
             return -1;
        }
        if (width <= 0 || height <= 0 || depth_width <= 0 || depth_height <= 0 ) {
            return -1;
        }

        DlibHandler* handler = static_cast<DlibHandler*>(detector_handle);
        try {
            // Wrap the input data in a cv::Mat.
            cv::Mat input_cv_frame(height, width, CV_8UC3, (void*)image_data, stride);
             if (input_cv_frame.empty()) {
                return -1;
            }

            dlib::cv_image<dlib::bgr_pixel> dlib_img_bgr;
             if (input_is_rgb) {
                cv::Mat temp_bgr_frame;
                cv::cvtColor(input_cv_frame, temp_bgr_frame, cv::COLOR_RGB2BGR);
                dlib_img_bgr = dlib::cv_image<dlib::bgr_pixel>(temp_bgr_frame);
            } else {
                dlib_img_bgr = dlib::cv_image<dlib::bgr_pixel>(input_cv_frame);
            }

            std::vector<dlib::rectangle> faces = handler->face_detector(dlib_img_bgr);
            if (faces.empty()) return 0;

            dlib::full_object_detection shape = handler->shape_predictor(dlib_img_bgr, faces[0]);
            if (shape.num_parts() != DLIB_LANDMARK_COUNT) {
                return 0;
            }

            for (unsigned long i = 0; i < shape.num_parts(); ++i) {
                dlib::point p = shape.part(i);
                float pixel_x = static_cast<float>(p.x());
                float pixel_y = static_cast<float>(p.y());
                pixel_x = std::max(0.0f, std::min(pixel_x, (float)width - 1));
                pixel_y = std::max(0.0f, std::min(pixel_y, (float)height - 1));
                int depth_pixel_x = static_cast<int>(pixel_x + 0.5f);
                int depth_pixel_y = static_cast<int>(pixel_y + 0.5f);
                uint16_t depth_units = 0;
                if (depth_pixel_x >= 0 && depth_pixel_x < depth_width && depth_pixel_y >= 0 && depth_pixel_y < depth_height) {
                    depth_units = depth_frame_data[depth_pixel_y * depth_width + depth_pixel_x];
                }
                float depth_m = (float)depth_units * depth_scale;
                if (depth_m <= 0.001f) {
                    detected_landmarks_arr[i] = {0.0f, 0.0f, 0.0f};
                } else {
                    float pixel_coords[2] = {pixel_x, pixel_y};
                    float point3d[3];
                    deproject_dlib_cpp(point3d, depth_intrinsics_c, pixel_coords, depth_m);
                    detected_landmarks_arr[i] = {point3d[0], point3d[1], point3d[2]};
                }
            }
            return 1;
        } catch (const dlib::error& de) {
            fprintf(stderr, "C++ Dlib: Dlib Error during detection: %s\n", de.what());
            return -1;
        } catch (const cv::Exception& cve) {
            fprintf(stderr, "C++ Dlib: OpenCV Error during detection: %s\n", cve.what());
            return -1;
        } catch (const std::exception& e) {
            fprintf(stderr, "C++ Dlib: Standard Exception during detection: %s\n", e.what());
            return -1;
        }
    }

    void release_dlib_landmark_detector(void* detector_handle) {
        if (detector_handle) {
            DlibHandler* handler = static_cast<DlibHandler*>(detector_handle);
            delete handler;
            fprintf(stdout, "C++ Dlib: Landmark detector released.\n");
        }
    }

    void process_bag_for_landmarks_c_interface(
        int sentence_num,
        const char* bag_filename_c,
        void* dlib_handler_c,
        const char* output_csv_dir_c,
        double trim_duration_sec)
    {
        std::string bag_filename(bag_filename_c ? bag_filename_c : "");
        std::string output_csv_dir(output_csv_dir_c ? output_csv_dir_c : ".");
        DlibHandler* dlib_h = static_cast<DlibHandler*>(dlib_handler_c);

        if (bag_filename.empty() || !dlib_h) {
            fprintf(stderr, "C Wrapper Error: Invalid arguments to process_bag_for_landmarks_c_interface.\n");
            return;
        }
        process_sentence_bag_cpp_logic(sentence_num, bag_filename, dlib_h, output_csv_dir, trim_duration_sec);
    }

} // extern "C"