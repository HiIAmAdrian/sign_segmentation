#include <librealsense2/rs.hpp> // RealSense C++ API
#include <opencv2/opencv.hpp>   // OpenCV
#include <iostream>
#include <string>
#include <vector>
#include <algorithm> // For std::transform, std::tolower
#include <thread>    // For std::this_thread::sleep_for
#include <chrono>    // For std::chrono

// Helper to check for RealSense errors (C++ style)
void check_rs_error_cpp(const rs2::error& e, const std::string& context_msg, bool exit_on_error = true) {
    // Only treat actual errors as fatal, not timeouts for wait_for_frames if handled by polling
    if (e.get_type() != RS2_EXCEPTION_TYPE_UNKNOWN &&
        std::string(e.what()).find("Frame didn't arrive within") == std::string::npos) {
        std::cerr << context_msg << ": RealSense error calling "
                  << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    "
                  << e.what() << std::endl;
        if (exit_on_error) {
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_bag_file> <output_video_file (e.g., video.avi or video.mp4)> [fps]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string input_bag_filename = argv[1];
    const std::string output_video_filename = argv[2];
    double output_fps = 60.0;

    if (argc > 3) {
        try {
            output_fps = std::stod(argv[3]);
            if (output_fps <= 0) {
                std::cerr << "Error: FPS must be positive." << std::endl;
                return EXIT_FAILURE;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid FPS value: " << argv[3] << " - " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "Converting '" << input_bag_filename << "' to '" << output_video_filename << "' at " << output_fps << " FPS." << std::endl;

    rs2::pipeline pipe;
    rs2::config cfg;
    cv::VideoWriter video_writer;
    bool writer_initialized = false;
    int frame_width = 0;
    int frame_height = 0;
    unsigned long long frame_count = 0; // Moved declaration here

    try {
        cfg.enable_device_from_file(input_bag_filename, false);
        rs2::pipeline_profile profile = pipe.start(cfg);

        rs2::device device = profile.get_device();
        if (!device || !device.is<rs2::playback>()) {
            std::cerr << "Error: Could not get playback device from bag file." << std::endl;
            return EXIT_FAILURE;
        }
        rs2::playback playback = device.as<rs2::playback>();
        playback.set_real_time(false);

        std::cout << "Pipeline started from bag file. Processing frames..." << std::endl;
        rs2_format color_format_from_bag = RS2_FORMAT_ANY;
        int consecutive_poll_timeouts = 0;
        const int MAX_CONSECUTIVE_POLL_TIMEOUTS = 300; // Approx 3 seconds if polling at 10ms

        while (true) {
            rs2::frameset frames;
            if (pipe.poll_for_frames(&frames)) {
                consecutive_poll_timeouts = 0; // Reset on successful frame retrieval

                rs2::video_frame color_frame = frames.get_color_frame();

                if (color_frame) {
                    if (!writer_initialized) {
                        frame_width = color_frame.get_width();
                        frame_height = color_frame.get_height();
                        color_format_from_bag = color_frame.get_profile().format();
                        std::cout << "First color frame received. Resolution: " << frame_width << "x" << frame_height
                                  << ", Format: " << rs2_format_to_string(color_format_from_bag)
                                  << ", Stride: " << color_frame.get_stride_in_bytes() << std::endl;

                        int fourcc = 0;
                        std::string ext = output_video_filename.substr(output_video_filename.find_last_of(".") + 1);
                        std::transform(ext.begin(), ext.end(), ext.begin(),
                                       [](unsigned char c){ return std::tolower(c); });


                        if (ext == "avi") {
                            fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                        } else if (ext == "mp4") {
                            fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                        } else {
                            std::cerr << "Warning: Unknown video extension '" << ext << "'. Defaulting to MJPG." << std::endl;
                            fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                        }

                        if (!video_writer.open(output_video_filename, fourcc, output_fps, cv::Size(frame_width, frame_height), true)) {
                            std::cerr << "Error: Could not open VideoWriter for output file: " << output_video_filename << std::endl;
                            pipe.stop(); // Stop before exiting
                            return EXIT_FAILURE;
                        }
                        writer_initialized = true;
                        std::cout << "VideoWriter initialized. Writing frames..." << std::endl;
                    }

                    cv::Mat frame_mat;
                    const void* frame_data = color_frame.get_data();

                    if (color_format_from_bag == RS2_FORMAT_BGR8) {
                        frame_mat = cv::Mat(cv::Size(frame_width, frame_height), CV_8UC3, (void*)frame_data, color_frame.get_stride_in_bytes());
                    } else if (color_format_from_bag == RS2_FORMAT_RGB8) {
                        cv::Mat rgb_mat(cv::Size(frame_width, frame_height), CV_8UC3, (void*)frame_data, color_frame.get_stride_in_bytes());
                        cv::cvtColor(rgb_mat, frame_mat, cv::COLOR_RGB2BGR);
                    } else if (color_format_from_bag == RS2_FORMAT_YUYV) {
                        cv::Mat yuv_mat(cv::Size(frame_width, frame_height), CV_8UC2, (void*)frame_data, color_frame.get_stride_in_bytes());
                        cv::cvtColor(yuv_mat, frame_mat, cv::COLOR_YUV2BGR_YUYV);
                    } else if (color_format_from_bag == RS2_FORMAT_UYVY) {
                        cv::Mat yuv_mat(cv::Size(frame_width, frame_height), CV_8UC2, (void*)frame_data, color_frame.get_stride_in_bytes());
                        cv::cvtColor(yuv_mat, frame_mat, cv::COLOR_YUV2BGR_UYVY);
                    } else {
                        std::cerr << "Warning: Unsupported color format in bag file: " << rs2_format_to_string(color_format_from_bag) << ". Skipping frame." << std::endl;
                        continue;
                    }

                    if (!frame_mat.empty()) {
                        video_writer.write(frame_mat);
                        frame_count++;
                        if (frame_count % 100 == 0) {
                            std::cout << "Processed " << frame_count << " frames..." << std::endl;
                        }
                    }
                } // if (color_frame)
            } else { // poll_for_frames returned false
                uint64_t current_pos_ns = 0;
                uint64_t duration_ns = 0;
                try { current_pos_ns = playback.get_position(); } catch(const rs2::error&){}
                try { duration_ns = playback.get_duration().count(); } catch(const rs2::error&){}

                // Fix for comparison warning: cast get_position() to long long if duration_ns is not 0
                // or compare std::chrono::nanoseconds objects directly
                bool end_of_stream = false;
                if (duration_ns > 0) {
                    if (current_pos_ns >= duration_ns) {
                        end_of_stream = true;
                    }
                } else { // If duration is 0, rely on timeouts
                    if (playback.current_status() == RS2_PLAYBACK_STATUS_STOPPED){ // Fallback check
                        end_of_stream = true;
                    }
                }


                if (end_of_stream) {
                     std::cout << "  C++ Playback reached end of stream. Exiting loop." << std::endl;
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
        } // while (true)

    } catch (const rs2::error & e) {
        check_rs_error_cpp(e, "RealSense exception during processing");
        // Ensure pipeline is stopped and writer released in case of error
        if (video_writer.isOpened()) { video_writer.release(); }
       // try { if(pipe) pipe.stop(); } catch(...) {} // Check if pipe was initialized (e.g. profile is valid)
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        if (video_writer.isOpened()) { video_writer.release(); }
//        try { if(pipe) pipe.stop(); } catch(...) {}
        return EXIT_FAILURE;
    }

    if (video_writer.isOpened()) {
        video_writer.release();
    }
    std::cout << "VideoWriter released. Total frames written: " << frame_count << std::endl;

    try {
        pipe.stop(); // Stop the pipeline if it was started
        std::cout << "\nPipeline stopped." << std::endl;
    } catch (const rs2::error& e) {
        // std::cerr << "Error stopping pipeline: " << e.what() << std::endl;
        // Potentially ignore if stop fails after processing or if it was never really started.
    }


    std::cout << "Conversion finished." << std::endl;
    return EXIT_SUCCESS;
}