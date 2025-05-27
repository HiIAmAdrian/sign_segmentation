#include <librealsense2/rs.hpp> // RealSense C++ API
#include <opencv2/opencv.hpp>   // OpenCV
#include <iostream>
#include <string>
#include <vector>
#include <algorithm> // For std::transform, std::tolower
#include <thread>    // For std::this_thread::sleep_for
#include <chrono>    // For std::chrono
#include <filesystem> // For directory iteration (C++17)

namespace fs = std::filesystem;

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

// Renamed helper to avoid conflict with SDK's rs2_format_to_string
std::string app_rs2_format_to_string(rs2_format format) {
    // You can call the SDK's C function if you prefer, and convert its char* to std::string
    // return rs2_format_to_string(format);
    // Or use your custom implementation:
    switch (format) {
        case RS2_FORMAT_ANY: return "ANY";
        case RS2_FORMAT_Z16: return "Z16";
        case RS2_FORMAT_DISPARITY16: return "DISPARITY16";
        case RS2_FORMAT_XYZ32F: return "XYZ32F";
        case RS2_FORMAT_YUYV: return "YUYV";
        case RS2_FORMAT_RGB8: return "RGB8";
        case RS2_FORMAT_BGR8: return "BGR8";
        case RS2_FORMAT_RGBA8: return "RGBA8";
        case RS2_FORMAT_BGRA8: return "BGRA8";
        case RS2_FORMAT_Y8: return "Y8";
        case RS2_FORMAT_Y16: return "Y16";
        case RS2_FORMAT_RAW10: return "RAW10";
        case RS2_FORMAT_RAW16: return "RAW16";
        case RS2_FORMAT_RAW8: return "RAW8";
        case RS2_FORMAT_UYVY: return "UYVY";
        case RS2_FORMAT_MOTION_RAW: return "MOTION_RAW";
        case RS2_FORMAT_MOTION_XYZ32F: return "MOTION_XYZ32F";
        case RS2_FORMAT_GPIO_RAW: return "GPIO_RAW";
        case RS2_FORMAT_6DOF: return "6DOF";
        case RS2_FORMAT_DISPARITY32: return "DISPARITY32";
        case RS2_FORMAT_Y10BPACK: return "Y10BPACK";
        case RS2_FORMAT_DISTANCE: return "DISTANCE";
        case RS2_FORMAT_MJPEG: return "MJPEG";
        case RS2_FORMAT_Y8I: return "Y8I";
        case RS2_FORMAT_Y12I: return "Y12I";
        case RS2_FORMAT_INZI: return "INZI";
        case RS2_FORMAT_INVI: return "INVI";
        case RS2_FORMAT_W10: return "W10";
        case RS2_FORMAT_Z16H: return "Z16H";
        default:
            // Fallback to SDK function if available and you want to ensure all formats are covered
            return rs2_format_to_string(format);
            //return "UNKNOWN_FORMAT (" + std::to_string(format) + ")";
    }
}


bool convert_bag_to_mp4(const std::string& input_bag_filename, const std::string& output_video_filename, double output_fps) {
    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "Processing: " << input_bag_filename << std::endl;
    std::cout << "Outputting to: " << output_video_filename << " at " << output_fps << " FPS." << std::endl;

    rs2::pipeline pipe;
    rs2::config cfg;
    cv::VideoWriter video_writer;
    bool writer_initialized = false;
    int frame_width = 0;
    int frame_height = 0;
    unsigned long long frame_count = 0;
    bool pipeline_started = false;

    try {
        cfg.enable_device_from_file(input_bag_filename, false); 
        
        std::cout << "Attempting to start pipeline for: " << input_bag_filename << std::endl;
        rs2::pipeline_profile profile = pipe.start(cfg);
        pipeline_started = true;
        std::cout << "Pipeline started successfully." << std::endl;

        std::cout << "Available streams in " << fs::path(input_bag_filename).filename().string() << ":" << std::endl;
        auto streams = profile.get_streams();
        if (streams.empty()) {
            std::cout << "  No streams found in the profile!" << std::endl;
        } else {
            for (const auto& stream_profile : streams) {
                std::cout << "  - Stream: " << rs2_stream_to_string(stream_profile.stream_type())
                          << ", Format: " << rs2_format_to_string(stream_profile.format())
                          << ", Index: " << stream_profile.stream_index()
                          << ", UID: " << stream_profile.unique_id()
                          << ", FPS: " << stream_profile.fps();
                if (auto vp = stream_profile.as<rs2::video_stream_profile>()) {
                    std::cout << ", Resolution: " << vp.width() << "x" << vp.height();
                }
                std::cout << std::endl;
            }
        }


        rs2::device device = profile.get_device();
        if (!device || !device.is<rs2::playback>()) {
            std::cerr << "Error: Could not get playback device from bag file: " << input_bag_filename << std::endl;
            if (pipeline_started) pipe.stop();
            return false;
        }
        rs2::playback playback = device.as<rs2::playback>();
        playback.set_real_time(false); 

        std::cout << "Playback device obtained. Processing frames..." << std::endl;
        rs2_format color_format_from_bag = RS2_FORMAT_ANY;
        int consecutive_poll_timeouts = 0;
        const int MAX_CONSECUTIVE_POLL_TIMEOUTS = 300; // ~15 seconds if bag is truly sparse and sleep is 50ms
                                                      // ~3 seconds if sleep is 10ms
        int poll_attempts = 0;

        while (true) {
            poll_attempts++;
            // std::cout << "Poll attempt #" << poll_attempts << std::endl; // Potentially too verbose
            rs2::frameset frames;
            if (pipe.poll_for_frames(&frames)) {
                 std::cout << "poll_for_frames returned true. Frameset size: " << frames.size() << std::endl;
                consecutive_poll_timeouts = 0; 

                rs2::video_frame color_frame = frames.get_color_frame();

                if (color_frame) {
                    // std::cout << "Color frame obtained." << std::endl;
                    if (!writer_initialized) {
                        frame_width = color_frame.get_width();
                        frame_height = color_frame.get_height();
                        color_format_from_bag = color_frame.get_profile().format();
                        std::cout << "First color frame received. Resolution: " << frame_width << "x" << frame_height
                                  << ", Format: " << rs2_format_to_string(color_format_from_bag) 
                                  << ", Stride: " << color_frame.get_stride_in_bytes() << std::endl;

                        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); 

                        if (!video_writer.open(output_video_filename, fourcc, output_fps, cv::Size(frame_width, frame_height), true)) {
                            std::cerr << "Error: Could not open VideoWriter for output file: " << output_video_filename << std::endl;
                            if (pipeline_started) pipe.stop();
                            return false;
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
                        if (frame_count % 300 == 0) { 
                            std::cout << "Processed " << frame_count << " frames for " << fs::path(input_bag_filename).filename().string() << "..." << std::endl;
                        }
                    }
                } else { // color_frame was null
					std::cout << "Poll successful, but no color_frame in frameset. Frameset size: " << frames.size() << std::endl;
					if(frames.size() > 0) {
    					std::cout << "  Available frames in this set:" << std::endl;
    					for(const auto& f : frames) {
      						  std::cout << "    - Type: " << rs2_stream_to_string(f.get_profile().stream_type())
                  					<< ", Format: " << rs2_format_to_string(f.get_profile().format()) << std::endl;
    					}
					}
                }
            } else { // poll_for_frames returned false
                rs2_playback_status status = RS2_PLAYBACK_STATUS_UNKNOWN;
                uint64_t pos = 0;
                std::chrono::nanoseconds dur(0);
                try { status = playback.current_status(); } catch (const rs2::error&e) { std::cerr << "Err getting status: " << e.what() << std::endl; }
                try { pos = playback.get_position(); } catch (const rs2::error&e) { std::cerr << "Err getting pos: " << e.what() << std::endl; }
                try { dur = playback.get_duration(); } catch (const rs2::error&e) { std::cerr << "Err getting dur: " << e.what() << std::endl; }


                std::cout << "poll_for_frames returned false. Status: " << rs2_playback_status_to_string(status) 
                           << ", Pos: " << pos << ", Dur: " << dur.count() << std::endl;


                if (status == RS2_PLAYBACK_STATUS_STOPPED) {
                     std::cout << "Playback stopped (end of stream or no more frames after " << poll_attempts << " polls). Processed " << frame_count << " frames." << std::endl;
                     break;
                } else if (status == RS2_PLAYBACK_STATUS_PAUSED) {
                     std::chrono::nanoseconds current_pos_ns(pos);
                     std::chrono::nanoseconds total_duration_ns = dur;

                     if (total_duration_ns.count() > 0 && current_pos_ns >= total_duration_ns) {
                         std::cout << "Playback paused at end of stream (after " << poll_attempts << " polls). Processed " << frame_count << " frames." << std::endl;
                         break;
                     }
                } else if (status == RS2_PLAYBACK_STATUS_UNKNOWN && consecutive_poll_timeouts > 0) {
                    // If status is unknown after some timeouts, it might also be an issue or end
                    if (dur.count() > 0 && pos >= dur.count()) {
                         std::cout << "Playback status unknown but position indicates end of stream. Processed " << frame_count << " frames." << std::endl;
                         break;
                    }
                }


                consecutive_poll_timeouts++;
                if (consecutive_poll_timeouts > MAX_CONSECUTIVE_POLL_TIMEOUTS) {
                    std::cout << "Warning: Polled for frames " << MAX_CONSECUTIVE_POLL_TIMEOUTS
                              << " times (" << poll_attempts << " total polls) without success. Assuming end or issue for " << fs::path(input_bag_filename).filename().string() << "." << std::endl;
                     std::cout << "Final playback status: " << rs2_playback_status_to_string(status) << ", position: " << pos 
                               << " ns, duration: " << dur.count() << " ns." << std::endl;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Increased sleep slightly
            }
        } // while (true)

    } catch (const rs2::error & e) {
        check_rs_error_cpp(e, "RealSense exception during processing " + input_bag_filename, false); 
        if (video_writer.isOpened()) { video_writer.release(); }
        if (pipeline_started) { try { pipe.stop(); } catch(...) {} }
        std::cerr << "Conversion FAILED for: " << input_bag_filename << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception during processing " << input_bag_filename << ": " << e.what() << std::endl;
        if (video_writer.isOpened()) { video_writer.release(); }
        if (pipeline_started) { try { pipe.stop(); } catch(...) {} }
        std::cerr << "Conversion FAILED for: " << input_bag_filename << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        return false;
    }

    if (video_writer.isOpened()) {
        video_writer.release();
    }
    std::cout << "VideoWriter released. Total frames written for " << fs::path(output_video_filename).filename().string() << ": " << frame_count << std::endl;

    if (pipeline_started) {
        try {
            pipe.stop(); 
            std::cout << "Pipeline stopped for " << fs::path(input_bag_filename).filename().string() << "." << std::endl;
        } catch (const rs2::error& e) {
            std::cerr << "Warning: Error stopping pipeline for " << fs::path(input_bag_filename).filename().string() << ": " << e.what() << std::endl;
        }
    }
    
    if (frame_count == 0 && writer_initialized) {
        std::cout << "Warning: No frames were written to " << output_video_filename << ". The file might be empty or the color stream was not processable." << std::endl;
    } else if (frame_count == 0 && !writer_initialized) {
        std::cout << "Warning: No color frames found or processed in " << input_bag_filename << ". Output file not created or empty." << std::endl;
        // If video_writer was never initialized, output_video_filename might not exist, 
        // or it might be a 0-byte file if video_writer.open() created it but no frames were written before release.
        // OpenCV's VideoWriter behavior on open() vs write() can vary slightly by backend/OS.
    }

    std::cout << "Conversion finished for: " << input_bag_filename << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_directory_with_bag_files> [output_fps (default: 60)]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string input_directory_str = argv[1];
    double output_fps = 60.0;

    if (argc > 2) {
        try {
            output_fps = std::stod(argv[2]);
            if (output_fps <= 0) {
                std::cerr << "Error: FPS must be positive." << std::endl;
                return EXIT_FAILURE;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid FPS value '" << argv[2] << "': " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    fs::path input_dir(input_directory_str);
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "Error: Input path '" << input_directory_str << "' is not a valid directory." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Scanning directory: " << input_dir.string() << " for .bag files." << std::endl;
    std::cout << "Output FPS set to: " << output_fps << std::endl;

    std::vector<fs::path> bag_files;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bag") {
            bag_files.push_back(entry.path());
        }
    }

    if (bag_files.empty()) {
        std::cout << "No .bag files found in directory: " << input_dir.string() << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "Found " << bag_files.size() << " .bag file(s) to process." << std::endl;
    int success_count = 0;
    int failure_count = 0;

    for (const auto& bag_file_path : bag_files) {
        fs::path output_video_path = bag_file_path;
        output_video_path.replace_extension(".mp4");

        if (convert_bag_to_mp4(bag_file_path.string(), output_video_path.string(), output_fps)) {
            success_count++;
        } else {
            failure_count++;
        }
    }

    std::cout << "\nBatch processing summary:" << std::endl;
    std::cout << "Successfully converted: " << success_count << " file(s)." << std::endl;
    std::cout << "Failed to convert: " << failure_count << " file(s)." << std::endl;

    if (failure_count > 0) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}