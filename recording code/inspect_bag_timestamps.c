#include <librealsense2/rs.h>
#include <librealsense2/h/rs_pipeline.h>
#include <librealsense2/h/rs_frame.h>
#include <librealsense2/h/rs_config.h>
#include <librealsense2/h/rs_device.h> // For playback device
#include <librealsense2/h/rs_record_playback.h> // For rs2_playback_status etc.
#include <librealsense2/h/rs_sensor.h> // For stream profile info

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h> // For PRIu64

// --- RealSense Error Checking ---
void check_rs_error(rs2_error* e, const char* context_msg) {
    if (e) {
        fprintf(stderr, "%s: RealSense error calling %s(%s):\n    %s\n",
                context_msg,
                rs2_get_failed_function(e),
                rs2_get_failed_args(e),
                rs2_get_error_message(e));
        rs2_free_error(e);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_bag_file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* bag_filename = argv[1];

    rs2_error* e = NULL;
    rs2_context* ctx = NULL;
    rs2_pipeline* pipeline = NULL;
    rs2_config* config = NULL;
    rs2_pipeline_profile* profile = NULL;
    rs2_device* playback_dev = NULL;

    printf("Inspecting timestamps in '%s'...\n", bag_filename);

    // --- Setup Pipeline for Playback ---
    ctx = rs2_create_context(RS2_API_VERSION, &e);
    check_rs_error(e, "Create context");

    config = rs2_create_config(&e);
    check_rs_error(e, "Create config");
    rs2_config_enable_device_from_file_repeat_option(config, bag_filename, 0, &e); // 0 = no repeat
    check_rs_error(e, "Enable device from file (no repeat)");
    rs2_config_enable_stream(config, RS2_STREAM_DEPTH, -1, 0, 0, RS2_FORMAT_ANY, 0, &e);
    if(e) { fprintf(stderr, "Warning: Could not configure depth stream: %s\n", rs2_get_error_message(e)); rs2_free_error(e); e=NULL; }
    rs2_config_enable_stream(config, RS2_STREAM_COLOR, -1, 0, 0, RS2_FORMAT_ANY, 0, &e);
     if(e) { fprintf(stderr, "Warning: Could not configure color stream: %s\n", rs2_get_error_message(e)); rs2_free_error(e); e=NULL; }

    pipeline = rs2_create_pipeline(ctx, &e);
    check_rs_error(e, "Create pipeline");

    profile = rs2_pipeline_start_with_config(pipeline, config, &e);
    if (e || !profile) { /* ... error handling ... */ return EXIT_FAILURE; }

    playback_dev = rs2_pipeline_profile_get_device(profile, &e);
    check_rs_error(e, "Get device from profile");
    if (!rs2_is_device_extendable_to(playback_dev, RS2_EXTENSION_PLAYBACK, &e)) { /* ... error handling ... */ exit(EXIT_FAILURE); }
    check_rs_error(e, "Check playback extension");
    rs2_playback_device_set_real_time(playback_dev, 0, &e); // 0 = false
    check_rs_error(e, "Set playback non-real-time");

    printf("Pipeline started. Reading frames...\n");
    printf("---------------------------------------------------------------------------\n");
    printf("Frame# | Depth TS (ms)    | Color TS (ms)    | D-C Diff (ms) | D-D Diff (ms) | C-C Diff (ms)\n");
    printf("---------------------------------------------------------------------------\n");

    unsigned long long frame_number = 0;
    double last_depth_ts = -1.0;
    double last_color_ts = -1.0;

    while (true) {
        rs2_frame* frames = NULL; // Holds frame(s) from wait_for_frames
        int success = rs2_pipeline_try_wait_for_frames(pipeline, &frames, 1000, &e);
        check_rs_error(e, "try_wait_for_frames");

        if (!success || !frames) { // Check end of stream or timeout
            rs2_playback_status current_status = rs2_playback_device_get_current_status(playback_dev, &e);
            check_rs_error(e, "get playback status after wait failure");
            if (current_status == RS2_PLAYBACK_STATUS_STOPPED) {
                printf("\nPlayback status STOPPED, likely end of file.\n");
                if (frames) rs2_release_frame(frames);
                break;
            } else {
                 if (frames) rs2_release_frame(frames);
                 continue;
            }
        }

        rs2_frame* depth_frame_ref = NULL; // Ptr to keep depth frame (if found)
        rs2_frame* color_frame_ref = NULL; // Ptr to keep color frame (if found)
        double current_depth_ts = -1.0;
        double current_color_ts = -1.0;

        // --- Process Frame(s) ---
        if (rs2_is_frame_extendable_to(frames, RS2_EXTENSION_COMPOSITE_FRAME, &e)) {
            int frame_count_in_set = rs2_embedded_frames_count(frames, &e);
            check_rs_error(e, "embedded frames count");

            for (int i = 0; i < frame_count_in_set; ++i) {
                rs2_frame* single_frame = rs2_extract_frame(frames, i, &e); // Ref count = 1
                check_rs_error(e, "extract frame in loop");
                if (!single_frame) continue;

                const rs2_stream_profile* f_profile = rs2_get_frame_stream_profile(single_frame, &e);
                check_rs_error(e, "get profile in loop");

                rs2_stream stream_type_local;
                rs2_format format_local; int index_local; int unique_id_local; int framerate_local;
                rs2_get_stream_profile_data(f_profile, &stream_type_local, &format_local, &index_local, &unique_id_local, &framerate_local, &e);
                check_rs_error(e, "get stream type data in loop");

                if (stream_type_local == RS2_STREAM_DEPTH && !depth_frame_ref) {
                    depth_frame_ref = single_frame; // Store pointer
                    rs2_frame_add_ref(depth_frame_ref, &e); // Add our ref (count = 2)
                    check_rs_error(e, "add ref depth composite");
                } else if (stream_type_local == RS2_STREAM_COLOR && !color_frame_ref) {
                    color_frame_ref = single_frame; // Store pointer
                    rs2_frame_add_ref(color_frame_ref, &e); // Add our ref (count = 2)
                    check_rs_error(e, "add ref color composite");
                }
                // Always release the reference from extract_frame
                rs2_release_frame(single_frame); // Ref count = 1 if kept, 0 if not
            }
        } else { // Handle non-composite frame case
             const rs2_stream_profile* f_profile = rs2_get_frame_stream_profile(frames, &e);
             check_rs_error(e, "get profile (single frame)");

             rs2_stream stream_type_local;
             rs2_format format_local; int index_local; int unique_id_local; int framerate_local;
             rs2_get_stream_profile_data(f_profile, &stream_type_local, &format_local, &index_local, &unique_id_local, &framerate_local, &e);
             check_rs_error(e, "get stream type data (single frame)");

             if (stream_type_local == RS2_STREAM_DEPTH) {
                 depth_frame_ref = frames; // Store pointer to original 'frames'
                 rs2_frame_add_ref(depth_frame_ref, &e); // Add our ref (count = 2)
                 check_rs_error(e, "add ref depth single");
             } else if (stream_type_local == RS2_STREAM_COLOR) {
                 color_frame_ref = frames; // Store pointer to original 'frames'
                 rs2_frame_add_ref(color_frame_ref, &e); // Add our ref (count = 2)
                 check_rs_error(e, "add ref color single");
             }
             // Original 'frames' ref (count = 1 or 2) released below
        }
        rs2_release_frame(frames); // Release original ref from try_wait_for_frames (count = 1 if kept, 0 if not)
        // --- End Process Frame(s) ---


        // --- Process the extracted frame references ---
        char depth_ts_str[20] = "N/A             ";
        char color_ts_str[20] = "N/A             ";
        char dc_diff_str[20] = "N/A          ";
        char dd_diff_str[20] = "N/A          ";
        char cc_diff_str[20] = "N/A          ";

        if (depth_frame_ref) { // Process if we found and kept a depth frame
            current_depth_ts = rs2_get_frame_timestamp(depth_frame_ref, &e); check_rs_error(e, "get depth ts");
            sprintf(depth_ts_str, "%-16.3f", current_depth_ts);
            if (last_depth_ts >= 0) {
                sprintf(dd_diff_str, "%-13.3f", current_depth_ts - last_depth_ts);
            }
            last_depth_ts = current_depth_ts;
            rs2_release_frame(depth_frame_ref); // Release our added reference
        }

        if (color_frame_ref) { // Process if we found and kept a color frame
            current_color_ts = rs2_get_frame_timestamp(color_frame_ref, &e); check_rs_error(e, "get color ts");
            sprintf(color_ts_str, "%-16.3f", current_color_ts);
            if (last_color_ts >= 0) {
                sprintf(cc_diff_str, "%-13.3f", current_color_ts - last_color_ts);
            }
            last_color_ts = current_color_ts;
             rs2_release_frame(color_frame_ref); // Release our added reference
        }

        if (current_depth_ts >= 0 && current_color_ts >= 0) {
             sprintf(dc_diff_str, "%-13.3f", current_depth_ts - current_color_ts);
        }

        printf("%-6llu | %s | %s | %s | %s | %s\n",
               frame_number, depth_ts_str, color_ts_str, dc_diff_str, dd_diff_str, cc_diff_str);

        frame_number++;

    } // End while loop

    printf("---------------------------------------------------------------------------\n");
    printf("Inspection complete.\n");

    // --- Cleanup ---
    printf("Cleaning up resources...\n");
    if (profile) rs2_delete_pipeline_profile(profile);
    if (config) rs2_delete_config(config);
    if (pipeline) {
        rs2_pipeline_stop(pipeline, &e);
        if(e) { fprintf(stderr,"Error stopping pipeline: %s\n", rs2_get_error_message(e)); rs2_free_error(e); e=NULL; }
        rs2_delete_pipeline(pipeline);
    }
    if (ctx) rs2_delete_context(ctx);
    printf("Cleanup complete.\n");

    return EXIT_SUCCESS;
}