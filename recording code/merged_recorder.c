#include <librealsense2/rs.h>
#include <librealsense2/h/rs_pipeline.h>
#include <librealsense2/h/rs_frame.h>
#include <librealsense2/h/rs_config.h>
// #include <librealsense2/h/rs_record_playback.h> // Not strictly needed by this main recorder
// #include <librealsense2/h/rs_sensor.h>       // Not strictly needed by this main recorder

// #include "face_detector.h"   // Dlib/RS processing - NOT CALLED BY THIS MAIN RECORDER
#include "teslasuit_handler.h"  // Interface for Teslasuit

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <signal.h>
#include <inttypes.h>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#define Sleep(x) usleep(x*1000)
int _kbhit() {
    struct termios oldt, newt; int ch; int oldf;
    tcgetattr(STDIN_FILENO, &oldt); newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);
    if(ch != EOF) { ungetc(ch, stdin); return 1; }
    return 0;
}
int _getch() { return getchar(); }
#endif

#define MAX_FILENAME_LEN 256
volatile sig_atomic_t app_should_run = 1;

void signal_handler(int dummy) { (void)dummy; app_should_run = 0; printf("\nCtrl+C pressed. Exiting gracefully...\n"); }

void check_rs_error(rs2_error* e, const char* context_msg) {
    if (e) {
        // Simplified error print for this recorder
        fprintf(stderr, "\n%s: RealSense error: %s\n",
                context_msg, rs2_get_error_message(e));
        rs2_free_error(e);
        // Decide if exit is appropriate here or if main loop should handle
        // For now, let main loop handle recording state based on function returns
    }
}

void delete_file_if_exists(const char* filename) {
    if (remove(filename) == 0) {
        printf("  INFO: Deleted previous file: %s\n", filename);
    } else {
        // printf("  INFO: File %s not found or could not be deleted.\n", filename);
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "\033[31mUsage: %s <output_directory> [start_sentence_num]\033[0m\n", argv[0]);
        fprintf(stderr, "\033[31mExample: %s ./output_data 10\033[0m\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* output_dir_base = argv[1];
    int start_sentence_num = 1;
    if (argc > 2) { // Changed to argc > 2 for start_sentence_num
        start_sentence_num = atoi(argv[2]);
        if (start_sentence_num < 1) {
            fprintf(stderr, "\033[33mWarning: Invalid start_sentence_num '%s', defaulting to 1.\033[0m\n", argv[2]);
            start_sentence_num = 1;
        }
    }

    signal(SIGINT, signal_handler);

    printf("Initializing Teslasuit System...\n");
    if (!ts_initialize_system()) {
        fprintf(stderr, "\033[31m-------------Fatal: Could not initialize Teslasuit API.\033[0m\n");
        return EXIT_FAILURE;
    }
    printf("Discovering and opening Teslasuit devices...\n");
    ts_discover_and_open_devices();
    printf("Teslasuit initialization complete.\n");

    rs2_error* e = NULL;
    rs2_context* rec_ctx = rs2_create_context(RS2_API_VERSION, &e);
    if (e) { // Check error immediately after creation
        fprintf(stderr, "\033[31mmain: ---------------RealSense error creating context: %s\033[0m\n", rs2_get_error_message(e));
        rs2_free_error(e);
        ts_close_all_devices();
        ts_uninitialize_system();
        return EXIT_FAILURE;
    }
    printf("RealSense context for recording initialized.\n");

    int sentence_count = start_sentence_num;
    bool is_recording = false;
    rs2_pipeline* rs_rec_pipeline = NULL;
    rs2_config* rs_rec_config = NULL;

    printf("Combined Teslasuit & RealSense Mocap Recorder\n");
    printf("Press [ENTER] to start/stop ALL recordings for a sentence.\n");
    printf("Press [B] (then Enter) to RE-RECORD previous sentence (when NOT recording).\n");
    printf("Press [G] (then Enter) to CALIBRATE BOTH GLOVES (when NOT recording).\n");
    printf("Press [S] (then Enter) for SUIT Calibration (when NOT recording).\n");
    printf("Press [Q] then [ENTER] to quit (when NOT recording).\n\n");

    while (app_should_run) {
        if (is_recording) {
            printf("\r\033[32mRECORDING Sentence %03d...\033[0m Press [ENTER] to STOP. ", sentence_count);
            fflush(stdout);
            if (rs_rec_pipeline) {
                rs2_frame* frames = rs2_pipeline_wait_for_frames(rs_rec_pipeline, 1000, &e);
                if (frames) {
                    rs2_release_frame(frames);
                } else if (e) {
                    fprintf(stderr, "\n\033[31m-----------RealSense ERROR during recording: %s\033[0m\n", rs2_get_error_message(e));
                    rs2_free_error(e); e = NULL;
                    printf("\033[33m---------------Critical RealSense error during recording. Attempting to stop all...\033[0m\n");
                    if(rs_rec_pipeline) { rs2_pipeline_stop(rs_rec_pipeline, NULL); rs2_delete_pipeline(rs_rec_pipeline); rs_rec_pipeline = NULL; }
                    if(rs_rec_config) { rs2_delete_config(rs_rec_config); rs_rec_config = NULL; }
                    ts_stop_mocap_recording();
                    is_recording = false; // Force stop
                }
            }
        } else {
            printf("\rReady for sentence %03d. [ENTER]=Start, [B]=ReRec, [G]=GlovesCalib, [S]=SuitCalib, [Q]=Quit. ", sentence_count);
            fflush(stdout);
        }

        if (_kbhit()) {
            char key = _getch();
            printf("\r%*s\r", 120, ""); // Clear the line more reliably

            if (!is_recording && (key == 'q' || key == 'Q')) {
                app_should_run = 0; printf("Quit command received.\n"); break;
            } else if (key == '\r' || key == '\n') { // Enter key
                if (!is_recording) {
                    // --- Start ALL Recordings ---
                    printf("Starting ALL recordings for sentence %03d...\n", sentence_count);

                    bool ts_started_ok = ts_start_mocap_recording(sentence_count, output_dir_base);
                    if (!ts_started_ok) {
                         fprintf(stderr, "\033[33m--------------Warning: Teslasuit Mocap failed to start for one or more devices. RealSense will still attempt to record.\033[0m\n");
                    }

                    rs_rec_config = rs2_create_config(&e); check_rs_error(e, "main: rs_cfg_create");
                    char rs_bag_filename[MAX_FILENAME_LEN];
                    if (strlen(output_dir_base) > 0 && (output_dir_base[strlen(output_dir_base) - 1] == '/' || output_dir_base[strlen(output_dir_base) - 1] == '\\')) {
                        sprintf(rs_bag_filename, "%ssentence_%03d_realsense.bag", output_dir_base, sentence_count);
                    } else {
                        sprintf(rs_bag_filename, "%s/sentence_%03d_realsense.bag", output_dir_base, sentence_count);
                    }
                    printf("  DEBUG: RealSense recording to: %s\n", rs_bag_filename);
                    rs2_config_enable_record_to_file(rs_rec_config, rs_bag_filename, &e); check_rs_error(e, "main: rs_rec_to_file");

                    rs2_config_enable_stream(rs_rec_config, RS2_STREAM_DEPTH, -1, 840, 480, RS2_FORMAT_Z16, 60, &e);
                    if(e) {fprintf(stderr, "\033[33m---------Warn: RS depth cfg: %s\033[0m\n", rs2_get_error_message(e)); rs2_free_error(e);e=NULL;}
                    rs2_config_enable_stream(rs_rec_config, RS2_STREAM_COLOR, -1, 840, 480, RS2_FORMAT_RGB8, 60, &e);
                    if(e) {
                        fprintf(stderr, "\033[33m------------Warn: RS BGR8 cfg: %s. Try RGB8.\033[0m\n", rs2_get_error_message(e)); rs2_free_error(e);e=NULL;
                        rs2_config_enable_stream(rs_rec_config, RS2_STREAM_COLOR, -1, 840, 480, RS2_FORMAT_BGR8, 60, &e);
                         if(e) {
                             fprintf(stderr, "\033[33m-----------Warn: RS RGB8 cfg: %s. Try ANY.\033[0m\n", rs2_get_error_message(e)); rs2_free_error(e);e=NULL;
                             rs2_config_enable_stream(rs_rec_config, RS2_STREAM_COLOR, -1, 840, 480, RS2_FORMAT_ANY, 60, &e);
                             if(e) {fprintf(stderr, "\033[31m------------FATAL: RS ANY color cfg: %s\033[0m\n", rs2_get_error_message(e)); rs2_free_error(e);e=NULL;}
                         }
                    }

                    rs_rec_pipeline = rs2_create_pipeline(rec_ctx, &e); check_rs_error(e, "main: rs_pipeline_create");
                    rs2_pipeline_profile* temp_rs_rec_profile = rs2_pipeline_start_with_config(rs_rec_pipeline, rs_rec_config, &e);
                    if(e) {
                        fprintf(stderr, "\033[31m---------Failed to start RealSense recording pipeline for sentence %d: %s\033[0m\n", sentence_count, rs2_get_error_message(e));
                        rs2_free_error(e); e=NULL;
                        if(rs_rec_pipeline) {rs2_delete_pipeline(rs_rec_pipeline); rs_rec_pipeline = NULL;}
                        if(rs_rec_config) {rs2_delete_config(rs_rec_config); rs_rec_config = NULL;}
                        if (ts_started_ok) ts_stop_mocap_recording();
                    } else {
                         printf("RealSense recording pipeline started successfully for %s\n", rs_bag_filename);
                         rs2_delete_pipeline_profile(temp_rs_rec_profile);
                         is_recording = true;
                    }

                } else { // Stop ALL recordings
                    printf("Stopping ALL recordings for sentence %03d...\n", sentence_count);
                    is_recording = false;

                    printf("  Stopping Teslasuit Mocap...\n");
                    ts_stop_mocap_recording();
                    printf("  Teslasuit Mocap stopped.\n");

                    if (rs_rec_pipeline) {
                        printf("  DEBUG: Calling RealSense rs2_pipeline_stop...\n"); fflush(stdout);
                        rs2_pipeline_stop(rs_rec_pipeline, &e);
                        if(e) { fprintf(stderr, "\033[31m---------------Error stopping RealSense pipeline: %s\033[0m\n", rs2_get_error_message(e)); rs2_free_error(e); e = NULL;}
                        else {
                            printf("  DEBUG: RealSense rs2_pipeline_stop completed. Adding small delay for file finalization...\n"); fflush(stdout);
                            Sleep(500);
                        }
                        rs2_delete_pipeline(rs_rec_pipeline);
                        rs_rec_pipeline = NULL;
                    }

                    if (rs_rec_config) { // rs_rec_config is deleted after this block
                        rs2_delete_config(rs_rec_config); rs_rec_config = NULL;
                        printf("  INFO: RealSense bag file for sentence %03d saved. Post-processing is separate.\n", sentence_count);
                    } else {
                         fprintf(stderr, "\033[33m---------Skipping RealSense Bag post-processing for sentence %d (no rs_rec_config).\033[0m\n", sentence_count);
                    }
                    sentence_count++;
                }
            } else if (!is_recording && (key == 'b' || key == 'B')) {
                if (sentence_count > start_sentence_num || (start_sentence_num == 1 && sentence_count > 1) ) {
                     int prev_sentence_to_delete = sentence_count - 1;

                    printf("\nRE-RECORDING: Deleting data for sentence %03d and preparing to record it again.\n", prev_sentence_to_delete);

                    char ts_suit_file[MAX_FILENAME_LEN];
                    char ts_lglove_file[MAX_FILENAME_LEN];
                    char ts_rglove_file[MAX_FILENAME_LEN];
                    char rs_bag_prev[MAX_FILENAME_LEN];
                    // Landmark CSV deletion is not needed here as it's a post-process step

                    // Corrected and safer filename construction
                    sprintf(ts_suit_file, "%s/sentence_%03d_ts_suit_mocap.csv", output_dir_base, prev_sentence_to_delete);
                    sprintf(ts_lglove_file, "%s/sentence_%03d_ts_glove_L_mocap.csv", output_dir_base, prev_sentence_to_delete);
                    sprintf(ts_rglove_file, "%s/sentence_%03d_ts_glove_R_mocap.csv", output_dir_base, prev_sentence_to_delete);
                    sprintf(rs_bag_prev, "%s/sentence_%03d_realsense.bag", output_dir_base, prev_sentence_to_delete);



                    delete_file_if_exists(ts_suit_file);
                    delete_file_if_exists(ts_lglove_file);
                    delete_file_if_exists(ts_rglove_file);
                    delete_file_if_exists(rs_bag_prev);

                    sentence_count = prev_sentence_to_delete;
                    printf("Ready to re-record sentence %03d. Press [ENTER] to start.\n", sentence_count);

                } else {
                    printf("\n-------------Cannot re-record. No previous sentence recorded in this session or at start_sentence_num.\n");
                }
            } else if (!is_recording) { // Teslasuit Calibration keys
                if (key == 's' || key == 'S') { printf("\nAttempting SUIT calibration...\n"); ts_calibrate_suit(); }
                else if (key== 'g' || key == 'G') { // New key for both gloves
                    printf("\nAttempting BOTH GLOVES calibration...\n");
                    bool left_ok = ts_calibrate_glove_left();
                    bool right_ok = ts_calibrate_glove_right();
                    if (left_ok || right_ok) printf("Glove calibration commands sent.\n");
                    else printf("---------------No gloves available or calibration failed to send.\n");
                }
            }}
        Sleep(50);
    }

    if (is_recording) {
        printf("Cleaning up active recordings due to exit...\n");
        if (rs_rec_pipeline) { rs2_pipeline_stop(rs_rec_pipeline, NULL); Sleep(500); rs2_delete_pipeline(rs_rec_pipeline); }
        if (rs_rec_config) rs2_delete_config(rs_rec_config);
        ts_stop_mocap_recording();
    }

    printf("Releasing global resources...\n");
    if (rec_ctx) rs2_delete_context(rec_ctx);
    ts_close_all_devices();
    ts_uninitialize_system();
    printf("Application finished.\n");
    return EXIT_SUCCESS;
}