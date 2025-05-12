// #include <librealsense2/rs.h> // Commented out RealSense includes
// #include <librealsense2/h/rs_pipeline.h>
// #include <librealsense2/h/rs_frame.h>
// #include <librealsense2/h/rs_config.h>
// #include <librealsense2/h/rs_record_playback.h>
// #include <librealsense2/h/rs_sensor.h>

// #include "face_detector.h"      // Dlib/RS processing - Not called by main
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

void signal_handler(int dummy) { (void)dummy; app_should_run = 0; printf("\nCtrl+C pressed...\n"); }

// RealSense error check not needed if RS calls are removed from main
// void check_rs_error(rs2_error* e, const char* context_msg) { ... }

int main(int argc, char* argv[]) {
    if (argc < 2) { // Now only needs output_directory
        fprintf(stderr, "Usage: %s <output_directory>\n", argv[0]);
        fprintf(stderr, "Example: %s ./output_data\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* output_dir = argv[1];
    // const char* dlib_model_path = argv[2]; // Dlib model not used by main directly
    // const double trim_sec = 0.0;

    signal(SIGINT, signal_handler);

    // --- Initialize Teslasuit ---
    if (!ts_initialize_system()) {
        fprintf(stderr, "Fatal: Could not initialize Teslasuit API.\n");
        return EXIT_FAILURE;
    }
    ts_discover_and_open_devices();
    // --- End Teslasuit Init ---

    // --- Dlib Initialization (Keep for now, as C++ part might still be linked, but not used by this main) ---
    // printf("Initializing Dlib landmark detector (not used by main loop)...\n");
    // void* landmark_detector = initialize_dlib_landmark_detector(dlib_model_path);
    // if (!landmark_detector) {
    //     fprintf(stderr, "Fatal: Could not initialize Dlib landmark detector.\n");
    //     ts_close_all_devices();
    //     ts_uninitialize_system();
    //     return EXIT_FAILURE;
    // }
    // printf("Dlib landmark detector initialized (not used by main loop).\n");
    // --- End Dlib Initialization ---


    // --- RealSense Context (Commented out) ---
    // rs2_error* e = NULL;
    // rs2_context* rec_ctx = rs2_create_context(RS2_API_VERSION, &e);
    // check_rs_error(e, "main: create recording context");
    // --- End RealSense Context ---

    int sentence_count = 1;
    bool is_recording = false;
    // rs2_pipeline* rec_pipeline = NULL; // Commented out
    // rs2_config* rec_config = NULL;   // Commented out

    printf("Teslasuit-Only Mocap Recorder\n");
    printf("Press [ENTER] to start/stop Teslasuit Mocap recording.[S/L/R for Calib]\n");
    printf("Press [Q] then [ENTER] to quit.\n\n");

    while (app_should_run) {
        if (is_recording) {
             printf("\rRECORDING Sentence %03d...  Press [ENTER] to STOP. ", sentence_count);
            fflush(stdout);
            // No RealSense frame polling needed here
            Sleep(100); // Just to keep the loop from spinning too fast
        } else {
            printf("\rReady for sentence %03d. Press [ENTER] to START, or [Q] then [ENTER] to quit. ", sentence_count);
            fflush(stdout);
        }

        if (_kbhit()) {
            char key = _getch();
            printf("\r%70s\r", "");

            if ((key == 'q' || key == 'Q') && !is_recording) {
                app_should_run = 0;
                printf("Quit command received.\n");
                break;
            } else if (key == '\r' || key == '\n') { // Enter key
                if (!is_recording) {
                    printf("Starting Teslasuit Mocap recording for sentence %03d...\n", sentence_count);
                    if (!ts_start_mocap_recording(sentence_count, output_dir)) {
                         fprintf(stderr, "Warning: Teslasuit Mocap failed to start for one or more devices.\n");
                         // Continue, allow user to try stopping/quitting
                    }
                    is_recording = true;
                } else { // Stop recording
                    printf("Stopping Teslasuit Mocap recording for sentence %03d...\n", sentence_count);
                    is_recording = false;
                    ts_stop_mocap_recording();
                    sentence_count++;
                }
            } else if (!is_recording) { // Calibration keys only active DURING recording
                if (key == 's' || key == 'S') {
                    printf("\nAttempting SUIT calibration...\n");
                    ts_calibrate_suit();
                } else if (key == 'l' || key == 'L') {
                    printf("\nAttempting LEFT GLOVE calibration...\n");
                    ts_calibrate_glove_left();
                } else if (key == 'r' || key == 'R') {
                    printf("\nAttempting RIGHT GLOVE calibration...\n");
                    ts_calibrate_glove_right();
                }
            }
        }
        Sleep(50);
    } // End main while loop

    if (is_recording) {
        printf("Cleaning up active Teslasuit recording due to exit...\n");
        ts_stop_mocap_recording();
        // No RealSense pipeline/config to clean up here
    }

    printf("Releasing resources...\n");
    // release_dlib_landmark_detector(landmark_detector); // Commented out Dlib cleanup
    // if (rec_ctx) rs2_delete_context(rec_ctx); // Commented out RS context cleanup
    ts_close_all_devices();
    ts_uninitialize_system();
    printf("Application finished.\n");
    return EXIT_SUCCESS;
}