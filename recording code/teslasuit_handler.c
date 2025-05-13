#include "teslasuit_handler.h" // Public interface declarations
#include <ts_api/ts_core_api.h>
#include <ts_api/ts_device_api.h>
#include <ts_api/ts_mocap_api.h>
// ts_types.h is included via teslasuit_handler.h

#include <stdio.h>
#include <stdlib.h> // For exit (if check_ts_error_internal decides to exit)
#include <string.h> // For sprintf
#include <time.h>
#ifdef _WIN32
#include <windows.h> // For CRITICAL_SECTION, Sleep, InitializeCriticalSection, DeleteCriticalSection
#else
#include <pthread.h>
#include <unistd.h> // For usleep
#define Sleep(x) usleep(x*1000)
#endif

#define MAX_TS_DEVICES_INTERNAL 5 // Max devices to check during discovery
#define MAX_FILENAME_LEN_TS 256   // Buffer for CSV filenames

// --- Static Global Variables for Teslasuit State (internal to this .c file) ---
static TsDeviceHandle* s_ts_suit_handle = NULL;
static TsDeviceHandle* s_ts_glove_left_handle = NULL;
static TsDeviceHandle* s_ts_glove_right_handle = NULL;

static FILE* s_ts_suit_csv_file = NULL;
static FILE* s_ts_glove_left_csv_file = NULL;
static FILE* s_ts_glove_right_csv_file = NULL;

// Volatile because they can be accessed by main thread and callback threads
static volatile int s_current_ts_sentence_id = 0;
static volatile uint64_t s_ts_mocap_frame_suit = 0;
static volatile uint64_t s_ts_mocap_frame_glove_l = 0;
static volatile uint64_t s_ts_mocap_frame_glove_r = 0;

#ifdef _WIN32
static CRITICAL_SECTION s_ts_suit_csv_mutex;
static CRITICAL_SECTION s_ts_glove_l_csv_mutex;
static CRITICAL_SECTION s_ts_glove_r_csv_mutex;
static LARGE_INTEGER qpc_frequency;
static BOOL qpc_init_done = FALSE;
static void initialize_qpc_ts_handler() { // Renamed to be specific
    if (!qpc_init_done) { // Use the static qpc_init_done
        if (!QueryPerformanceFrequency(&qpc_frequency)) {
            qpc_frequency.QuadPart = 0;
            fprintf(stderr, "Warning: QueryPerformanceFrequency failed. Timestamps might be less precise.\n");
        }
        qpc_init_done = TRUE;
    }
}
#else
static pthread_mutex_t s_ts_suit_csv_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t s_ts_glove_l_csv_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t s_ts_glove_r_csv_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif
// --- End Static Global Variables ---





static uint64_t get_system_timestamp_us() {
    if (!qpc_init_done) {
        initialize_qpc_ts_handler(); // Ensure it's initialized
    }

    LARGE_INTEGER current_count;
    if (!QueryPerformanceCounter(&current_count)) {
        // Fallback if QPC fails after init (highly unlikely)
        FILETIME ft;
        ULARGE_INTEGER uli;
        GetSystemTimeAsFileTime(&ft);
        uli.LowPart = ft.dwLowDateTime;
        uli.HighPart = ft.dwHighDateTime;
        return uli.QuadPart / 10; // Convert 100ns intervals to microseconds
    }

    if (qpc_frequency.QuadPart == 0) { // If frequency could not be obtained
        return 0; // Or another error indicator
    }

    // Calculate seconds and fractional seconds separately to avoid overflow
    uint64_t seconds = current_count.QuadPart / qpc_frequency.QuadPart;
    uint64_t remainder_counts = current_count.QuadPart % qpc_frequency.QuadPart;
    uint64_t microseconds_part = (remainder_counts * 1000000ULL) / qpc_frequency.QuadPart;

    return (seconds * 1000000ULL) + microseconds_part;
}


static const char* ts_bone_index_to_name_str(TsBoneIndex index) {
    switch (index) {
        case TsBoneIndex_Hips: return "root";
        case TsBoneIndex_LeftUpperLeg: return "left_upper_leg";
        case TsBoneIndex_RightUpperLeg: return "right_upper_leg";
        case TsBoneIndex_LeftLowerLeg: return "left_lower_leg";
        case TsBoneIndex_RightLowerLeg: return "right_lower_leg";
        case TsBoneIndex_LeftFoot: return "left_foot";
        case TsBoneIndex_RightFoot: return "right_foot";
        case TsBoneIndex_Spine: return "spine";
        case TsBoneIndex_Chest: return "chest";
        case TsBoneIndex_UpperSpine: return "upper_spine"; // Corrected from "UpperSpine"
        case TsBoneIndex_Neck: return "neck";
        case TsBoneIndex_Head: return "head";
        case TsBoneIndex_LeftShoulder: return "left_shoulder";
        case TsBoneIndex_RightShoulder: return "right_shoulder";
        case TsBoneIndex_LeftUpperArm: return "left_upper_arm";
        case TsBoneIndex_RightUpperArm: return "right_upper_arm";
        case TsBoneIndex_LeftLowerArm: return "left_lower_arm";
        case TsBoneIndex_RightLowerArm: return "right_lower_arm";
        case TsBoneIndex_LeftHand: return "left_hand";
        case TsBoneIndex_RightHand: return "right_hand";
        case TsBoneIndex_LeftThumbProximal: return "left_thumb_proximal";
        case TsBoneIndex_LeftThumbIntermediate: return "left_thumb_intermediate";
        case TsBoneIndex_LeftThumbDistal: return "left_thumb_distal";
        case TsBoneIndex_LeftIndexProximal: return "left_index_proximal";
        case TsBoneIndex_LeftIndexIntermediate: return "left_index_intermediate";
        case TsBoneIndex_LeftIndexDistal: return "left_index_distal";
        case TsBoneIndex_LeftMiddleProximal: return "left_middle_proximal";
        case TsBoneIndex_LeftMiddleIntermediate: return "left_middle_intermediate";
        case TsBoneIndex_LeftMiddleDistal: return "left_middle_distal";
        case TsBoneIndex_LeftRingProximal: return "left_ring_proximal";
        case TsBoneIndex_LeftRingIntermediate: return "left_ring_intermediate";
        case TsBoneIndex_LeftRingDistal: return "left_ring_distal";
        case TsBoneIndex_LeftLittleProximal: return "left_little_proximal";
        case TsBoneIndex_LeftLittleIntermediate: return "left_little_intermediate";
        case TsBoneIndex_LeftLittleDistal: return "left_little_distal";
        case TsBoneIndex_RightThumbProximal: return "right_thumb_proximal";
        case TsBoneIndex_RightThumbIntermediate: return "right_thumb_intermediate";
        case TsBoneIndex_RightThumbDistal: return "right_thumb_distal";
        case TsBoneIndex_RightIndexProximal: return "right_index_proximal";
        case TsBoneIndex_RightIndexIntermediate: return "right_index_intermediate";
        case TsBoneIndex_RightIndexDistal: return "right_index_distal";
        case TsBoneIndex_RightMiddleProximal: return "right_middle_proximal";
        case TsBoneIndex_RightMiddleIntermediate: return "right_middle_intermediate";
        case TsBoneIndex_RightMiddleDistal: return "right_middle_distal";
        case TsBoneIndex_RightRingProximal: return "right_ring_proximal";
        case TsBoneIndex_RightRingIntermediate: return "right_ring_intermediate";
        case TsBoneIndex_RightRingDistal: return "right_ring_distal";
        case TsBoneIndex_RightLittleProximal: return "right_little_proximal";
        case TsBoneIndex_RightLittleIntermediate: return "right_little_intermediate";
        case TsBoneIndex_RightLittleDistal: return "right_little_distal";
        default: return "unknown_bone";
    }
}

static const char* ts_biomech_index_to_name_str(TsBiomechanicalIndex index) {
    switch (index) {
        case TsBiomechanicalIndex_PelvisTilt: return "PelvisTilt";
        case TsBiomechanicalIndex_PelvisList: return "PelvisList";
        case TsBiomechanicalIndex_PelvisRotation: return "PelvisRotation";
        case TsBiomechanicalIndex_HipFlexExtR: return "HipFlexExtR";
        case TsBiomechanicalIndex_HipAddAbdR: return "HipAddAbdR";
        case TsBiomechanicalIndex_HipRotR: return "HipRotR";
        case TsBiomechanicalIndex_KneeFlexExtR: return "KneeFlexExtR";
        case TsBiomechanicalIndex_AnkleFlexExtR: return "AnkleFlexExtR";
        case TsBiomechanicalIndex_AnkleProSupR: return "AnkleProSupR";
        case TsBiomechanicalIndex_HipFlexExtL: return "HipFlexExtL";
        case TsBiomechanicalIndex_HipAddAbdL: return "HipAddAbdL";
        case TsBiomechanicalIndex_HipRotL: return "HipRotL";
        case TsBiomechanicalIndex_KneeFlexExtL: return "KneeFlexExtL";
        case TsBiomechanicalIndex_AnkleFlexExtL: return "AnkleFlexExtL";
        case TsBiomechanicalIndex_AnkleProSupL: return "AnkleProSupL";
        case TsBiomechanicalIndex_ElbowFlexExtR: return "ElbowFlexExtR";
        case TsBiomechanicalIndex_ForearmProSupR: return "ForearmProSupR";
        case TsBiomechanicalIndex_WristFlexExtR: return "WristFlexExtR";
        case TsBiomechanicalIndex_WristDeviationR: return "WristDeviationR";
        case TsBiomechanicalIndex_ElbowFlexExtL: return "ElbowFlexExtL";
        case TsBiomechanicalIndex_ForearmProSupL: return "ForearmProSupL";
        case TsBiomechanicalIndex_WristFlexExtL: return "WristFlexExtL";
        case TsBiomechanicalIndex_WristDeviationL: return "WristDeviationL";
        // Note: Missing Lumbar, Thorax, Scapula from your example, add if API supports them
        case TsBiomechanicalIndex_ShoulderAddAbdR: return "ShoulderAddAbdR";
        case TsBiomechanicalIndex_ShoulderRotR: return "ShoulderRotR";
        case TsBiomechanicalIndex_ShoulderFlexExtR: return "ShoulderFlexExtR";
        case TsBiomechanicalIndex_ShoulderAddAbdL: return "ShoulderAddAbdL";
        case TsBiomechanicalIndex_ShoulderRotL: return "ShoulderRotL";
        case TsBiomechanicalIndex_ShoulderFlexExtL: return "ShoulderFlexExtL";
        default: return "UnknownBiomech";
    }
}

// Define the order of bones for the CSV output
static const TsBoneIndex suit_bone_order[] = {
    TsBoneIndex_Hips, // Often considered "root"
    TsBoneIndex_LeftUpperLeg, TsBoneIndex_RightUpperLeg,
    TsBoneIndex_LeftLowerLeg, TsBoneIndex_RightLowerLeg,
    TsBoneIndex_LeftFoot, TsBoneIndex_RightFoot,
    TsBoneIndex_Spine, TsBoneIndex_Chest, TsBoneIndex_UpperSpine,
    TsBoneIndex_Neck, TsBoneIndex_Head,
    TsBoneIndex_LeftShoulder, TsBoneIndex_RightShoulder,
    TsBoneIndex_LeftUpperArm, TsBoneIndex_RightUpperArm,
    TsBoneIndex_LeftLowerArm, TsBoneIndex_RightLowerArm,
    TsBoneIndex_LeftHand, TsBoneIndex_RightHand
};
static const int num_suit_bones_to_log = sizeof(suit_bone_order) / sizeof(suit_bone_order[0]);

static const TsBoneIndex left_glove_bone_order[] = {
    TsBoneIndex_LeftHand, TsBoneIndex_LeftThumbProximal, TsBoneIndex_LeftThumbIntermediate, TsBoneIndex_LeftThumbDistal,
    TsBoneIndex_LeftIndexProximal, TsBoneIndex_LeftIndexIntermediate, TsBoneIndex_LeftIndexDistal,
    TsBoneIndex_LeftMiddleProximal, TsBoneIndex_LeftMiddleIntermediate, TsBoneIndex_LeftMiddleDistal,
    TsBoneIndex_LeftRingProximal, TsBoneIndex_LeftRingIntermediate, TsBoneIndex_LeftRingDistal,
    TsBoneIndex_LeftLittleProximal, TsBoneIndex_LeftLittleIntermediate, TsBoneIndex_LeftLittleDistal
};
static const int num_left_glove_bones_to_log = sizeof(left_glove_bone_order) / sizeof(left_glove_bone_order[0]);

static const TsBoneIndex right_glove_bone_order[] = {
    // TsBoneIndex_Hips, // No need for Hips in glove-only file unless you want it for context
    TsBoneIndex_RightHand, TsBoneIndex_RightThumbProximal, TsBoneIndex_RightThumbIntermediate, TsBoneIndex_RightThumbDistal,
    TsBoneIndex_RightIndexProximal, TsBoneIndex_RightIndexIntermediate, TsBoneIndex_RightIndexDistal,
    TsBoneIndex_RightMiddleProximal, TsBoneIndex_RightMiddleIntermediate, TsBoneIndex_RightMiddleDistal,
    TsBoneIndex_RightRingProximal, TsBoneIndex_RightRingIntermediate, TsBoneIndex_RightRingDistal,
    TsBoneIndex_RightLittleProximal, TsBoneIndex_RightLittleIntermediate, TsBoneIndex_RightLittleDistal
};
static const int num_right_glove_bones_to_log = sizeof(right_glove_bone_order) / sizeof(right_glove_bone_order[0]);
static const TsBiomechanicalIndex biomech_indices_to_log[] = {
    TsBiomechanicalIndex_PelvisTilt, TsBiomechanicalIndex_PelvisList, TsBiomechanicalIndex_PelvisRotation,
    TsBiomechanicalIndex_HipFlexExtR, TsBiomechanicalIndex_HipAddAbdR, TsBiomechanicalIndex_HipRotR,
    TsBiomechanicalIndex_KneeFlexExtR, TsBiomechanicalIndex_AnkleFlexExtR, TsBiomechanicalIndex_AnkleProSupR,
    TsBiomechanicalIndex_HipFlexExtL, TsBiomechanicalIndex_HipAddAbdL, TsBiomechanicalIndex_HipRotL,
    TsBiomechanicalIndex_KneeFlexExtL, TsBiomechanicalIndex_AnkleFlexExtL, TsBiomechanicalIndex_AnkleProSupL,
    TsBiomechanicalIndex_ElbowFlexExtR, TsBiomechanicalIndex_ForearmProSupR, TsBiomechanicalIndex_WristFlexExtR, TsBiomechanicalIndex_WristDeviationR,
    TsBiomechanicalIndex_ElbowFlexExtL, TsBiomechanicalIndex_ForearmProSupL, TsBiomechanicalIndex_WristFlexExtL, TsBiomechanicalIndex_WristDeviationL,
    // Lumbar, Thorax, Scapula are missing from your enum but in CSV example - will skip for now as they are not in TsBiomechanicalIndex
    TsBiomechanicalIndex_ShoulderAddAbdR, TsBiomechanicalIndex_ShoulderRotR, TsBiomechanicalIndex_ShoulderFlexExtR,
    TsBiomechanicalIndex_ShoulderAddAbdL, TsBiomechanicalIndex_ShoulderRotL, TsBiomechanicalIndex_ShoulderFlexExtL
};
static const int num_biomech_indices_to_log = sizeof(biomech_indices_to_log) / sizeof(biomech_indices_to_log[0]);
// --- Internal Teslasuit Status Code Check ---
static void check_ts_error_internal(TsStatusCode status_code, const char* context_msg) {
    if (status_code != 0) { // 0 (Good) is success for Teslasuit
        fprintf(stderr, "-----Teslasuit API Error in %s: %s (Code: %d)\n",
                context_msg,
                ts_get_status_code_message(status_code),
                status_code);
        // For critical errors during initialization or device opening, exiting might be appropriate.
        // For errors during streaming, a warning might be better to allow graceful shutdown.
        // Example: if (status_code < 0 && (strstr(context_msg, "ts_initialize") || strstr(context_msg, "ts_device_open"))) {
        //     exit(EXIT_FAILURE);
        // }
    }
}

// Public version if needed by other files (though main.c can just check return values)
void check_ts_error_public(TsStatusCode status_code, const char* context_msg) {
    check_ts_error_internal(status_code, context_msg);
}

bool ts_calibrate_device(TsDeviceHandle* device_handle, const char* device_name_for_log) {
    if (!device_handle) {
        printf("------------Cannot calibrate %s: device handle is NULL.\n", device_name_for_log);
        return false;
    }

    // The API docs say mocap streaming should be active.
    // We assume the main loop will manage starting streaming if needed,
    // or this is called when streaming is already active.
    // A more robust way would be to check if streaming is active and start it if not,
    // but that adds complexity here. For now, let's assume the user starts streaming first.
    printf("Attempting to calibrate %s...\n", device_name_for_log);
    printf("Ensure user is in the required calibration pose (e.g., I-Pose).\n");

    // Add a small delay or prompt for user to get into pose
    printf("Calibration will start in 3 seconds...\n");
    Sleep(1000); printf("2...\n"); Sleep(1000); printf("1...\n"); Sleep(1000);
    printf("Calibrating %s NOW!\n", device_name_for_log);


    TsStatusCode status = ts_mocap_skeleton_calibrate(device_handle);
    check_ts_error_internal(status, device_name_for_log); // check_ts_error_internal won't exit on non-fatal

    if (status == 0) { // Good
        printf("%s calibration command sent successfully.\n", device_name_for_log);
        // Note: Calibration might take some time on the device.
        // The API call might just trigger it.
        return true;
    } else {
        fprintf(stderr, "Failed to send calibration command for %s.\n", device_name_for_log);
        return false;
    }
}

bool ts_calibrate_suit() {
    if (!s_ts_suit_handle) {
        printf("Suit not available for calibration.\n");
        return false;
    }
    // Important: Ensure Mocap is streaming for the suit *before* calling this
    // if the API truly requires it. The current program starts streaming
    // only when "recording" a sentence. Calibration might need to happen
    // during a recording session, or we might need a separate "calibration mode"
    // where streaming is started just for calibration.
    // For simplicity, let's assume it's called while a recording is active.
    printf("Ensure Teslasuit Suit Mocap is streaming to calibrate.\n");
    return ts_calibrate_device(s_ts_suit_handle, "Teslasuit Suit");
}

bool ts_calibrate_glove_left() {
    if (!s_ts_glove_left_handle) {
        printf("Left Glove not available for calibration.\n");
        return false;
    }
    printf("Ensure Teslasuit Left Glove Mocap is streaming to calibrate.\n");
    return ts_calibrate_device(s_ts_glove_left_handle, "Left Glove");
}

bool ts_calibrate_glove_right() {
    if (!s_ts_glove_right_handle) {
        printf("Right Glove not available for calibration.\n");
        return false;
    }
    printf("Ensure Teslasuit Right Glove Mocap is streaming to calibrate.\n");
    return ts_calibrate_device(s_ts_glove_right_handle, "Right Glove");
}


// --- Teslasuit Mocap Callback Functions (Static) ---
static void ts_mocap_suit_callback_internal(TsDeviceHandle* dev, TsMocapSkeleton skeleton, void* user_data) {
    (void)dev; (void)user_data;
    if (!s_ts_suit_csv_file) return;
    uint64_t timestamp_us = get_system_timestamp_us();
#ifdef _WIN32
    EnterCriticalSection(&s_ts_suit_csv_mutex);
#else
    pthread_mutex_lock(&s_ts_suit_csv_mutex);
#endif
    fprintf(s_ts_suit_csv_file, "%d,%llu,%llu", s_current_ts_sentence_id, s_ts_mocap_frame_suit, timestamp_us);
    for (int i = 0; i < num_suit_bones_to_log; ++i) {
        TsBoneIndex bone_idx = suit_bone_order[i];
        TsMocapBone bone_data;
        if (ts_mocap_skeleton_get_bone(skeleton, bone_idx, &bone_data) == 0) {
            fprintf(s_ts_suit_csv_file, ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                    bone_data.position.x, bone_data.position.y, bone_data.position.z,
                    bone_data.rotation.w, bone_data.rotation.x, bone_data.rotation.y, bone_data.rotation.z);
        } else { fprintf(s_ts_suit_csv_file, ",0,0,0,1,0,0,0"); }
    }
    fprintf(s_ts_suit_csv_file, ",0.0,0.0,0.0");
    for (int i = 0; i < num_biomech_indices_to_log; ++i) {
        float angle;
        if (ts_mocap_skeleton_get_biomechanical_angle(skeleton, biomech_indices_to_log[i], &angle) == 0) {
            fprintf(s_ts_suit_csv_file, ",%.6f,0.0,0.0", angle);
        } else { fprintf(s_ts_suit_csv_file, ",0.0,0.0,0.0"); }
    }
    fprintf(s_ts_suit_csv_file, ",0,0\n");
    s_ts_mocap_frame_suit++;
#ifdef _WIN32
    LeaveCriticalSection(&s_ts_suit_csv_mutex);
#else
    pthread_mutex_unlock(&s_ts_suit_csv_mutex);
#endif
}

static void write_glove_data_to_csv_with_ts(FILE* csv_file, int sentence_id, uint64_t frame_id,
                                           uint64_t timestamp_us, TsMocapSkeleton skeleton,
                                           const TsBoneIndex bone_order[], int num_bones) {
    if (!csv_file) return;
    fprintf(csv_file, "%d,%llu,%llu", sentence_id, frame_id, timestamp_us);
    for (int i = 0; i < num_bones; ++i) {
        TsBoneIndex bone_idx = bone_order[i];
        TsMocapBone bone_data;
        if (ts_mocap_skeleton_get_bone(skeleton, bone_idx, &bone_data) == 0) {
            fprintf(csv_file, ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                    bone_data.position.x, bone_data.position.y, bone_data.position.z,
                    bone_data.rotation.w, bone_data.rotation.x, bone_data.rotation.y, bone_data.rotation.z);
        } else { fprintf(csv_file, ",0,0,0,1,0,0,0"); }
    }
    fprintf(csv_file, "\n");
}

static void ts_mocap_glove_left_callback_internal(TsDeviceHandle* dev, TsMocapSkeleton skeleton, void* user_data) {
    (void)dev; (void)user_data;
    uint64_t timestamp_us = get_system_timestamp_us();
#ifdef _WIN32
    EnterCriticalSection(&s_ts_glove_l_csv_mutex);
#else
    pthread_mutex_lock(&s_ts_glove_l_csv_mutex);
#endif
    write_glove_data_to_csv_with_ts(s_ts_glove_left_csv_file, s_current_ts_sentence_id,
                                    s_ts_mocap_frame_glove_l, timestamp_us, skeleton,
                                    left_glove_bone_order, num_left_glove_bones_to_log);
    s_ts_mocap_frame_glove_l++;
#ifdef _WIN32
    LeaveCriticalSection(&s_ts_glove_l_csv_mutex);
#else
    pthread_mutex_unlock(&s_ts_glove_l_csv_mutex);
#endif
}

static void ts_mocap_glove_right_callback_internal(TsDeviceHandle* dev, TsMocapSkeleton skeleton, void* user_data) {
    (void)dev; (void)user_data;
    uint64_t timestamp_us = get_system_timestamp_us();
#ifdef _WIN32
    EnterCriticalSection(&s_ts_glove_r_csv_mutex);
#else
    pthread_mutex_lock(&s_ts_glove_r_csv_mutex);
#endif
    write_glove_data_to_csv_with_ts(s_ts_glove_right_csv_file, s_current_ts_sentence_id,
                                    s_ts_mocap_frame_glove_r, timestamp_us, skeleton,
                                    right_glove_bone_order, num_right_glove_bones_to_log);
    s_ts_mocap_frame_glove_r++;
#ifdef _WIN32
    LeaveCriticalSection(&s_ts_glove_r_csv_mutex);
#else
    pthread_mutex_unlock(&s_ts_glove_r_csv_mutex);
#endif
}

bool ts_initialize_system() {
    printf("Initializing Teslasuit API...\n");
#ifdef _WIN32
    initialize_qpc_ts_handler(); // Initialize QPC for Windows timestamps
#endif
    TsStatusCode ts_status = ts_initialize();
    check_ts_error_internal(ts_status, "ts_initialize");
    if (ts_status != 0) return false;

#ifdef _WIN32
    InitializeCriticalSection(&s_ts_suit_csv_mutex);
    InitializeCriticalSection(&s_ts_glove_l_csv_mutex);
    InitializeCriticalSection(&s_ts_glove_r_csv_mutex);
#endif
    printf("Teslasuit API initialized.\n");
    return true;
}


void ts_discover_and_open_devices() {
    TsDevice devices[MAX_TS_DEVICES_INTERNAL];
    uint32_t device_count = MAX_TS_DEVICES_INTERNAL; // Input: size of buffer
    TsStatusCode ts_status = ts_get_device_list(devices, &device_count); // Output: actual number of devices
    check_ts_error_internal(ts_status, "ts_get_device_list");
    if (ts_status != 0) return;


    printf("Found %u Teslasuit devices.\n", device_count);
    for (uint32_t i = 0; i < device_count; ++i) {
        TsDeviceHandle* temp_handle = ts_device_open(&devices[i]);
        if (temp_handle) {
            TsProductType type = ts_device_get_product_type(temp_handle);
            TsDeviceSide side = ts_device_get_device_side(temp_handle);
            const char* name = ts_device_get_name(temp_handle);
            printf("  Device %u: Type=%d, Side=%d, Name=%s\n", i, type, side, name ? name : "N/A");

            if (type == TsProductType_Suit && !s_ts_suit_handle) {
                s_ts_suit_handle = temp_handle;
                printf("    Suit assigned.\n");
                ts_status = ts_mocap_set_skeleton_update_callback(s_ts_suit_handle, ts_mocap_suit_callback_internal, NULL);
                check_ts_error_internal(ts_status, "ts_mocap_set_skeleton_update_callback for Suit");
            } else if (type == TsProductType_Glove) {
                if (side == TsDeviceSide_Left && !s_ts_glove_left_handle) {
                    s_ts_glove_left_handle = temp_handle;
                    printf("    Left Glove assigned.\n");
                    ts_status = ts_mocap_set_skeleton_update_callback(s_ts_glove_left_handle, ts_mocap_glove_left_callback_internal, NULL);
                    check_ts_error_internal(ts_status, "ts_mocap_set_skeleton_update_callback for Left Glove");
                } else if (side == TsDeviceSide_Right && !s_ts_glove_right_handle) {
                    s_ts_glove_right_handle = temp_handle;
                    printf("    Right Glove assigned.\n");
                    ts_status = ts_mocap_set_skeleton_update_callback(s_ts_glove_right_handle, ts_mocap_glove_right_callback_internal, NULL);
                    check_ts_error_internal(ts_status, "ts_mocap_set_skeleton_update_callback for Right Glove");
                } else {
                    printf("    Glove (Type:%d, Side:%d) not assigned (already have one or side undefined).\n", type, side);
                    ts_device_close(temp_handle);
                }
            } else {
                 if(type == TsProductType_Suit && s_ts_suit_handle) {
                     printf("    Another suit found but one is already assigned.\n");
                 } else {
                     printf("    Device type %d not a Suit or Glove, or Suit already assigned.\n", type);
                 }
                ts_device_close(temp_handle);
            }
        } else {
             fprintf(stderr, "  Failed to open Teslasuit device %u.\n", i);
        }
    }
    if (!s_ts_suit_handle && !s_ts_glove_left_handle && !s_ts_glove_right_handle) {
        printf("-------Warning: No Teslasuit devices (Suit or Gloves) were assigned for Mocap recording.\n");
    }
}

void ts_close_all_devices() {
    printf("Closing Teslasuit device handles...\n");
    if (s_ts_suit_handle) { ts_device_close(s_ts_suit_handle); s_ts_suit_handle = NULL; }
    if (s_ts_glove_left_handle) { ts_device_close(s_ts_glove_left_handle); s_ts_glove_left_handle = NULL; }
    if (s_ts_glove_right_handle) { ts_device_close(s_ts_glove_right_handle); s_ts_glove_right_handle = NULL; }
}

void ts_uninitialize_system() {
    printf("Uninitializing Teslasuit API...\n");
    ts_uninitialize();
#ifdef _WIN32
    DeleteCriticalSection(&s_ts_suit_csv_mutex);
    DeleteCriticalSection(&s_ts_glove_l_csv_mutex);
    DeleteCriticalSection(&s_ts_glove_r_csv_mutex);
#else
    // For PTHREAD_MUTEX_INITIALIZER, destroy is often not strictly needed
    pthread_mutex_destroy(&s_ts_suit_csv_mutex);
    pthread_mutex_destroy(&s_ts_glove_l_csv_mutex);
    pthread_mutex_destroy(&s_ts_glove_r_csv_mutex);
#endif
    printf("Teslasuit API uninitialized.\n");
}

bool ts_start_mocap_recording(int sentence_id, const char* output_dir) {
    s_current_ts_sentence_id = sentence_id;
    s_ts_mocap_frame_suit = 0;
    s_ts_mocap_frame_glove_l = 0;
    s_ts_mocap_frame_glove_r = 0;
    bool started_any = false;
    TsStatusCode status;
    char ts_csv_filename[MAX_FILENAME_LEN_TS];

    if (s_ts_suit_handle) {
        sprintf(ts_csv_filename, "%s/sentence_%03d_ts_suit_mocap.csv", output_dir, sentence_id);
        s_ts_suit_csv_file = fopen(ts_csv_filename, "w");
        if (s_ts_suit_csv_file) {
            fprintf(s_ts_suit_csv_file, "sentence_id,frame_id,frame_timestamp_us");
            for (int i = 0; i < num_suit_bones_to_log; ++i) {
                const char* bone_name = ts_bone_index_to_name_str(suit_bone_order[i]);
                fprintf(s_ts_suit_csv_file, ",%s.position.x,%s.position.y,%s.position.z,%s.rotation.w,%s.rotation.x,%s.rotation.y,%s.rotation.z",
                        bone_name, bone_name, bone_name, bone_name, bone_name, bone_name, bone_name);
            }
            fprintf(s_ts_suit_csv_file, ",mass_center.x,mass_center.y,mass_center.z");
            for (int i = 0; i < num_biomech_indices_to_log; ++i) {
                const char* biomech_name = ts_biomech_index_to_name_str(biomech_indices_to_log[i]);
                fprintf(s_ts_suit_csv_file, ",%s.angle,%s.angular_v,%s.angular_acc", biomech_name, biomech_name, biomech_name);
            }
            fprintf(s_ts_suit_csv_file, ",left_foot.contact,right_foot.contact\n");
        } else { fprintf(stderr, "Error opening suit CSV: %s\n", ts_csv_filename); }
        status = ts_mocap_start_streaming(s_ts_suit_handle);  check_ts_error_internal(status, "start suit mocap"); if(status == 0) {printf("  Teslasuit Suit Mocap streaming started.\n"); started_any = true;}
    }
    if (s_ts_glove_left_handle) {
        sprintf(ts_csv_filename, "%s/sentence_%03d_ts_glove_L_mocap.csv", output_dir, sentence_id);
        s_ts_glove_left_csv_file = fopen(ts_csv_filename, "w");
        if (s_ts_glove_left_csv_file) {
             fprintf(s_ts_glove_left_csv_file, "sentence_id,frame_id,frame_timestamp_us");
             for (int i = 0; i < num_left_glove_bones_to_log; ++i) {
                const char* bone_name = ts_bone_index_to_name_str(left_glove_bone_order[i]);
                 fprintf(s_ts_glove_left_csv_file, ",%s.position.x,%s.position.y,%s.position.z,%s.rotation.w,%s.rotation.x,%s.rotation.y,%s.rotation.z",
                         bone_name, bone_name, bone_name, bone_name, bone_name, bone_name, bone_name);
             }
             fprintf(s_ts_glove_left_csv_file, "\n");
        } else { fprintf(stderr, "Error opening L-glove CSV: %s\n", ts_csv_filename); }
        status = ts_mocap_start_streaming(s_ts_glove_left_handle); check_ts_error_internal(status, "start L-glove mocap"); if(status == 0) {printf("  Teslasuit Left Glove Mocap streaming started.\n"); started_any = true;}
    }
    if (s_ts_glove_right_handle) {
        sprintf(ts_csv_filename, "%s/sentence_%03d_ts_glove_R_mocap.csv", output_dir, sentence_id);
        s_ts_glove_right_csv_file = fopen(ts_csv_filename, "w");
        if (s_ts_glove_right_csv_file) {
            fprintf(s_ts_glove_right_csv_file, "sentence_id,frame_id,frame_timestamp_us");
            for (int i = 0; i < num_right_glove_bones_to_log; ++i) {
                const char* bone_name = ts_bone_index_to_name_str(right_glove_bone_order[i]);
                fprintf(s_ts_glove_right_csv_file, ",%s.position.x,%s.position.y,%s.position.z,%s.rotation.w,%s.rotation.x,%s.rotation.y,%s.rotation.z",
                        bone_name, bone_name, bone_name, bone_name, bone_name, bone_name, bone_name);
            }
            fprintf(s_ts_glove_right_csv_file, "\n");
        } else { fprintf(stderr, "Error opening R-glove CSV: %s\n", ts_csv_filename); }
        status = ts_mocap_start_streaming(s_ts_glove_right_handle); check_ts_error_internal(status, "start R-glove mocap"); if(status == 0) {printf("  Teslasuit Right Glove Mocap streaming started.\n"); started_any = true;}
    }
    return started_any;
}

void ts_stop_mocap_recording() {
    if (s_ts_suit_handle) {
        TsStatusCode status = ts_mocap_stop_streaming(s_ts_suit_handle);
        check_ts_error_internal(status, "ts_mocap_stop_streaming for Suit");
        if (s_ts_suit_csv_file) { fflush(s_ts_suit_csv_file); fclose(s_ts_suit_csv_file); s_ts_suit_csv_file = NULL; }
        printf("  Teslasuit Suit Mocap streaming stopped.\n");
    }
    if (s_ts_glove_left_handle) {
        TsStatusCode status = ts_mocap_stop_streaming(s_ts_glove_left_handle);
        check_ts_error_internal(status, "ts_mocap_stop_streaming for Left Glove");
        if (s_ts_glove_left_csv_file) { fflush(s_ts_glove_left_csv_file); fclose(s_ts_glove_left_csv_file); s_ts_glove_left_csv_file = NULL; }
        printf("  Teslasuit Left Glove Mocap streaming stopped.\n");
    }
    if (s_ts_glove_right_handle) {
        TsStatusCode status = ts_mocap_stop_streaming(s_ts_glove_right_handle);
        check_ts_error_internal(status, "ts_mocap_stop_streaming for Right Glove");
        if (s_ts_glove_right_csv_file) { fflush(s_ts_glove_right_csv_file); fclose(s_ts_glove_right_csv_file); s_ts_glove_right_csv_file = NULL; }
        printf("  Teslasuit Right Glove Mocap streaming stopped.\n");
    }
}