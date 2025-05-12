#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <librealsense2/rs.h>
// ... other RealSense includes ...
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <time.h>
#endif
#include <pthread.h>

#include "face_detector.h" // Your C-compatible face landmark header

// --- TESLASUIT SDK HEADERS ---
#include <ts_api/ts_core_api.h>
#include <ts_api/ts_device_api.h>
#include <ts_api/ts_mocap_api.h>
#include <ts_api/ts_force_feedback_api.h>
#include <ts_api/ts_types.h>
// Potentially ts_mapping_api.h if you need channel IDs for haptics later

#define MAX_FILENAME_LEN 256
#define MAX_TS_DEVICES 10 // Max Teslasuit devices we might try to handle

// --- Global Teslasuit Device Handles ---
TsDeviceHandle* ts_suit_dev_handle = NULL;
TsDeviceHandle* ts_glove_left_dev_handle = NULL;
TsDeviceHandle* ts_glove_right_dev_handle = NULL;

// --- Teslasuit Threading and Callback Data Structures ---

// Data shared between a Teslasuit acquisition thread and its SDK callback
typedef struct {
    FILE* csv_file;
    int sentence_id;
    volatile long frame_counter; // Incremented by the callback for this stream
    pthread_mutex_t* p_mutex;   // Pointer to the mutex in teslasuit_pthread_args_t
} ts_sdk_callback_shared_data_t;

// Arguments passed to each Teslasuit pthread
typedef struct {
    TsDeviceHandle* device_handle;
    char output_csv_path[MAX_FILENAME_LEN];
    int sentence_id;
    volatile bool* p_master_should_record; // Pointer to the main loop's recording flag

    TsProductType device_product_type;
    TsDeviceSide device_side; // Relevant for gloves

    ts_sdk_callback_shared_data_t callback_shared_data; // Initialized by thread, used by callback
    pthread_mutex_t callback_mutex;                     // Owned by the thread
} teslasuit_pthread_args_t;


// --- Helper Functions ---
void check_rs_error(rs2_error* e); // You have this

void check_ts_status(TsStatusCode code, const char* function_name, const char* details) {
    if (code != 0) { // Assuming 0 is TS_SUCCESS (Good)
        fprintf(stderr, "Teslasuit API Error in %s: %s (Code: %d). Details: %s\n",
                function_name, ts_get_status_code_message(code), code, details ? details : "");
        // Potentially exit or handle more gracefully
        if (code < 0) { // Typically, negative codes are errors
             // ts_uninitialize(); // Maybe cleanup before exit
             // exit(EXIT_FAILURE);
        }
    }
}

uint64_t get_high_res_timestamp_us() {
#ifdef _WIN32
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (uint64_t)(counter.QuadPart * 1000000.0 / frequency.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + (uint64_t)ts.tv_nsec / 1000;
#endif
}

const char* get_bone_name_str(TsBoneIndex bone_index) {
    switch (bone_index) {
        case TsBoneIndex_Hips: return "root"; // As per Teslasuit Studio CSV
        case TsBoneIndex_LeftUpperLeg: return "left_upper_leg";
        case TsBoneIndex_RightUpperLeg: return "right_upper_leg";
        case TsBoneIndex_LeftLowerLeg: return "left_lower_leg";
        case TsBoneIndex_RightLowerLeg: return "right_lower_leg";
        case TsBoneIndex_LeftFoot: return "left_foot";
        case TsBoneIndex_RightFoot: return "right_foot";
        case TsBoneIndex_Spine: return "spine";
        case TsBoneIndex_Chest: return "chest";
        case TsBoneIndex_UpperSpine: return "upper_spine"; // Or "spine1", "spine2" if studio uses that
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
        // TsBoneIndex_BonesCount is not a real bone
        default: return "unknown_bone";
    }
}

const char* get_biomechanical_angle_name_str(TsBiomechanicalIndex biomech_index) {
    switch (biomech_index) {
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
        case TsBiomechanicalIndex_ShoulderAddAbdR: return "ShoulderAddAbdR";
        case TsBiomechanicalIndex_ShoulderRotR: return "ShoulderRotR";
        case TsBiomechanicalIndex_ShoulderFlexExtR: return "ShoulderFlexExtR";
        case TsBiomechanicalIndex_ShoulderAddAbdL: return "ShoulderAddAbdL";
        case TsBiomechanicalIndex_ShoulderRotL: return "ShoulderRotL";
        case TsBiomechanicalIndex_ShoulderFlexExtL: return "ShoulderFlexExtL";
        default: return "UnknownBiomechanicalAngle";
    }
}


// --- Teslasuit SDK Callbacks ---

static void suit_mocap_callback(TsDeviceHandle* dev_handle, TsMocapSkeleton skeleton_data, void* user_data) {
    if (!user_data) return;
    ts_sdk_callback_shared_data_t* cb_data = (ts_sdk_callback_shared_data_t*)user_data;

    uint64_t system_ts_us = get_high_res_timestamp_us();
    long current_frame_idx;

    pthread_mutex_lock(cb_data->p_mutex);
    current_frame_idx = cb_data->frame_counter++;
    pthread_mutex_unlock(cb_data->p_mutex);

    if (!cb_data->csv_file) return;

    fprintf(cb_data->csv_file, "%d,%ld,%llu",
            cb_data->sentence_id,
            current_frame_idx,
            system_ts_us);

    TsMocapBone bone_transform;
    TsStatusCode status;

    // Mocap Bone Data - iterate all bones and append to the current line
    for (int i = 0; i < TsBoneIndex_BonesCount; ++i) {
        TsBoneIndex current_bone_idx = (TsBoneIndex)i;
        // Skip TsBoneIndex_BonesCount itself if it's causing "unknown_bone"
        const char* bone_name_check = get_bone_name_str(current_bone_idx);
        if (strcmp(bone_name_check, "unknown_bone") == 0 && current_bone_idx == TsBoneIndex_BonesCount) {
             // If we want exactly 50 bones data columns even if the last one is dummy:
             fprintf(cb_data->csv_file, ",0,0,0,0,0,0,0"); // Dummy data for "BonesCount"
             continue;
        }


        status = ts_mocap_skeleton_get_bone(skeleton_data, current_bone_idx, &bone_transform);
        if (status == 0) { // Success
            fprintf(cb_data->csv_file, ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                    bone_transform.position.x, bone_transform.position.y, bone_transform.position.z,
                    bone_transform.rotation.w, bone_transform.rotation.x, bone_transform.rotation.y, bone_transform.rotation.z);
        } else {
            // Output placeholder data if bone is not available, to keep CSV structure consistent
            fprintf(cb_data->csv_file, ",0,0,0,0,0,0,0"); // Or NaN, or empty strings if preferred
        }
    }

    // Biomechanical Angle Data - iterate and append
    TsBiomechanicalIndex biomech_indices_to_log[] = {
            TsBiomechanicalIndex_PelvisTilt, TsBiomechanicalIndex_PelvisList, TsBiomechanicalIndex_PelvisRotation,
            TsBiomechanicalIndex_HipFlexExtR, TsBiomechanicalIndex_HipAddAbdR, TsBiomechanicalIndex_HipRotR,
            TsBiomechanicalIndex_KneeFlexExtR, TsBiomechanicalIndex_AnkleFlexExtR, TsBiomechanicalIndex_AnkleProSupR,
            TsBiomechanicalIndex_HipFlexExtL, TsBiomechanicalIndex_HipAddAbdL, TsBiomechanicalIndex_HipRotL,
            TsBiomechanicalIndex_KneeFlexExtL, TsBiomechanicalIndex_AnkleFlexExtL, TsBiomechanicalIndex_AnkleProSupL,
            TsBiomechanicalIndex_ElbowFlexExtR, TsBiomechanicalIndex_ForearmProSupR,
            TsBiomechanicalIndex_WristFlexExtR, TsBiomechanicalIndex_WristDeviationR,
            TsBiomechanicalIndex_ElbowFlexExtL, TsBiomechanicalIndex_ForearmProSupL,
            TsBiomechanicalIndex_WristFlexExtL, TsBiomechanicalIndex_WristDeviationL,
            TsBiomechanicalIndex_ShoulderAddAbdR, TsBiomechanicalIndex_ShoulderRotR, TsBiomechanicalIndex_ShoulderFlexExtR,
            TsBiomechanicalIndex_ShoulderAddAbdL, TsBiomechanicalIndex_ShoulderRotL, TsBiomechanicalIndex_ShoulderFlexExtL
    };
    int num_biomech_indices = sizeof(biomech_indices_to_log) / sizeof(biomech_indices_to_log[0]);
    for(int i=0; i < num_biomech_indices; ++i) {
        float angle_value = 0.0f;
        status = ts_mocap_skeleton_get_biomechanical_angle(skeleton_data, biomech_indices_to_log[i], &angle_value);
        if (status == 0) { // Success
            fprintf(cb_data->csv_file, ",%.6f", angle_value);
        } else {
            fprintf(cb_data->csv_file, ",0"); // Placeholder
        }
    }

    fprintf(cb_data->csv_file, "\n"); // End of data line for this frame
    fflush(cb_data->csv_file);
}

static void glove_position_callback(TsDeviceHandle* dev_handle, TsForceFeedbackPositionContainer position_data, void* user_data) {
    if (!user_data) return;
    ts_sdk_callback_shared_data_t* cb_data = (ts_sdk_callback_shared_data_t*)user_data;

    uint64_t system_ts_us = get_high_res_timestamp_us();
    long current_frame_idx;

    pthread_mutex_lock(cb_data->p_mutex);
    current_frame_idx = cb_data->frame_counter++;
    pthread_mutex_unlock(cb_data->p_mutex);

    if (!cb_data->csv_file) return;

    // Determine which glove based on dev_handle if needed, or assume thread context implies it
    // For this example, we rely on the thread args to know if it's left/right implicitly for bone iteration
    // A more robust way would be to store product_type/side in cb_data if the same callback is used for L/R.

    TsBoneIndex finger_bones_start, finger_bones_end;

    // This is a bit tricky: The callback gets a generic dev_handle.
    // We need to know if it's left or right to iterate the correct bones.
    // We'll assume the calling thread sets up the callback_shared_data knowing which glove it is.
    // For now, let's make a simplification or pass more info in user_data.
    // Let's assume the calling thread passes product type / side in a richer user_data or the
    // thread only deals with one glove. The current `teslasuit_pthread_args_t` has this info.
    // The `cb_data` is part of `teslasuit_pthread_args_t`.
    // This callback is set by a specific thread for a specific glove.
    // We need to know which glove it is to iterate over correct bones.
    // One way is to pass more specific info in user_data or have separate callbacks.
    // For now, let's assume the `user_data` has enough context, or the parent `teslasuit_pthread_args_t` is accessible.
    // Simpler: the thread function knows its device side. It could set a flag in cb_data.
    // Or, we check the dev_handle against our global glove handles.

    TsDeviceSide glove_side = TsDeviceSide_Undefined;
    if(dev_handle == ts_glove_left_dev_handle) glove_side = TsDeviceSide_Left;
    else if(dev_handle == ts_glove_right_dev_handle) glove_side = TsDeviceSide_Right;

    if (glove_side == TsDeviceSide_Left) {
        finger_bones_start = TsBoneIndex_LeftThumbProximal;
        finger_bones_end = TsBoneIndex_LeftLittleDistal;
    } else if (glove_side == TsDeviceSide_Right) {
        finger_bones_start = TsBoneIndex_RightThumbProximal;
        finger_bones_end = TsBoneIndex_RightLittleDistal;
    } else {
        return; // Not a recognized glove
    }

    for (int i = finger_bones_start; i <= finger_bones_end; ++i) {
        TsBoneIndex current_finger_bone_idx = (TsBoneIndex)i;
        float flexion_angle = 0.0f, abduction_angle = 0.0f;
        TsStatusCode status_flex, status_abd;

        status_flex = ts_force_feedback_get_flexion_angle(position_data, current_finger_bone_idx, &flexion_angle);
        status_abd = ts_force_feedback_get_abduction_angle(position_data, current_finger_bone_idx, &abduction_angle);

        // It's possible not all bones provide both, or any. Check status.
        // For simplicity, we write even if one fails, with the value being 0.
        fprintf(cb_data->csv_file, "%d,%ld,%llu,%s,%d,%.6f,%.6f\n",
                cb_data->sentence_id,
                current_frame_idx,
                system_ts_us,
                get_bone_name_str(current_finger_bone_idx),
                current_finger_bone_idx,
                (status_flex == 0) ? flexion_angle : 0.0f, // Assuming 0 is success
                (status_abd == 0) ? abduction_angle : 0.0f);
    }
    fflush(cb_data->csv_file);
}


// --- Teslasuit Acquisition Thread Function ---
// --- Teslasuit Acquisition Thread Function ---
void* teslasuit_acquisition_thread_func(void* arg) {
    teslasuit_pthread_args_t* data = (teslasuit_pthread_args_t*)arg;
    TsStatusCode status;

    // 1. Initialize callback shared data
    data->callback_shared_data.sentence_id = data->sentence_id;
    data->callback_shared_data.frame_counter = 0;
    data->callback_shared_data.p_mutex = &data->callback_mutex;

    data->callback_shared_data.csv_file = fopen(data->output_csv_path, "w");
    if (!data->callback_shared_data.csv_file) {
        fprintf(stderr, "Error: Could not open Teslasuit CSV file: %s\n", data->output_csv_path);
        return NULL;
    }

    // Write CSV Header
    if (data->device_product_type == TsProductType_Suit) {
        fprintf(data->callback_shared_data.csv_file, "sentence_id,ts_frame_idx,system_capture_ts_us");
        // Add Mocap Bone Data Headers
        for (int i = 0; i < TsBoneIndex_BonesCount; ++i) {
            TsBoneIndex current_bone_idx = (TsBoneIndex)i;
            const char* bone_name = get_bone_name_str(current_bone_idx);
            // Skip "unknown_bone" if TsBoneIndex_BonesCount itself is processed or default case hit
            if (strcmp(bone_name, "unknown_bone") == 0 && current_bone_idx == TsBoneIndex_BonesCount) continue;


            fprintf(data->callback_shared_data.csv_file, ",%s.position.x,%s.position.y,%s.position.z",
                    bone_name, bone_name, bone_name);
            fprintf(data->callback_shared_data.csv_file, ",%s.rotation.w,%s.rotation.x,%s.rotation.y,%s.rotation.z",
                    bone_name, bone_name, bone_name, bone_name);
        }
        // Add Biomechanical Angle Headers
        // Need to iterate through known biomechanical indices. This is harder as there isn't a "Count" enum.
        // We'll explicitly list them based on the get_biomechanical_angle_name_str or assume a range if documented.
        // For simplicity, let's list the ones in our helper.
        TsBiomechanicalIndex biomech_indices_to_log[] = {
            TsBiomechanicalIndex_PelvisTilt, TsBiomechanicalIndex_PelvisList, TsBiomechanicalIndex_PelvisRotation,
            TsBiomechanicalIndex_HipFlexExtR, TsBiomechanicalIndex_HipAddAbdR, TsBiomechanicalIndex_HipRotR,
            TsBiomechanicalIndex_KneeFlexExtR, TsBiomechanicalIndex_AnkleFlexExtR, TsBiomechanicalIndex_AnkleProSupR,
            TsBiomechanicalIndex_HipFlexExtL, TsBiomechanicalIndex_HipAddAbdL, TsBiomechanicalIndex_HipRotL,
            TsBiomechanicalIndex_KneeFlexExtL, TsBiomechanicalIndex_AnkleFlexExtL, TsBiomechanicalIndex_AnkleProSupL,
            TsBiomechanicalIndex_ElbowFlexExtR, TsBiomechanicalIndex_ForearmProSupR,
            TsBiomechanicalIndex_WristFlexExtR, TsBiomechanicalIndex_WristDeviationR,
            TsBiomechanicalIndex_ElbowFlexExtL, TsBiomechanicalIndex_ForearmProSupL,
            TsBiomechanicalIndex_WristFlexExtL, TsBiomechanicalIndex_WristDeviationL,
            TsBiomechanicalIndex_ShoulderAddAbdR, TsBiomechanicalIndex_ShoulderRotR, TsBiomechanicalIndex_ShoulderFlexExtR,
            TsBiomechanicalIndex_ShoulderAddAbdL, TsBiomechanicalIndex_ShoulderRotL, TsBiomechanicalIndex_ShoulderFlexExtL
        };
        int num_biomech_indices = sizeof(biomech_indices_to_log) / sizeof(biomech_indices_to_log[0]);
        for(int i=0; i < num_biomech_indices; ++i) {
            const char* angle_name = get_biomechanical_angle_name_str(biomech_indices_to_log[i]);
            fprintf(data->callback_shared_data.csv_file, ",%s.angle", angle_name);
        }

        fprintf(data->callback_shared_data.csv_file, "\n"); // End of header line

    } else if (data->device_product_type == TsProductType_Glove) {
        // Glove CSV header remains one-row-per-finger-bone per frame
        fprintf(data->callback_shared_data.csv_file, "sentence_id,ts_frame_idx,system_capture_ts_us,bone_name,bone_idx_val,flexion_angle,abduction_angle\n");
    }

    // ... (rest of the function: mutex init, callback registration, streaming loop, teardown) ...
    // (This part remains the same as in the previous good answer)
    if (pthread_mutex_init(&data->callback_mutex, NULL) != 0) {
        fprintf(stderr, "Error initializing Teslasuit callback mutex for %s\n", data->output_csv_path);
        fclose(data->callback_shared_data.csv_file);
        data->callback_shared_data.csv_file = NULL;
        return NULL;
    }

    // 3. Register SDK callback and start streaming
    if (data->device_product_type == TsProductType_Suit) {
        status = ts_mocap_set_skeleton_update_callback(data->device_handle, suit_mocap_callback, &data->callback_shared_data);
        check_ts_status(status, "ts_mocap_set_skeleton_update_callback", data->output_csv_path);
        if (status == 0) { // Assuming 0 is success
            status = ts_mocap_start_streaming(data->device_handle);
            check_ts_status(status, "ts_mocap_start_streaming", data->output_csv_path);
             if (status != 0) { // Failed to start streaming
                ts_mocap_set_skeleton_update_callback(data->device_handle, NULL, NULL); // Deregister
                fclose(data->callback_shared_data.csv_file);
                pthread_mutex_destroy(&data->callback_mutex);
                return NULL;
            }
        } else { // Failed to set callback
            fclose(data->callback_shared_data.csv_file);
            pthread_mutex_destroy(&data->callback_mutex);
            return NULL;
        }
    } else if (data->device_product_type == TsProductType_Glove) {
        status = ts_force_feedback_set_position_update_callback(data->device_handle, glove_position_callback, &data->callback_shared_data);
        check_ts_status(status, "ts_force_feedback_set_position_update_callback", data->output_csv_path);
        if (status == 0) { // Assuming 0 is success
            status = ts_force_feedback_start_position_streaming(data->device_handle);
            check_ts_status(status, "ts_force_feedback_start_position_streaming", data->output_csv_path);
            if (status != 0) { // Failed to start streaming
                ts_force_feedback_set_position_update_callback(data->device_handle, NULL, NULL); // Deregister
                fclose(data->callback_shared_data.csv_file);
                pthread_mutex_destroy(&data->callback_mutex);
                return NULL;
            }
        } else { // Failed to set callback
            fclose(data->callback_shared_data.csv_file);
            pthread_mutex_destroy(&data->callback_mutex);
            return NULL;
        }
    }

    printf("Teslasuit acquisition thread started for %s. Waiting for stop signal.\n", data->output_csv_path);

    // 4. Idle loop checking master recording flag
    while (*(data->p_master_should_record)) {
        Sleep(100); // Check flag every 100ms
    }

    printf("Teslasuit acquisition thread stopping for %s.\n", data->output_csv_path);

    // 5. Teardown
    if (data->device_product_type == TsProductType_Suit) {
        status = ts_mocap_stop_streaming(data->device_handle);
        check_ts_status(status, "ts_mocap_stop_streaming", data->output_csv_path);
        ts_mocap_set_skeleton_update_callback(data->device_handle, NULL, NULL);
    } else if (data->device_product_type == TsProductType_Glove) {
        status = ts_force_feedback_stop_position_streaming(data->device_handle);
        check_ts_status(status, "ts_force_feedback_stop_position_streaming", data->output_csv_path);
        ts_force_feedback_set_position_update_callback(data->device_handle, NULL, NULL);
    }

    if (data->callback_shared_data.csv_file) {
        fclose(data->callback_shared_data.csv_file);
        data->callback_shared_data.csv_file = NULL;
    }
    pthread_mutex_destroy(&data->callback_mutex);

    printf("Teslasuit acquisition thread finished for %s.\n", data->output_csv_path);
    return NULL;
}

// --- Main Application ---
int main(int argc, char* argv[]) {
    rs2_error* e = NULL;
    // ... (RealSense variables) ...
    int sentence_id = 0;
    char output_dir[MAX_FILENAME_LEN] = "recorded_data";
    float skip_seconds = 0.5f; // For RealSense post-processing
    volatile bool master_should_record = false; // Controls all recording threads

    printf("Sign Language Sentence Recorder with Teslasuit\n");

    // --- Teslasuit Initialization ---
    TsStatusCode ts_status;
    ts_status = ts_initialize();
    check_ts_status(ts_status, "ts_initialize", NULL);
    if (ts_status != 0) return EXIT_FAILURE;

    TsVersion sdk_version = ts_get_version();
    printf("Teslasuit SDK Initialized. Version: %d.%d.%d (Build %d)\n",
           sdk_version.major, sdk_version.minor, sdk_version.patch, sdk_version.build);

    // Discover and open Teslasuit devices
    uint32_t ts_device_count = MAX_TS_DEVICES;
    TsDevice ts_device_list[MAX_TS_DEVICES]; // Array of TsDevice structs
    ts_status = ts_get_device_list(ts_device_list, &ts_device_count);
    check_ts_status(ts_status, "ts_get_device_list", NULL);

    if (ts_status == 0 && ts_device_count > 0) {
        printf("Found %u Teslasuit devices:\n", ts_device_count);
        for (uint32_t i = 0; i < ts_device_count; ++i) {
            // Note: ts_device_open takes const TsDevice*, so &ts_device_list[i]
            TsDeviceHandle* current_dev_handle = ts_device_open(&ts_device_list[i]);
            if (current_dev_handle) {
                const char* dev_name = ts_device_get_name(current_dev_handle);
                TsProductType prod_type = ts_device_get_product_type(current_dev_handle);
                TsDeviceSide dev_side = ts_device_get_device_side(current_dev_handle);
                printf("  - Device %u: %s, Type: %d, Side: %d\n", i, dev_name ? dev_name : "N/A", prod_type, dev_side);

                if (prod_type == TsProductType_Suit && !ts_suit_dev_handle) {
                    ts_suit_dev_handle = current_dev_handle;
                    printf("    Assigned as Bodysuit.\n");
                } else if (prod_type == TsProductType_Glove) {
                    if (dev_side == TsDeviceSide_Left && !ts_glove_left_dev_handle) {
                        ts_glove_left_dev_handle = current_dev_handle;
                        printf("    Assigned as Left Glove.\n");
                    } else if (dev_side == TsDeviceSide_Right && !ts_glove_right_dev_handle) {
                        ts_glove_right_dev_handle = current_dev_handle;
                        printf("    Assigned as Right Glove.\n");
                    } else {
                        // Already assigned or undefined side, close it if we don't need it
                         printf("    Glove not assigned (already have one for this side or side undefined). Closing.\n");
                         ts_device_close(current_dev_handle);
                    }
                } else {
                    // Not a suit or glove we are looking for, or already assigned
                    if(prod_type == TsProductType_Suit && ts_suit_dev_handle && current_dev_handle != ts_suit_dev_handle) {
                         printf("    Another suit found, but one is already assigned. Closing.\n");
                         ts_device_close(current_dev_handle);
                    } else if (prod_type != TsProductType_Suit && prod_type != TsProductType_Glove){
                         printf("    Unknown product type. Closing.\n");
                         ts_device_close(current_dev_handle);
                    }
                }
            } else {
                check_ts_status(-1, "ts_device_open", "Failed to open a listed device"); // Assuming non-zero is error
            }
        }
    } else if (ts_status == 0 && ts_device_count == 0) {
        printf("No Teslasuit devices found.\n");
    }
    // --- RealSense Initialization (your existing code) ---
    rs2_context* ctx = rs2_create_context(RS2_API_VERSION, &e);
    // ... (rest of RealSense init)
    // Initialize Face Landmark Detector (your existing code)
    FaceLandmarkDetector* landmark_detector = create_landmark_detector("shape_predictor_68_face_landmarks.dat");


    // --- Main Recording Loop ---
    char input_buffer[10];
    while (1) {
        printf("\nPress [Enter] to start recording sentence %d, or 'q' [Enter] to quit: ", sentence_id);
        if (fgets(input_buffer, sizeof(input_buffer), stdin) == NULL || input_buffer[0] == 'q') {
            master_should_record = false; // Ensure threads trying to start would stop
            break;
        }
        master_should_record = true; // Signal threads to start/continue recording

        // --- Start RealSense Recording (your existing logic to record to .bag) ---
        printf("Starting RealSense recording for sentence %d...\n", sentence_id);
        // ... (code to setup pipeline, config, start recording to bag_filename)
        // Example snippet:
        rs2_pipeline* pipeline_rec = rs2_create_pipeline(ctx, &e); check_rs_error(e);
        rs2_config* config_rec = rs2_create_config(&e); check_rs_error(e);
        char bag_filename[MAX_FILENAME_LEN];
        sprintf(bag_filename, "%s/sentence_%03d.bag", output_dir, sentence_id);
        rs2_config_enable_record_to_file(config_rec, bag_filename, &e); check_rs_error(e);
        rs2_pipeline_profile* profile_rec = rs2_pipeline_start_with_config(pipeline_rec, config_rec, &e);
        if (e) {
            fprintf(stderr, "Failed to start RealSense recording pipeline.\n");
            check_rs_error(e); // This will exit or handle
            master_should_record = false; // Reset flag
            // Clean up pipeline_rec, config_rec if needed
            continue;
        }
        printf("RealSense recording to %s...\n", bag_filename);


        // --- Start Teslasuit Recording Threads ---
        pthread_t suit_thread_id, glove_l_thread_id, glove_r_thread_id;
        teslasuit_pthread_args_t suit_args, glove_l_args, glove_r_args;
        bool suit_thread_started = false, glove_l_thread_started = false, glove_r_thread_started = false;

        if (ts_suit_dev_handle) {
            suit_args.device_handle = ts_suit_dev_handle;
            sprintf(suit_args.output_csv_path, "%s/sentence_%03d_teslasuit_suit.csv", output_dir, sentence_id);
            suit_args.sentence_id = sentence_id;
            suit_args.p_master_should_record = &master_should_record;
            suit_args.device_product_type = TsProductType_Suit;
            suit_args.device_side = TsDeviceSide_Undefined;
            if (pthread_create(&suit_thread_id, NULL, teslasuit_acquisition_thread_func, &suit_args) == 0) {
                suit_thread_started = true;
            } else {
                fprintf(stderr, "Error creating Teslasuit suit acquisition thread.\n");
            }
        }
        if (ts_glove_left_dev_handle) {
            glove_l_args.device_handle = ts_glove_left_dev_handle;
            sprintf(glove_l_args.output_csv_path, "%s/sentence_%03d_teslasuit_glove_left.csv", output_dir, sentence_id);
            glove_l_args.sentence_id = sentence_id;
            glove_l_args.p_master_should_record = &master_should_record;
            glove_l_args.device_product_type = TsProductType_Glove;
            glove_l_args.device_side = TsDeviceSide_Left;
             if (pthread_create(&glove_l_thread_id, NULL, teslasuit_acquisition_thread_func, &glove_l_args) == 0) {
                glove_l_thread_started = true;
            } else {
                fprintf(stderr, "Error creating Teslasuit left glove acquisition thread.\n");
            }
        }
        if (ts_glove_right_dev_handle) {
            // Similar setup for right glove
            glove_r_args.device_handle = ts_glove_right_dev_handle;
            sprintf(glove_r_args.output_csv_path, "%s/sentence_%03d_teslasuit_glove_right.csv", output_dir, sentence_id);
            glove_r_args.sentence_id = sentence_id;
            glove_r_args.p_master_should_record = &master_should_record;
            glove_r_args.device_product_type = TsProductType_Glove;
            glove_r_args.device_side = TsDeviceSide_Right;
             if (pthread_create(&glove_r_thread_id, NULL, teslasuit_acquisition_thread_func, &glove_r_args) == 0) {
                glove_r_thread_started = true;
            } else {
                fprintf(stderr, "Error creating Teslasuit right glove acquisition thread.\n");
            }
        }

        if (!suit_thread_started && ts_suit_dev_handle ||
            !glove_l_thread_started && ts_glove_left_dev_handle ||
            !glove_r_thread_started && ts_glove_right_dev_handle) {
            fprintf(stderr, "One or more Teslasuit threads failed to start. Stopping RealSense.\n");
            master_should_record = false; // Signal any started threads to stop
            if (suit_thread_started) pthread_join(suit_thread_id, NULL);
            if (glove_l_thread_started) pthread_join(glove_l_thread_id, NULL);
            if (glove_r_thread_started) pthread_join(glove_r_thread_id, NULL);

            rs2_pipeline_stop(pipeline_rec, &e); check_rs_error(e);
            rs2_delete_pipeline_profile(profile_rec);
            rs2_delete_config(config_rec);
            rs2_delete_pipeline(pipeline_rec);
            continue; // Skip to next sentence prompt
        }


        printf("All recordings started. Press [Enter] to stop recording sentence %d.\n", sentence_id);
        getchar(); // Wait for Enter to stop

        // --- Stop Recording ---
        printf("Stopping recording for sentence %d...\n", sentence_id);
        master_should_record = false; // Signal all threads to stop

        // Join Teslasuit threads
        if (suit_thread_started) pthread_join(suit_thread_id, NULL);
        if (glove_l_thread_started) pthread_join(glove_l_thread_id, NULL);
        if (glove_r_thread_started) pthread_join(glove_r_thread_id, NULL);
        printf("Teslasuit data acquisition stopped.\n");

        // Stop RealSense recording
        rs2_pipeline_stop(pipeline_rec, &e); check_rs_error(e);
        printf("RealSense recording stopped. Bag file: %s\n", bag_filename);
        rs2_delete_pipeline_profile(profile_rec);
        rs2_delete_config(config_rec);
        rs2_delete_pipeline(pipeline_rec);


        // --- Post-processing .bag file (your existing logic) ---
        printf("Processing %s for landmarks...\n", bag_filename);
        // ... (your existing RealSense .bag processing and landmark extraction code) ...
        // Make sure to use a different pipeline/config for playback if needed.

        sentence_id++;
    }

    // --- Cleanup ---
    printf("Shutting down...\n");
    master_should_record = false; // Final ensure for any lingering threads (shouldn't be any if joined)

    destroy_landmark_detector(landmark_detector);

    // Teslasuit Cleanup
    if (ts_suit_dev_handle) ts_device_close(ts_suit_dev_handle);
    if (ts_glove_left_dev_handle) ts_device_close(ts_glove_left_dev_handle);
    if (ts_glove_right_dev_handle) ts_device_close(ts_glove_right_dev_handle);
    ts_uninitialize();
    printf("Teslasuit SDK Uninitialized.\n");

    // RealSense Cleanup
    // ... (your existing RealSense context cleanup, e.g., rs2_delete_context(ctx)) ...
    if (ctx) rs2_delete_context(ctx);


    printf("Application finished.\n");
    return EXIT_SUCCESS;
}

void check_rs_error(rs2_error* e) { // Ensure this is defined
    if (e) {
        fprintf(stderr, "RealSense error calling %s(%s):\n  %s\n",
                rs2_get_failed_function(e), rs2_get_failed_args(e), rs2_get_error_message(e));
        rs2_free_error(e);
        // Consider if exit is always appropriate or if the calling code should handle it
        // exit(EXIT_FAILURE);
    }
}