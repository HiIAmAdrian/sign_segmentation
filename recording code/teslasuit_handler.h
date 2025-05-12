#ifndef TESLASUIT_HANDLER_H
#define TESLASUIT_HANDLER_H

#include <ts_api/ts_types.h> // For TsDeviceHandle, TsStatusCode etc.
#include <stdbool.h>         // For bool type

// Initialization and Cleanup
bool ts_initialize_system();
void ts_discover_and_open_devices();
void ts_close_all_devices();
void ts_uninitialize_system();

// Recording Control
// Returns true if at least one device started streaming, false otherwise.
bool ts_start_mocap_recording(int sentence_id, const char* output_dir);
void ts_stop_mocap_recording();
bool ts_calibrate_suit();
bool ts_calibrate_glove_left();
bool ts_calibrate_glove_right();

// Utility (publicly accessible error checker if needed, though internal is also used)
void check_ts_error_public(TsStatusCode status_code, const char* context_msg);

#endif // TESLASUIT_HANDLER_H