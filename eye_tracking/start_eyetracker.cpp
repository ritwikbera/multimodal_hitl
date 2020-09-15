// Compile it by running:
// gcc start_eyetracker.cpp -o start_eyetracker.sh -pthread /usr/lib/tobii/libtobii_stream_engine.so -lstdc++
#include <tobii/tobii.h>
#include <tobii/tobii_streams.h>
#include <tuple>
#include <stdio.h>
#include <assert.h>
#include <cstring>
#include <iostream>

// global eye tracker object to return data directly from callback
// setup globals to pass data between Cpp and Python
float global_gaze[2]; // [x,y]


void gaze_point_callback(tobii_gaze_point_t const *gaze_point, void *user_data) {
    // start reading
    if (gaze_point->validity == TOBII_VALIDITY_VALID) {
        // printf("%f,%f\n",
        //        gaze_point->position_xy[0],
        //        gaze_point->position_xy[1]);
        
        // pass gaze data to global eyetracker class
        ::global_gaze[0] = gaze_point->position_xy[0];
        ::global_gaze[1] = gaze_point->position_xy[1];
    }
}

static void url_receiver(char const *url, void *user_data) {
    char *buffer = (char *) user_data;
    if (*buffer != '\0') return; // only keep first value

    if (strlen(url) < 256)
        strcpy(buffer, url);
}


int main() {    
    // create API
    tobii_api_t *api;
    tobii_error_t result = tobii_api_create(&api, NULL, NULL);
    assert(result == TOBII_ERROR_NO_ERROR);

    // enumerate devices to find connected eye trackers, keep the first
    char url[256] = {0};
    result = tobii_enumerate_local_device_urls(api, url_receiver, url);
    assert(result == TOBII_ERROR_NO_ERROR && *url != '\0');

    // connect to device
    tobii_device_t *device;
    result = tobii_device_create(api, url, &device);
    assert(result == TOBII_ERROR_NO_ERROR);

    result = tobii_gaze_point_subscribe(device, gaze_point_callback, 0);
    assert(result == TOBII_ERROR_NO_ERROR);

    // start device and read data using callbacks
    int cnt_loop = 0;  // count gaze samples
    while (true) {

        // optionally block this thread until data is available. Especially useful if running in a separate thread.
        result = tobii_wait_for_callbacks(1, &device);
        assert(result == TOBII_ERROR_NO_ERROR || result == TOBII_ERROR_TIMED_OUT);

        // process callbacks on this thread if data is available
        result = tobii_device_process_callbacks(device);
        assert(result == TOBII_ERROR_NO_ERROR);

        // display value stored in eyetracker class
        printf("%f,%f\n", global_gaze[0], global_gaze[1]);
        std::cout.flush();

        // update counter loop
        // printf("cnt_loop: %i\n", cnt_loop);
        cnt_loop++;
    }

    result = tobii_gaze_point_unsubscribe(device);
    assert(result == TOBII_ERROR_NO_ERROR);

    result = tobii_device_destroy(device);
    assert(result == TOBII_ERROR_NO_ERROR);

    result = tobii_api_destroy(api);
    assert(result == TOBII_ERROR_NO_ERROR);

    return 0;
}