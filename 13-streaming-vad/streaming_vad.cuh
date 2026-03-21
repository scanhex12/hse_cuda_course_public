#pragma once

void run_streaming_vad(const float *host_input, unsigned char *host_vad,
                       int num_chunks, int samples_per_chunk, float threshold,
                       int num_streams);
