/* Small, fixed ABI bridge for the explicitly side-loaded, qualified oneMKL
 * 2024.2 Linux x86-64 LP64/sequential DFTI profile. No vendor code is included.
 * Public API declarations follow Intel's C Developer Reference (2024.2):
 * DftiCreateDescriptor, DftiSetValue, DftiCommitDescriptor,
 * DftiComputeForward, DftiFreeDescriptor, and the thread/version services.
 * The generic DftiCreateDescriptor function is intentionally used, not the
 * header macro's private specialized targets.
 */
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if !defined(__linux__) || !defined(__x86_64__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error Unsupported bridge ABI
#endif
_Static_assert(sizeof(long) == 8 && sizeof(int) == 4 && sizeof(void *) == 8 &&
               sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "Linux x86-64 LP64 and IEEE float32 required");

typedef struct DFTI_DESCRIPTOR *Descriptor;
extern long DftiCreateDescriptor(Descriptor *, int, int, long, ...);
extern long DftiSetValue(Descriptor, int, ...);
extern long DftiCommitDescriptor(Descriptor);
extern long DftiComputeForward(Descriptor, void *, ...);
extern long DftiFreeDescriptor(Descriptor *);
extern int MKL_Set_Num_Threads_Local(int);
extern int MKL_Get_Max_Threads(void);
extern void MKL_Get_Version_String(char *, int);

enum {
    DFTI_NUMBER_OF_TRANSFORMS = 7, DFTI_CONJUGATE_EVEN_STORAGE = 10,
    DFTI_PLACEMENT = 11, DFTI_INPUT_STRIDES = 12, DFTI_OUTPUT_STRIDES = 13,
    DFTI_INPUT_DISTANCE = 14, DFTI_OUTPUT_DISTANCE = 15, DFTI_REAL = 33,
    DFTI_SINGLE = 35, DFTI_COMPLEX_COMPLEX = 39, DFTI_NOT_INPLACE = 44,
    FFT_LENGTH = 320, COMPLEX_BINS = 161, MAX_FRAMES = 3000
};

struct MklStatus {
    int64_t native_status;
    int64_t cleanup_status;
    int32_t stage;
    int32_t prior_threads;
    int32_t effective_threads;
    int32_t restored_previous;
    int32_t descriptor_released;
};
_Static_assert(sizeof(struct MklStatus) == 40, "fixed bridge status ABI");

#define EXPORTED __attribute__((visibility("default")))
EXPORTED uint32_t gigastt_v4_mkl_abi(void) { return 1; }

EXPORTED int gigastt_v4_mkl_version(char *buffer, size_t length) {
    if (!buffer || length < 512 || length > INT_MAX) return 1;
    memset(buffer, 0, length);
    MKL_Get_Version_String(buffer, (int)length);
    buffer[length - 1] = 0;
    return 0;
}

/* Stage 0 success; 1/2 input; 3 thread; 4 create; 5 null descriptor;
 * 6..12 configuration; 13 commit; 14 transform; 15 cleanup; 16 restore.
 * Rust owns both disjoint, aligned buffers. No descriptor or C allocation
 * outlives this synchronous call. All failures after a successful create are
 * followed by exactly one free attempt and restoration of the thread setting.
 */
EXPORTED int gigastt_v4_mkl_forward(float *input, size_t input_len,
                                   float *output, size_t output_len,
                                   long frames, struct MklStatus *status) {
    if (!status) return 1;
    memset(status, 0, sizeof(*status));
    if (!input || !output || frames < 1 || frames > MAX_FRAMES ||
        input_len != (size_t)frames * FFT_LENGTH ||
        output_len != (size_t)frames * COMPLEX_BINS * 2 ||
        (uintptr_t)input % 64 || (uintptr_t)output % 64) {
        status->stage = 1;
        return 1;
    }
    for (size_t i = 0; i < input_len; ++i) {
        if (!isfinite(input[i])) { status->stage = 2; return 1; }
    }

    Descriptor descriptor = NULL;
    int created = 0;
    long strides[2] = {0, 1};
    status->prior_threads = MKL_Set_Num_Threads_Local(1);
    status->effective_threads = MKL_Get_Max_Threads();
    if (status->effective_threads != 1 || status->prior_threads < 0) {
        status->stage = 3;
        goto cleanup;
    }
#define CHECK(stage_number, expression) do { \
    status->stage = (stage_number); \
    status->native_status = (expression); \
    if (status->native_status != 0) goto cleanup; \
} while (0)
    CHECK(4, DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_REAL, (long)1, (long)FFT_LENGTH));
    if (!descriptor) { status->stage = 5; goto cleanup; }
    created = 1;
    CHECK(6, DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
    CHECK(7, DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, frames));
    CHECK(8, DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, (long)FFT_LENGTH));
    CHECK(9, DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, (long)COMPLEX_BINS));
    CHECK(10, DftiSetValue(descriptor, DFTI_INPUT_STRIDES, strides));
    CHECK(11, DftiSetValue(descriptor, DFTI_OUTPUT_STRIDES, strides));
    CHECK(12, DftiSetValue(descriptor, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
    CHECK(13, DftiCommitDescriptor(descriptor));
    CHECK(14, DftiComputeForward(descriptor, input, output));
    status->stage = 0;
cleanup:
    if (created) {
        status->cleanup_status = DftiFreeDescriptor(&descriptor);
        status->descriptor_released = status->cleanup_status == 0 && !descriptor;
        if (!status->descriptor_released && status->stage == 0) status->stage = 15;
    }
    /* A failed create does not establish a valid descriptor. An anomalous
     * non-null handle in that case is not reused or dereferenced; the caller
     * marks the process-lifetime runtime unusable after any native error. */
    status->restored_previous = status->prior_threads >= 0
        ? MKL_Set_Num_Threads_Local(status->prior_threads) : INT32_MIN;
    if (status->restored_previous != 1 && status->stage == 0) status->stage = 16;
    return status->stage != 0;
#undef CHECK
}
