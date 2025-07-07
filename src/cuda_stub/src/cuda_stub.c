/*
 * CUDA Driver API Stub Implementation
 * 
 * This file provides stub implementations of CUDA driver API functions
 * for cross-compilation and development on systems without CUDA hardware.
 * 
 * All functions return CUDA_SUCCESS and provide minimal mock functionality.
 */

#include "../include/cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Enable/disable debug output
#define CUDA_STUB_DEBUG 0

#if CUDA_STUB_DEBUG
#define STUB_LOG(...) printf("[CUDA_STUB] " __VA_ARGS__)
#else
#define STUB_LOG(...) ((void)0)
#endif

// Global state for mock objects
static int g_initialized = 0;
static int g_device_count = 1;
static int g_context_count = 0;

// Mock device and context handles
#define MOCK_DEVICE_BASE    0x1000
#define MOCK_CONTEXT_BASE   0x2000
#define MOCK_MODULE_BASE    0x3000
#define MOCK_FUNCTION_BASE  0x4000
#define MOCK_MEMORY_BASE    0x5000

/*
 * Helper function to create mock handles
 */
static void* make_mock_handle(int base, int id) {
    return (void*)(uintptr_t)(base + id);
}

/*
 * CUDA Driver API Stub Implementations
 */

CUresult cuInit(unsigned int Flags) {
    STUB_LOG("cuInit(flags=%u)\n", Flags);
    
    if (g_initialized) {
        return CUDA_SUCCESS; // Already initialized
    }
    
    g_initialized = 1;
    STUB_LOG("CUDA Driver API initialized (stub)\n");
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    STUB_LOG("cuDeviceGet(device=%p, ordinal=%d)\n", device, ordinal);
    
    if (!g_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    if (!device) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    if (ordinal < 0 || ordinal >= g_device_count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    
    *device = MOCK_DEVICE_BASE + ordinal;
    STUB_LOG("Mock device handle: %d\n", *device);
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    STUB_LOG("cuCtxCreate(pctx=%p, flags=%u, dev=%d)\n", pctx, flags, dev);
    
    if (!g_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    if (!pctx) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    if (dev < MOCK_DEVICE_BASE || dev >= MOCK_DEVICE_BASE + g_device_count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    
    *pctx = make_mock_handle(MOCK_CONTEXT_BASE, g_context_count++);
    STUB_LOG("Mock context handle: %p\n", *pctx);
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx) {
    STUB_LOG("cuCtxDestroy(ctx=%p)\n", ctx);
    
    if (!g_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    if (!ctx) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    STUB_LOG("Mock context destroyed: %p\n", ctx);
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    STUB_LOG("cuModuleLoadData(module=%p, image=%p)\n", module, image);
    
    if (!g_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    if (!module || !image) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Mock PTX parsing - just check if it looks like PTX
    const char* ptx_data = (const char*)image;
    if (strstr(ptx_data, ".version") == NULL) {
        STUB_LOG("Warning: PTX data doesn't contain .version directive\n");
    }
    
    static int module_counter = 0;
    *module = make_mock_handle(MOCK_MODULE_BASE, module_counter++);
    STUB_LOG("Mock module loaded: %p\n", *module);
    return CUDA_SUCCESS;
}

CUresult cuModuleUnload(CUmodule hmod) {
    STUB_LOG("cuModuleUnload(hmod=%p)\n", hmod);
    
    if (!g_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    if (!hmod) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    STUB_LOG("Mock module unloaded: %p\n", hmod);
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
    STUB_LOG("cuDevicePrimaryCtxReset(dev=%d)\n", dev);
    
    if (!g_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    if (dev < MOCK_DEVICE_BASE || dev >= MOCK_DEVICE_BASE + g_device_count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    
    STUB_LOG("Mock device primary context reset: %d\n", dev);
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    STUB_LOG("cuModuleGetFunction(hfunc=%p, hmod=%p, name=%s)\n", hfunc, hmod, name ? name : "NULL");
    
    if (!g_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    if (!hfunc || !hmod || !name) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Mock function lookup
    static int function_counter = 0;
    *hfunc = make_mock_handle(MOCK_FUNCTION_BASE, function_counter++);
    STUB_LOG("Mock function handle: %p for '%s'\n", *hfunc, name);
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    STUB_LOG("cuMemAlloc(dptr=%p, bytesize=%zu)\n", dptr, bytesize);
    
    if (!g_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    if (!dptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Use regular malloc for mock GPU memory
    void* host_ptr = malloc(bytesize);
    if (!host_ptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    
    *dptr = host_ptr;
    STUB_LOG("Mock GPU memory allocated: %p (%zu bytes)\n", *dptr, bytesize);
    return CUDA_SUCCESS;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    STUB_LOG("cuMemFree(dptr=%p)\n", dptr);
    
    if (!g_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    if (!dptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Free the host memory used for mock GPU memory
    free(dptr);
    STUB_LOG("Mock GPU memory freed: %p\n", dptr);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    STUB_LOG("cuMemcpyHtoD(dstDevice=%p, srcHost=%p, ByteCount=%zu)\n", dstDevice, srcHost, ByteCount);
    
    if (!g_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    if (!dstDevice || !srcHost) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Copy from host to mock GPU memory (which is actually host memory)
    memcpy(dstDevice, srcHost, ByteCount);
    STUB_LOG("Mock H2D copy completed: %zu bytes\n", ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    STUB_LOG("cuMemcpyDtoH(dstHost=%p, srcDevice=%p, ByteCount=%zu)\n", dstHost, srcDevice, ByteCount);
    
    if (!g_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    if (!dstHost || !srcDevice) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Copy from mock GPU memory to host
    memcpy(dstHost, srcDevice, ByteCount);
    STUB_LOG("Mock D2H copy completed: %zu bytes\n", ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra
) {
    STUB_LOG("cuLaunchKernel(f=%p, grid=(%u,%u,%u), block=(%u,%u,%u), shared=%u, stream=%p, params=%p, extra=%p)\n",
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    
    if (!g_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    if (!f) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Mock kernel launch - just log the parameters
    unsigned int total_threads = gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ;
    STUB_LOG("Mock kernel launch: %u total threads\n", total_threads);
    
    // In a real implementation, we could:
    // 1. Parse and simulate the PTX kernel
    // 2. Execute equivalent CPU code
    // 3. Call user-provided simulation callbacks
    // For now, we just return success
    
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
    STUB_LOG("cuCtxSynchronize()\n");
    
    if (!g_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    // Mock synchronization - nothing to do in stub
    STUB_LOG("Mock context synchronized\n");
    return CUDA_SUCCESS;
}

/*
 * Additional utility functions for stub management
 */

void cuda_stub_enable_debug(int enable) {
    // This would be used to enable/disable debug output at runtime
    // For now, debug output is controlled by compile-time flag
    (void)enable;
}

void cuda_stub_reset_state(void) {
    // Reset all global state - useful for testing
    g_initialized = 0;
    g_context_count = 0;
}

int cuda_stub_is_initialized(void) {
    return g_initialized;
} 