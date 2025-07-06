const std = @import("std");

// CUDA Driver API via official headers
const cuda = @cImport({
    @cInclude("cuda.h");
});

// Embedded PTX code loaded from test.ptx at compile time
const PTX_CODE = @embedFile("./test.ptx");

fn cudaErrorString(error_code: cuda.CUresult) []const u8 {
    return switch (error_code) {
        cuda.CUDA_SUCCESS => "CUDA_SUCCESS",
        cuda.CUDA_ERROR_INVALID_VALUE => "CUDA_ERROR_INVALID_VALUE (invalid parameter passed to API call)",
        cuda.CUDA_ERROR_OUT_OF_MEMORY => "CUDA_ERROR_OUT_OF_MEMORY (insufficient GPU memory)",
        cuda.CUDA_ERROR_NOT_INITIALIZED => "CUDA_ERROR_NOT_INITIALIZED (CUDA driver not initialized)",
        cuda.CUDA_ERROR_DEINITIALIZED => "CUDA_ERROR_DEINITIALIZED (CUDA driver is shutting down)",
        cuda.CUDA_ERROR_NO_DEVICE => "CUDA_ERROR_NO_DEVICE (no CUDA-capable device found)",
        cuda.CUDA_ERROR_INVALID_DEVICE => "CUDA_ERROR_INVALID_DEVICE (invalid device ordinal)",
        cuda.CUDA_ERROR_INVALID_IMAGE => "CUDA_ERROR_INVALID_IMAGE (device kernel image is invalid)",
        cuda.CUDA_ERROR_INVALID_CONTEXT => "CUDA_ERROR_INVALID_CONTEXT (invalid device context)",
        cuda.CUDA_ERROR_CONTEXT_ALREADY_CURRENT => "CUDA_ERROR_CONTEXT_ALREADY_CURRENT (context already current)",
        cuda.CUDA_ERROR_MAP_FAILED => "CUDA_ERROR_MAP_FAILED (memory mapping failed)",
        cuda.CUDA_ERROR_UNMAP_FAILED => "CUDA_ERROR_UNMAP_FAILED (memory unmapping failed)",
        cuda.CUDA_ERROR_ARRAY_IS_MAPPED => "CUDA_ERROR_ARRAY_IS_MAPPED (array is mapped)",
        cuda.CUDA_ERROR_ALREADY_MAPPED => "CUDA_ERROR_ALREADY_MAPPED (resource already mapped)",
        cuda.CUDA_ERROR_NO_BINARY_FOR_GPU => "CUDA_ERROR_NO_BINARY_FOR_GPU (no kernel image for device)",
        cuda.CUDA_ERROR_ALREADY_ACQUIRED => "CUDA_ERROR_ALREADY_ACQUIRED (resource already acquired)",
        cuda.CUDA_ERROR_NOT_MAPPED => "CUDA_ERROR_NOT_MAPPED (resource not mapped)",
        cuda.CUDA_ERROR_NOT_MAPPED_AS_ARRAY => "CUDA_ERROR_NOT_MAPPED_AS_ARRAY (resource not mapped as array)",
        cuda.CUDA_ERROR_NOT_MAPPED_AS_POINTER => "CUDA_ERROR_NOT_MAPPED_AS_POINTER (resource not mapped as pointer)",
        cuda.CUDA_ERROR_ECC_UNCORRECTABLE => "CUDA_ERROR_ECC_UNCORRECTABLE (uncorrectable ECC error)",
        cuda.CUDA_ERROR_UNSUPPORTED_LIMIT => "CUDA_ERROR_UNSUPPORTED_LIMIT (unsupported limit)",
        cuda.CUDA_ERROR_CONTEXT_ALREADY_IN_USE => "CUDA_ERROR_CONTEXT_ALREADY_IN_USE (context already in use)",
        cuda.CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED (peer access unsupported)",
        cuda.CUDA_ERROR_INVALID_PTX => "CUDA_ERROR_INVALID_PTX (PTX JIT compilation failed - check PTX syntax)",
        cuda.CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT (invalid OpenGL/DirectX context)",
        cuda.CUDA_ERROR_NVLINK_UNCORRECTABLE => "CUDA_ERROR_NVLINK_UNCORRECTABLE (uncorrectable NVLink error)",
        cuda.CUDA_ERROR_JIT_COMPILER_NOT_FOUND => "CUDA_ERROR_JIT_COMPILER_NOT_FOUND (PTX JIT compiler not available)",
        cuda.CUDA_ERROR_INVALID_SOURCE => "CUDA_ERROR_INVALID_SOURCE (device kernel source is invalid)",
        cuda.CUDA_ERROR_FILE_NOT_FOUND => "CUDA_ERROR_FILE_NOT_FOUND (file not found)",
        cuda.CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND (symbol not found)",
        cuda.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED (shared object initialization failed)",
        cuda.CUDA_ERROR_OPERATING_SYSTEM => "CUDA_ERROR_OPERATING_SYSTEM (OS call failed)",
        cuda.CUDA_ERROR_INVALID_HANDLE => "CUDA_ERROR_INVALID_HANDLE (invalid resource handle)",
        cuda.CUDA_ERROR_NOT_FOUND => "CUDA_ERROR_NOT_FOUND (named symbol not found)",
        cuda.CUDA_ERROR_NOT_READY => "CUDA_ERROR_NOT_READY (asynchronous operations not complete)",
        cuda.CUDA_ERROR_ILLEGAL_ADDRESS => "CUDA_ERROR_ILLEGAL_ADDRESS (illegal memory access)",
        cuda.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES (insufficient resources for launch)",
        cuda.CUDA_ERROR_LAUNCH_TIMEOUT => "CUDA_ERROR_LAUNCH_TIMEOUT (kernel launch timeout)",
        cuda.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING (incompatible texturing mode)",
        cuda.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED (peer access already enabled)",
        cuda.CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED (peer access not enabled)",
        cuda.CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE (primary context already active)",
        cuda.CUDA_ERROR_CONTEXT_IS_DESTROYED => "CUDA_ERROR_CONTEXT_IS_DESTROYED (context is destroyed)",
        cuda.CUDA_ERROR_ASSERT => "CUDA_ERROR_ASSERT (device assert triggered)",
        cuda.CUDA_ERROR_TOO_MANY_PEERS => "CUDA_ERROR_TOO_MANY_PEERS (too many peer devices)",
        cuda.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED (host memory already registered)",
        cuda.CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED (host memory not registered)",
        cuda.CUDA_ERROR_HARDWARE_STACK_ERROR => "CUDA_ERROR_HARDWARE_STACK_ERROR (hardware stack error)",
        cuda.CUDA_ERROR_ILLEGAL_INSTRUCTION => "CUDA_ERROR_ILLEGAL_INSTRUCTION (illegal instruction)",
        cuda.CUDA_ERROR_MISALIGNED_ADDRESS => "CUDA_ERROR_MISALIGNED_ADDRESS (misaligned address)",
        cuda.CUDA_ERROR_INVALID_ADDRESS_SPACE => "CUDA_ERROR_INVALID_ADDRESS_SPACE (invalid address space)",
        cuda.CUDA_ERROR_INVALID_PC => "CUDA_ERROR_INVALID_PC (invalid program counter)",
        cuda.CUDA_ERROR_LAUNCH_FAILED => "CUDA_ERROR_LAUNCH_FAILED (kernel launch failed)",
        cuda.CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE => "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE (cooperative launch exceeds resources)",
        cuda.CUDA_ERROR_NOT_PERMITTED => "CUDA_ERROR_NOT_PERMITTED (operation not permitted)",
        cuda.CUDA_ERROR_NOT_SUPPORTED => "CUDA_ERROR_NOT_SUPPORTED (operation not supported)",
        cuda.CUDA_ERROR_SYSTEM_NOT_READY => "CUDA_ERROR_SYSTEM_NOT_READY (system not ready)",
        cuda.CUDA_ERROR_SYSTEM_DRIVER_MISMATCH => "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH (system driver version mismatch)",
        cuda.CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE => "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE (compatibility not supported)",
        cuda.CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED => "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED (stream capture unsupported)",
        cuda.CUDA_ERROR_STREAM_CAPTURE_INVALIDATED => "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED (stream capture invalidated)",
        cuda.CUDA_ERROR_STREAM_CAPTURE_MERGE => "CUDA_ERROR_STREAM_CAPTURE_MERGE (stream capture merge error)",
        cuda.CUDA_ERROR_STREAM_CAPTURE_UNMATCHED => "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED (stream capture unmatched)",
        cuda.CUDA_ERROR_STREAM_CAPTURE_UNJOINED => "CUDA_ERROR_STREAM_CAPTURE_UNJOINED (stream capture unjoined)",
        cuda.CUDA_ERROR_STREAM_CAPTURE_ISOLATION => "CUDA_ERROR_STREAM_CAPTURE_ISOLATION (stream capture isolation violation)",
        cuda.CUDA_ERROR_STREAM_CAPTURE_IMPLICIT => "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT (stream capture implicit dependency)",
        cuda.CUDA_ERROR_CAPTURED_EVENT => "CUDA_ERROR_CAPTURED_EVENT (captured event error)",
        cuda.CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD => "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD (stream capture wrong thread)",
        cuda.CUDA_ERROR_TIMEOUT => "CUDA_ERROR_TIMEOUT (operation timeout)",
        cuda.CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE => "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE (graph execution update failed)",
        cuda.CUDA_ERROR_UNKNOWN => "CUDA_ERROR_UNKNOWN (unknown error)",
        else => |err_code| blk: {
            // For unknown error codes, provide the numeric value
            _ = err_code;
            break :blk "UNKNOWN_CUDA_ERROR (check CUDA documentation for error code)";
        },
    };
}

fn checkCudaError(result: cuda.CUresult, operation: []const u8) !void {
    if (result != cuda.CUDA_SUCCESS) {
        std.debug.print("CUDA Error in {s}: {d} ({s})\n", .{ operation, result, cudaErrorString(result) });
        return error.CudaError;
    }
}

pub fn main() !void {
    std.debug.print("=== DCC CUDA PTX Kernel Execution Test ===\n", .{});
    std.debug.print("Testing DCC-generated PTX kernel execution...\n", .{});

    // Initialize CUDA
    std.debug.print("Initializing CUDA...\n", .{});
    try checkCudaError(cuda.cuInit(0), "cuInit");

    // Get first device
    std.debug.print("Getting first CUDA device...\n", .{});
    var device: cuda.CUdevice = undefined;
    try checkCudaError(cuda.cuDeviceGet(&device, 0), "cuDeviceGet");

    // Get device name
    var device_name: [256]u8 = undefined;
    try checkCudaError(cuda.cuDeviceGetName(@ptrCast(device_name[0..].ptr), 256, device), "cuDeviceGetName");
    const device_name_slice = std.mem.sliceTo(&device_name, 0);
    std.debug.print("Found device: {s}\n", .{device_name_slice});

    // Create context
    std.debug.print("Creating CUDA context...\n", .{});
    var context: cuda.CUcontext = undefined;
    try checkCudaError(cuda.cuCtxCreate(&context, 0, device), "cuCtxCreate");

    // Load PTX module
    std.debug.print("Loading DCC-generated PTX module...\n", .{});
    var module: cuda.CUmodule = undefined;
    try checkCudaError(cuda.cuModuleLoadData(&module, PTX_CODE.ptr), "cuModuleLoadData");

    // Get kernel function
    std.debug.print("Getting gpu_add_kernel kernel function...\n", .{});
    var kernel_func: cuda.CUfunction = undefined;
    try checkCudaError(cuda.cuModuleGetFunction(&kernel_func, module, "gpu_vector_mul"), "cuModuleGetFunction");

    // Set up test data
    const array_size: u32 = 1024;
    const byte_size: usize = array_size * @sizeOf(f32);

    std.debug.print("Setting up test data (arrays of {d} f32 elements)...\n", .{array_size});

    // Allocate host memory
    var allocator = std.heap.page_allocator;
    const host_a = try allocator.alloc(f32, array_size);
    const host_b = try allocator.alloc(f32, array_size);
    const host_a_original = try allocator.alloc(f32, array_size); // Save original for verification
    defer allocator.free(host_a);
    defer allocator.free(host_b);
    defer allocator.free(host_a_original);

    // Initialize test data
    for (0..array_size) |i| {
        host_a[i] = @floatFromInt(i); // a = [0, 1, 2, 3, ...]
        host_b[i] = @floatFromInt(i * 2); // b = [0, 2, 4, 6, ...]
        host_a_original[i] = host_a[i]; // Save original a values
    }

    std.debug.print("Test vectors initialized: a[0:3] = [{d:.1}, {d:.1}, {d:.1}, ...], b[0:3] = [{d:.1}, {d:.1}, {d:.1}, ...]\n", .{ host_a[0], host_a[1], host_a[2], host_b[0], host_b[1], host_b[2] });
    std.debug.print("In-place addition: a[i] will be updated to a[i] + b[i]\n", .{});

    // Allocate GPU memory
    std.debug.print("Allocating GPU memory...\n", .{});
    var gpu_a: cuda.CUdeviceptr = undefined;
    var gpu_b: cuda.CUdeviceptr = undefined;
    try checkCudaError(cuda.cuMemAlloc(&gpu_a, byte_size), "cuMemAlloc for a");
    try checkCudaError(cuda.cuMemAlloc(&gpu_b, byte_size), "cuMemAlloc for b");

    // Copy data to GPU
    std.debug.print("Copying input data to GPU...\n", .{});
    try checkCudaError(cuda.cuMemcpyHtoD(gpu_a, host_a.ptr, byte_size), "cuMemcpyHtoD for a");
    try checkCudaError(cuda.cuMemcpyHtoD(gpu_b, host_b.ptr, byte_size), "cuMemcpyHtoD for b");

    // Set up kernel parameters (matching in-place PTX signature)
    std.debug.print("Setting up kernel parameters...\n", .{});
    var params = [_]?*anyopaque{
        @ptrCast(&gpu_a), // a_ptr (will be modified in-place)
        @ptrCast(&gpu_b), // b_ptr
    };

    // Launch kernel
    const block_size: u32 = 256;
    const grid_size: u32 = (array_size + block_size - 1) / block_size;

    std.debug.print("Launching kernel: grid=({d},1,1), block=({d},1,1)...\n", .{ grid_size, block_size });
    try checkCudaError(cuda.cuLaunchKernel(kernel_func, grid_size, 1, 1, // Grid dimensions
        block_size, 1, 1, // Block dimensions
        0, // Shared memory
        null, // Stream
        @ptrCast(&params), // Parameters
        null // Extra
    ), "cuLaunchKernel");

    // Synchronize
    std.debug.print("Synchronizing GPU execution...\n", .{});
    try checkCudaError(cuda.cuCtxSynchronize(), "cuCtxSynchronize");

    // Copy modified array 'a' back from GPU
    std.debug.print("Copying modified array 'a' back from GPU...\n", .{});
    try checkCudaError(cuda.cuMemcpyDtoH(host_a.ptr, gpu_a, byte_size), "cuMemcpyDtoH for a");

    // Verify results
    std.debug.print("Verifying in-place addition results...\n", .{});
    var errors: u32 = 0;
    for (0..@min(10, array_size)) |i| {
        const expected = host_a_original[i] + host_b[i]; // original_a[i] + b[i]
        const actual = host_a[i]; // modified a[i]
        std.debug.print("  a[{d}] = {d:.1} (expected {d:.1} = {d:.1} + {d:.1}) {s}\n", .{ i, actual, expected, host_a_original[i], host_b[i], if (@abs(actual - expected) < 0.001) "✅" else "❌" });
        if (@abs(actual - expected) >= 0.001) errors += 1;
    }

    if (errors == 0) {
        std.debug.print("🎉 All results correct! DCC PTX in-place addition executed successfully!\n", .{});
        std.debug.print("Array 'a' was successfully modified: a[i] = a[i] + b[i]\n", .{});
    } else {
        std.debug.print("❌ {d} errors found in in-place addition results\n", .{errors});
    }

    // Cleanup
    std.debug.print("Cleaning up GPU memory...\n", .{});
    try checkCudaError(cuda.cuMemFree(gpu_a), "cuMemFree for a");
    try checkCudaError(cuda.cuMemFree(gpu_b), "cuMemFree for b");

    // Destroy context
    std.debug.print("Destroying CUDA context...\n", .{});
    try checkCudaError(cuda.cuCtxDestroy(context), "cuCtxDestroy");

    std.debug.print("✅ DCC PTX kernel execution test completed successfully!\n", .{});
    std.debug.print("Your DCC compiler can now generate working GPU kernels! 🚀\n", .{});
}
