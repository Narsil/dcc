const std = @import("std");
const llvm_types = @import("llvm_types.zig");
const LLVM = llvm_types.LLVM;

/// Helper functions for GPU memory operations
pub const GpuMemoryOps = struct {
    context: LLVM.LLVMContextRef,
    module: LLVM.LLVMModuleRef,
    builder: LLVM.LLVMBuilderRef,
    verbose: bool,

    // CUDA function types
    cuMemAlloc_func: LLVM.LLVMValueRef,
    cuMemcpyHtoD_func: LLVM.LLVMValueRef,
    cuMemcpyDtoH_func: LLVM.LLVMValueRef,
    cuMemFree_func: LLVM.LLVMValueRef,
    cuCtxSynchronize_func: LLVM.LLVMValueRef,

    pub fn init(context: LLVM.LLVMContextRef, module: LLVM.LLVMModuleRef, builder: LLVM.LLVMBuilderRef, verbose: bool) GpuMemoryOps {
        return .{
            .context = context,
            .module = module,
            .builder = builder,
            .verbose = verbose,
            .cuMemAlloc_func = LLVM.LLVMGetNamedFunction(module, "cuMemAlloc_v2"),
            .cuMemcpyHtoD_func = LLVM.LLVMGetNamedFunction(module, "cuMemcpyHtoD_v2"),
            .cuMemcpyDtoH_func = LLVM.LLVMGetNamedFunction(module, "cuMemcpyDtoH_v2"),
            .cuMemFree_func = LLVM.LLVMGetNamedFunction(module, "cuMemFree_v2"),
            .cuCtxSynchronize_func = LLVM.LLVMGetNamedFunction(module, "cuCtxSynchronize"),
        };
    }

    /// Allocate GPU memory for a tensor
    pub fn allocateGpuMemory(self: *GpuMemoryOps, size_bytes: LLVM.LLVMValueRef, name: []const u8) LLVM.LLVMValueRef {
        const ptr_type = LLVM.LLVMPointerType(LLVM.LLVMInt8TypeInContext(self.context), 0);
        
        // Allocate space for GPU pointer
        const alloc_name = std.fmt.allocPrintZ(std.heap.page_allocator, "{s}_gpu_ptr", .{name}) catch "gpu_ptr";
        defer std.heap.page_allocator.free(alloc_name);
        const gpu_ptr = LLVM.LLVMBuildAlloca(self.builder, ptr_type, alloc_name.ptr);
        
        // Call cuMemAlloc_v2
        const cuMemAlloc_type = LLVM.LLVMGlobalGetValueType(self.cuMemAlloc_func);
        var alloc_args = [_]LLVM.LLVMValueRef{ gpu_ptr, size_bytes };
        _ = LLVM.LLVMBuildCall2(self.builder, cuMemAlloc_type, self.cuMemAlloc_func, &alloc_args, 2, "");
        
        // Load and return the allocated pointer
        const load_name = std.fmt.allocPrintZ(std.heap.page_allocator, "{s}_gpu_ptr_val", .{name}) catch "gpu_ptr_val";
        defer std.heap.page_allocator.free(load_name);
        return LLVM.LLVMBuildLoad2(self.builder, ptr_type, gpu_ptr, load_name.ptr);
    }

    /// Copy data from host to device
    pub fn copyHostToDevice(self: *GpuMemoryOps, gpu_ptr: LLVM.LLVMValueRef, host_ptr: LLVM.LLVMValueRef, size_bytes: LLVM.LLVMValueRef) void {
        const cuMemcpyHtoD_type = LLVM.LLVMGlobalGetValueType(self.cuMemcpyHtoD_func);
        var copy_args = [_]LLVM.LLVMValueRef{ gpu_ptr, host_ptr, size_bytes };
        _ = LLVM.LLVMBuildCall2(self.builder, cuMemcpyHtoD_type, self.cuMemcpyHtoD_func, &copy_args, 3, "");
    }

    /// Copy data from device to host
    pub fn copyDeviceToHost(self: *GpuMemoryOps, host_ptr: LLVM.LLVMValueRef, gpu_ptr: LLVM.LLVMValueRef, size_bytes: LLVM.LLVMValueRef) void {
        const cuMemcpyDtoH_type = LLVM.LLVMGlobalGetValueType(self.cuMemcpyDtoH_func);
        var copy_args = [_]LLVM.LLVMValueRef{ host_ptr, gpu_ptr, size_bytes };
        _ = LLVM.LLVMBuildCall2(self.builder, cuMemcpyDtoH_type, self.cuMemcpyDtoH_func, &copy_args, 3, "");
    }

    /// Free GPU memory
    pub fn freeGpuMemory(self: *GpuMemoryOps, gpu_ptr: LLVM.LLVMValueRef) void {
        const cuMemFree_type = LLVM.LLVMGlobalGetValueType(self.cuMemFree_func);
        var free_args = [_]LLVM.LLVMValueRef{gpu_ptr};
        _ = LLVM.LLVMBuildCall2(self.builder, cuMemFree_type, self.cuMemFree_func, &free_args, 1, "");
    }

    /// Synchronize GPU operations
    pub fn synchronize(self: *GpuMemoryOps) void {
        const cuCtxSynchronize_type = LLVM.LLVMGlobalGetValueType(self.cuCtxSynchronize_func);
        _ = LLVM.LLVMBuildCall2(self.builder, cuCtxSynchronize_type, self.cuCtxSynchronize_func, null, 0, "sync_result");
    }

    /// Generate optimized memory transfers for a GPU function call
    /// Returns GPU pointers for each tensor parameter
    pub fn prepareGpuCall(
        self: *GpuMemoryOps,
        allocator: std.mem.Allocator,
        func_name: []const u8,
        parameters: []const @import("parser.zig").FunctionParameter,
        arguments: []LLVM.LLVMValueRef,
        tracker: *@import("gpu_memory_tracker.zig").GpuMemoryTracker,
    ) ![]LLVM.LLVMValueRef {
        const gpu_ptrs = try allocator.alloc(LLVM.LLVMValueRef, arguments.len);
        const ptr_type = LLVM.LLVMPointerType(LLVM.LLVMInt8TypeInContext(self.context), 0);

        for (parameters, arguments, 0..) |param, arg, i| {
            if (param.type == .tensor) {
                // Get variable name (simplified - in real code would track properly)
                const var_name = std.fmt.allocPrint(allocator, "{s}_arg{}", .{ func_name, i }) catch "unknown";
                defer allocator.free(var_name);

                // Check if already on GPU
                if (tracker.getGpuPtr(var_name)) |existing_gpu_ptr| {
                    gpu_ptrs[i] = existing_gpu_ptr;
                    if (self.verbose) {
                        std.debug.print("♻️  Reusing GPU memory for '{s}'\n", .{var_name});
                    }
                } else {
                    // Need to allocate and transfer
                    const elem_size: u32 = switch (param.type.tensor.element_type.*) {
                        .f32, .i32 => 4,
                        .f64, .i64 => 8,
                        else => 4,
                    };
                    const size_t_type = LLVM.LLVMInt64TypeInContext(self.context);
                    const tensor_size = LLVM.LLVMConstInt(size_t_type, param.type.tensor.shape[0] * elem_size, 0);

                    // Allocate GPU memory
                    const gpu_ptr = self.allocateGpuMemory(tensor_size, var_name);
                    gpu_ptrs[i] = gpu_ptr;

                    // Copy to GPU
                    const host_ptr = LLVM.LLVMBuildBitCast(self.builder, arg, ptr_type, "host_ptr");
                    self.copyHostToDevice(gpu_ptr, host_ptr, tensor_size);

                    // Update tracker
                    try tracker.markGpuAllocated(var_name, gpu_ptr);
                    try tracker.markCopiedToGpu(var_name);
                }
            } else {
                gpu_ptrs[i] = arg; // Non-tensor arguments passed as-is
            }
        }

        return gpu_ptrs;
    }

    /// Handle post-call synchronization and memory transfers
    pub fn finishGpuCall(
        self: *GpuMemoryOps,
        allocator: std.mem.Allocator,
        func_name: []const u8,
        parameters: []const @import("parser.zig").FunctionParameter,
        arguments: []LLVM.LLVMValueRef,
        gpu_ptrs: []LLVM.LLVMValueRef,
        tracker: *@import("gpu_memory_tracker.zig").GpuMemoryTracker,
        needs_sync: bool,
    ) !void {
        
        // Mark all tensor parameters as modified on GPU
        for (parameters, 0..) |param, i| {
            if (param.type == .tensor) {
                const var_name = std.fmt.allocPrint(allocator, "{s}_arg{}", .{ func_name, i }) catch "unknown";
                defer allocator.free(var_name);
                try tracker.markModifiedOnGpu(var_name);
            }
        }

        // Only sync if needed (e.g., before CPU access)
        if (needs_sync) {
            self.synchronize();
            
            // Copy back data that needs to be on CPU
            const ptr_type = LLVM.LLVMPointerType(LLVM.LLVMInt8TypeInContext(self.context), 0);
            
            for (parameters, arguments, gpu_ptrs, 0..) |param, arg, gpu_ptr, i| {
                if (param.type == .tensor) {
                    const var_name = std.fmt.allocPrint(allocator, "{s}_arg{}", .{ func_name, i }) catch "unknown";
                    defer allocator.free(var_name);
                    
                    if (tracker.needsTransferToCpu(var_name)) {
                        const elem_size: u32 = switch (param.type.tensor.element_type.*) {
                            .f32, .i32 => 4,
                            .f64, .i64 => 8,
                            else => 4,
                        };
                        const size_t_type = LLVM.LLVMInt64TypeInContext(self.context);
                        const tensor_size = LLVM.LLVMConstInt(size_t_type, param.type.tensor.shape[0] * elem_size, 0);
                        
                        const host_ptr = LLVM.LLVMBuildBitCast(self.builder, arg, ptr_type, "host_ptr");
                        self.copyDeviceToHost(host_ptr, gpu_ptr, tensor_size);
                        
                        try tracker.markCopiedToCpu(var_name);
                    }
                }
            }
        }
    }
};