const std = @import("std");
const parser = @import("../parser.zig");
const typechecker = @import("../typechecker.zig");
const mlir_codegen = @import("mlir.zig");
const gpu_memory_tracker = @import("gpu_memory_tracker.zig");
const gpu_memory_ops = @import("gpu_memory_ops.zig");
const core = @import("core.zig");
const LLVM = core.LLVM;

// Import CodeGenError from the main codegen module
const CodeGen = core.CodeGen;
const CodeGenError = core.CodeGenError;

/// Generate all GPU functions together in a single MLIR module
pub fn generateAllGpuFunctions(self: *CodeGen, statements: []parser.ASTNode) CodeGenError!void {
    if (self.accelerator) |*accel| {
        if (self.verbose) {
            std.debug.print("🚀 Compiling all GPU functions together in a single module\n", .{});
        }

        // Collect all GPU function declarations
        var gpu_functions = std.ArrayList(@TypeOf(@as(parser.ASTNode, undefined).function_declaration)).init(self.allocator);
        defer gpu_functions.deinit();

        for (statements) |stmt| {
            if (stmt == .function_declaration) {
                const func = stmt.function_declaration;
                if (std.mem.startsWith(u8, func.name, "gpu_")) {
                    try gpu_functions.append(func);
                    if (self.verbose) {
                        std.debug.print("🔧 Collected GPU function: {s}\n", .{func.name});
                    }
                }
            }
        }

        if (gpu_functions.items.len == 0) {
            if (self.verbose) {
                std.debug.print("❌ No GPU functions found to compile\n", .{});
            }
            return;
        }

        // Step 1: Generate MLIR for all GPU functions in a single module
        accel.codegen.generateGpuModule(gpu_functions.items) catch |err| {
            if (self.verbose) {
                std.debug.print("MLIR GPU module compilation failed: {}\n", .{err});
            }
            // Fall back to generating CPU wrappers for all functions
            for (gpu_functions.items) |func| {
                generateGpuHostWrapper(self, func) catch {};
            }
            return;
        };

        if (self.verbose) {
            accel.codegen.printMLIR();
        }

        // Step 2: Generate PTX from the combined MLIR module
        // Use the first function name as a representative for the module
        const ptx_code = accel.codegen.lowerMLIRToPTX(gpu_functions.items[0].name) catch |err| {
            if (self.verbose) {
                std.debug.print("PTX generation failed: {}\n", .{err});
            }
            // Fall back to generating CPU wrappers for all functions
            for (gpu_functions.items) |func| {
                generateGpuHostWrapper(self, func) catch {};
            }
            return;
        };
        defer self.allocator.free(ptx_code);

        if (self.verbose) {
            std.debug.print("✅ Generated PTX code for all GPU functions ({d} bytes)\n", .{ptx_code.len});
        }

        // Step 3: Generate CUDA LLVM IR wrapper using the combined PTX
        // Use the first function for the wrapper but the PTX contains all functions
        try generateCudaLLVMIRWrapper(self, gpu_functions.items[0], ptx_code);

        // Step 4: Generate host wrapper functions for all GPU functions
        for (gpu_functions.items) |func| {
            // Generate both the old wrapper (for compatibility) and new launcher
            try generateGpuHostWrapper(self, func);
            try generateGpuKernelLauncher(self, func);
            if (self.verbose) {
                std.debug.print("✅ Generated host wrapper and launcher for GPU function: {s}\n", .{func.name});
            }
        }

        if (self.verbose) {
            std.debug.print("✅ Successfully compiled all {} GPU functions together\n", .{gpu_functions.items.len});
        }
    } else {
        std.debug.print("Error: Cannot compile GPU functions without --gpu flag\n", .{});
        std.debug.print("GPU functions require GPU compilation support. Use --gpu flag to enable.\n", .{});
        return error.GpuCompilationNotImplemented;
    }
}

// pub fn generateGpuFunction(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) CodeGenError!void {
//     if (self.accelerator) |*accel| {
//         if (self.verbose) {
//             std.debug.print("🚀 Compiling GPU function: {s}\n", .{func.name});
//         }
//
//         // Step 1: Generate MLIR for the GPU function
//         accel.codegen.generateGpuFunction(func) catch |err| {
//             if (self.verbose) {
//                 std.debug.print("MLIR GPU compilation failed: {}\n", .{err});
//             }
//             // Fall through to generate host wrapper anyway
//         };
//
//         if (self.verbose) {
//             accel.codegen.printMLIR();
//         }
//
//         // Step 2: Generate PTX from MLIR
//         const ptx_code = accel.codegen.lowerMLIRToPTX(func.name) catch |err| {
//             if (self.verbose) {
//                 std.debug.print("PTX generation failed: {}\n", .{err});
//             }
//             // Fall back to generating CPU wrapper
//             return self.generateGpuHostWrapper(func);
//         };
//         defer self.allocator.free(ptx_code);
//
//         if (self.verbose) {
//             std.debug.print("✅ Generated PTX code ({d} bytes)\n", .{ptx_code.len});
//         }
//
//         // Step 3: Generate CUDA LLVM IR wrapper using the new generator
//         try self.generateCudaLLVMIRWrapper(func, ptx_code);
//
//         // Step 4: Generate host wrapper function that can be called from main
//         try self.generateGpuHostWrapper(func);
//     } else {
//         std.debug.print("Error: Cannot compile GPU function '{s}' without --gpu flag\n", .{func.name});
//         std.debug.print("GPU functions require GPU compilation support. Use --gpu flag to enable.\n", .{});
//         return error.GpuCompilationNotImplemented;
//     }
// }

/// Generates a simple GPU kernel launcher without memory transfers
/// Memory transfers will be handled separately at call sites
pub fn generateGpuKernelLauncher(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) CodeGenError!void {
    if (self.verbose) {
        std.debug.print("🚀 Generating GPU kernel launcher (no memory transfers): {s}\n", .{func.name});
    }

    // Create launcher function with GPU pointer parameters
    const param_types = try self.allocator.alloc(LLVM.LLVMTypeRef, func.parameters.len);
    defer self.allocator.free(param_types);

    // All tensor parameters become GPU device pointers (i8*)
    const ptr_type = LLVM.LLVMPointerType(LLVM.LLVMInt8TypeInContext(self.context), 0);
    for (func.parameters, 0..) |param, i| {
        if (param.type == .tensor) {
            param_types[i] = ptr_type; // GPU device pointer
        } else {
            param_types[i] = self.toLLVMType(param.type);
        }
    }

    // Create launcher function with _gpu_launch suffix
    const launcher_name = try std.fmt.allocPrint(self.allocator, "{s}_gpu_launch", .{func.name});
    defer self.allocator.free(launcher_name);

    const return_type = self.toLLVMType(func.return_type);
    const launcher_type = LLVM.LLVMFunctionType(return_type, param_types.ptr, @intCast(param_types.len), 0);
    const launcher_func = LLVM.LLVMAddFunction(self.module, launcher_name.ptr, launcher_type);

    // Store the launcher function
    const launcher_name_dup = try self.allocator.dupe(u8, launcher_name);
    try self.functions.put(launcher_name_dup, launcher_func);

    // Create entry block
    const entry_block = LLVM.LLVMAppendBasicBlockInContext(self.context, launcher_func, "entry");
    LLVM.LLVMPositionBuilderAtEnd(self.builder, entry_block);

    // Get global CUDA context and module
    const cuda_context_global = LLVM.LLVMGetNamedGlobal(self.module, "cuda_context") orelse return error.CudaFunctionNotFound;
    const cuda_module_global = LLVM.LLVMGetNamedGlobal(self.module, "cuda_module") orelse return error.CudaFunctionNotFound;

    _ = LLVM.LLVMBuildLoad2(self.builder, ptr_type, cuda_context_global, "cuda_context");
    _ = LLVM.LLVMBuildLoad2(self.builder, ptr_type, cuda_module_global, "cuda_module");

    // Get pre-obtained kernel function pointer
    const func_global_name = try std.fmt.allocPrint(self.allocator, "cuda_func_{s}", .{func.name});
    defer self.allocator.free(func_global_name);

    const func_global = LLVM.LLVMGetNamedGlobal(self.module, func_global_name.ptr);
    if (func_global == null) {
        return CodeGenError.CudaFunctionNotFound;
    }

    const kernel_func = LLVM.LLVMBuildLoad2(self.builder, ptr_type, func_global.?, "kernel_func_val");

    // Setup kernel launch parameters
    const int32_type = LLVM.LLVMInt32TypeInContext(self.context);
    const grid_x = LLVM.LLVMConstInt(int32_type, 1, 0);
    const grid_y = LLVM.LLVMConstInt(int32_type, 1, 0);
    const grid_z = LLVM.LLVMConstInt(int32_type, 1, 0);
    const block_x = LLVM.LLVMConstInt(int32_type, 1024, 0);
    const block_y = LLVM.LLVMConstInt(int32_type, 1, 0);
    const block_z = LLVM.LLVMConstInt(int32_type, 1, 0);
    const shared_mem = LLVM.LLVMConstInt(int32_type, 0, 0);
    const stream = LLVM.LLVMConstNull(ptr_type);

    // Create kernel arguments array from launcher parameters
    const ptr_ptr_type = LLVM.LLVMPointerType(ptr_type, 0);
    var kernel_param_ptrs = std.ArrayList(LLVM.LLVMValueRef).init(self.allocator);
    defer kernel_param_ptrs.deinit();

    for (func.parameters, 0..) |param, i| {
        if (param.type == .tensor) {
            const gpu_ptr = LLVM.LLVMGetParam(launcher_func, @intCast(i));

            // Allocate space for the device pointer value
            const param_ptr = LLVM.LLVMBuildAlloca(self.builder, ptr_type, "param_ptr");
            _ = LLVM.LLVMBuildStore(self.builder, gpu_ptr, param_ptr);

            try kernel_param_ptrs.append(param_ptr);
        }
    }

    // Create array of parameter pointers
    const param_array_type = LLVM.LLVMArrayType(ptr_type, @intCast(kernel_param_ptrs.items.len));
    const param_array = LLVM.LLVMBuildAlloca(self.builder, param_array_type, "param_array");

    for (kernel_param_ptrs.items, 0..) |param_ptr, i| {
        var indices = [_]LLVM.LLVMValueRef{ LLVM.LLVMConstInt(int32_type, 0, 0), LLVM.LLVMConstInt(int32_type, @intCast(i), 0) };
        const elem_ptr = LLVM.LLVMBuildGEP2(self.builder, param_array_type, param_array, &indices, 2, "param_elem_ptr");
        _ = LLVM.LLVMBuildStore(self.builder, param_ptr, elem_ptr);
    }

    const kernel_args = LLVM.LLVMBuildBitCast(self.builder, param_array, ptr_ptr_type, "kernel_args");

    // Get CUDA function declarations
    const cuLaunchKernel_func = LLVM.LLVMGetNamedFunction(self.module, "cuLaunchKernel");
    const cuLaunchKernel_type = LLVM.LLVMGlobalGetValueType(cuLaunchKernel_func);

    // Launch kernel (no synchronization here!)
    var launch_args = [_]LLVM.LLVMValueRef{ kernel_func, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem, stream, kernel_args, LLVM.LLVMConstNull(ptr_type) };
    _ = LLVM.LLVMBuildCall2(self.builder, cuLaunchKernel_type, cuLaunchKernel_func, &launch_args, 11, "launch_result");

    // Return void (no synchronization, no memory copies)
    if (func.return_type == .void) {
        _ = LLVM.LLVMBuildRetVoid(self.builder);
    } else {
        // For non-void functions, we can't return a value without synchronization
        // Return a dummy value for now
        const zero = LLVM.LLVMConstInt(self.toLLVMType(func.return_type), 0, 0);
        _ = LLVM.LLVMBuildRet(self.builder, zero);
    }

    if (self.verbose) {
        std.debug.print("✅ Generated GPU kernel launcher: {s}\n", .{launcher_name});
    }
}

pub fn generateGpuHostWrapper(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) CodeGenError!void {
    if (self.verbose) {
        std.debug.print("🔧 Generating GPU host wrapper function: {s}\n", .{func.name});
    }

    // Create a host function that launches the CUDA kernel
    const param_types = try self.allocator.alloc(LLVM.LLVMTypeRef, func.parameters.len);
    defer self.allocator.free(param_types);

    for (func.parameters, 0..) |param, i| {
        if (param.type == .tensor) {
            // Use pointer type for tensor parameters to allow in-place modification
            const array_type = self.toLLVMType(param.type);
            param_types[i] = LLVM.LLVMPointerType(array_type, 0);
        } else {
            param_types[i] = self.toLLVMType(param.type);
        }
    }

    // Get the already declared function instead of creating a new one
    const llvm_function = self.functions.get(func.name) orelse return error.UndefinedFunction;

    // Create entry basic block
    const entry_block = LLVM.LLVMAppendBasicBlockInContext(self.context, llvm_function, "entry");
    LLVM.LLVMPositionBuilderAtEnd(self.builder, entry_block);

    // Generate actual CUDA kernel launch sequence
    try generateCudaKernelLaunch(self, llvm_function, func);

    if (self.verbose) {
        std.debug.print("Generated GPU host wrapper with CPU simulation for function: {s}\n", .{func.name});
    }
}

pub fn generateCudaLLVMIRWrapper(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration), ptx_code: []const u8) CodeGenError!void {
    if (self.verbose) {
        std.debug.print("🔧 Generating CUDA LLVM IR wrapper for function: {s}\n", .{func.name});
    }

    // Check if we should use CUDA stubs and extract them if needed
    if (self.accelerator) |*accel| {
        accel.stub.extractAndCompile() catch |err| {
            if (self.verbose) {
                std.debug.print("⚠️  Warning: Failed to extract CUDA stub files: {}\n", .{err});
                std.debug.print("   Proceeding with CUDA LLVM IR generation only\n", .{});
            }
            // Continue without stub files - just generate the IR
        };

        if (self.verbose) {
            std.debug.print("✅ CUDA stub files extracted and ready\n", .{});
            if (accel.stub.getIncludePath()) |include_path| {
                std.debug.print("   Include path: {s}\n", .{include_path});
            }
            if (accel.stub.getLibPath()) |lib_path| {
                std.debug.print("   Library path: {s}\n", .{lib_path});
            }
        }
    }

    // Instead of creating a separate CUDA module, integrate CUDA functions directly into the main module
    try addCudaFunctionsToMainModule(self, ptx_code);

    if (self.verbose) {
        std.debug.print("✅ CUDA functions integrated into main module\n", .{});
    }
}

fn addCudaFunctionsToMainModule(self: *CodeGen, ptx_code: []const u8) CodeGenError!void {
    if (self.verbose) {
        std.debug.print("🔧 Adding CUDA functions to main module\n", .{});
    }

    // Declare CUDA runtime functions in the main module
    const int32_type = LLVM.LLVMInt32TypeInContext(self.context);
    const int8_type = LLVM.LLVMInt8TypeInContext(self.context);
    const ptr_type = LLVM.LLVMPointerType(int8_type, 0);
    const void_type = LLVM.LLVMVoidTypeInContext(self.context);

    // Declare CUDA runtime functions
    var cuInit_params = [_]LLVM.LLVMTypeRef{int32_type};
    const cuInit_type = LLVM.LLVMFunctionType(int32_type, &cuInit_params, 1, 0);
    const cuInit_func = LLVM.LLVMAddFunction(self.module, "cuInit", cuInit_type);

    var cuDeviceGet_params = [_]LLVM.LLVMTypeRef{ ptr_type, int32_type };
    const cuDeviceGet_type = LLVM.LLVMFunctionType(int32_type, &cuDeviceGet_params, 2, 0);
    const cuDeviceGet_func = LLVM.LLVMAddFunction(self.module, "cuDeviceGet", cuDeviceGet_type);

    var cuCtxCreate_params = [_]LLVM.LLVMTypeRef{ ptr_type, int32_type, int32_type };
    const cuCtxCreate_type = LLVM.LLVMFunctionType(int32_type, &cuCtxCreate_params, 3, 0);
    const cuCtxCreate_func = LLVM.LLVMAddFunction(self.module, "cuCtxCreate_v2", cuCtxCreate_type);

    // Add cuCtxDestroy for proper cleanup
    var cuCtxDestroy_params = [_]LLVM.LLVMTypeRef{ptr_type};
    const cuCtxDestroy_type = LLVM.LLVMFunctionType(int32_type, &cuCtxDestroy_params, 1, 0);
    const cuCtxDestroy_func = LLVM.LLVMAddFunction(self.module, "cuCtxDestroy_v2", cuCtxDestroy_type);

    // Add cuDevicePrimaryCtxReset for thorough cleanup
    var cuDevicePrimaryCtxReset_params = [_]LLVM.LLVMTypeRef{int32_type};
    const cuDevicePrimaryCtxReset_type = LLVM.LLVMFunctionType(int32_type, &cuDevicePrimaryCtxReset_params, 1, 0);
    const cuDevicePrimaryCtxReset_func = LLVM.LLVMAddFunction(self.module, "cuDevicePrimaryCtxReset_v2", cuDevicePrimaryCtxReset_type);

    // Add pthread functions for thread cleanup
    var pthread_kill_params = [_]LLVM.LLVMTypeRef{ LLVM.LLVMInt64TypeInContext(self.context), int32_type };
    const pthread_kill_type = LLVM.LLVMFunctionType(int32_type, &pthread_kill_params, 2, 0);
    _ = LLVM.LLVMAddFunction(self.module, "pthread_kill", pthread_kill_type);

    // No exit function needed - using Linux syscall for cross-compilation

    var cuModuleLoadData_params = [_]LLVM.LLVMTypeRef{ ptr_type, ptr_type };
    const cuModuleLoadData_type = LLVM.LLVMFunctionType(int32_type, &cuModuleLoadData_params, 2, 0);
    const cuModuleLoadData_func = LLVM.LLVMAddFunction(self.module, "cuModuleLoadData", cuModuleLoadData_type);

    var cuModuleUnload_params = [_]LLVM.LLVMTypeRef{ptr_type};
    const cuModuleUnload_type = LLVM.LLVMFunctionType(int32_type, &cuModuleUnload_params, 1, 0);
    const cuModuleUnload_func = LLVM.LLVMAddFunction(self.module, "cuModuleUnload", cuModuleUnload_type);

    // Additional CUDA functions needed for initialization
    var cuDeviceGetCount_params = [_]LLVM.LLVMTypeRef{ptr_type};
    const cuDeviceGetCount_type = LLVM.LLVMFunctionType(int32_type, &cuDeviceGetCount_params, 1, 0);
    _ = LLVM.LLVMAddFunction(self.module, "cuDeviceGetCount", cuDeviceGetCount_type);

    var cuDeviceGetName_params = [_]LLVM.LLVMTypeRef{ ptr_type, int32_type, int32_type };
    const cuDeviceGetName_type = LLVM.LLVMFunctionType(int32_type, &cuDeviceGetName_params, 3, 0);
    _ = LLVM.LLVMAddFunction(self.module, "cuDeviceGetName", cuDeviceGetName_type);

    // Additional CUDA functions needed for kernel launch
    const size_t_type = LLVM.LLVMInt64TypeInContext(self.context);

    var cuModuleGetFunction_params = [_]LLVM.LLVMTypeRef{ ptr_type, ptr_type, ptr_type };
    const cuModuleGetFunction_type = LLVM.LLVMFunctionType(int32_type, &cuModuleGetFunction_params, 3, 0);
    _ = LLVM.LLVMAddFunction(self.module, "cuModuleGetFunction", cuModuleGetFunction_type);

    var cuMemAlloc_params = [_]LLVM.LLVMTypeRef{ ptr_type, size_t_type };
    const cuMemAlloc_type = LLVM.LLVMFunctionType(int32_type, &cuMemAlloc_params, 2, 0);
    _ = LLVM.LLVMAddFunction(self.module, "cuMemAlloc_v2", cuMemAlloc_type);

    var cuMemcpyHtoD_params = [_]LLVM.LLVMTypeRef{ ptr_type, ptr_type, size_t_type };
    const cuMemcpyHtoD_type = LLVM.LLVMFunctionType(int32_type, &cuMemcpyHtoD_params, 3, 0);
    _ = LLVM.LLVMAddFunction(self.module, "cuMemcpyHtoD_v2", cuMemcpyHtoD_type);

    var cuLaunchKernel_params = [_]LLVM.LLVMTypeRef{ ptr_type, int32_type, int32_type, int32_type, int32_type, int32_type, int32_type, int32_type, ptr_type, ptr_type, ptr_type };
    const cuLaunchKernel_type = LLVM.LLVMFunctionType(int32_type, &cuLaunchKernel_params, 11, 0);
    _ = LLVM.LLVMAddFunction(self.module, "cuLaunchKernel", cuLaunchKernel_type);

    const cuCtxSynchronize_type = LLVM.LLVMFunctionType(int32_type, null, 0, 0);
    _ = LLVM.LLVMAddFunction(self.module, "cuCtxSynchronize", cuCtxSynchronize_type);

    var cuMemcpyDtoH_params = [_]LLVM.LLVMTypeRef{ ptr_type, ptr_type, size_t_type };
    const cuMemcpyDtoH_type = LLVM.LLVMFunctionType(int32_type, &cuMemcpyDtoH_params, 3, 0);
    _ = LLVM.LLVMAddFunction(self.module, "cuMemcpyDtoH_v2", cuMemcpyDtoH_type);

    var cuMemFree_params = [_]LLVM.LLVMTypeRef{ptr_type};
    const cuMemFree_type = LLVM.LLVMFunctionType(int32_type, &cuMemFree_params, 1, 0);
    _ = LLVM.LLVMAddFunction(self.module, "cuMemFree_v2", cuMemFree_type);

    // Embed PTX data as global constant
    try embedPTXDataInMainModule(self, ptx_code);

    // Create global variables to store CUDA context and module
    const cuda_context_global = LLVM.LLVMAddGlobal(self.module, ptr_type, "cuda_context");
    LLVM.LLVMSetInitializer(cuda_context_global, LLVM.LLVMConstNull(ptr_type));
    LLVM.LLVMSetLinkage(cuda_context_global, LLVM.LLVMInternalLinkage);

    const cuda_module_global = LLVM.LLVMAddGlobal(self.module, ptr_type, "cuda_module");
    LLVM.LLVMSetInitializer(cuda_module_global, LLVM.LLVMConstNull(ptr_type));
    LLVM.LLVMSetLinkage(cuda_module_global, LLVM.LLVMInternalLinkage);

    // Create global variables to store GPU function pointers
    if (self.gpu_function_names) |gpu_names| {
        for (gpu_names.items) |func_name| {
            const global_name = try std.fmt.allocPrintZ(self.allocator, "cuda_func_{s}", .{func_name});
            defer self.allocator.free(global_name);

            const func_global = LLVM.LLVMAddGlobal(self.module, ptr_type, global_name.ptr);
            LLVM.LLVMSetInitializer(func_global, LLVM.LLVMConstNull(ptr_type));
            LLVM.LLVMSetLinkage(func_global, LLVM.LLVMInternalLinkage);

            if (self.verbose) {
                std.debug.print("🔧 Created global variable for GPU function: {s}\n", .{func_name});
            }
        }
    }

    // Generate CUDA wrapper functions in the main module
    try generateCudaInitInMainModule(self, cuInit_func, cuInit_type);
    try generateCudaCreateContextInMainModule(self, cuDeviceGet_func, cuDeviceGet_type, cuCtxCreate_func, cuCtxCreate_type);
    try generateCudaLoadModuleInMainModule(self, cuModuleLoadData_func, cuModuleLoadData_type);
    try generateCudaCleanupInMainModule(self, cuCtxDestroy_func, cuCtxDestroy_type, cuModuleUnload_func, cuModuleUnload_type, cuDevicePrimaryCtxReset_func, cuDevicePrimaryCtxReset_type, void_type);

    // Initialize GPU memory operations now that CUDA functions are declared
    if (self.gpu_memory_tracker != null) {
        self.gpu_memory_ops = gpu_memory_ops.GpuMemoryOps.init(self.context, self.module, self.builder, self.verbose);
    }

    if (self.verbose) {
        std.debug.print("✅ All CUDA functions added to main module\n", .{});
    }
}

fn embedPTXDataInMainModule(self: *CodeGen, ptx_code: []const u8) CodeGenError!void {
    const int8_type = LLVM.LLVMInt8TypeInContext(self.context);

    // Create array type for PTX data + null terminator
    // PTX data MUST be null-terminated for cuModuleLoadData
    const array_type = LLVM.LLVMArrayType(int8_type, @intCast(ptx_code.len + 1));

    // Create global variable for PTX data
    const ptx_global = LLVM.LLVMAddGlobal(self.module, array_type, "embedded_ptx_data");
    LLVM.LLVMSetLinkage(ptx_global, LLVM.LLVMPrivateLinkage);
    LLVM.LLVMSetGlobalConstant(ptx_global, 1);

    // Create initializer from PTX code + null terminator
    const ptx_values = try self.allocator.alloc(LLVM.LLVMValueRef, ptx_code.len + 1);
    defer self.allocator.free(ptx_values);

    for (ptx_code, 0..) |byte, i| {
        ptx_values[i] = LLVM.LLVMConstInt(int8_type, byte, 0);
    }
    // Add null terminator - this is CRITICAL for cuModuleLoadData
    ptx_values[ptx_code.len] = LLVM.LLVMConstInt(int8_type, 0, 0);

    const ptx_array = LLVM.LLVMConstArray(int8_type, ptx_values.ptr, @intCast(ptx_code.len + 1));
    LLVM.LLVMSetInitializer(ptx_global, ptx_array);

    if (self.verbose) {
        std.debug.print("✅ PTX data embedded in main module ({d} bytes + null terminator)\n", .{ptx_code.len});
    }
}

pub fn generateCudaInitInMainModule(self: *CodeGen, cuInit_func: LLVM.LLVMValueRef, cuInit_type: LLVM.LLVMTypeRef) CodeGenError!void {
    const int32_type = LLVM.LLVMInt32TypeInContext(self.context);

    // Create cuda_init function
    const function_type = LLVM.LLVMFunctionType(int32_type, null, 0, 0);
    const function = LLVM.LLVMAddFunction(self.module, "cuda_init", function_type);

    // Create entry block
    const entry_block = LLVM.LLVMAppendBasicBlockInContext(self.context, function, "entry");
    const old_position = LLVM.LLVMGetInsertBlock(self.builder);
    LLVM.LLVMPositionBuilderAtEnd(self.builder, entry_block);

    // Call cuInit(0)
    const zero = LLVM.LLVMConstInt(int32_type, 0, 0);
    var args = [_]LLVM.LLVMValueRef{zero};
    const cuInit_result = LLVM.LLVMBuildCall2(self.builder, cuInit_type, cuInit_func, &args, 1, "cuInit_result");

    // Return the result
    _ = LLVM.LLVMBuildRet(self.builder, cuInit_result);

    // Restore builder position
    if (old_position != null) {
        LLVM.LLVMPositionBuilderAtEnd(self.builder, old_position);
    }
}

pub fn generateCudaCreateContextInMainModule(self: *CodeGen, cuDeviceGet_func: LLVM.LLVMValueRef, cuDeviceGet_type: LLVM.LLVMTypeRef, cuCtxCreate_func: LLVM.LLVMValueRef, cuCtxCreate_type: LLVM.LLVMTypeRef) CodeGenError!void {
    const int32_type = LLVM.LLVMInt32TypeInContext(self.context);
    const int8_type = LLVM.LLVMInt8TypeInContext(self.context);
    const ptr_type = LLVM.LLVMPointerType(int8_type, 0);

    // Create cuda_create_context function
    var param_types = [_]LLVM.LLVMTypeRef{ ptr_type, int32_type };
    const function_type = LLVM.LLVMFunctionType(int32_type, &param_types, 2, 0);
    const function = LLVM.LLVMAddFunction(self.module, "cuda_create_context", function_type);

    // Get parameters
    const context_param = LLVM.LLVMGetParam(function, 0);
    const device_id_param = LLVM.LLVMGetParam(function, 1);

    // Create blocks
    const entry_block = LLVM.LLVMAppendBasicBlockInContext(self.context, function, "entry");
    const success_block = LLVM.LLVMAppendBasicBlockInContext(self.context, function, "success");
    const error_block = LLVM.LLVMAppendBasicBlockInContext(self.context, function, "error");

    const old_position = LLVM.LLVMGetInsertBlock(self.builder);
    LLVM.LLVMPositionBuilderAtEnd(self.builder, entry_block);

    // Get device
    const device_var = LLVM.LLVMBuildAlloca(self.builder, int32_type, "device");
    var cuDeviceGet_args = [_]LLVM.LLVMValueRef{ device_var, device_id_param };
    const cuDeviceGet_result = LLVM.LLVMBuildCall2(self.builder, cuDeviceGet_type, cuDeviceGet_func, &cuDeviceGet_args, 2, "cuDeviceGet_result");

    // Check if device get was successful
    const zero = LLVM.LLVMConstInt(int32_type, 0, 0);
    const device_get_success = LLVM.LLVMBuildICmp(self.builder, LLVM.LLVMIntEQ, cuDeviceGet_result, zero, "device_get_success");
    _ = LLVM.LLVMBuildCondBr(self.builder, device_get_success, success_block, error_block);

    // Success block: create context
    LLVM.LLVMPositionBuilderAtEnd(self.builder, success_block);
    const device_value = LLVM.LLVMBuildLoad2(self.builder, int32_type, device_var, "device_value");
    var cuCtxCreate_args = [_]LLVM.LLVMValueRef{ context_param, zero, device_value };
    const cuCtxCreate_result = LLVM.LLVMBuildCall2(self.builder, cuCtxCreate_type, cuCtxCreate_func, &cuCtxCreate_args, 3, "cuCtxCreate_result");
    _ = LLVM.LLVMBuildRet(self.builder, cuCtxCreate_result);

    // Error block: return error
    LLVM.LLVMPositionBuilderAtEnd(self.builder, error_block);
    _ = LLVM.LLVMBuildRet(self.builder, cuDeviceGet_result);

    // Restore builder position
    if (old_position != null) {
        LLVM.LLVMPositionBuilderAtEnd(self.builder, old_position);
    }
}

pub fn generateCudaLoadModuleInMainModule(self: *CodeGen, cuModuleLoadData_func: LLVM.LLVMValueRef, cuModuleLoadData_type: LLVM.LLVMTypeRef) CodeGenError!void {
    const int32_type = LLVM.LLVMInt32TypeInContext(self.context);
    const int8_type = LLVM.LLVMInt8TypeInContext(self.context);
    const ptr_type = LLVM.LLVMPointerType(int8_type, 0);

    // Create cuda_load_module function
    var param_types = [_]LLVM.LLVMTypeRef{ ptr_type, ptr_type };
    const function_type = LLVM.LLVMFunctionType(int32_type, &param_types, 2, 0);
    const function = LLVM.LLVMAddFunction(self.module, "cuda_load_module", function_type);

    // Get parameters
    const module_param = LLVM.LLVMGetParam(function, 0);
    const ptx_data_param = LLVM.LLVMGetParam(function, 1);

    // Create entry block
    const entry_block = LLVM.LLVMAppendBasicBlockInContext(self.context, function, "entry");
    const old_position = LLVM.LLVMGetInsertBlock(self.builder);
    LLVM.LLVMPositionBuilderAtEnd(self.builder, entry_block);

    // Call cuModuleLoadData(module, ptx_data)
    var cuModuleLoadData_args = [_]LLVM.LLVMValueRef{ module_param, ptx_data_param };
    const cuModuleLoadData_result = LLVM.LLVMBuildCall2(self.builder, cuModuleLoadData_type, cuModuleLoadData_func, &cuModuleLoadData_args, 2, "cuModuleLoadData_result");

    // Return the result
    _ = LLVM.LLVMBuildRet(self.builder, cuModuleLoadData_result);

    // Restore builder position
    if (old_position != null) {
        LLVM.LLVMPositionBuilderAtEnd(self.builder, old_position);
    }
}

pub fn generateCudaCleanupInMainModule(self: *CodeGen, cuCtxDestroy_func: LLVM.LLVMValueRef, cuCtxDestroy_type: LLVM.LLVMTypeRef, cuModuleUnload_func: LLVM.LLVMValueRef, cuModuleUnload_type: LLVM.LLVMTypeRef, cuDevicePrimaryCtxReset_func: LLVM.LLVMValueRef, cuDevicePrimaryCtxReset_type: LLVM.LLVMTypeRef, void_type: LLVM.LLVMTypeRef) CodeGenError!void {
    const int8_type = LLVM.LLVMInt8TypeInContext(self.context);
    const ptr_type = LLVM.LLVMPointerType(int8_type, 0);

    // Create cuda_cleanup function
    var param_types = [_]LLVM.LLVMTypeRef{ ptr_type, ptr_type };
    const function_type = LLVM.LLVMFunctionType(void_type, &param_types, 2, 0);
    const function = LLVM.LLVMAddFunction(self.module, "cuda_cleanup", function_type);

    // Get parameters
    const context_param = LLVM.LLVMGetParam(function, 0);
    const module_param = LLVM.LLVMGetParam(function, 1);

    // Create entry block
    const entry_block = LLVM.LLVMAppendBasicBlockInContext(self.context, function, "entry");
    const old_position = LLVM.LLVMGetInsertBlock(self.builder);
    LLVM.LLVMPositionBuilderAtEnd(self.builder, entry_block);

    // Call cuModuleUnload(module)
    var module_args = [_]LLVM.LLVMValueRef{module_param};
    _ = LLVM.LLVMBuildCall2(self.builder, cuModuleUnload_type, cuModuleUnload_func, &module_args, 1, "");

    // Call cuCtxDestroy(context)
    var context_args = [_]LLVM.LLVMValueRef{context_param};
    _ = LLVM.LLVMBuildCall2(self.builder, cuCtxDestroy_type, cuCtxDestroy_func, &context_args, 1, "");

    // Call cuDevicePrimaryCtxReset(0) to thoroughly clean up device state
    const int32_type = LLVM.LLVMInt32TypeInContext(self.context);
    const zero = LLVM.LLVMConstInt(int32_type, 0, 0);
    var device_reset_args = [_]LLVM.LLVMValueRef{zero};
    _ = LLVM.LLVMBuildCall2(self.builder, cuDevicePrimaryCtxReset_type, cuDevicePrimaryCtxReset_func, &device_reset_args, 1, "");

    // Return void to let the main function return with its intended exit code
    _ = LLVM.LLVMBuildRetVoid(self.builder);

    // Restore builder position
    if (old_position != null) {
        LLVM.LLVMPositionBuilderAtEnd(self.builder, old_position);
    }
}

pub fn injectCudaInitializationIntoMain(self: *CodeGen) CodeGenError!void {
    if (self.verbose) {
        std.debug.print("🔧 Injecting CUDA one-time initialization into main function\n", .{});
    }

    // Basic types
    const int32_type = LLVM.LLVMInt32TypeInContext(self.context);
    const int8_type = LLVM.LLVMInt8TypeInContext(self.context);
    const ptr_type = LLVM.LLVMPointerType(int8_type, 0);

    // Get CUDA functions that should already be declared
    const cuInit_func = LLVM.LLVMGetNamedFunction(self.module, "cuInit");
    const cuDeviceGet_func = LLVM.LLVMGetNamedFunction(self.module, "cuDeviceGet");
    const cuCtxCreate_func = LLVM.LLVMGetNamedFunction(self.module, "cuCtxCreate_v2");
    const cuModuleLoadData_func = LLVM.LLVMGetNamedFunction(self.module, "cuModuleLoadData");

    if (cuInit_func == null or cuDeviceGet_func == null or cuCtxCreate_func == null or cuModuleLoadData_func == null) {
        if (self.verbose) {
            std.debug.print("❌ CUDA functions not found - they should be declared first\n", .{});
        }
        return CodeGenError.CudaFunctionNotFound;
    }

    const zero = LLVM.LLVMConstInt(int32_type, 0, 0);

    // Get function types from existing functions
    const cuInit_type = LLVM.LLVMGlobalGetValueType(cuInit_func.?);
    const cuDeviceGet_type = LLVM.LLVMGlobalGetValueType(cuDeviceGet_func.?);
    const cuCtxCreate_type = LLVM.LLVMGlobalGetValueType(cuCtxCreate_func.?);
    const cuModuleLoadData_type = LLVM.LLVMGlobalGetValueType(cuModuleLoadData_func.?);

    // Get global variables that should already exist
    const cuda_context_global = LLVM.LLVMGetNamedGlobal(self.module, "cuda_context");
    const cuda_module_global = LLVM.LLVMGetNamedGlobal(self.module, "cuda_module");

    if (cuda_context_global == null or cuda_module_global == null) {
        if (self.verbose) {
            std.debug.print("❌ Global CUDA variables not found during initialization\n", .{});
        }
        return CodeGenError.CudaFunctionNotFound;
    }

    // 1. Initialize CUDA
    var cuda_init_args = [_]LLVM.LLVMValueRef{zero};
    _ = LLVM.LLVMBuildCall2(self.builder, cuInit_type, cuInit_func.?, &cuda_init_args, 1, "cuda_init_result");

    // 2. Get device
    const device_var = LLVM.LLVMBuildAlloca(self.builder, int32_type, "device");
    var device_args = [_]LLVM.LLVMValueRef{ device_var, zero };
    _ = LLVM.LLVMBuildCall2(self.builder, cuDeviceGet_type, cuDeviceGet_func.?, &device_args, 2, "device_result");

    // 3. Create context
    const device = LLVM.LLVMBuildLoad2(self.builder, int32_type, device_var, "device_val");
    const context_var = LLVM.LLVMBuildAlloca(self.builder, ptr_type, "context");
    var context_args = [_]LLVM.LLVMValueRef{ context_var, zero, device };
    _ = LLVM.LLVMBuildCall2(self.builder, cuCtxCreate_type, cuCtxCreate_func.?, &context_args, 3, "context_result");

    // Store context in global variable
    const context_val = LLVM.LLVMBuildLoad2(self.builder, ptr_type, context_var, "context_val");
    _ = LLVM.LLVMBuildStore(self.builder, context_val, cuda_context_global.?);

    // 4. Load PTX module
    const ptx_data_global = LLVM.LLVMGetNamedGlobal(self.module, "embedded_ptx_data");
    if (ptx_data_global != null) {
        // Cast PTX data array to char pointer
        var ptx_gep_indices = [_]LLVM.LLVMValueRef{ zero, zero };
        const ptx_data = LLVM.LLVMBuildGEP2(self.builder, LLVM.LLVMGlobalGetValueType(ptx_data_global.?), ptx_data_global.?, &ptx_gep_indices, 2, "ptx_data_ptr");

        const module_var = LLVM.LLVMBuildAlloca(self.builder, ptr_type, "module");
        var module_args = [_]LLVM.LLVMValueRef{ module_var, ptx_data };
        _ = LLVM.LLVMBuildCall2(self.builder, cuModuleLoadData_type, cuModuleLoadData_func.?, &module_args, 2, "module_result");

        // Store module in global variable
        const module_val = LLVM.LLVMBuildLoad2(self.builder, ptr_type, module_var, "module_val");
        _ = LLVM.LLVMBuildStore(self.builder, module_val, cuda_module_global.?);

        // 5. Get all GPU function pointers from the PTX module and store them in global variables
        try getAllGpuFunctionPointers(self, module_val);
    }

    if (self.verbose) {
        std.debug.print("✅ CUDA one-time initialization injected into main function\n", .{});
    }
}

pub fn getAllGpuFunctionPointers(self: *CodeGen, cuda_module: LLVM.LLVMValueRef) CodeGenError!void {
    if (self.verbose) {
        std.debug.print("🔧 Getting all GPU function pointers from PTX module\n", .{});
    }

    // Get cuModuleGetFunction that should already be declared
    const cuModuleGetFunction_func = LLVM.LLVMGetNamedFunction(self.module, "cuModuleGetFunction");
    if (cuModuleGetFunction_func == null) {
        if (self.verbose) {
            std.debug.print("❌ cuModuleGetFunction not found\n", .{});
        }
        return CodeGenError.CudaFunctionNotFound;
    }

    const cuModuleGetFunction_type = LLVM.LLVMGlobalGetValueType(cuModuleGetFunction_func.?);

    // Basic types
    const int8_type = LLVM.LLVMInt8TypeInContext(self.context);
    const ptr_type = LLVM.LLVMPointerType(int8_type, 0);

    // Get all GPU function pointers
    if (self.gpu_function_names) |gpu_names| {
        for (gpu_names.items) |func_name| {
            // Get the global variable for this function
            const global_name = try std.fmt.allocPrintZ(self.allocator, "cuda_func_{s}", .{func_name});
            defer self.allocator.free(global_name);

            const func_global = LLVM.LLVMGetNamedGlobal(self.module, global_name.ptr);
            if (func_global == null) {
                if (self.verbose) {
                    std.debug.print("❌ Global variable not found for GPU function: {s}\n", .{func_name});
                }
                continue;
            }

            // Allocate space for the function pointer
            const func_ptr_var = LLVM.LLVMBuildAlloca(self.builder, ptr_type, "func_ptr");

            // Create kernel name string
            const kernel_name = try std.fmt.allocPrintZ(self.allocator, "{s}", .{func_name});
            defer self.allocator.free(kernel_name);
            const kernel_name_global = LLVM.LLVMBuildGlobalStringPtr(self.builder, kernel_name.ptr, "kernel_name");

            // Call cuModuleGetFunction(func_ptr, module, kernel_name)
            var get_func_args = [_]LLVM.LLVMValueRef{ func_ptr_var, cuda_module, kernel_name_global };
            _ = LLVM.LLVMBuildCall2(self.builder, cuModuleGetFunction_type, cuModuleGetFunction_func.?, &get_func_args, 3, "get_func_result");

            // Load the function pointer and store it in the global variable
            const func_ptr = LLVM.LLVMBuildLoad2(self.builder, ptr_type, func_ptr_var, "func_ptr_val");
            _ = LLVM.LLVMBuildStore(self.builder, func_ptr, func_global.?);

            if (self.verbose) {
                std.debug.print("✅ Retrieved function pointer for GPU function: {s}\n", .{func_name});
            }
        }
    }

    if (self.verbose) {
        std.debug.print("✅ All GPU function pointers retrieved and stored\n", .{});
    }
}

pub fn freeAllocatedGpuMemory(self: *CodeGen) CodeGenError!void {
    if (self.gpu_memory_tracker == null or self.gpu_memory_ops == null) {
        return;
    }

    var tracker = &self.gpu_memory_tracker.?;
    var ops = &self.gpu_memory_ops.?;

    if (self.verbose) {
        std.debug.print("🗑️  Freeing allocated GPU memory\n", .{});
    }

    // Iterate through all tracked variables and free GPU memory
    var it = tracker.variables.iterator();
    while (it.next()) |entry| {
        const info = entry.value_ptr.*;
        if (info.gpu_ptr) |gpu_ptr| {
            ops.freeGpuMemory(gpu_ptr);
            if (self.verbose) {
                std.debug.print("  🗑️  Freed GPU memory for '{s}'\n", .{entry.key_ptr.*});
            }
        }
    }
}

pub fn injectCudaCleanupBeforeReturn(self: *CodeGen) CodeGenError!void {
    if (self.verbose) {
        std.debug.print("🧹 Injecting CUDA cleanup before main function return\n", .{});
    }

    // Basic types
    const int8_type = LLVM.LLVMInt8TypeInContext(self.context);
    const ptr_type = LLVM.LLVMPointerType(int8_type, 0);

    // Get the cuda_cleanup function that should already be declared
    const cuda_cleanup_func = LLVM.LLVMGetNamedFunction(self.module, "cuda_cleanup");
    if (cuda_cleanup_func == null) {
        if (self.verbose) {
            std.debug.print("❌ cuda_cleanup function not found\n", .{});
        }
        return CodeGenError.CudaFunctionNotFound;
    }

    // Get global variables
    const cuda_context_global = LLVM.LLVMGetNamedGlobal(self.module, "cuda_context");
    const cuda_module_global = LLVM.LLVMGetNamedGlobal(self.module, "cuda_module");

    if (cuda_context_global == null or cuda_module_global == null) {
        if (self.verbose) {
            std.debug.print("❌ Global CUDA variables not found during cleanup\n", .{});
        }
        return CodeGenError.CudaFunctionNotFound;
    }

    // Load the global context and module
    const cuda_context = LLVM.LLVMBuildLoad2(self.builder, ptr_type, cuda_context_global.?, "cuda_context_val");
    const cuda_module = LLVM.LLVMBuildLoad2(self.builder, ptr_type, cuda_module_global.?, "cuda_module_val");

    // Get function type
    const cuda_cleanup_type = LLVM.LLVMGlobalGetValueType(cuda_cleanup_func.?);

    // Call cuda_cleanup(context, module)
    var cleanup_args = [_]LLVM.LLVMValueRef{ cuda_context, cuda_module };
    _ = LLVM.LLVMBuildCall2(self.builder, cuda_cleanup_type, cuda_cleanup_func.?, &cleanup_args, 2, "");

    if (self.verbose) {
        std.debug.print("✅ CUDA cleanup injected before main function return\n", .{});
    }
}

pub fn generateSimpleCudaHostFunction(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) CodeGenError!void {
    if (self.verbose) {
        std.debug.print("🔧 Generating simple CUDA host function for: {s}\n", .{func.name});
    }

    // Generate a CPU fallback implementation that simply calls the regular CPU version
    // This allows the program to compile and run without requiring actual CUDA hardware

    // For now, we'll just generate a stub function that prints a message
    // and falls back to CPU simulation
    return generateGpuHostWrapper(self, func);
}

fn generateCudaKernelLaunch(self: *CodeGen, llvm_function: LLVM.LLVMValueRef, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) CodeGenError!void {
    // Get type references for CUDA API
    const int32_type = LLVM.LLVMInt32TypeInContext(self.context);
    const int8_type = LLVM.LLVMInt8TypeInContext(self.context);
    const ptr_type = LLVM.LLVMPointerType(int8_type, 0);
    const size_t_type = LLVM.LLVMInt64TypeInContext(self.context);

    if (self.verbose) {
        std.debug.print("🚀 Generating CUDA kernel launch for {s} (using global context/module)\n", .{func.name});
    }

    // Get global CUDA context and module (initialized once in main)
    const cuda_context_global = LLVM.LLVMGetNamedGlobal(self.module, "cuda_context");
    const cuda_module_global = LLVM.LLVMGetNamedGlobal(self.module, "cuda_module");

    if (cuda_context_global == null or cuda_module_global == null) {
        if (self.verbose) {
            std.debug.print("❌ Global CUDA context/module not found\n", .{});
        }
        return CodeGenError.CodeGenError;
    }

    // Load the global context and module
    // Note: cuda_context not used in current implementation but available if needed
    _ = LLVM.LLVMBuildLoad2(self.builder, ptr_type, cuda_context_global.?, "cuda_context");
    _ = LLVM.LLVMBuildLoad2(self.builder, ptr_type, cuda_module_global.?, "cuda_module");

    // Get additional CUDA functions that should already be declared in addCudaFunctionsToMainModule
    const cuMemAlloc_func = LLVM.LLVMGetNamedFunction(self.module, "cuMemAlloc_v2");
    const cuMemcpyHtoD_func = LLVM.LLVMGetNamedFunction(self.module, "cuMemcpyHtoD_v2");
    const cuLaunchKernel_func = LLVM.LLVMGetNamedFunction(self.module, "cuLaunchKernel");
    const cuCtxSynchronize_func = LLVM.LLVMGetNamedFunction(self.module, "cuCtxSynchronize");
    const cuMemcpyDtoH_func = LLVM.LLVMGetNamedFunction(self.module, "cuMemcpyDtoH_v2");
    const cuMemFree_func = LLVM.LLVMGetNamedFunction(self.module, "cuMemFree_v2");

    if (cuMemAlloc_func == null or cuMemcpyHtoD_func == null or
        cuLaunchKernel_func == null or cuCtxSynchronize_func == null or cuMemcpyDtoH_func == null or cuMemFree_func == null)
    {
        if (self.verbose) {
            std.debug.print("❌ Additional CUDA functions not found - they should be declared first\n", .{});
        }
        return CodeGenError.CudaFunctionNotFound;
    }

    // Get function types from existing functions
    const cuMemAlloc_type = LLVM.LLVMGlobalGetValueType(cuMemAlloc_func.?);
    const cuMemcpyHtoD_type = LLVM.LLVMGlobalGetValueType(cuMemcpyHtoD_func.?);
    const cuLaunchKernel_type = LLVM.LLVMGlobalGetValueType(cuLaunchKernel_func.?);
    const cuCtxSynchronize_type = LLVM.LLVMGlobalGetValueType(cuCtxSynchronize_func.?);
    const cuMemcpyDtoH_type = LLVM.LLVMGlobalGetValueType(cuMemcpyDtoH_func.?);
    const cuMemFree_type = LLVM.LLVMGlobalGetValueType(cuMemFree_func.?);

    const zero = LLVM.LLVMConstInt(int32_type, 0, 0);

    // 1. Get kernel function from the pre-obtained global variable
    const global_name = try std.fmt.allocPrintZ(self.allocator, "cuda_func_{s}", .{func.name});
    defer self.allocator.free(global_name);

    const func_global = LLVM.LLVMGetNamedGlobal(self.module, global_name.ptr);
    if (func_global == null) {
        if (self.verbose) {
            std.debug.print("❌ Pre-obtained function pointer not found for GPU function: {s}\n", .{func.name});
        }
        return CodeGenError.CudaFunctionNotFound;
    }

    // Load the pre-obtained function pointer
    const kernel_func = LLVM.LLVMBuildLoad2(self.builder, ptr_type, func_global.?, "kernel_func_val");

    if (self.verbose) {
        std.debug.print("✅ Using pre-obtained function pointer for GPU function: {s}\n", .{func.name});
    }

    // 6. Allocate GPU memory and copy data for each parameter
    var gpu_ptrs = std.ArrayList(LLVM.LLVMValueRef).init(self.allocator);
    defer gpu_ptrs.deinit();

    for (func.parameters, 0..) |param, i| {
        if (param.type == .tensor) {
            const llvm_param = LLVM.LLVMGetParam(llvm_function, @intCast(i));

            // Calculate size for tensor
            const elem_size: u32 = switch (param.type.tensor.element_type.*) {
                .f32 => 4,
                .i32 => 4,
                .f64 => 8,
                .i64 => 8,
                else => 4,
            };
            const tensor_size = LLVM.LLVMConstInt(size_t_type, param.type.tensor.shape[0] * elem_size, 0);

            // Allocate GPU memory
            const gpu_ptr = LLVM.LLVMBuildAlloca(self.builder, ptr_type, "gpu_ptr");
            try gpu_ptrs.append(gpu_ptr);

            var alloc_args = [_]LLVM.LLVMValueRef{ gpu_ptr, tensor_size };
            _ = LLVM.LLVMBuildCall2(self.builder, cuMemAlloc_type, cuMemAlloc_func.?, &alloc_args, 2, "");

            // Copy data to GPU
            const gpu_ptr_val = LLVM.LLVMBuildLoad2(self.builder, ptr_type, gpu_ptr, "gpu_ptr_val");
            const host_ptr = LLVM.LLVMBuildBitCast(self.builder, llvm_param, ptr_type, "host_ptr");

            var copy_args = [_]LLVM.LLVMValueRef{ gpu_ptr_val, host_ptr, tensor_size };
            _ = LLVM.LLVMBuildCall2(self.builder, cuMemcpyHtoD_type, cuMemcpyHtoD_func.?, &copy_args, 3, "");
        }
    }

    // 7. Create kernel arguments array with simple GPU device pointers
    const ptr_ptr_type = LLVM.LLVMPointerType(ptr_type, 0);

    // For simple CUDA kernels, just pass device pointers directly
    // The PTX kernel expects simple u64 parameters, not complex memref descriptors
    var kernel_param_ptrs = std.ArrayList(LLVM.LLVMValueRef).init(self.allocator);
    defer kernel_param_ptrs.deinit();

    var tensor_param_idx: usize = 0;
    for (func.parameters) |param| {
        if (param.type == .tensor) {
            const gpu_ptr_val = LLVM.LLVMBuildLoad2(self.builder, ptr_type, gpu_ptrs.items[tensor_param_idx], "gpu_ptr_val");

            // Allocate space for the device pointer value and store it
            const param_ptr = LLVM.LLVMBuildAlloca(self.builder, ptr_type, "param_ptr");
            _ = LLVM.LLVMBuildStore(self.builder, gpu_ptr_val, param_ptr);

            // Add pointer to this device pointer value to the parameter array
            try kernel_param_ptrs.append(param_ptr);

            tensor_param_idx += 1;
        }
    }

    // Create array of parameter pointers (should be 2 elements for 2 tensors)
    const param_array_type = LLVM.LLVMArrayType(ptr_type, @intCast(kernel_param_ptrs.items.len));
    const param_array = LLVM.LLVMBuildAlloca(self.builder, param_array_type, "param_array");

    // Store each parameter pointer in the array
    for (kernel_param_ptrs.items, 0..) |param_ptr, i| {
        var indices = [_]LLVM.LLVMValueRef{ LLVM.LLVMConstInt(int32_type, 0, 0), LLVM.LLVMConstInt(int32_type, @intCast(i), 0) };
        const elem_ptr = LLVM.LLVMBuildGEP2(self.builder, param_array_type, param_array, &indices, 2, "param_elem_ptr");
        _ = LLVM.LLVMBuildStore(self.builder, param_ptr, elem_ptr);
    }

    // Cast to void** for cuLaunchKernel
    const kernel_args = LLVM.LLVMBuildBitCast(self.builder, param_array, ptr_ptr_type, "kernel_args");

    // 3. Launch kernel
    const grid_x = LLVM.LLVMConstInt(int32_type, 1, 0);
    const grid_y = LLVM.LLVMConstInt(int32_type, 1, 0);
    const grid_z = LLVM.LLVMConstInt(int32_type, 1, 0);
    const block_x = LLVM.LLVMConstInt(int32_type, 1024, 0); // 1024 threads per block
    const block_y = LLVM.LLVMConstInt(int32_type, 1, 0);
    const block_z = LLVM.LLVMConstInt(int32_type, 1, 0);
    const shared_mem = LLVM.LLVMConstInt(int32_type, 0, 0);
    const stream = LLVM.LLVMConstNull(ptr_type);
    const extra = LLVM.LLVMConstNull(ptr_type);

    var launch_args = [_]LLVM.LLVMValueRef{ kernel_func, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem, stream, kernel_args, extra };
    _ = LLVM.LLVMBuildCall2(self.builder, cuLaunchKernel_type, cuLaunchKernel_func.?, &launch_args, 11, "launch_result");

    // 4. Synchronize to wait for kernel completion
    _ = LLVM.LLVMBuildCall2(self.builder, cuCtxSynchronize_type, cuCtxSynchronize_func.?, null, 0, "sync_result");

    // 9. Copy results back from GPU to host
    for (func.parameters, 0..) |param, i| {
        if (param.type == .tensor) {
            const llvm_param = LLVM.LLVMGetParam(llvm_function, @intCast(i));
            const gpu_ptr = gpu_ptrs.items[i];

            const elem_size: u32 = switch (param.type.tensor.element_type.*) {
                .f32 => 4,
                .i32 => 4,
                .f64 => 8,
                .i64 => 8,
                else => 4,
            };
            const tensor_size = LLVM.LLVMConstInt(size_t_type, param.type.tensor.shape[0] * elem_size, 0);

            const gpu_ptr_val = LLVM.LLVMBuildLoad2(self.builder, ptr_type, gpu_ptr, "gpu_ptr_val");
            const host_ptr = LLVM.LLVMBuildBitCast(self.builder, llvm_param, ptr_type, "host_ptr");

            var copy_back_args = [_]LLVM.LLVMValueRef{ host_ptr, gpu_ptr_val, tensor_size };
            _ = LLVM.LLVMBuildCall2(self.builder, cuMemcpyDtoH_type, cuMemcpyDtoH_func.?, &copy_back_args, 3, "");

            // 10. Free GPU memory
            var free_args = [_]LLVM.LLVMValueRef{gpu_ptr_val};
            _ = LLVM.LLVMBuildCall2(self.builder, cuMemFree_type, cuMemFree_func.?, &free_args, 1, "");
        }
    }

    // 5. Handle return value
    if (func.return_type == .void) {
        _ = LLVM.LLVMBuildRetVoid(self.builder);
    } else {
        // For non-void functions, return first element of first tensor parameter
        if (func.parameters.len > 0 and func.parameters[0].type == .tensor) {
            const first_param = LLVM.LLVMGetParam(llvm_function, 0);
            var indices = [_]LLVM.LLVMValueRef{zero};
            const first_elem_ptr = LLVM.LLVMBuildGEP2(self.builder, self.toLLVMType(func.parameters[0].type), first_param, &indices, 1, "first_elem_ptr");
            const first_elem = LLVM.LLVMBuildLoad2(self.builder, self.toLLVMType(func.return_type), first_elem_ptr, "first_elem");
            _ = LLVM.LLVMBuildRet(self.builder, first_elem);
        } else {
            // Return dummy value if no tensor parameters
            const return_value = switch (func.return_type) {
                .i32 => LLVM.LLVMConstInt(LLVM.LLVMInt32TypeInContext(self.context), 0, 0),
                .i64 => LLVM.LLVMConstInt(LLVM.LLVMInt64TypeInContext(self.context), 0, 0),
                .f32 => LLVM.LLVMConstReal(LLVM.LLVMFloatTypeInContext(self.context), 0.0),
                .f64 => LLVM.LLVMConstReal(LLVM.LLVMDoubleTypeInContext(self.context), 0.0),
                else => LLVM.LLVMConstInt(LLVM.LLVMInt32TypeInContext(self.context), 0, 0),
            };
            _ = LLVM.LLVMBuildRet(self.builder, return_value);
        }
    }

    if (self.verbose) {
        std.debug.print("✅ CUDA kernel launch generated for {s} (using global context/module)\n", .{func.name});
    }
}
