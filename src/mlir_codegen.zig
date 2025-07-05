const std = @import("std");
const parser = @import("parser.zig");

// MLIR C API bindings
const MLIR = @cImport({
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/BuiltinTypes.h");
    @cInclude("mlir-c/BuiltinAttributes.h");
    @cInclude("mlir-c/Dialect/Func.h");
    @cInclude("mlir-c/Dialect/GPU.h");
    @cInclude("mlir-c/Dialect/Arith.h");
    @cInclude("mlir-c/Dialect/MemRef.h");
    @cInclude("mlir-c/Dialect/NVVM.h");
    @cInclude("mlir-c/Dialect/SCF.h");
    @cInclude("mlir-c/Pass.h");
    @cInclude("mlir-c/ExecutionEngine.h");
    @cInclude("mlir-c/Transforms.h");
    @cInclude("mlir-c/Target/LLVMIR.h");
    @cInclude("llvm-c/Core.h");
    @cInclude("llvm-c/Target.h");
    @cInclude("llvm-c/TargetMachine.h");
    @cInclude("llvm-c/Support.h");
});

pub const MLIRCodeGenError = error{
    MLIRError,
    InvalidGpuFunction,
    UnsupportedOperation,
    MLIRNotAvailable,
    PassPipelineError,
    GPUDialectNotSupported,
    NVVMDialectNotAvailable,
    NVVMError,
    NVVMFailed,
    InvalidCharacter,
    Overflow,
} || std.mem.Allocator.Error;

/// GPU function analysis result
const GPUFunctionInfo = struct {
    has_parallel_assignment: bool,
    tensor_params: [][]const u8, // names of tensor parameters
    parallel_assignment: ?parser.ASTNode, // the parallel assignment if found

    pub fn deinit(self: *GPUFunctionInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.tensor_params);
    }
};

/// Error callback for MLIR pass pipeline parsing
fn pipelineErrorCallback(message: MLIR.MlirStringRef, userData: ?*anyopaque) callconv(.C) void {
    _ = message; // Message parsing not available in current MLIR C API
    _ = userData; // Unused

    std.debug.print("MLIR Pass Pipeline Error: Parse or execution error occurred\n", .{});
}

pub const MLIRCodeGen = struct {
    context: MLIR.MlirContext,
    module: MLIR.MlirModule,
    location: MLIR.MlirLocation,
    allocator: std.mem.Allocator,
    verbose: bool,

    pub fn init(allocator: std.mem.Allocator, module_name: []const u8, verbose: bool) !MLIRCodeGen {
        // Initialize MLIR context
        const context = MLIR.mlirContextCreate();

        // Register required dialects
        MLIR.mlirDialectHandleRegisterDialect(MLIR.mlirGetDialectHandle__func__(), context);
        MLIR.mlirDialectHandleRegisterDialect(MLIR.mlirGetDialectHandle__gpu__(), context);
        MLIR.mlirDialectHandleRegisterDialect(MLIR.mlirGetDialectHandle__arith__(), context);
        MLIR.mlirDialectHandleRegisterDialect(MLIR.mlirGetDialectHandle__memref__(), context);
        MLIR.mlirDialectHandleRegisterDialect(MLIR.mlirGetDialectHandle__scf__(), context);
        MLIR.mlirDialectHandleRegisterDialect(MLIR.mlirGetDialectHandle__nvvm__(), context);

        // Create location and module
        const location = MLIR.mlirLocationUnknownGet(context);
        _ = module_name; // For now, we'll use a default name
        const module = MLIR.mlirModuleCreateEmpty(location);

        return MLIRCodeGen{
            .context = context,
            .module = module,
            .location = location,
            .allocator = allocator,
            .verbose = verbose,
        };
    }

    pub fn deinit(self: *MLIRCodeGen) void {
        MLIR.mlirModuleDestroy(self.module);
        MLIR.mlirContextDestroy(self.context);
    }

    /// Print the MLIR module (for debugging)
    pub fn printMLIR(self: *MLIRCodeGen) void {
        if (self.verbose) {
            // Use mlirOperationDump for proper printing to stderr
            MLIR.mlirOperationDump(MLIR.mlirModuleGetOperation(self.module));
        }
    }

    /// Lower MLIR module to PTX assembly through GPU → NVVM → PTX pipeline
    pub fn lowerMLIRToPTX(self: *MLIRCodeGen) MLIRCodeGenError![]const u8 {
        if (self.verbose) {
            std.debug.print("Lowering MLIR to PTX using official GPU compilation pipeline\n", .{});
        }

        // Step 1: Create pass manager for the official GPU compilation pipeline
        const pass_manager = MLIR.mlirPassManagerCreate(self.context);
        defer MLIR.mlirPassManagerDestroy(pass_manager);

        // Step 2: Get module operation
        const module_op = MLIR.mlirModuleGetOperation(self.module);

        // Step 3: Print MLIR before pass pipeline (for debugging)
        if (self.verbose) {
            std.debug.print("=== MLIR Before GPU Pass Pipeline ===\n", .{});
            MLIR.mlirOperationDump(module_op);
            std.debug.print("\n=== End MLIR ===\n", .{});
        }

        if (self.verbose) {
            std.debug.print("Building official MLIR GPU → NVVM → PTX pass pipeline...\n", .{});
        }

        // Get the builtin.module operation pass manager
        const builtin_module_name = MLIR.mlirStringRefCreateFromCString("builtin.module");
        const op_pass_manager = MLIR.mlirPassManagerGetNestedUnder(pass_manager, builtin_module_name);

        // Build the complete pipeline string according to MLIR documentation
        const pipeline_str =
            \\convert-gpu-to-nvvm,
            \\gpu-to-llvm
        ;

        if (self.verbose) {
            std.debug.print("Parsing GPU pass pipeline: {s}\n", .{pipeline_str});
        }

        // TODO: Re-enable once we have the right pass names and libraries
        // Parse the pass pipeline
        const pipeline_string_ref = MLIR.mlirStringRefCreateFromCString(pipeline_str);
        const parse_result = MLIR.mlirParsePassPipeline(op_pass_manager, pipeline_string_ref, pipelineErrorCallback, null);

        if (!MLIR.mlirLogicalResultIsSuccess(parse_result)) {
            if (self.verbose) {
                std.debug.print("Warning: Pass pipeline parsing failed, proceeding without GPU→NVVM conversion\n", .{});
                std.debug.print("This is expected until we have all required MLIR libraries\n", .{});
            }
            return MLIRCodeGenError.NVVMError;
        }
        if (self.verbose) {
            std.debug.print("Successfully parsed GPU pass pipeline\n", .{});
        }

        // Step 5: Run the pass pipeline
        if (self.verbose) {
            std.debug.print("Running GPU → NVVM → LLVM pass pipeline...\n", .{});
        }

        const pass_result = MLIR.mlirPassManagerRunOnOp(pass_manager, module_op);
        if (!MLIR.mlirLogicalResultIsSuccess(pass_result)) {
            if (self.verbose) {
                std.debug.print("Warning: GPU pass pipeline failed, continuing anyway\n", .{});
            }
            return MLIRCodeGenError.NVVMFailed;
        }

        if (self.verbose) {
            std.debug.print("GPU pass pipeline completed successfully\n", .{});
            std.debug.print("=== MLIR After GPU Pass Pipeline ===\n", .{});
            MLIR.mlirOperationDump(module_op);
            std.debug.print("\n=== End MLIR ===\n", .{});
        }

        // Step 6: Translate MLIR to LLVM IR (the module should now contain gpu.binary ops with PTX)
        if (self.verbose) {
            std.debug.print("Translating MLIR to LLVM IR...\n", .{});
        }

        const llvm_context = MLIR.LLVMContextCreate();
        defer MLIR.LLVMContextDispose(llvm_context);

        const llvm_module = MLIR.mlirTranslateModuleToLLVMIR(module_op, llvm_context);
        defer if (llvm_module != null) MLIR.LLVMDisposeModule(llvm_module);

        if (llvm_module == null) {
            if (self.verbose) {
                std.debug.print("Error: Failed to translate MLIR to LLVM IR\n", .{});
            }
            return MLIRCodeGenError.MLIRError;
        }

        // Step 7: The gpu-module-to-binary pass should have embedded PTX in gpu.binary operations
        // For now, let's extract PTX from the LLVM module using the NVPTX backend
        if (self.verbose) {
            std.debug.print("Extracting PTX from compiled GPU modules...\n", .{});
        }

        // Initialize NVPTX target
        MLIR.LLVMInitializeNVPTXTargetInfo();
        MLIR.LLVMInitializeNVPTXTarget();
        MLIR.LLVMInitializeNVPTXTargetMC();
        MLIR.LLVMInitializeNVPTXAsmPrinter();

        // Get NVPTX target
        const target_triple = "nvptx64-nvidia-cuda";
        var target: MLIR.LLVMTargetRef = undefined;
        var error_msg: [*c]u8 = undefined;

        if (MLIR.LLVMGetTargetFromTriple(target_triple, &target, &error_msg) != 0) {
            if (self.verbose) {
                std.debug.print("Error: Failed to get NVPTX target: {s}\n", .{error_msg});
            }
            MLIR.LLVMDisposeMessage(error_msg);
            return MLIRCodeGenError.NVVMDialectNotAvailable;
        }

        // Create target machine
        const target_machine = MLIR.LLVMCreateTargetMachine(target, target_triple, "sm_50", // GPU capability
            "", // features
            MLIR.LLVMCodeGenLevelDefault, MLIR.LLVMRelocDefault, MLIR.LLVMCodeModelDefault);
        defer MLIR.LLVMDisposeTargetMachine(target_machine);

        // Generate PTX assembly
        var output_buffer: MLIR.LLVMMemoryBufferRef = undefined;
        var error_message: [*c]u8 = undefined;

        const result = MLIR.LLVMTargetMachineEmitToMemoryBuffer(target_machine, llvm_module, MLIR.LLVMAssemblyFile, &error_message, &output_buffer);

        if (result != 0) {
            if (self.verbose) {
                std.debug.print("Error: Failed to generate PTX assembly: {s}\n", .{error_message});
            }
            MLIR.LLVMDisposeMessage(error_message);
            return MLIRCodeGenError.PassPipelineError;
        }

        // Extract PTX string from memory buffer
        const ptx_data = MLIR.LLVMGetBufferStart(output_buffer);
        const ptx_size = MLIR.LLVMGetBufferSize(output_buffer);

        // Copy PTX to our allocator-managed memory
        const ptx_string = try self.allocator.alloc(u8, ptx_size);
        @memcpy(ptx_string, ptx_data[0..ptx_size]);

        MLIR.LLVMDisposeMemoryBuffer(output_buffer);

        if (self.verbose) {
            std.debug.print("Successfully generated PTX using official MLIR GPU compilation pipeline\n", .{});
            std.debug.print("PTX size: {d} bytes\n", .{ptx_size});
        }

        return ptx_string;
    }

    /// Main entry point for generating GPU functions
    pub fn generateGpuFunction(self: *MLIRCodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) MLIRCodeGenError!void {
        if (self.verbose) {
            std.debug.print("Analyzing function for GPU compilation: {s}\n", .{func.name});
        }

        // Analyze function to determine if it's suitable for GPU compilation
        var gpu_info = try self.analyzeGPUFunction(func);
        defer gpu_info.deinit(self.allocator);

        if (!gpu_info.has_parallel_assignment) {
            if (self.verbose) {
                std.debug.print("Function {s} has no parallel assignments, generating host function\n", .{func.name});
            }
            try self.generateHostFunction(func);
            return;
        }

        if (self.verbose) {
            std.debug.print("Generating GPU kernel for function: {s}\n", .{func.name});
        }

        // Generate GPU module and kernel using proper MLIR operations
        try self.generateGPUModule(func, gpu_info);

        // Generate host wrapper function with gpu.launch
        try self.generateHostWrapper(func, gpu_info);
    }

    /// Analyze function to determine GPU compilation strategy
    fn analyzeGPUFunction(self: *MLIRCodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) MLIRCodeGenError!GPUFunctionInfo {
        var tensor_params = std.ArrayList([]const u8).init(self.allocator);
        var has_parallel = false;
        var parallel_assignment: ?parser.ASTNode = null;

        // Find tensor parameters
        for (func.parameters) |param| {
            if (param.type == .tensor) {
                try tensor_params.append(param.name);
            }
        }

        // Look for parallel assignments in function body (a[i] = expr)
        for (func.body) |stmt| {
            if (stmt == .parallel_assignment) {
                has_parallel = true;
                parallel_assignment = stmt;
                break;
            }
        }

        return GPUFunctionInfo{
            .has_parallel_assignment = has_parallel,
            .tensor_params = try tensor_params.toOwnedSlice(),
            .parallel_assignment = parallel_assignment,
        };
    }

    /// Generate GPU module using proper MLIR GPU dialect
    fn generateGPUModule(self: *MLIRCodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration), gpu_info: GPUFunctionInfo) MLIRCodeGenError!void {
        if (self.verbose) {
            std.debug.print("Creating GPU module with MLIR GPU dialect for: {s}\n", .{func.name});
        }

        // Get module level for operation insertion
        const module_op = MLIR.mlirModuleGetOperation(self.module);
        const module_body = MLIR.mlirOperationGetFirstRegion(module_op);
        const module_block = MLIR.mlirRegionGetFirstBlock(module_body);

        // Create GPU module operation (gpu.module)
        const gpu_module_name = try std.fmt.allocPrintZ(self.allocator, "{s}_gpu", .{func.name});
        defer self.allocator.free(gpu_module_name);

        var gpu_module_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("gpu.module"), self.location);

        // Add sym_name attribute to GPU module
        const gpu_module_name_attr = MLIR.mlirStringAttrGet(self.context, MLIR.mlirStringRefCreateFromCString(gpu_module_name.ptr));
        const sym_name_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("sym_name"));
        const gpu_module_name_named_attr = MLIR.mlirNamedAttributeGet(sym_name_id, gpu_module_name_attr);
        MLIR.mlirOperationStateAddAttributes(&gpu_module_state, 1, &gpu_module_name_named_attr);

        // Add a region to the GPU module operation
        const gpu_module_region = MLIR.mlirRegionCreate();
        MLIR.mlirOperationStateAddOwnedRegions(&gpu_module_state, 1, &gpu_module_region);

        const gpu_module_op = MLIR.mlirOperationCreate(&gpu_module_state);
        MLIR.mlirBlockAppendOwnedOperation(module_block, gpu_module_op);

        // Get the body of the GPU module and create kernel function
        const gpu_module_body = MLIR.mlirOperationGetFirstRegion(gpu_module_op);

        // Check if region is valid before proceeding
        if (MLIR.mlirRegionIsNull(gpu_module_body)) {
            if (self.verbose) {
                std.debug.print("Error: GPU module has no regions\n", .{});
            }
            return MLIRCodeGenError.MLIRError;
        }

        // Create a block for the GPU module if none exists
        var gpu_module_block = MLIR.mlirRegionGetFirstBlock(gpu_module_body);
        if (MLIR.mlirBlockIsNull(gpu_module_block)) {
            if (self.verbose) {
                std.debug.print("Creating block for GPU module\n", .{});
            }
            gpu_module_block = MLIR.mlirBlockCreate(0, null, null);
            MLIR.mlirRegionAppendOwnedBlock(gpu_module_body, gpu_module_block);
        }

        // Create GPU kernel function (gpu.func with kernel attribute)
        try self.createGPUKernelFunction(func, gpu_info, gpu_module_block);
    }

    /// Create GPU kernel function with proper memref types and GPU operations
    fn createGPUKernelFunction(self: *MLIRCodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration), gpu_info: GPUFunctionInfo, gpu_module_block: MLIR.MlirBlock) MLIRCodeGenError!void {
        if (self.verbose) {
            std.debug.print("Creating GPU kernel function with memref parameters\n", .{});
        }

        // Convert parameters to memref types for GPU (tensors become memrefs)
        const param_types = try self.allocator.alloc(MLIR.MlirType, func.parameters.len);
        defer self.allocator.free(param_types);

        for (func.parameters, 0..) |param, i| {
            if (self.verbose) {
                std.debug.print("Converting parameter {d}: {s}\n", .{ i, param.name });
            }
            param_types[i] = try self.convertTypeToMemRef(param.type);
        }

        if (self.verbose) {
            std.debug.print("Created {d} parameter types for function block\n", .{param_types.len});
        }

        // GPU kernels return void
        const void_type = MLIR.mlirNoneTypeGet(self.context);
        const kernel_func_type = MLIR.mlirFunctionTypeGet(self.context, @intCast(param_types.len), param_types.ptr, 1, &void_type);

        // Create GPU function operation (gpu.func)
        const kernel_name = try std.fmt.allocPrintZ(self.allocator, "{s}_kernel", .{func.name});
        defer self.allocator.free(kernel_name);

        var gpu_func_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("gpu.func"), self.location);

        // Add function attributes
        const kernel_name_attr = MLIR.mlirStringAttrGet(self.context, MLIR.mlirStringRefCreateFromCString(kernel_name.ptr));
        const sym_name_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("sym_name"));
        const sym_name_named_attr = MLIR.mlirNamedAttributeGet(sym_name_id, kernel_name_attr);

        const func_type_attr = MLIR.mlirTypeAttrGet(kernel_func_type);
        const function_type_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("function_type"));
        const function_type_named_attr = MLIR.mlirNamedAttributeGet(function_type_id, func_type_attr);

        // Add kernel attribute (marks this as a GPU kernel)
        const kernel_attr = MLIR.mlirUnitAttrGet(self.context);
        const kernel_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("gpu.kernel"));
        const kernel_named_attr = MLIR.mlirNamedAttributeGet(kernel_id, kernel_attr);

        const attrs = [_]MLIR.MlirNamedAttribute{ sym_name_named_attr, function_type_named_attr, kernel_named_attr };
        MLIR.mlirOperationStateAddAttributes(&gpu_func_state, attrs.len, &attrs[0]);

        // Add a region to the GPU function operation
        const gpu_func_region = MLIR.mlirRegionCreate();
        MLIR.mlirOperationStateAddOwnedRegions(&gpu_func_state, 1, &gpu_func_region);

        const gpu_func_op = MLIR.mlirOperationCreate(&gpu_func_state);
        MLIR.mlirBlockAppendOwnedOperation(gpu_module_block, gpu_func_op);

        // Create function body
        const func_body = MLIR.mlirOperationGetFirstRegion(gpu_func_op);
        // Create block with no arguments first to avoid segfault with MLIR types
        const func_block = MLIR.mlirBlockCreate(0, null, null);
        MLIR.mlirRegionAppendOwnedBlock(func_body, func_block);

        // Generate kernel body with GPU indexing
        try self.generateKernelBody(func, gpu_info, func_block);

        // Add gpu.return
        var return_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("gpu.return"), self.location);
        const return_op = MLIR.mlirOperationCreate(&return_state);
        MLIR.mlirBlockAppendOwnedOperation(func_block, return_op);
    }

    /// Generate the body of the GPU kernel with proper GPU indexing
    fn generateKernelBody(self: *MLIRCodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration), gpu_info: GPUFunctionInfo, block: MLIR.MlirBlock) MLIRCodeGenError!void {
        _ = func; // Will be used for more complex operations

        if (self.verbose) {
            std.debug.print("Generating GPU kernel body with thread indexing\n", .{});
        }

        if (gpu_info.parallel_assignment) |assignment| {
            // Generate GPU thread indexing operations
            const global_idx = try self.generateGPUThreadIndexing(block);

            // Generate bounds-checked memory operations
            try self.generateBoundsCheckedAssignment(assignment.parallel_assignment, global_idx, block);
        }
    }

    /// Generate GPU thread indexing: global_idx = blockIdx.x * blockDim.x + threadIdx.x
    fn generateGPUThreadIndexing(self: *MLIRCodeGen, block: MLIR.MlirBlock) MLIRCodeGenError!MLIR.MlirValue {
        const index_type = MLIR.mlirIndexTypeGet(self.context);

        // Get thread index using gpu.thread_id x
        var thread_id_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("gpu.thread_id"), self.location);
        MLIR.mlirOperationStateAddResults(&thread_id_state, 1, &index_type);

        // Add dimension attribute (x)
        const dim_attr = MLIR.mlirStringAttrGet(self.context, MLIR.mlirStringRefCreateFromCString("x"));
        const dimension_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("dimension"));
        const dimension_named_attr = MLIR.mlirNamedAttributeGet(dimension_id, dim_attr);
        MLIR.mlirOperationStateAddAttributes(&thread_id_state, 1, &dimension_named_attr);

        const thread_id_op = MLIR.mlirOperationCreate(&thread_id_state);
        MLIR.mlirBlockAppendOwnedOperation(block, thread_id_op);
        const thread_idx = MLIR.mlirOperationGetResult(thread_id_op, 0);

        // Get block index using gpu.block_id x
        var block_id_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("gpu.block_id"), self.location);
        MLIR.mlirOperationStateAddResults(&block_id_state, 1, &index_type);
        MLIR.mlirOperationStateAddAttributes(&block_id_state, 1, &dimension_named_attr);

        const block_id_op = MLIR.mlirOperationCreate(&block_id_state);
        MLIR.mlirBlockAppendOwnedOperation(block, block_id_op);
        const block_idx = MLIR.mlirOperationGetResult(block_id_op, 0);

        // Get block dimension using gpu.block_dim x
        var block_dim_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("gpu.block_dim"), self.location);
        MLIR.mlirOperationStateAddResults(&block_dim_state, 1, &index_type);
        MLIR.mlirOperationStateAddAttributes(&block_dim_state, 1, &dimension_named_attr);

        const block_dim_op = MLIR.mlirOperationCreate(&block_dim_state);
        MLIR.mlirBlockAppendOwnedOperation(block, block_dim_op);
        const block_dim = MLIR.mlirOperationGetResult(block_dim_op, 0);

        // Calculate global index: global_idx = block_idx * block_dim + thread_idx
        return try self.generateGlobalIndex(thread_idx, block_idx, block_dim, block);
    }

    /// Generate global index calculation using arith operations
    fn generateGlobalIndex(self: *MLIRCodeGen, thread_idx: MLIR.MlirValue, block_idx: MLIR.MlirValue, block_dim: MLIR.MlirValue, block: MLIR.MlirBlock) MLIRCodeGenError!MLIR.MlirValue {
        const index_type = MLIR.mlirIndexTypeGet(self.context);

        // block_idx * block_dim
        var mul_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("arith.muli"), self.location);
        MLIR.mlirOperationStateAddOperands(&mul_state, 2, &[_]MLIR.MlirValue{ block_idx, block_dim });
        MLIR.mlirOperationStateAddResults(&mul_state, 1, &index_type);

        const mul_op = MLIR.mlirOperationCreate(&mul_state);
        MLIR.mlirBlockAppendOwnedOperation(block, mul_op);
        const block_offset = MLIR.mlirOperationGetResult(mul_op, 0);

        // block_offset + thread_idx
        var add_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("arith.addi"), self.location);
        MLIR.mlirOperationStateAddOperands(&add_state, 2, &[_]MLIR.MlirValue{ block_offset, thread_idx });
        MLIR.mlirOperationStateAddResults(&add_state, 1, &index_type);

        const add_op = MLIR.mlirOperationCreate(&add_state);
        MLIR.mlirBlockAppendOwnedOperation(block, add_op);

        return MLIR.mlirOperationGetResult(add_op, 0);
    }

    /// Generate bounds-checked assignment with scf.if operation
    fn generateBoundsCheckedAssignment(self: *MLIRCodeGen, assignment: @TypeOf(@as(parser.ASTNode, undefined).parallel_assignment), global_idx: MLIR.MlirValue, block: MLIR.MlirBlock) MLIRCodeGenError!void {
        if (self.verbose) {
            std.debug.print("Generating bounds-checked assignment with scf.if\n", .{});
        }

        // Create array size constant (1024 for our example)
        const index_type = MLIR.mlirIndexTypeGet(self.context);
        const size_attr = MLIR.mlirIntegerAttrGet(index_type, 1024);

        var size_const_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("arith.constant"), self.location);
        MLIR.mlirOperationStateAddResults(&size_const_state, 1, &index_type);

        const value_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("value"));
        const value_named_attr = MLIR.mlirNamedAttributeGet(value_id, size_attr);
        MLIR.mlirOperationStateAddAttributes(&size_const_state, 1, &value_named_attr);

        const size_const_op = MLIR.mlirOperationCreate(&size_const_state);
        MLIR.mlirBlockAppendOwnedOperation(block, size_const_op);
        const array_size = MLIR.mlirOperationGetResult(size_const_op, 0);

        // Compare: global_idx < array_size
        const i1_type = MLIR.mlirIntegerTypeGet(self.context, 1);
        var cmp_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("arith.cmpi"), self.location);
        MLIR.mlirOperationStateAddOperands(&cmp_state, 2, &[_]MLIR.MlirValue{ global_idx, array_size });
        MLIR.mlirOperationStateAddResults(&cmp_state, 1, &i1_type);

        // Add predicate attribute "slt" (signed less than)
        const predicate_attr = MLIR.mlirIntegerAttrGet(MLIR.mlirIntegerTypeGet(self.context, 64), 2); // slt = 2
        const predicate_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("predicate"));
        const predicate_named_attr = MLIR.mlirNamedAttributeGet(predicate_id, predicate_attr);
        MLIR.mlirOperationStateAddAttributes(&cmp_state, 1, &predicate_named_attr);

        const cmp_op = MLIR.mlirOperationCreate(&cmp_state);
        MLIR.mlirBlockAppendOwnedOperation(block, cmp_op);
        const condition = MLIR.mlirOperationGetResult(cmp_op, 0);

        // Create scf.if operation for bounds checking
        var if_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("scf.if"), self.location);
        MLIR.mlirOperationStateAddOperands(&if_state, 1, &condition);

        // Add regions to scf.if operation
        const then_region = MLIR.mlirRegionCreate();
        MLIR.mlirOperationStateAddOwnedRegions(&if_state, 1, &then_region);

        const if_op = MLIR.mlirOperationCreate(&if_state);
        MLIR.mlirBlockAppendOwnedOperation(block, if_op);

        // Get the then region and create block
        const first_region = MLIR.mlirOperationGetFirstRegion(if_op);
        const then_block = MLIR.mlirBlockCreate(0, null, null);
        MLIR.mlirRegionAppendOwnedBlock(first_region, then_block);

        // Generate the actual memory operations inside the if
        try self.generateMemRefOperations(assignment, global_idx, then_block);

        // Add scf.yield to the then block
        var yield_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("scf.yield"), self.location);
        const yield_op = MLIR.mlirOperationCreate(&yield_state);
        MLIR.mlirBlockAppendOwnedOperation(then_block, yield_op);
    }

    /// Generate memref load/store operations (this is where the actual computation happens)
    fn generateMemRefOperations(self: *MLIRCodeGen, assignment: @TypeOf(@as(parser.ASTNode, undefined).parallel_assignment), global_idx: MLIR.MlirValue, block: MLIR.MlirBlock) MLIRCodeGenError!void {
        if (self.verbose) {
            std.debug.print("Generating memref load/store operations for parallel assignment\n", .{});
        }

        // For the example a[i] = a[i] + b[i], we generate:
        // %val_a = memref.load %arg0[%global_idx] : memref<1024xf32>
        // %val_b = memref.load %arg1[%global_idx] : memref<1024xf32>
        // %result = arith.addf %val_a, %val_b : f32
        // memref.store %result, %arg0[%global_idx] : memref<1024xf32>

        const f32_type = MLIR.mlirF32TypeGet(self.context);

        // For now, let's create a simple computation that doesn't depend on function arguments
        // This demonstrates the MLIR operation creation pattern without complex region operations

        // Create a constant value to store
        const const_attr = MLIR.mlirFloatAttrDoubleGet(self.context, f32_type, 42.0);
        var const_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("arith.constant"), self.location);
        MLIR.mlirOperationStateAddResults(&const_state, 1, &f32_type);

        const value_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("value"));
        const value_named_attr = MLIR.mlirNamedAttributeGet(value_id, const_attr);
        MLIR.mlirOperationStateAddAttributes(&const_state, 1, &value_named_attr);

        const const_op = MLIR.mlirOperationCreate(&const_state);
        MLIR.mlirBlockAppendOwnedOperation(block, const_op);

        if (self.verbose) {
            std.debug.print("Generated memref operations: constant value creation\n", .{});
            std.debug.print("Note: Full memref load/store requires function argument access\n", .{});
        }

        // Note: To implement full memref.load and memref.store operations, we need:
        // 1. Access to function block arguments (the memref parameters)
        // 2. Proper analysis of the assignment expression to generate the right operations
        // 3. Support for different arithmetic operations based on the assignment

        _ = assignment; // Will be used for more complex expressions
        _ = global_idx; // Will be used for indexing operations
    }

    /// Generate host wrapper function that launches the GPU kernel
    fn generateHostWrapper(self: *MLIRCodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration), gpu_info: GPUFunctionInfo) MLIRCodeGenError!void {
        if (self.verbose) {
            std.debug.print("Generating host wrapper with gpu.launch for: {s}\n", .{func.name});
        }

        // Get module level for operation insertion
        const module_op = MLIR.mlirModuleGetOperation(self.module);
        const module_body = MLIR.mlirOperationGetFirstRegion(module_op);
        const module_block = MLIR.mlirRegionGetFirstBlock(module_body);

        // Convert parameters to regular types (not memref) for host function
        const param_types = try self.allocator.alloc(MLIR.MlirType, func.parameters.len);
        defer self.allocator.free(param_types);

        for (func.parameters, 0..) |param, i| {
            param_types[i] = try self.convertType(param.type);
        }

        // Host function returns the same type as declared
        const return_type = try self.convertType(func.return_type);
        const host_func_type = MLIR.mlirFunctionTypeGet(self.context, @intCast(param_types.len), param_types.ptr, 1, &return_type);

        // Create host function operation (func.func)
        var host_func_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("func.func"), self.location);

        // Add function attributes - create proper null-terminated string
        const func_name_z = try std.fmt.allocPrintZ(self.allocator, "{s}", .{func.name});
        defer self.allocator.free(func_name_z);
        const host_name_attr = MLIR.mlirStringAttrGet(self.context, MLIR.mlirStringRefCreateFromCString(func_name_z.ptr));
        const sym_name_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("sym_name"));
        const sym_name_named_attr = MLIR.mlirNamedAttributeGet(sym_name_id, host_name_attr);

        const func_type_attr = MLIR.mlirTypeAttrGet(host_func_type);
        const function_type_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("function_type"));
        const function_type_named_attr = MLIR.mlirNamedAttributeGet(function_type_id, func_type_attr);

        const attrs = [_]MLIR.MlirNamedAttribute{ sym_name_named_attr, function_type_named_attr };
        MLIR.mlirOperationStateAddAttributes(&host_func_state, attrs.len, &attrs[0]);

        // Add a region to the host function operation
        const host_func_region = MLIR.mlirRegionCreate();
        MLIR.mlirOperationStateAddOwnedRegions(&host_func_state, 1, &host_func_region);

        const host_func_op = MLIR.mlirOperationCreate(&host_func_state);
        MLIR.mlirBlockAppendOwnedOperation(module_block, host_func_op);

        // Create function body
        const host_func_body = MLIR.mlirOperationGetFirstRegion(host_func_op);
        const host_func_block = MLIR.mlirBlockCreate(0, null, null);
        MLIR.mlirRegionAppendOwnedBlock(host_func_body, host_func_block);

        // Generate a simple return for now (demonstrates the structure)
        try self.generateHostFunctionBody(func, gpu_info, host_func_block);

        if (self.verbose) {
            std.debug.print("Generated host wrapper function: {s}\n", .{func.name});
        }
    }

    /// Generate the body of the host function (simplified version)
    fn generateHostFunctionBody(self: *MLIRCodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration), gpu_info: GPUFunctionInfo, block: MLIR.MlirBlock) MLIRCodeGenError!void {
        _ = gpu_info; // Will be used for gpu.launch parameters

        if (self.verbose) {
            std.debug.print("Generating host function body for: {s}\n", .{func.name});
        }

        // For now, generate a simple return of the appropriate type
        // In a full implementation, this would:
        // 1. Set up GPU launch grid/block dimensions
        // 2. Call gpu.launch_func to invoke the kernel
        // 3. Return the result

        if (func.return_type == .void) {
            // Add func.return with no operands for void functions
            var return_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("func.return"), self.location);

            const return_op = MLIR.mlirOperationCreate(&return_state);
            MLIR.mlirBlockAppendOwnedOperation(block, return_op);
        } else {
            const return_value = switch (func.return_type) {
                .i32 => blk: {
                    const i32_type = MLIR.mlirIntegerTypeGet(self.context, 32);
                    const const_attr = MLIR.mlirIntegerAttrGet(i32_type, 0);

                    var const_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("arith.constant"), self.location);
                    MLIR.mlirOperationStateAddResults(&const_state, 1, &i32_type);

                    const value_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("value"));
                    const value_named_attr = MLIR.mlirNamedAttributeGet(value_id, const_attr);
                    MLIR.mlirOperationStateAddAttributes(&const_state, 1, &value_named_attr);

                    const const_op = MLIR.mlirOperationCreate(&const_state);
                    MLIR.mlirBlockAppendOwnedOperation(block, const_op);

                    break :blk MLIR.mlirOperationGetResult(const_op, 0);
                },
                else => blk: {
                    // Default to i32 for other types for now
                    const i32_type = MLIR.mlirIntegerTypeGet(self.context, 32);
                    const const_attr = MLIR.mlirIntegerAttrGet(i32_type, 0);

                    var const_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("arith.constant"), self.location);
                    MLIR.mlirOperationStateAddResults(&const_state, 1, &i32_type);

                    const value_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("value"));
                    const value_named_attr = MLIR.mlirNamedAttributeGet(value_id, const_attr);
                    MLIR.mlirOperationStateAddAttributes(&const_state, 1, &value_named_attr);

                    const const_op = MLIR.mlirOperationCreate(&const_state);
                    MLIR.mlirBlockAppendOwnedOperation(block, const_op);

                    break :blk MLIR.mlirOperationGetResult(const_op, 0);
                },
            };

            // Add func.return
            var return_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("func.return"), self.location);
            MLIR.mlirOperationStateAddOperands(&return_state, 1, &return_value);

            const return_op = MLIR.mlirOperationCreate(&return_state);
            MLIR.mlirBlockAppendOwnedOperation(block, return_op);
        }

        if (self.verbose) {
            std.debug.print("Generated host function body with return statement\n", .{});
        }
    }

    /// Generate simple host function (no GPU operations)
    fn generateHostFunction(self: *MLIRCodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) MLIRCodeGenError!void {
        if (self.verbose) {
            std.debug.print("Generating simple host function: {s}\n", .{func.name});
        }

        // Get module level for operation insertion
        const module_op = MLIR.mlirModuleGetOperation(self.module);
        const module_body = MLIR.mlirOperationGetFirstRegion(module_op);
        const module_block = MLIR.mlirRegionGetFirstBlock(module_body);

        // Convert parameters to MLIR types
        const param_types = try self.allocator.alloc(MLIR.MlirType, func.parameters.len);
        defer self.allocator.free(param_types);

        for (func.parameters, 0..) |param, i| {
            param_types[i] = try self.convertType(param.type);
        }

        // Convert return type
        const return_type = try self.convertType(func.return_type);
        const host_func_type = MLIR.mlirFunctionTypeGet(self.context, @intCast(param_types.len), param_types.ptr, 1, &return_type);

        // Create host function operation (func.func)
        var host_func_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("func.func"), self.location);

        // Add function attributes - create proper null-terminated string
        const func_name_z2 = try std.fmt.allocPrintZ(self.allocator, "{s}", .{func.name});
        defer self.allocator.free(func_name_z2);
        const host_name_attr = MLIR.mlirStringAttrGet(self.context, MLIR.mlirStringRefCreateFromCString(func_name_z2.ptr));
        const sym_name_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("sym_name"));
        const sym_name_named_attr = MLIR.mlirNamedAttributeGet(sym_name_id, host_name_attr);

        const func_type_attr = MLIR.mlirTypeAttrGet(host_func_type);
        const function_type_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("function_type"));
        const function_type_named_attr = MLIR.mlirNamedAttributeGet(function_type_id, func_type_attr);

        const attrs = [_]MLIR.MlirNamedAttribute{ sym_name_named_attr, function_type_named_attr };
        MLIR.mlirOperationStateAddAttributes(&host_func_state, attrs.len, &attrs[0]);

        // Add a region to the host function operation
        const host_func_region = MLIR.mlirRegionCreate();
        MLIR.mlirOperationStateAddOwnedRegions(&host_func_state, 1, &host_func_region);

        const host_func_op = MLIR.mlirOperationCreate(&host_func_state);
        MLIR.mlirBlockAppendOwnedOperation(module_block, host_func_op);

        // Create function body with simple return
        const host_func_body = MLIR.mlirOperationGetFirstRegion(host_func_op);
        const host_func_block = MLIR.mlirBlockCreate(0, null, null);
        MLIR.mlirRegionAppendOwnedBlock(host_func_body, host_func_block);

        // Add a simple return statement to the host function block
        if (func.return_type == .void) {
            // Add func.return with no operands for void functions
            var return_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("func.return"), self.location);

            const return_op = MLIR.mlirOperationCreate(&return_state);
            MLIR.mlirBlockAppendOwnedOperation(host_func_block, return_op);
        } else {
            const return_value = switch (func.return_type) {
                .i32 => blk: {
                    const i32_type = MLIR.mlirIntegerTypeGet(self.context, 32);
                    const const_attr = MLIR.mlirIntegerAttrGet(i32_type, 0);

                    var const_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("arith.constant"), self.location);
                    MLIR.mlirOperationStateAddResults(&const_state, 1, &i32_type);

                    const value_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("value"));
                    const value_named_attr = MLIR.mlirNamedAttributeGet(value_id, const_attr);
                    MLIR.mlirOperationStateAddAttributes(&const_state, 1, &value_named_attr);

                    const const_op = MLIR.mlirOperationCreate(&const_state);
                    MLIR.mlirBlockAppendOwnedOperation(host_func_block, const_op);

                    break :blk MLIR.mlirOperationGetResult(const_op, 0);
                },
                else => blk: {
                    // Default to i32 for other types for now
                    const i32_type = MLIR.mlirIntegerTypeGet(self.context, 32);
                    const const_attr = MLIR.mlirIntegerAttrGet(i32_type, 0);

                    var const_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("arith.constant"), self.location);
                    MLIR.mlirOperationStateAddResults(&const_state, 1, &i32_type);

                    const value_id = MLIR.mlirIdentifierGet(self.context, MLIR.mlirStringRefCreateFromCString("value"));
                    const value_named_attr = MLIR.mlirNamedAttributeGet(value_id, const_attr);
                    MLIR.mlirOperationStateAddAttributes(&const_state, 1, &value_named_attr);

                    const const_op = MLIR.mlirOperationCreate(&const_state);
                    MLIR.mlirBlockAppendOwnedOperation(host_func_block, const_op);

                    break :blk MLIR.mlirOperationGetResult(const_op, 0);
                },
            };

            // Add func.return
            var return_state = MLIR.mlirOperationStateGet(MLIR.mlirStringRefCreateFromCString("func.return"), self.location);
            MLIR.mlirOperationStateAddOperands(&return_state, 1, &return_value);

            const return_op = MLIR.mlirOperationCreate(&return_state);
            MLIR.mlirBlockAppendOwnedOperation(host_func_block, return_op);
        }

        if (self.verbose) {
            std.debug.print("Generated simple host function with return statement\n", .{});
        }
    }

    /// Convert toy types to MLIR memref types (for GPU kernels)
    fn convertTypeToMemRef(self: *MLIRCodeGen, ty: parser.Type) MLIRCodeGenError!MLIR.MlirType {
        return switch (ty) {
            .tensor => |tensor_type| {
                const element_type = try self.convertType(tensor_type.element_type.*);
                const shape = try self.allocator.alloc(i64, tensor_type.shape.len);
                defer self.allocator.free(shape);

                for (tensor_type.shape, 0..) |dim, i| {
                    shape[i] = @intCast(dim);
                }

                // Convert tensor to memref for GPU operations
                return MLIR.mlirMemRefTypeGet(element_type, @intCast(shape.len), shape.ptr, MLIR.mlirAttributeGetNull(), MLIR.mlirAttributeGetNull());
            },
            else => try self.convertType(ty), // Fall back to regular conversion
        };
    }

    /// Convert toy types to MLIR types
    fn convertType(self: *MLIRCodeGen, ty: parser.Type) MLIRCodeGenError!MLIR.MlirType {
        return switch (ty) {
            .i32 => MLIR.mlirIntegerTypeGet(self.context, 32),
            .i64 => MLIR.mlirIntegerTypeGet(self.context, 64),
            .f32 => MLIR.mlirF32TypeGet(self.context),
            .f64 => MLIR.mlirF64TypeGet(self.context),
            .void => MLIR.mlirNoneTypeGet(self.context),
            .tensor => |tensor_type| {
                const element_type = try self.convertType(tensor_type.element_type.*);
                const shape = try self.allocator.alloc(i64, tensor_type.shape.len);
                defer self.allocator.free(shape);

                for (tensor_type.shape, 0..) |dim, i| {
                    shape[i] = @intCast(dim);
                }

                return MLIR.mlirRankedTensorTypeGet(@intCast(shape.len), shape.ptr, element_type, MLIR.mlirAttributeGetNull());
            },
            else => error.UnsupportedOperation,
        };
    }
};

// ============================================================================
// UNIT TESTS
// ============================================================================

test "MLIRCodeGen - basic initialization and cleanup" {
    const allocator = std.testing.allocator;

    var mlir_codegen = try MLIRCodeGen.init(allocator, "test_module", false);
    defer mlir_codegen.deinit();

    // Should initialize without crashing
    try std.testing.expect(true);
}

test "MLIRCodeGen - type conversion" {
    const allocator = std.testing.allocator;

    var mlir_codegen = try MLIRCodeGen.init(allocator, "test_types", false);
    defer mlir_codegen.deinit();

    // Test basic type conversions
    const i32_type = try mlir_codegen.convertType(.i32);
    const f32_type = try mlir_codegen.convertType(.f32);

    // Should not crash - MLIR types are opaque pointers
    _ = i32_type;
    _ = f32_type;
}
