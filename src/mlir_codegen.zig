const std = @import("std");
const parser = @import("parser.zig");

// MLIR C API bindings
const MLIR = @cImport({
    @cInclude("mlir-c/Support.h"); // Must come first - defines MlirStringRef
    @cInclude("mlir-c/IR.h");
    @cInclude("mlir-c/BuiltinTypes.h");
    @cInclude("mlir-c/BuiltinAttributes.h");
    @cInclude("mlir-c/Dialect/Func.h");
    @cInclude("mlir-c/Dialect/GPU.h");
    @cInclude("mlir-c/Dialect/Arith.h");
    @cInclude("mlir-c/Dialect/MemRef.h");
    @cInclude("mlir-c/Dialect/SCF.h");
    @cInclude("mlir-c/Dialect/ControlFlow.h");
    @cInclude("mlir-c/Pass.h");
    @cInclude("mlir-c/Conversion.h");
    @cInclude("mlir-c/Transforms.h");
    @cInclude("mlir-c/RegisterEverything.h"); // For mlirRegisterAllDialects
    @cInclude("mlir-c/Target/LLVMIR.h"); // For mlirTranslateModuleToLLVMIR
    @cInclude("llvm-c/Core.h"); // For LLVM IR manipulation
    @cInclude("llvm-c/IRReader.h"); // For LLVM IR parsing
    @cInclude("llvm-c/Target.h"); // For LLVM target machine
    @cInclude("llvm-c/TargetMachine.h"); // For LLVM target machine creation
    @cInclude("gpu_to_nvvm_wrapper.h"); // Our custom wrapper for GPU to NVVM with options
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
    PipelineError,
} || std.mem.Allocator.Error;

pub const MLIRCodeGen = struct {
    allocator: std.mem.Allocator,
    verbose: bool,
    generated_mlir: std.ArrayList(u8),
    sm_version: u32,

    pub fn init(allocator: std.mem.Allocator, sm_version: u32, verbose: bool) !MLIRCodeGen {
        return MLIRCodeGen{
            .allocator = allocator,
            .verbose = verbose,
            .generated_mlir = std.ArrayList(u8).init(allocator),
            .sm_version = sm_version,
        };
    }

    pub fn deinit(self: *MLIRCodeGen) void {
        self.generated_mlir.deinit();
    }

    /// Print the MLIR module (for debugging)
    pub fn printMLIR(self: *MLIRCodeGen) void {
        if (self.verbose) {
            std.debug.print("=== MLIR Module ===\n", .{});
            std.debug.print("{s}", .{self.generated_mlir.items});
            std.debug.print("\n=== End MLIR ===\n", .{});
        }
    }

    /// Main entry point for generating GPU functions
    pub fn generateGpuFunction(self: *MLIRCodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) MLIRCodeGenError!void {
        if (self.verbose) {
            std.debug.print("🔧 Generating GPU function: {s}\n", .{func.name});
        }

        // Generate the exact MLIR structure from simple_vector_add_gpu.mlir
        try self.generateGpuModuleMLIR(func);

        if (self.verbose) {
            std.debug.print("✅ Successfully generated GPU MLIR for function: {s}\n", .{func.name});
        }
    }

    /// Generate the GPU module MLIR based on the actual function declaration
    fn generateGpuModuleMLIR(self: *MLIRCodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) MLIRCodeGenError!void {
        const writer = self.generated_mlir.writer();

        // Analyze function parameters to determine dimensions and types
        const param_info = try self.analyzeParameters(func.parameters);
        defer self.allocator.free(param_info);

        // Analyze function body to determine the operation
        const operation_info = try self.analyzeOperation(func.body);
        defer self.allocator.free(operation_info.source_params);

        // Generate MLIR based on the analyzed information
        try writer.writeAll("gpu.module @kernels {\n");
        
        // Create the function signature based on actual parameters
        try writer.print("  gpu.func @{s}(", .{func.name});
        for (param_info, 0..) |param, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("%arg{d}: memref<{d}x{s}>", .{i, param.dimension, param.mlir_type});
        }
        try writer.writeAll(") kernel {\n");

        // Generate constants based on the actual dimensions
        try writer.writeAll("    %c0 = arith.constant 0 : index\n");
        if (param_info.len > 0) {
            try writer.print("    %c{d} = arith.constant {d} : index\n", .{param_info[0].dimension, param_info[0].dimension});
        }
        try writer.writeAll("    \n");

        // Generate GPU thread indexing code
        try writer.writeAll("    // Calculate global thread index: blockIdx.x * blockDim.x + threadIdx.x\n");
        try writer.writeAll("    %block_id = gpu.block_id x\n");
        try writer.writeAll("    %block_dim = gpu.block_dim x\n");
        try writer.writeAll("    %thread_id = gpu.thread_id x\n");
        try writer.writeAll("    %block_offset = arith.muli %block_id, %block_dim : index\n");
        try writer.writeAll("    %global_id = arith.addi %block_offset, %thread_id : index\n");
        try writer.writeAll("    \n");

        // Generate bounds check
        if (param_info.len > 0) {
            try writer.print("    // Bounds check: if (global_id >= {d}) return\n", .{param_info[0].dimension});
            try writer.print("    %cond = arith.cmpi ult, %global_id, %c{d} : index\n", .{param_info[0].dimension});
        }
        try writer.writeAll("    scf.if %cond {\n");

        // Generate the actual operation based on function body analysis
        try self.generateOperationMLIR(writer, param_info, operation_info);

        try writer.writeAll("    }\n");
        try writer.writeAll("    gpu.return\n");
        try writer.writeAll("  }\n");
        try writer.writeAll("} \n");

        // Store the generated MLIR for debugging
        if (self.verbose) {
            const generated_mlir_content = self.generated_mlir.items;
            std.fs.cwd().writeFile(.{ .sub_path = "generated_mlir.mlir", .data = generated_mlir_content }) catch |err| {
                std.debug.print("Warning: Could not save generated MLIR to file: {}\n", .{err});
            };
            std.debug.print("📄 Generated MLIR saved to generated_mlir.mlir\n", .{});
        }
    }

    /// Parameter information for MLIR generation
    const ParameterInfo = struct {
        dimension: u32,
        mlir_type: []const u8,
        element_type: parser.Type,
    };

    /// Operation information extracted from function body
    const OperationInfo = struct {
        operation: parser.BinaryOperator,
        target_param: usize,
        source_params: []usize,
    };

    /// Analyze function parameters to extract dimension and type information
    fn analyzeParameters(self: *MLIRCodeGen, parameters: []parser.Parameter) ![]ParameterInfo {
        var param_info = try self.allocator.alloc(ParameterInfo, parameters.len);
        
        for (parameters, 0..) |param, i| {
            switch (param.type) {
                .tensor => |tensor_type| {
                    // Extract dimension (assume 1D tensor for now)
                    const dimension = if (tensor_type.shape.len > 0) tensor_type.shape[0] else 1024;
                    
                    // Convert element type to MLIR type
                    const mlir_type = switch (tensor_type.element_type.*) {
                        .i32 => "i32",
                        .i64 => "i64",
                        .f32 => "f32",
                        .f64 => "f64",
                        .u32 => "i32", // MLIR uses signed types
                        .u64 => "i64", // MLIR uses signed types
                        else => "f32", // Default fallback
                    };
                    
                    param_info[i] = ParameterInfo{
                        .dimension = dimension,
                        .mlir_type = mlir_type,
                        .element_type = tensor_type.element_type.*,
                    };
                },
                else => {
                    // Default for non-tensor types
                    param_info[i] = ParameterInfo{
                        .dimension = 1024,
                        .mlir_type = "f32",
                        .element_type = parser.Type.f32,
                    };
                }
            }
        }
        
        return param_info;
    }

    /// Analyze function body to determine the operation being performed
    fn analyzeOperation(self: *MLIRCodeGen, body: []parser.ASTNode) !OperationInfo {
        // Look for parallel assignment pattern: a[i] = a[i] + b[i]
        for (body) |stmt| {
            if (stmt == .parallel_assignment) {
                const pa = stmt.parallel_assignment;
                
                // Check if the value is a binary expression
                if (pa.value.* == .binary_expression) {
                    const bin_expr = pa.value.*.binary_expression;
                    
                    // Create source params slice
                    const source_params = try self.allocator.alloc(usize, 2);
                    source_params[0] = 0;
                    source_params[1] = 1;
                    
                    // For now, assume the first parameter is the target
                    // and extract the operation type
                    return OperationInfo{
                        .operation = bin_expr.operator,
                        .target_param = 0,
                        .source_params = source_params,
                    };
                }
            }
        }
        
        // Default to addition if no clear pattern found
        const default_source_params = try self.allocator.alloc(usize, 2);
        default_source_params[0] = 0;
        default_source_params[1] = 1;
        
        return OperationInfo{
            .operation = parser.BinaryOperator.add,
            .target_param = 0,
            .source_params = default_source_params,
        };
    }

    /// Generate MLIR code for the specific operation
    fn generateOperationMLIR(self: *MLIRCodeGen, writer: anytype, param_info: []ParameterInfo, operation_info: OperationInfo) !void {
        _ = self; // Unused parameter for now
        // Generate load operations for source parameters
        for (operation_info.source_params, 0..) |param_idx, i| {
            if (param_idx < param_info.len) {
                try writer.print("      %val{d} = memref.load %arg{d}[%global_id] : memref<{d}x{s}>\n", 
                    .{i + 1, param_idx, param_info[param_idx].dimension, param_info[param_idx].mlir_type});
            }
        }
        try writer.writeAll("      \n");

        // Generate the operation based on the binary operator
        const op_name = switch (operation_info.operation) {
            .add => if (std.mem.eql(u8, param_info[0].mlir_type, "f32") or std.mem.eql(u8, param_info[0].mlir_type, "f64")) "arith.addf" else "arith.addi",
            .subtract => if (std.mem.eql(u8, param_info[0].mlir_type, "f32") or std.mem.eql(u8, param_info[0].mlir_type, "f64")) "arith.subf" else "arith.subi",
            .multiply => if (std.mem.eql(u8, param_info[0].mlir_type, "f32") or std.mem.eql(u8, param_info[0].mlir_type, "f64")) "arith.mulf" else "arith.muli",
            .divide => if (std.mem.eql(u8, param_info[0].mlir_type, "f32") or std.mem.eql(u8, param_info[0].mlir_type, "f64")) "arith.divf" else "arith.divsi",
        };

        const op_comment = switch (operation_info.operation) {
            .add => "addition",
            .subtract => "subtraction", 
            .multiply => "multiplication",
            .divide => "division",
        };

        try writer.print("      // Perform {s}: result = a[i] {s} b[i]\n", .{op_comment, switch (operation_info.operation) {
            .add => "+",
            .subtract => "-", 
            .multiply => "*",
            .divide => "/",
        }});

        // Generate the operation instruction
        if (operation_info.source_params.len >= 2) {
            try writer.print("      %result = {s} %val1, %val2 : {s}\n", .{op_name, param_info[0].mlir_type});
        } else {
            // Fallback for single operand (shouldn't happen in normal cases)
            try writer.print("      %result = {s} %val1, %val1 : {s}\n", .{op_name, param_info[0].mlir_type});
        }
        try writer.writeAll("      \n");

        // Generate store operation (in-place modification)
        const target_param = operation_info.target_param;
        if (target_param < param_info.len) {
            try writer.writeAll("      // Store result back to a[i] (in-place)\n");
            try writer.print("      memref.store %result, %arg{d}[%global_id] : memref<{d}x{s}>\n", 
                .{target_param, param_info[target_param].dimension, param_info[target_param].mlir_type});
        }
    }

    /// Lower MLIR to PTX using the integrated pipeline
    pub fn lowerMLIRToPTX(self: *MLIRCodeGen, function_name: []const u8) ![]const u8 {
        if (self.verbose) {
            std.debug.print("🎯 Starting integrated PTX generation pipeline...\n", .{});
        }

        const mlir_content = self.generated_mlir.items;

        // Step 1: Apply MLIR passes to the input MLIR content
        const transformed_mlir = try self.canonicalizeMLIRContent(mlir_content);
        defer self.allocator.free(transformed_mlir);

        // Step 2: Extract kernel function for standalone compilation
        if (self.verbose) std.debug.print("🔧 Extracting kernel function...\n", .{});
        const standalone_kernel = try self.extractKernelFunction(transformed_mlir, function_name);
        defer self.allocator.free(standalone_kernel);

        // Step 3: Fix NVVM operations for mlir-translate
        if (self.verbose) std.debug.print("🔧 Fixing NVVM operations...\n", .{});
        const fixed_mlir = try self.fixNVVMOperations(standalone_kernel);
        defer self.allocator.free(fixed_mlir);

        // Step 4: Translate MLIR to LLVM IR using MLIR C API (replaces external mlir-translate)
        const llvm_ir_content = try self.translateMLIRToLLVMIR(fixed_mlir);
        defer self.allocator.free(llvm_ir_content);

        // Step 5: Compile LLVM IR to PTX using LLVM C API (replaces external llc)
        const ptx_content = try self.compileLLVMIRToPTX(llvm_ir_content);

        if (self.verbose) std.debug.print("✅ PTX generation complete\n", .{});
        return ptx_content;
    }

    /// Run integrated MLIR passes on MLIR content
    fn canonicalizeMLIRContent(self: *MLIRCodeGen, input_content: []const u8) ![]const u8 {
        if (self.verbose) {
            std.debug.print("🔧 Steps 1-7: Integrated MLIR passes using C API (part of fully integrated pipeline)\n", .{});
        }

        // Create null-terminated version for MLIR C API
        const null_terminated_content = try self.allocator.allocSentinel(u8, input_content.len, 0);
        defer self.allocator.free(null_terminated_content);
        @memcpy(null_terminated_content, input_content);

        // Use bulk registration for better compatibility
        const registry = MLIR.mlirDialectRegistryCreate();
        if (self.verbose) {
            std.debug.print("✅ Registry created\n", .{});
        }
        defer MLIR.mlirDialectRegistryDestroy(registry);

        // Register only the specific dialects we need
        MLIR.mlirDialectHandleInsertDialect(MLIR.mlirGetDialectHandle__func__(), registry);
        MLIR.mlirDialectHandleInsertDialect(MLIR.mlirGetDialectHandle__gpu__(), registry);
        MLIR.mlirDialectHandleInsertDialect(MLIR.mlirGetDialectHandle__arith__(), registry);
        MLIR.mlirDialectHandleInsertDialect(MLIR.mlirGetDialectHandle__memref__(), registry);
        MLIR.mlirDialectHandleInsertDialect(MLIR.mlirGetDialectHandle__scf__(), registry);

        if (self.verbose) {
            std.debug.print("✅ Registered specific dialects\n", .{});
        }

        // Register transform passes we need
        MLIR.mlirRegisterTransformsCanonicalizer();
        MLIR.mlirRegisterAllPasses();

        if (self.verbose) {
            std.debug.print("✅ Registered canonicalizer pass\n", .{});
        }

        // Initialize MLIR context with all dialects and passes registered
        const context = MLIR.mlirContextCreateWithRegistry(registry, false);
        defer MLIR.mlirContextDestroy(context);

        // Register LLVM translations for mlirTranslateModuleToLLVMIR
        MLIR.mlirRegisterAllLLVMTranslations(context);

        if (self.verbose) {
            std.debug.print("✅ Created MLIR context with pre-registered dialects\n", .{});
        }

        // Create pass manager for builtin.module operations
        const pass_manager = MLIR.mlirPassManagerCreate(context);
        defer MLIR.mlirPassManagerDestroy(pass_manager);

        // Add passes to the pipeline
        const canonicalize_pass = MLIR.mlirCreateTransformsCanonicalizer();
        MLIR.mlirPassManagerAddOwnedPass(pass_manager, canonicalize_pass);
        const kernel_pass = MLIR.mlirCreateGPUGpuKernelOutlining();
        MLIR.mlirPassManagerAddOwnedPass(pass_manager, kernel_pass);
        const scf_to_cf_pass = MLIR.mlirCreateConversionSCFToControlFlow();
        MLIR.mlirPassManagerAddOwnedPass(pass_manager, scf_to_cf_pass);

        // Get the default operation pass manager and create nested GPU module pass manager
        const default_op_pm = MLIR.mlirPassManagerGetAsOpPassManager(pass_manager);
        const gpu_module_pm = MLIR.mlirOpPassManagerGetNestedUnder(default_op_pm, MLIR.mlirStringRefCreateFromCString("gpu.module"));
        // Use our custom wrapper that supports bare pointer call convention
        const gpu_to_nvvm_pass = MLIR.mlirCreateConversionConvertGpuOpsToNVVMOpsWithBarePtr();
        MLIR.mlirOpPassManagerAddOwnedPass(gpu_module_pm, gpu_to_nvvm_pass);

        // Add NVVM to LLVM conversion pass to the main module pass manager
        const nvvm_to_llvm_pass = MLIR.mlirCreateConversionConvertNVVMToLLVMPass();
        MLIR.mlirPassManagerAddOwnedPass(pass_manager, nvvm_to_llvm_pass);

        // Add finalize MemRef to LLVM conversion pass
        const finalize_memref_pass = MLIR.mlirCreateConversionFinalizeMemRefToLLVMConversionPass();
        MLIR.mlirPassManagerAddOwnedPass(pass_manager, finalize_memref_pass);

        // Add convert Func to LLVM pass with bare pointer call convention
        const func_to_llvm_pass = MLIR.mlirCreateConversionConvertFuncToLLVMPassWithBarePtr();
        MLIR.mlirPassManagerAddOwnedPass(pass_manager, func_to_llvm_pass);

        // Add reconcile unrealized casts pass
        const reconcile_casts_pass = MLIR.mlirCreateConversionReconcileUnrealizedCasts();
        MLIR.mlirPassManagerAddOwnedPass(pass_manager, reconcile_casts_pass);

        if (self.verbose) {
            std.debug.print("✅ Created pass manager with integrated passes\n", .{});
        }

        // Parse the actual input MLIR file
        const input_str_ref = MLIR.mlirStringRefCreateFromCString(null_terminated_content);
        const module = MLIR.mlirModuleCreateParse(context, input_str_ref);

        if (MLIR.mlirModuleIsNull(module)) {
            if (self.verbose) {
                std.debug.print("❌ Failed to parse MLIR - using fallback\n", .{});
            }
            // Fallback: just return input content (no transformation)
            return try self.allocator.dupe(u8, input_content);
        }
        defer MLIR.mlirModuleDestroy(module);

        if (self.verbose) {
            std.debug.print("✅ Successfully parsed input MLIR file\n", .{});
        }

        // Get module operation and run passes
        const module_op = MLIR.mlirModuleGetOperation(module);
        _ = MLIR.mlirPassManagerRunOnOp(pass_manager, module_op);

        // Capture MLIR operation dump
        const temp_file_path = "temp_mlir_dump.txt";

        // Save current stderr and redirect to temp file
        const original_stderr = std.posix.dup(std.posix.STDERR_FILENO) catch |err| {
            std.debug.print("Error duplicating stderr: {}\n", .{err});
            return MLIRCodeGenError.PipelineError;
        };
        defer std.posix.close(original_stderr);

        const temp_file = std.fs.cwd().createFile(temp_file_path, .{}) catch |err| {
            std.debug.print("Error creating temp file: {}\n", .{err});
            return MLIRCodeGenError.PipelineError;
        };
        defer std.fs.cwd().deleteFile(temp_file_path) catch {};

        const temp_fd = temp_file.handle;
        std.posix.dup2(temp_fd, std.posix.STDERR_FILENO) catch |err| {
            std.debug.print("Error redirecting stderr: {}\n", .{err});
            return MLIRCodeGenError.PipelineError;
        };

        // Dump the MLIR
        MLIR.mlirOperationDump(module_op);

        // Restore original stderr
        std.posix.dup2(original_stderr, std.posix.STDERR_FILENO) catch |err| {
            std.debug.print("Error restoring stderr: {}\n", .{err});
            return MLIRCodeGenError.PipelineError;
        };

        temp_file.close();

        // Read the captured content
        const captured_mlir = std.fs.cwd().readFileAlloc(self.allocator, temp_file_path, 1024 * 1024) catch |err| {
            std.debug.print("Error reading captured MLIR: {}\n", .{err});
            return MLIRCodeGenError.PipelineError;
        };

        if (self.verbose) {
            std.debug.print("✅ Successfully captured MLIR dump\n", .{});
        }

        return captured_mlir;
    }

    /// Extract kernel function for standalone compilation
    fn extractKernelFunction(self: *MLIRCodeGen, input_content: []const u8, function_name: []const u8) ![]const u8 {
        // Create the kernel function name to search for (without _kernel suffix)
        const kernel_name = try std.fmt.allocPrint(self.allocator, "llvm.func @{s}", .{function_name});
        defer self.allocator.free(kernel_name);

        // Find the kernel function and extract it
        const kernel_start = std.mem.indexOf(u8, input_content, kernel_name) orelse {
            std.debug.print("Error: Could not find kernel function '{s}'\n", .{kernel_name});
            return MLIRCodeGenError.PipelineError;
        };

        const kernel_end = std.mem.indexOf(u8, input_content[kernel_start..], "  }") orelse {
            std.debug.print("Error: Could not find end of kernel function\n", .{});
            return MLIRCodeGenError.PipelineError;
        };

        // Extract the kernel function
        const kernel_function = input_content[kernel_start .. kernel_start + kernel_end + 3];

        // Remove the gpu.kernel attribute as it's not needed in standalone module
        const kernel_with_attrs = std.mem.replacementSize(u8, kernel_function, "gpu.kernel, ", "");
        const cleaned_kernel = try self.allocator.alloc(u8, kernel_with_attrs);
        defer self.allocator.free(cleaned_kernel);
        _ = std.mem.replace(u8, kernel_function, "gpu.kernel, ", "", cleaned_kernel);

        // Also remove if it's at the end
        const kernel_with_attrs2 = std.mem.replacementSize(u8, cleaned_kernel, ", gpu.kernel", "");
        const cleaned_kernel2 = try self.allocator.alloc(u8, kernel_with_attrs2);
        defer self.allocator.free(cleaned_kernel2);
        _ = std.mem.replace(u8, cleaned_kernel, ", gpu.kernel", "", cleaned_kernel2);

        // Final cleanup
        const final_kernel = try self.allocator.dupe(u8, cleaned_kernel2);

        // Create a standalone module with the kernel function
        const standalone_module = try std.fmt.allocPrint(self.allocator,
            \\module attributes {{nvvm.target = "cuda"}} {{
            \\  {s}
            \\  llvm.func @llvm.nvvm.read.ptx.sreg.tid.x() -> i32 attributes {{passthrough = ["nounwind", "readnone"]}}
            \\  llvm.func @llvm.nvvm.read.ptx.sreg.ntid.x() -> i32 attributes {{passthrough = ["nounwind", "readnone"]}}
            \\  llvm.func @llvm.nvvm.read.ptx.sreg.ctaid.x() -> i32 attributes {{passthrough = ["nounwind", "readnone"]}}
            \\}}
            \\
        , .{final_kernel});
        defer self.allocator.free(final_kernel);

        return standalone_module;
    }

    /// Fix NVVM operations for translation
    fn fixNVVMOperations(self: *MLIRCodeGen, input_content: []const u8) ![]const u8 {
        // Replace NVVM operations with LLVM intrinsic calls
        const content1 = std.mem.replaceOwned(u8, self.allocator, input_content, "nvvm.read.ptx.sreg.ctaid.x : i32", "llvm.call @llvm.nvvm.read.ptx.sreg.ctaid.x() : () -> i32") catch |err| {
            std.debug.print("Error replacing NVVM operations: {}\n", .{err});
            return MLIRCodeGenError.PipelineError;
        };
        defer self.allocator.free(content1);

        const content2 = std.mem.replaceOwned(u8, self.allocator, content1, "nvvm.read.ptx.sreg.ntid.x : i32", "llvm.call @llvm.nvvm.read.ptx.sreg.ntid.x() : () -> i32") catch |err| {
            std.debug.print("Error replacing NVVM operations: {}\n", .{err});
            return MLIRCodeGenError.PipelineError;
        };
        defer self.allocator.free(content2);

        const content3 = std.mem.replaceOwned(u8, self.allocator, content2, "nvvm.read.ptx.sreg.tid.x : i32", "llvm.call @llvm.nvvm.read.ptx.sreg.tid.x() : () -> i32") catch |err| {
            std.debug.print("Error replacing NVVM operations: {}\n", .{err});
            return MLIRCodeGenError.PipelineError;
        };
        defer self.allocator.free(content3);

        // Remove invalid function attributes and nvvm.target
        const content4 = std.mem.replaceOwned(u8, self.allocator, content3, "attributes {passthrough = [\"nounwind\", \"readnone\"]}", "") catch |err| {
            std.debug.print("Error removing invalid attributes: {}\n", .{err});
            return MLIRCodeGenError.PipelineError;
        };
        defer self.allocator.free(content4);

        const clean_content = std.mem.replaceOwned(u8, self.allocator, content4, "module attributes {nvvm.target = \"cuda\"} {", "module {") catch |err| {
            std.debug.print("Error removing nvvm.target: {}\n", .{err});
            return MLIRCodeGenError.PipelineError;
        };

        return clean_content;
    }

    /// Translate MLIR to LLVM IR using the MLIR C API
    fn translateMLIRToLLVMIR(self: *MLIRCodeGen, mlir_content: []const u8) ![]const u8 {
        if (self.verbose) {
            std.debug.print("🔄 Translating MLIR to LLVM IR using MLIR C API...\n", .{});
        }

        // Create null-terminated version for MLIR C API
        const null_terminated_content = try self.allocator.allocSentinel(u8, mlir_content.len, 0);
        defer self.allocator.free(null_terminated_content);
        @memcpy(null_terminated_content, mlir_content);

        // Initialize MLIR context
        const context = MLIR.mlirContextCreate();
        defer MLIR.mlirContextDestroy(context);

        // Register LLVM translations
        MLIR.mlirRegisterAllLLVMTranslations(context);

        if (self.verbose) {
            std.debug.print("✅ Created MLIR context for translation\n", .{});
        }

        // Parse the MLIR content
        const input_str_ref = MLIR.mlirStringRefCreateFromCString(null_terminated_content);
        const module = MLIR.mlirModuleCreateParse(context, input_str_ref);

        if (MLIR.mlirModuleIsNull(module)) {
            if (self.verbose) {
                std.debug.print("❌ Failed to parse MLIR for translation\n", .{});
            }
            return MLIRCodeGenError.PipelineError;
        }
        defer MLIR.mlirModuleDestroy(module);

        if (self.verbose) {
            std.debug.print("✅ Successfully parsed MLIR for translation\n", .{});
        }

        // Get module operation
        const module_op = MLIR.mlirModuleGetOperation(module);

        // Create LLVM context for translation
        const llvm_context = MLIR.LLVMContextCreate();
        defer MLIR.LLVMContextDispose(llvm_context);

        // Translate MLIR to LLVM IR
        const llvm_module = MLIR.mlirTranslateModuleToLLVMIR(module_op, llvm_context);
        if (llvm_module == null) {
            if (self.verbose) {
                std.debug.print("❌ Failed to translate MLIR to LLVM IR\n", .{});
            }
            return MLIRCodeGenError.PipelineError;
        }
        defer MLIR.LLVMDisposeModule(llvm_module);

        if (self.verbose) {
            std.debug.print("✅ Successfully translated MLIR to LLVM IR\n", .{});
        }

        // Convert LLVM module to string
        const llvm_ir_cstr = MLIR.LLVMPrintModuleToString(llvm_module);
        defer MLIR.LLVMDisposeMessage(llvm_ir_cstr);

        // Copy to our allocator-managed memory
        const llvm_ir_len = std.mem.len(llvm_ir_cstr);
        const llvm_ir_content = try self.allocator.alloc(u8, llvm_ir_len);
        @memcpy(llvm_ir_content, llvm_ir_cstr[0..llvm_ir_len]);

        if (self.verbose) {
            std.debug.print("✅ Converted LLVM module to string ({d} bytes)\n", .{llvm_ir_len});
        }

        return llvm_ir_content;
    }

    /// Compile LLVM IR to PTX using the LLVM C API
    fn compileLLVMIRToPTX(self: *MLIRCodeGen, llvm_ir_content: []const u8) ![]const u8 {
        if (self.verbose) {
            std.debug.print("🎯 Compiling LLVM IR to PTX using LLVM C API...\n", .{});
        }

        // Initialize NVPTX target
        MLIR.LLVMInitializeNVPTXTargetInfo();
        MLIR.LLVMInitializeNVPTXTarget();
        MLIR.LLVMInitializeNVPTXTargetMC();
        MLIR.LLVMInitializeNVPTXAsmPrinter();

        if (self.verbose) {
            std.debug.print("✅ Initialized NVPTX target\n", .{});
        }

        // Create LLVM context and parse the IR using memory buffer
        const llvm_context = MLIR.LLVMContextCreate();
        defer MLIR.LLVMContextDispose(llvm_context);

        // Create memory buffer copy
        const memory_buffer = MLIR.LLVMCreateMemoryBufferWithMemoryRangeCopy(llvm_ir_content.ptr, llvm_ir_content.len, "llvm_ir");

        // Parse LLVM IR from memory buffer
        var llvm_module: MLIR.LLVMModuleRef = undefined;
        var error_msg: [*c]u8 = undefined;

        if (MLIR.LLVMParseIRInContext(llvm_context, memory_buffer, &llvm_module, &error_msg) != 0) {
            defer MLIR.LLVMDisposeMessage(error_msg);
            if (self.verbose) {
                std.debug.print("❌ Failed to parse LLVM IR: {s}\n", .{error_msg});
            }
            return MLIRCodeGenError.PipelineError;
        }
        defer MLIR.LLVMDisposeModule(llvm_module);

        if (self.verbose) {
            std.debug.print("✅ Successfully parsed LLVM IR\n", .{});
        }

        // Get NVPTX target
        const target_triple = "nvptx64-nvidia-cuda";
        var target: MLIR.LLVMTargetRef = undefined;
        var target_error_msg: [*c]u8 = undefined;

        if (MLIR.LLVMGetTargetFromTriple(target_triple, &target, &target_error_msg) != 0) {
            defer MLIR.LLVMDisposeMessage(target_error_msg);
            if (self.verbose) {
                std.debug.print("❌ Failed to get NVPTX target: {s}\n", .{target_error_msg});
            }
            return MLIRCodeGenError.PipelineError;
        }

        if (self.verbose) {
            std.debug.print("✅ Got NVPTX target\n", .{});
        }

        // Create target machine
        const cpu_str = try std.fmt.allocPrintZ(self.allocator, "sm_{d}", .{self.sm_version});
        defer self.allocator.free(cpu_str);

        const target_machine = MLIR.LLVMCreateTargetMachine(target, target_triple, cpu_str.ptr, "", // features
            MLIR.LLVMCodeGenLevelDefault, MLIR.LLVMRelocDefault, MLIR.LLVMCodeModelDefault);
        defer MLIR.LLVMDisposeTargetMachine(target_machine);

        if (self.verbose) {
            std.debug.print("✅ Created target machine for SM {d}\n", .{self.sm_version});
        }

        // Create memory buffer for PTX output
        var ptx_memory_buffer: MLIR.LLVMMemoryBufferRef = undefined;
        var emit_error_msg: [*c]u8 = undefined;

        if (MLIR.LLVMTargetMachineEmitToMemoryBuffer(target_machine, llvm_module, MLIR.LLVMAssemblyFile, &emit_error_msg, &ptx_memory_buffer) != 0) {
            defer MLIR.LLVMDisposeMessage(emit_error_msg);
            if (self.verbose) {
                std.debug.print("❌ Failed to emit PTX: {s}\n", .{emit_error_msg});
            }
            return MLIRCodeGenError.PipelineError;
        }
        defer MLIR.LLVMDisposeMemoryBuffer(ptx_memory_buffer);

        // Get PTX content from memory buffer
        const ptx_data = MLIR.LLVMGetBufferStart(ptx_memory_buffer);
        const ptx_size = MLIR.LLVMGetBufferSize(ptx_memory_buffer);

        // Copy PTX to our allocator-managed memory
        const ptx_content = try self.allocator.alloc(u8, ptx_size);
        @memcpy(ptx_content, ptx_data[0..ptx_size]);

        if (self.verbose) {
            std.debug.print("✅ Successfully compiled LLVM IR to PTX ({d} bytes)\n", .{ptx_size});
        }

        return ptx_content;
    }
};

// ============================================================================
// UNIT TESTS
// ============================================================================

test "MLIRCodeGen - basic initialization and cleanup" {
    const allocator = std.testing.allocator;

    var mlir_codegen = try MLIRCodeGen.init(allocator, 50, false);
    defer mlir_codegen.deinit();

    // Should initialize without crashing
    try std.testing.expect(true);
}

test "MLIRCodeGen - generate GPU function" {
    const allocator = std.testing.allocator;

    var mlir_codegen = try MLIRCodeGen.init(allocator, 50, false);
    defer mlir_codegen.deinit();

    // Create a dummy function declaration for testing
    const func_decl = @TypeOf(@as(parser.ASTNode, undefined).function_declaration){
        .offset = 0,
        .name = "test_func",
        .parameters = &[_]parser.Parameter{},
        .return_type = .void,
        .body = &[_]parser.ASTNode{},
    };

    // Should generate without crashing
    try mlir_codegen.generateGpuFunction(func_decl);

    // Check that some MLIR was generated
    try std.testing.expect(mlir_codegen.generated_mlir.items.len > 0);
}
