const std = @import("std");
const parser = @import("../parser.zig");

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

/// Extract SM version from std.Target for CUDA/NVPTX targets
/// Returns the SM version as u32, or a default value if not NVPTX or parsing fails
fn extractSmVersionFromTarget(target: std.Target) !u32 {
    // Check if this is an NVPTX target
    if (!target.cpu.arch.isNvptx()) {
        return 50; // Default SM version for non-NVPTX targets
    }

    // Parse the CPU model name to extract SM version
    // NVPTX CPU model names are typically like "sm_50", "sm_52", etc.
    const model_name = target.cpu.model.name;

    if (std.mem.startsWith(u8, model_name, "sm_")) {
        const version_str = model_name[3..]; // Skip "sm_" prefix
        if (std.fmt.parseInt(u32, version_str, 10)) |version| {
            return version;
        } else |_| {
            // If parsing fails, return a reasonable default
            return 50;
        }
    }
    std.debug.panic("Cannot detect cuda sm version on {}", .{target});
    return error.SMNotFound;
}

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
    target: std.Target,

    pub fn init(allocator: std.mem.Allocator, target: std.Target, verbose: bool) !MLIRCodeGen {
        return MLIRCodeGen{
            .allocator = allocator,
            .verbose = verbose,
            .generated_mlir = std.ArrayList(u8).init(allocator),
            .target = target,
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
        try self.generateGpuFunctionMLIR(func);

        if (self.verbose) {
            std.debug.print("✅ Successfully generated GPU MLIR for function: {s}\n", .{func.name});
        }
    }

    /// Generate multiple GPU functions in a single MLIR module
    pub fn generateGpuModule(self: *MLIRCodeGen, functions: []@TypeOf(@as(parser.ASTNode, undefined).function_declaration)) MLIRCodeGenError!void {
        if (self.verbose) {
            std.debug.print("🔧 Generating GPU module with {} functions\n", .{functions.len});
        }

        const writer = self.generated_mlir.writer();

        // Start the GPU module
        try writer.writeAll("gpu.module @kernels {\n");

        // Generate all GPU functions within the same module
        for (functions) |func| {
            if (self.verbose) {
                std.debug.print("🔧 Adding GPU function to module: {s}\n", .{func.name});
            }
            try self.generateSingleGpuFunction(writer, func);
        }

        // Close the GPU module
        try writer.writeAll("}\n");

        // Store the generated MLIR for debugging
        if (self.verbose) {
            const generated_mlir_content = self.generated_mlir.items;
            std.fs.cwd().writeFile(.{ .sub_path = "generated_mlir.mlir", .data = generated_mlir_content }) catch |err| {
                std.debug.print("Warning: Could not save generated MLIR to file: {}\n", .{err});
            };
            std.debug.print("📄 Generated MLIR saved to generated_mlir.mlir\n", .{});
        }

        if (self.verbose) {
            std.debug.print("✅ Successfully generated GPU module with {} functions\n", .{functions.len});
        }
    }

    /// Generate the GPU function MLIR (with module wrapper for single function)
    fn generateGpuFunctionMLIR(self: *MLIRCodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) MLIRCodeGenError!void {
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
            // Handle multi-dimensional tensors
            if (param.tensor_shape) |shape| {
                if (shape.len > 1) {
                    // Multi-dimensional tensor: generate proper multi-dimensional memref
                    try writer.print("%arg{d}: memref<", .{i});
                    for (shape, 0..) |dim, j| {
                        if (j > 0) try writer.writeAll("x");
                        try writer.print("{d}", .{dim});
                    }
                    try writer.print("x{s}>", .{param.mlir_type});
                } else {
                    // 1D tensor
                    try writer.print("%arg{d}: memref<{d}x{s}>", .{ i, param.dimension, param.mlir_type });
                }
            } else {
                // Default case
                try writer.print("%arg{d}: memref<{d}x{s}>", .{ i, param.dimension, param.mlir_type });
            }
        }
        try writer.writeAll(") kernel {\n");

        // Generate constants based on the actual dimensions
        try writer.writeAll("    %c0 = arith.constant 0 : index\n");

        // For reduce operations, generate constant for target dimension
        const bounds_dim = if (operation_info.is_reduce and operation_info.target_param < param_info.len)
            param_info[operation_info.target_param].dimension
        else if (param_info.len > 0)
            param_info[0].dimension
        else
            1;

        try writer.print("    %c{d} = arith.constant {d} : index\n", .{ bounds_dim, bounds_dim });
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
        try writer.print("    // Bounds check: if (global_id >= {d}) return\n", .{bounds_dim});
        try writer.print("    %cond = arith.cmpi ult, %global_id, %c{d} : index\n", .{bounds_dim});
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

    /// Generate a single GPU function inside an existing module
    fn generateSingleGpuFunction(self: *MLIRCodeGen, writer: anytype, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) MLIRCodeGenError!void {
        // Analyze function parameters to determine dimensions and types
        const param_info = try self.analyzeParameters(func.parameters);
        defer self.allocator.free(param_info);

        // Analyze function body to determine the operation
        const operation_info = try self.analyzeOperation(func.body);
        defer self.allocator.free(operation_info.source_params);

        // Create the function signature based on actual parameters
        try writer.print("  gpu.func @{s}(", .{func.name});
        for (param_info, 0..) |param, i| {
            if (i > 0) try writer.writeAll(", ");
            // Handle multi-dimensional tensors
            if (param.tensor_shape) |shape| {
                if (shape.len > 1) {
                    // Multi-dimensional tensor: generate proper multi-dimensional memref
                    try writer.print("%arg{d}: memref<", .{i});
                    for (shape, 0..) |dim, j| {
                        if (j > 0) try writer.writeAll("x");
                        try writer.print("{d}", .{dim});
                    }
                    try writer.print("x{s}>", .{param.mlir_type});
                } else {
                    // 1D tensor
                    try writer.print("%arg{d}: memref<{d}x{s}>", .{ i, param.dimension, param.mlir_type });
                }
            } else {
                // Default case
                try writer.print("%arg{d}: memref<{d}x{s}>", .{ i, param.dimension, param.mlir_type });
            }
        }
        try writer.writeAll(") kernel {\n");

        // Generate constants based on the actual dimensions
        try writer.writeAll("    %c0 = arith.constant 0 : index\n");

        // For reduce operations, generate constant for target dimension
        const bounds_dim = if (operation_info.is_reduce and operation_info.target_param < param_info.len)
            param_info[operation_info.target_param].dimension
        else if (param_info.len > 0)
            param_info[0].dimension
        else
            1;

        try writer.print("    %c{d} = arith.constant {d} : index\n", .{ bounds_dim, bounds_dim });
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
        try writer.print("    // Bounds check: if (global_id >= {d}) return\n", .{bounds_dim});
        try writer.print("    %cond = arith.cmpi ult, %global_id, %c{d} : index\n", .{bounds_dim});
        try writer.writeAll("    scf.if %cond {\n");

        // Generate the actual operation based on function body analysis
        try self.generateOperationMLIR(writer, param_info, operation_info);

        try writer.writeAll("    }\n");
        try writer.writeAll("    gpu.return\n");
        try writer.writeAll("  }\n");
    }

    /// Parameter information for MLIR generation
    const ParameterInfo = struct {
        dimension: u32,
        mlir_type: []const u8,
        element_type: parser.Type,
        tensor_shape: ?[]const u32 = null, // For multi-dimensional tensors
    };

    /// Operation information extracted from function body
    const OperationInfo = struct {
        operation: parser.BinaryOperator,
        target_param: usize,
        source_params: []usize,
        is_reduce: bool = false,
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
                        .tensor_shape = tensor_type.shape,
                    };
                },
                else => {
                    // Default for non-tensor types
                    param_info[i] = ParameterInfo{
                        .dimension = 1024,
                        .mlir_type = "f32",
                        .element_type = parser.Type.f32,
                    };
                },
            }
        }

        return param_info;
    }

    /// Analyze function body to determine the operation being performed
    fn analyzeOperation(self: *MLIRCodeGen, body: []parser.ASTNode) !OperationInfo {
        // Look for parallel assignment pattern: a[i] = a[i] + b[i] or reduce pattern
        for (body) |stmt| {
            if (stmt == .parallel_assignment) {
                const pa = stmt.parallel_assignment;

                // Check if the value is a reduce expression
                if (pa.value.* == .reduce_expression) {
                    const reduce_expr = pa.value.*.reduce_expression;

                    // For reduce operations, we need to identify which parameter is being reduced
                    // For now, assume first parameter is source, second is target
                    const source_params = try self.allocator.alloc(usize, 1);
                    source_params[0] = 0; // Source tensor to reduce

                    return OperationInfo{
                        .operation = reduce_expr.operator,
                        .target_param = 1, // Target is second parameter
                        .source_params = source_params,
                        .is_reduce = true,
                    };
                }
                // Check if the value is a binary expression
                else if (pa.value.* == .binary_expression) {
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
                        .is_reduce = false,
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
            .is_reduce = false,
        };
    }

    /// Generate MLIR code for the specific operation
    fn generateOperationMLIR(self: *MLIRCodeGen, writer: anytype, param_info: []ParameterInfo, operation_info: OperationInfo) !void {

        // Early return if no parameters to work with
        if (param_info.len == 0) {
            try writer.writeAll("      // No parameters provided for operation\n");
            return;
        }

        if (operation_info.is_reduce) {
            // Handle reduce operations differently
            try self.generateReduceOperationMLIR(writer, param_info, operation_info);
        } else {
            // Original binary operation code
            // Generate load operations for source parameters
            for (operation_info.source_params, 0..) |param_idx, i| {
                if (param_idx < param_info.len) {
                    try writer.print("      %val{d} = memref.load %arg{d}[%global_id] : memref<{d}x{s}>\n", .{ i + 1, param_idx, param_info[param_idx].dimension, param_info[param_idx].mlir_type });
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

            try writer.print("      // Perform {s}: result = a[i] {s} b[i]\n", .{ op_comment, switch (operation_info.operation) {
                .add => "+",
                .subtract => "-",
                .multiply => "*",
                .divide => "/",
            } });

            // Generate the operation instruction
            if (operation_info.source_params.len >= 2) {
                try writer.print("      %result = {s} %val1, %val2 : {s}\n", .{ op_name, param_info[0].mlir_type });
            } else {
                // Fallback for single operand (shouldn't happen in normal cases)
                try writer.print("      %result = {s} %val1, %val1 : {s}\n", .{ op_name, param_info[0].mlir_type });
            }
            try writer.writeAll("      \n");

            // Generate store operation (in-place modification)
            const target_param = operation_info.target_param;
            if (target_param < param_info.len) {
                try writer.writeAll("      // Store result back to a[i] (in-place)\n");
                try writer.print("      memref.store %result, %arg{d}[%global_id] : memref<{d}x{s}>\n", .{ target_param, param_info[target_param].dimension, param_info[target_param].mlir_type });
            }
        }
    }

    /// Generate MLIR code for reduce operations
    fn generateReduceOperationMLIR(self: *MLIRCodeGen, writer: anytype, param_info: []ParameterInfo, operation_info: OperationInfo) !void {
        _ = self;

        // For GPU reduce, we need to:
        // 1. Each thread handles one element of the output (dimension 0)
        // 2. Each thread loops over the reduction dimension (dimension 1)

        const source_param = operation_info.source_params[0];
        const target_param = operation_info.target_param;

        if (source_param >= param_info.len or target_param >= param_info.len) {
            try writer.writeAll("      // Invalid parameter indices for reduce\n");
            return;
        }

        const source_info = param_info[source_param];
        const target_info = param_info[target_param];
        const element_type = target_info.mlir_type;

        // For 2D to 1D reduction, source should be 2D tensor type
        // We need to extract the second dimension for the reduction loop
        const reduction_dim = if (source_info.tensor_shape) |shape|
            if (shape.len > 1) shape[1] else 1
        else
            1;

        try writer.writeAll("      // Reduce operation\n");

        // Initialize accumulator based on operation type
        const init_value = switch (operation_info.operation) {
            .add => if (std.mem.eql(u8, element_type, "f32")) "0.0" else if (std.mem.eql(u8, element_type, "f64")) "0.0" else "0",
            .multiply => if (std.mem.eql(u8, element_type, "f32")) "1.0" else if (std.mem.eql(u8, element_type, "f64")) "1.0" else "1",
            else => "0", // Default for unsupported operations
        };

        try writer.print("      %init = arith.constant {s} : {s}\n", .{ init_value, element_type });
        try writer.print("      %c1_reduce = arith.constant 1 : index\n", .{});
        try writer.print("      %c{d}_reduce = arith.constant {d} : index\n", .{ reduction_dim, reduction_dim });

        // Create the reduction loop
        try writer.writeAll("      %reduced = scf.for %j = %c0 to %c");
        try writer.print("{d}_reduce", .{reduction_dim});
        try writer.writeAll(" step %c1_reduce iter_args(%acc = %init) -> (");
        try writer.print("{s}", .{element_type});
        try writer.writeAll(") {\n");

        // Load element from multi-dimensional source tensor
        if (source_info.tensor_shape) |shape| {
            if (shape.len > 1) {
                // Multi-dimensional access
                try writer.print("        %elem = memref.load %arg{d}[%global_id, %j] : memref<", .{source_param});
                for (shape, 0..) |dim, k| {
                    if (k > 0) try writer.writeAll("x");
                    try writer.print("{d}", .{dim});
                }
                try writer.print("x{s}>\n", .{element_type});
            } else {
                // 1D access
                try writer.print("        %idx1 = arith.muli %global_id, %c{d} : index\n", .{reduction_dim});
                try writer.writeAll("        %idx = arith.addi %idx1, %j : index\n");
                try writer.print("        %elem = memref.load %arg{d}[%idx] : memref<{d}x{s}>\n", .{ source_param, source_info.dimension, element_type });
            }
        } else {
            // No shape info - fallback
            try writer.print("        %idx1 = arith.muli %global_id, %c{d} : index\n", .{reduction_dim});
            try writer.writeAll("        %idx = arith.addi %idx1, %j : index\n");
            try writer.print("        %elem = memref.load %arg{d}[%idx] : memref<{d}x{s}>\n", .{ source_param, source_info.dimension, element_type });
        }

        // Perform the reduction operation
        const op_name = switch (operation_info.operation) {
            .add => if (std.mem.eql(u8, element_type, "f32") or std.mem.eql(u8, element_type, "f64")) "arith.addf" else "arith.addi",
            .multiply => if (std.mem.eql(u8, element_type, "f32") or std.mem.eql(u8, element_type, "f64")) "arith.mulf" else "arith.muli",
            else => "arith.addi", // Default
        };

        try writer.print("        %new_acc = {s} %acc, %elem : {s}\n", .{ op_name, element_type });
        try writer.print("        scf.yield %new_acc : {s}\n", .{element_type});
        try writer.writeAll("      }\n");

        // Store the reduced value
        try writer.print("      memref.store %reduced, %arg{d}[%global_id] : memref<{d}x{s}>\n", .{ target_param, target_info.dimension, element_type });
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

        // Use bare pointer conversion to match emit_ptx behavior (2 parameters instead of 10)
        const gpu_to_nvvm_pass = MLIR.mlirCreateConversionConvertGpuOpsToNVVMOpsWithBarePtr();
        MLIR.mlirOpPassManagerAddOwnedPass(gpu_module_pm, gpu_to_nvvm_pass);

        // Add NVVM to LLVM conversion pass to the main module pass manager
        const nvvm_to_llvm_pass = MLIR.mlirCreateConversionConvertNVVMToLLVMPass();
        MLIR.mlirPassManagerAddOwnedPass(pass_manager, nvvm_to_llvm_pass);

        // Add finalize MemRef to LLVM conversion pass with bare pointer conversion
        // This constrains memref parameters to simple pointers instead of full descriptors
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
        _ = function_name; // We'll extract ALL GPU functions, not just one

        // Find all GPU kernel functions (they start with "llvm.func @gpu_")
        var functions = std.ArrayList([]const u8).init(self.allocator);
        defer functions.deinit();
        defer for (functions.items) |func| {
            self.allocator.free(func);
        };

        var search_pos: usize = 0;
        while (true) {
            // Search for GPU functions
            const func_start = std.mem.indexOf(u8, input_content[search_pos..], "llvm.func @gpu_") orelse break;
            const actual_start = search_pos + func_start;

            // Find the end of this function
            const func_end = std.mem.indexOf(u8, input_content[actual_start..], "  }") orelse {
                std.debug.print("Error: Could not find end of GPU function\n", .{});
                return MLIRCodeGenError.PipelineError;
            };

            // Extract the function
            const function = input_content[actual_start .. actual_start + func_end + 3];

            // Clean up the function (remove gpu.kernel attributes)
            var cleaned_func = try self.allocator.dupe(u8, function);

            // Remove the gpu.kernel attribute as it's not needed in standalone module
            const kernel_with_attrs = std.mem.replacementSize(u8, cleaned_func, "gpu.kernel, ", "");
            if (kernel_with_attrs < cleaned_func.len) {
                const temp = try self.allocator.alloc(u8, kernel_with_attrs);
                _ = std.mem.replace(u8, cleaned_func, "gpu.kernel, ", "", temp);
                self.allocator.free(cleaned_func);
                cleaned_func = temp;
            }

            // Also remove if it's at the end
            const kernel_with_attrs2 = std.mem.replacementSize(u8, cleaned_func, ", gpu.kernel", "");
            if (kernel_with_attrs2 < cleaned_func.len) {
                const temp = try self.allocator.alloc(u8, kernel_with_attrs2);
                _ = std.mem.replace(u8, cleaned_func, ", gpu.kernel", "", temp);
                self.allocator.free(cleaned_func);
                cleaned_func = temp;
            }

            try functions.append(cleaned_func);

            // Move search position forward
            search_pos = actual_start + func_end + 3;
        }

        if (functions.items.len == 0) {
            std.debug.print("Error: No GPU kernel functions found in MLIR\n", .{});
            return MLIRCodeGenError.PipelineError;
        }

        if (self.verbose) {
            std.debug.print("Found {} GPU kernel functions to extract\n", .{functions.items.len});
        }

        // Combine all functions into a single module
        var combined_functions = std.ArrayList(u8).init(self.allocator);
        defer combined_functions.deinit();

        for (functions.items) |func| {
            try combined_functions.appendSlice(func);
            try combined_functions.appendSlice("\n");
        }

        // Create a standalone module with all kernel functions
        const standalone_module = try std.fmt.allocPrint(self.allocator,
            \\module attributes {{nvvm.target = "cuda"}} {{
            \\{s}
            \\  llvm.func @llvm.nvvm.read.ptx.sreg.tid.x() -> i32 attributes {{passthrough = ["nounwind", "readnone"]}}
            \\  llvm.func @llvm.nvvm.read.ptx.sreg.ntid.x() -> i32 attributes {{passthrough = ["nounwind", "readnone"]}}
            \\  llvm.func @llvm.nvvm.read.ptx.sreg.ctaid.x() -> i32 attributes {{passthrough = ["nounwind", "readnone"]}}
            \\}}
            \\
        , .{combined_functions.items});

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
        var target: MLIR.LLVMTargetRef = undefined;
        var target_error_msg: [*c]u8 = undefined;
        const query = std.Target.Query.fromTarget(self.target);
        const target_triple = try query.zigTriple(self.allocator);
        defer self.allocator.free(target_triple);

        if (MLIR.LLVMGetTargetFromTriple(@as([*c]const u8, @ptrCast(target_triple.ptr)), &target, &target_error_msg) != 0) {
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
        // Extract SM version from target and create CPU string
        const sm_version = try extractSmVersionFromTarget(self.target);
        const cpu_str = try std.fmt.allocPrintZ(self.allocator, "sm_{d}", .{sm_version});
        defer self.allocator.free(cpu_str);

        const target_machine = MLIR.LLVMCreateTargetMachine(target, @as([*c]const u8, @ptrCast(target_triple.ptr)), cpu_str.ptr, "", // features
            MLIR.LLVMCodeGenLevelDefault, MLIR.LLVMRelocDefault, MLIR.LLVMCodeModelDefault);
        defer MLIR.LLVMDisposeTargetMachine(target_machine);

        if (self.verbose) {
            std.debug.print("✅ Created target machine for SM {d}\n", .{sm_version});
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

    // Create a basic CUDA target for testing
    const query = std.Target.Query.parse(.{ .arch_os_abi = "nvptx64-cuda", .cpu_features = "sm_50" }) catch return;
    const cuda_target = std.zig.system.resolveTargetQuery(query) catch return;

    var mlir_codegen = try MLIRCodeGen.init(allocator, cuda_target, false);
    defer mlir_codegen.deinit();

    // Should initialize without crashing
    try std.testing.expect(true);
}

test "MLIRCodeGen - generate GPU function" {
    const allocator = std.testing.allocator;

    // Create a basic CUDA target for testing
    const query = std.Target.Query.parse(.{ .arch_os_abi = "nvptx64-cuda", .cpu_features = "sm_50" }) catch return;
    const cuda_target = std.zig.system.resolveTargetQuery(query) catch return;

    var mlir_codegen = try MLIRCodeGen.init(allocator, cuda_target, false);
    defer mlir_codegen.deinit();

    // Create a dummy function declaration for testing
    const func_decl = @TypeOf(@as(parser.ASTNode, undefined).function_declaration){
        .offset = 0,
        .name = "test_func",
        .parameters = &[_]parser.Parameter{},
        .return_type = .void,
        .body = &[_]parser.ASTNode{},
        .is_public = false,
    };

    // Should generate without crashing
    try mlir_codegen.generateGpuFunction(func_decl);

    // Check that some MLIR was generated
    try std.testing.expect(mlir_codegen.generated_mlir.items.len > 0);
}
