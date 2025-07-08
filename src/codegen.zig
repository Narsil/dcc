const std = @import("std");
const parser = @import("parser.zig");
const mlir_codegen = @import("mlir_codegen.zig");
const cuda_llvm_ir_gen = @import("cuda_llvm_ir_gen.zig");
const cuda_stub_manager = @import("cuda_stub_manager.zig");
const LLVM = @cImport({
    @cInclude("llvm-c/Core.h");
    @cInclude("llvm-c/TargetMachine.h");
    @cInclude("llvm-c/Target.h");
    @cInclude("llvm-c/Support.h");
    @cInclude("llvm-c/BitReader.h");
    @cInclude("llvm-c/Object.h");
});

pub const CodeGenError = error{ InvalidTopLevelNode, InvalidStatement, InvalidExpression, InvalidCallee, UndefinedVariable, UndefinedFunction, TargetError, CodeGenError, MainFunctionNotFound, MissingMainFunction, LinkingFailed, GpuCompilationNotImplemented, InvalidGpuTriplet, InvalidTargetTriple, InvalidCharacter, Overflow, CudaFunctionNotFound } || std.mem.Allocator.Error;

const VarInfo = struct {
    alloca: LLVM.LLVMValueRef,
    ty: parser.Type,
};

// LLVM types are now available from the LLVM module

// LLVM functions are now available from the LLVM module

// Function attribute management functions are now available from the LLVM module

// LLVM attribute types
// Function attribute indices
// const LLVM.LLVMAttributeFunctionIndex = 0xffffffff;

// Target initialization functions are now available from the LLVM module

// Target, memory buffer, and object file functions are now available from the LLVM module

// LLVM Inline Assembly APIs
// LLD (LLVM Linker) C wrapper API
extern fn lld_main(args: [*]const [*:0]const u8, argc: c_int) c_int;

// Additional LLVM types for memory buffers and object files
// Darwin ARM64 system call numbers
const SYS_EXIT_DARWIN = 1; // exit() system call on Darwin

pub const CodeGen = struct {
    context: LLVM.LLVMContextRef,
    module: LLVM.LLVMModuleRef,
    builder: LLVM.LLVMBuilderRef,
    allocator: std.mem.Allocator,
    verbose: bool,
    variables: std.HashMap([]const u8, VarInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    functions: std.HashMap([]const u8, LLVM.LLVMValueRef, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    mlir_codegen: ?mlir_codegen.MLIRCodeGen,
    cuda_stub_mgr: ?cuda_stub_manager.CudaStubManager,

    /// Parse GPU triplet and extract SM version
    /// Expected format: nvidia-ptx-smXX (e.g., nvidia-ptx-sm50)
    fn parseGpuTriplet(triplet: []const u8) !u32 {
        // Split by '-' to get parts
        var parts = std.mem.splitSequence(u8, triplet, "-");

        // Part 1: vendor (must be "nvidia")
        const vendor = parts.next() orelse return error.InvalidGpuTriplet;
        if (!std.mem.eql(u8, vendor, "nvidia")) {
            return error.InvalidGpuTriplet;
        }

        // Part 2: target (must be "ptx")
        const target = parts.next() orelse return error.InvalidGpuTriplet;
        if (!std.mem.eql(u8, target, "ptx")) {
            return error.InvalidGpuTriplet;
        }

        // Part 3: SM version (must be "smXX")
        const sm_part = parts.next() orelse return error.InvalidGpuTriplet;
        if (!std.mem.startsWith(u8, sm_part, "sm")) {
            return error.InvalidGpuTriplet;
        }

        // Make sure there are no more parts
        if (parts.next() != null) {
            return error.InvalidGpuTriplet;
        }

        // Extract the numeric part after "sm"
        const sm_version_str = sm_part[2..]; // Skip "sm"
        const sm_version = std.fmt.parseInt(u32, sm_version_str, 10) catch return error.InvalidGpuTriplet;

        // Validate SM version is reasonable (20-90)
        if (sm_version < 20 or sm_version > 90) {
            return error.InvalidGpuTriplet;
        }

        return sm_version;
    }

    pub fn init(allocator: std.mem.Allocator, module_name: []const u8, verbose: bool, target: std.Target, gpu_triplet: ?[]const u8) !CodeGen {
        // Initialize LLVM X86 target (for x86_64 support)
        LLVM.LLVMInitializeX86TargetInfo();
        LLVM.LLVMInitializeX86Target();
        LLVM.LLVMInitializeX86TargetMC();
        LLVM.LLVMInitializeX86AsmParser();
        LLVM.LLVMInitializeX86AsmPrinter();

        // Initialize LLVM AArch64 target (for arm64 support)
        LLVM.LLVMInitializeAArch64TargetInfo();
        LLVM.LLVMInitializeAArch64Target();
        LLVM.LLVMInitializeAArch64TargetMC();
        LLVM.LLVMInitializeAArch64AsmParser();
        LLVM.LLVMInitializeAArch64AsmPrinter();

        const context = LLVM.LLVMContextCreate();
        const module_name_z = try allocator.dupeZ(u8, module_name);
        defer allocator.free(module_name_z);

        const module = LLVM.LLVMModuleCreateWithNameInContext(module_name_z.ptr, context);
        const builder = LLVM.LLVMCreateBuilderInContext(context);

        // Parse GPU triplet if provided
        var mlir_gen: ?mlir_codegen.MLIRCodeGen = null;
        if (gpu_triplet) |triplet| {
            const sm_version = parseGpuTriplet(triplet) catch |err| {
                std.debug.print("Error: Invalid GPU triplet '{s}': {}\n", .{ triplet, err });
                std.debug.print("Expected format: nvidia-ptx-smXX (e.g., nvidia-ptx-sm50)\n", .{});
                return error.CodeGenError;
            };

            if (target.os.tag == .macos) {
                std.debug.print("Error: NVIDIA GPU compilation is not supported on macOS targets\n", .{});
                std.debug.print("NVIDIA GPUs are not available on macOS. GPU compilation with '{s}' is only supported on Linux targets.\n", .{triplet});
                std.debug.print("To compile for GPU targets, use a Linux target (e.g., --target x86_64-unknown-linux-gnu).\n", .{});
                return error.CodeGenError;
            }

            mlir_gen = mlir_codegen.MLIRCodeGen.init(allocator, sm_version, verbose) catch null;
        }

        // Initialize CUDA stub manager if GPU compilation is enabled
        var cuda_stub_mgr: ?cuda_stub_manager.CudaStubManager = null;
        if (gpu_triplet != null) {
            cuda_stub_mgr = cuda_stub_manager.CudaStubManager.init(allocator, verbose) catch |err| blk: {
                if (verbose) {
                    std.debug.print("⚠️  Warning: Failed to initialize CUDA stub manager: {}\n", .{err});
                }
                break :blk null;
            };
        }

        return CodeGen{
            .context = context,
            .module = module,
            .builder = builder,
            .allocator = allocator,
            .verbose = verbose,
            .variables = std.HashMap([]const u8, VarInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .functions = std.HashMap([]const u8, LLVM.LLVMValueRef, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .mlir_codegen = mlir_gen,
            .cuda_stub_mgr = cuda_stub_mgr,
        };
    }

    pub fn deinit(self: *CodeGen) void {
        // Free all duplicated strings used as keys in the functions HashMap
        var func_iter = self.functions.iterator();
        while (func_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.functions.deinit();

        // Variables are cleaned up after each function, so just deinit the HashMap
        self.variables.deinit();

        // Clean up MLIR resources
        if (self.mlir_codegen) |*mlir| {
            mlir.deinit();
        }

        // Clean up CUDA stub manager
        if (self.cuda_stub_mgr) |*cuda_mgr| {
            cuda_mgr.deinit();
        }

        // Clean up LLVM resources
        LLVM.LLVMDisposeBuilder(self.builder);
        LLVM.LLVMDisposeModule(self.module);
        LLVM.LLVMContextDispose(self.context);
    }

    /// Convert a Type to an LLVM type
    pub fn toLLVMType(self: *CodeGen, type_to_convert: parser.Type) LLVM.LLVMTypeRef {
        return switch (type_to_convert) {
            .u8, .i8 => LLVM.LLVMInt8TypeInContext(self.context),
            .u16, .i16 => LLVM.LLVMInt16TypeInContext(self.context),
            .u32, .i32 => LLVM.LLVMInt32TypeInContext(self.context),
            .u64, .i64 => LLVM.LLVMInt64TypeInContext(self.context),
            .f32 => LLVM.LLVMFloatTypeInContext(self.context),
            .f64 => LLVM.LLVMDoubleTypeInContext(self.context),
            .void => LLVM.LLVMVoidTypeInContext(self.context),
            .tensor => |tensor_type| {
                // For now, create a simple array type
                // In a full implementation, we'd want to handle multi-dimensional arrays properly
                const element_type = self.toLLVMType(tensor_type.element_type.*);
                return LLVM.LLVMArrayType(element_type, @intCast(tensor_type.total_elements()));
            },
        };
    }

    fn hasGpuFunctions(self: *CodeGen) bool {
        // Check if any function names start with "gpu_"
        // Since functions may not be registered yet when this is called,
        // we'll use a simple check for the MLIR codegen being non-null
        // which indicates GPU compilation support is active
        return self.mlir_codegen != null;
    }

    pub fn generate(self: *CodeGen, ast: parser.ASTNode) CodeGenError!void {
        return self.generateWithMode(ast, .executable);
    }

    pub fn generateWithMode(self: *CodeGen, ast: parser.ASTNode, mode: enum { executable, library }) CodeGenError!void {
        // Generate LLVM IR for the program AST node
        switch (ast) {
            .program => |prog| {
                // Two-pass approach to handle forward function calls:
                // Pass 1: Declare all functions (signatures only)
                for (prog.statements) |stmt| {
                    if (stmt == .function_declaration) {
                        try self.declareFunctionSignature(stmt.function_declaration);
                    }
                }

                // Pass 2: Process all function bodies (GPU functions first, then main)
                var main_processed = false;
                var main_func_decl: ?@TypeOf(@as(parser.ASTNode, undefined).function_declaration) = null;

                // First, process GPU functions
                for (prog.statements) |stmt| {
                    if (stmt == .function_declaration) {
                        const func = stmt.function_declaration;
                        if (std.mem.eql(u8, func.name, "main")) {
                            main_func_decl = func;
                        } else {
                            try self.generateFunctionBody(func);
                        }
                    }
                }

                // Then, process main function after GPU functions are done
                if (main_func_decl) |main_func| {
                    try self.generateFunctionBody(main_func);
                    main_processed = true;
                }

                // Process other statements
                for (prog.statements) |stmt| {
                    switch (stmt) {
                        .function_declaration => {
                            // Already processed
                        },
                        else => try self.generateNode(stmt),
                    }
                }

                // Only require main function for executable mode
                if (mode == .executable and !main_processed) {
                    return error.MissingMainFunction;
                }
                // No need for _start function - using main directly as entry point
                // This avoids symbol naming issues with double underscores on macOS
            },
            else => return error.InvalidTopLevelNode,
        }
    }

    fn generateNode(self: *CodeGen, node: parser.ASTNode) CodeGenError!void {
        switch (node) {
            .function_declaration => try self.generateFunctionBody(node.function_declaration),
            .variable_declaration => try self.generateVariableDeclaration(node.variable_declaration),
            .return_statement => try self.generateReturn(node.return_statement),
            .expression_statement => _ = try self.generateExpression(node.expression_statement.expression.*),
            .parallel_assignment => try self.generateParallelAssignment(node.parallel_assignment),
            else => return error.InvalidStatement,
        }
    }

    fn declareFunctionSignature(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) CodeGenError!void {
        // Create parameter types array
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

        // Create function type
        const actual_return_type = self.toLLVMType(func.return_type);
        const function_type = LLVM.LLVMFunctionType(actual_return_type, param_types.ptr, @intCast(param_types.len), 0);

        // Create function
        const name_z = try self.allocator.dupeZ(u8, func.name);
        defer self.allocator.free(name_z);

        const llvm_function = LLVM.LLVMAddFunction(self.module, name_z.ptr, function_type);

        // Set proper stack alignment attributes for x86-64 System V ABI compliance
        self.setStackAlignmentAttributes(llvm_function);

        try self.functions.put(try self.allocator.dupe(u8, func.name), llvm_function);

        if (self.verbose) {
            std.debug.print("🔧 Declared function signature: {s}\n", .{func.name});
        }
    }

    fn generateFunctionBody(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) CodeGenError!void {
        // Detect GPU functions by naming convention and bail out for now
        if (std.mem.startsWith(u8, func.name, "gpu_")) {
            return self.generateGpuFunction(func);
        }

        // Get the already declared function
        const llvm_function = self.functions.get(func.name) orelse return error.UndefinedFunction;

        // Use i32 return type for main function to match C runtime expectations
        const is_main = std.mem.eql(u8, func.name, "main");

        // Create entry basic block
        const entry_block = LLVM.LLVMAppendBasicBlockInContext(self.context, llvm_function, "entry");
        LLVM.LLVMPositionBuilderAtEnd(self.builder, entry_block);

        // Create allocas for parameters
        for (func.parameters, 0..) |param, i| {
            const param_value = LLVM.LLVMGetParam(llvm_function, @intCast(i));
            const param_name_z = try self.allocator.dupeZ(u8, param.name);
            defer self.allocator.free(param_name_z);

            if (param.type == .tensor) {
                // For tensor parameters, use the passed pointer directly
                // (no need to create a local copy)
                try self.variables.put(try self.allocator.dupe(u8, param.name), VarInfo{ .alloca = param_value, .ty = param.type });
            } else {
                const param_type = self.toLLVMType(param.type);
                const alloca = LLVM.LLVMBuildAlloca(self.builder, param_type, param_name_z.ptr);
                LLVM.LLVMSetAlignment(alloca, 16); // 16-byte alignment for x86-64 System V ABI
                const store_inst = LLVM.LLVMBuildStore(self.builder, param_value, alloca);
                LLVM.LLVMSetAlignment(store_inst, 16); // Match alloca alignment
                try self.variables.put(try self.allocator.dupe(u8, param.name), VarInfo{ .alloca = alloca, .ty = param.type });
            }
        }

        // Add dummy alloca for stack alignment in main
        if (is_main) {
            // Allocate 16 bytes of padding so total local size is multiple of 16 (2x i64 = 16 + padding = 32)
            const int8_type = LLVM.LLVMInt8TypeInContext(self.context);
            const dummy_array_type = LLVM.LLVMArrayType(int8_type, 16); // 16 bytes
            const dummy_name = "dummy_padding";
            const dummy_alloca = LLVM.LLVMBuildAlloca(self.builder, dummy_array_type, dummy_name);
            LLVM.LLVMSetAlignment(dummy_alloca, 16);
        }

        // Inject CUDA initialization into main function ONLY when we have GPU functions
        // This ensures CUDA is initialized once, not every time a GPU function is called
        if (is_main) {
            if (self.verbose) {
                std.debug.print("🔧 In main function, checking for GPU functions...\n", .{});
                std.debug.print("   self.mlir_codegen != null: {}\n", .{self.mlir_codegen != null});
                std.debug.print("   hasGpuFunctions(): {}\n", .{self.hasGpuFunctions()});
            }
            if (self.hasGpuFunctions()) {
                if (self.verbose) {
                    std.debug.print("🔧 Injecting CUDA initialization into main function\n", .{});
                }
                try self.injectCudaInitializationIntoMain();
            } else {
                if (self.verbose) {
                    std.debug.print("❌ No GPU functions detected, skipping CUDA initialization\n", .{});
                }
            }
        }

        // Generate function body
        for (func.body) |stmt| {
            try self.generateNode(stmt);
        }

        // Ensure void functions have proper terminators
        // Check if the current basic block has a terminator
        const current_block = LLVM.LLVMGetInsertBlock(self.builder);
        const terminator = LLVM.LLVMGetBasicBlockTerminator(current_block);

        if (terminator == null) {
            // No terminator found, add appropriate return based on function type
            if (func.return_type == .void) {
                _ = LLVM.LLVMBuildRetVoid(self.builder);
            } else {
                // For non-void functions without explicit return, this is an error
                // that should have been caught by the type checker, but add a default return
                // to prevent LLVM IR errors
                const default_value = switch (func.return_type) {
                    .i32 => LLVM.LLVMConstInt(LLVM.LLVMInt32TypeInContext(self.context), 0, 0),
                    .i64 => LLVM.LLVMConstInt(LLVM.LLVMInt64TypeInContext(self.context), 0, 0),
                    .f32 => LLVM.LLVMConstReal(LLVM.LLVMFloatTypeInContext(self.context), 0.0),
                    .f64 => LLVM.LLVMConstReal(LLVM.LLVMDoubleTypeInContext(self.context), 0.0),
                    else => LLVM.LLVMConstInt(LLVM.LLVMInt32TypeInContext(self.context), 0, 0),
                };
                _ = LLVM.LLVMBuildRet(self.builder, default_value);
            }
        }

        // Clear variables after function generation to avoid conflicts
        // Functions should persist across the entire compilation process
        self.clearVariables();
    }

    fn generateVariableDeclaration(self: *CodeGen, var_decl: @TypeOf(@as(parser.ASTNode, undefined).variable_declaration)) CodeGenError!void {
        const var_type = self.toLLVMType(var_decl.type);
        const name_z = try self.allocator.dupeZ(u8, var_decl.name);
        defer self.allocator.free(name_z);

        const alloca = LLVM.LLVMBuildAlloca(self.builder, var_type, name_z.ptr);
        LLVM.LLVMSetAlignment(alloca, 16); // 16-byte alignment for x86-64 System V ABI
        const value = try self.generateExpression(var_decl.value.*);
        const store_inst = LLVM.LLVMBuildStore(self.builder, value, alloca);
        LLVM.LLVMSetAlignment(store_inst, 16); // Match alloca alignment

        try self.variables.put(try self.allocator.dupe(u8, var_decl.name), VarInfo{ .alloca = alloca, .ty = var_decl.type });
    }

    fn generateReturn(self: *CodeGen, ret: @TypeOf(@as(parser.ASTNode, undefined).return_statement)) CodeGenError!void {
        // Check if we're in the main function with GPU functions - if so, inject cleanup before return
        const current_function = LLVM.LLVMGetBasicBlockParent(LLVM.LLVMGetInsertBlock(self.builder));
        const function_name = LLVM.LLVMGetValueName(current_function);
        const is_main = std.mem.eql(u8, std.mem.span(function_name), "main");

        if (is_main and self.hasGpuFunctions()) {
            if (self.verbose) {
                std.debug.print("🧹 Injecting CUDA cleanup before return in main function\n", .{});
            }
            try self.injectCudaCleanupBeforeReturn();
        }

        if (ret.value) |value| {
            const llvm_value = try self.generateExpression(value.*);
            _ = LLVM.LLVMBuildRet(self.builder, llvm_value);
        } else {
            // Return statement has no value - must be a void function
            _ = LLVM.LLVMBuildRetVoid(self.builder);
        }
    }

    fn generateParallelAssignment(self: *CodeGen, pa: @TypeOf(@as(parser.ASTNode, undefined).parallel_assignment)) CodeGenError!void {
        // For now only support target of the form identifier[i]
        if (pa.target.* != .implicit_tensor_index) {
            return error.CodeGenError;
        }
        const ti = pa.target.*.implicit_tensor_index;
        // Expect tensor to be an identifier
        if (ti.tensor.* != .identifier) {
            return error.CodeGenError;
        }
        const tensor_name = ti.tensor.*.identifier.name;
        // Fetch alloca of tensor variable
        const tensor_info = self.variables.get(tensor_name) orelse return error.UndefinedVariable;

        // Determine tensor length from stored type info (first dimension)
        const len: u64 = switch (tensor_info.ty) {
            .tensor => |t| t.shape[0],
            else => 0,
        };
        const elem_type = self.toLLVMType(tensor_info.ty);

        const i64_type = LLVM.LLVMInt64TypeInContext(self.context);
        const zero_const = LLVM.LLVMConstInt(i64_type, 0, 0);
        const one_const = LLVM.LLVMConstInt(i64_type, 1, 0);
        const bound_const = LLVM.LLVMConstInt(i64_type, len, 0);

        // Current function and basic blocks
        const func = LLVM.LLVMGetBasicBlockParent(LLVM.LLVMGetInsertBlock(self.builder));
        const loop_bb = LLVM.LLVMAppendBasicBlockInContext(self.context, func, "par_loop");
        const body_bb = LLVM.LLVMAppendBasicBlockInContext(self.context, func, "par_body");
        const inc_bb = LLVM.LLVMAppendBasicBlockInContext(self.context, func, "par_inc");
        const after_bb = LLVM.LLVMAppendBasicBlockInContext(self.context, func, "par_after");

        // Create induction variable on stack
        const idx_alloc = LLVM.LLVMBuildAlloca(self.builder, i64_type, "idx");
        _ = LLVM.LLVMBuildStore(self.builder, zero_const, idx_alloc);
        _ = LLVM.LLVMBuildBr(self.builder, loop_bb);

        // Loop condition
        LLVM.LLVMPositionBuilderAtEnd(self.builder, loop_bb);
        const idx_val = LLVM.LLVMBuildLoad2(self.builder, i64_type, idx_alloc, "idx_val");
        const cond = LLVM.LLVMBuildICmp(self.builder, LLVM.LLVMIntULT, idx_val, bound_const, "loop_cond");
        _ = LLVM.LLVMBuildCondBr(self.builder, cond, body_bb, after_bb);

        // Body
        LLVM.LLVMPositionBuilderAtEnd(self.builder, body_bb);
        if (self.verbose) {
            std.debug.print("Parallel assignment: IR before generating RHS\n", .{});
            self.printIR();
        }
        const rhs_val = try self.generateTensorExpression(pa.value.*, idx_val);
        var gep_indices = [_]LLVM.LLVMValueRef{ zero_const, idx_val };
        const elem_ptr = LLVM.LLVMBuildGEP2(self.builder, elem_type, tensor_info.alloca, &gep_indices[0], 2, "elem_ptr");
        if (elem_ptr == null) {
            std.debug.print("ERROR: LLVMBuildGEP2 returned null\n", .{});
        }
        _ = LLVM.LLVMBuildStore(self.builder, rhs_val, elem_ptr);
        _ = LLVM.LLVMBuildBr(self.builder, inc_bb);

        // Increment
        LLVM.LLVMPositionBuilderAtEnd(self.builder, inc_bb);
        const next_idx = LLVM.LLVMBuildAdd(self.builder, idx_val, one_const, "next_idx");
        _ = LLVM.LLVMBuildStore(self.builder, next_idx, idx_alloc);
        _ = LLVM.LLVMBuildBr(self.builder, loop_bb);

        // After
        LLVM.LLVMPositionBuilderAtEnd(self.builder, after_bb);
    }

    fn generateExpression(self: *CodeGen, node: parser.ASTNode) CodeGenError!LLVM.LLVMValueRef {
        switch (node) {
            .number_literal => |num| {
                const llvm_type = self.toLLVMType(num.type);

                if (num.type.isFloat()) {
                    // Strip type suffix for float parsing
                    const value_without_suffix = if (std.mem.endsWith(u8, num.value, "f32") or std.mem.endsWith(u8, num.value, "f64"))
                        num.value[0..(num.value.len - 3)]
                    else
                        num.value;
                    const float_value = try std.fmt.parseFloat(f64, value_without_suffix);
                    return LLVM.LLVMConstReal(llvm_type, float_value);
                } else {
                    // Strip type suffix for integer parsing
                    const value_without_suffix = if (std.mem.endsWith(u8, num.value, "u8") or std.mem.endsWith(u8, num.value, "i8"))
                        num.value[0..(num.value.len - 2)]
                    else if (std.mem.endsWith(u8, num.value, "u16") or std.mem.endsWith(u8, num.value, "i16") or
                        std.mem.endsWith(u8, num.value, "u32") or std.mem.endsWith(u8, num.value, "i32") or
                        std.mem.endsWith(u8, num.value, "u64") or std.mem.endsWith(u8, num.value, "i64"))
                        num.value[0..(num.value.len - 3)]
                    else
                        num.value;
                    const int_value = try std.fmt.parseInt(u64, value_without_suffix, 10);
                    return LLVM.LLVMConstInt(llvm_type, int_value, 0);
                }
            },
            .identifier => |ident| {
                if (self.variables.get(ident.name)) |info| {
                    if (info.ty == .tensor) {
                        return info.alloca; // use pointer for tensors
                    }

                    const elem_type = self.toLLVMType(info.ty);
                    const name_z = try self.allocator.dupeZ(u8, ident.name);
                    defer self.allocator.free(name_z);

                    if (self.verbose) {
                        std.debug.print("Identifier load '{s}', printing IR before load\n", .{ident.name});
                        self.printIR();
                    }

                    const load_inst = LLVM.LLVMBuildLoad2(self.builder, elem_type, info.alloca, name_z.ptr);
                    LLVM.LLVMSetAlignment(load_inst, 16);
                    return load_inst;
                } else {
                    return error.UndefinedVariable;
                }
            },
            .binary_expression => |bin_expr| {
                const left = try self.generateExpression(bin_expr.left.*);
                const right = try self.generateExpression(bin_expr.right.*);

                // Decide whether we are dealing with floating-point or integer ops based on LLVM type of the LHS.
                const type_kind = LLVM.LLVMGetTypeKind(LLVM.LLVMTypeOf(left));
                const is_float = switch (type_kind) {
                    LLVM.LLVMFloatTypeKind, LLVM.LLVMDoubleTypeKind, LLVM.LLVMHalfTypeKind, LLVM.LLVMBFloatTypeKind, LLVM.LLVMFP128TypeKind => true,
                    else => false,
                };

                if (is_float) {
                    return switch (bin_expr.operator) {
                        .add => LLVM.LLVMBuildFAdd(self.builder, left, right, "fadd"),
                        .subtract => LLVM.LLVMBuildFSub(self.builder, left, right, "fsub"),
                        .multiply => LLVM.LLVMBuildFMul(self.builder, left, right, "fmul"),
                        .divide => LLVM.LLVMBuildFDiv(self.builder, left, right, "fdiv"),
                    };
                } else {
                    // Integer path (assume signed until we propagate sign info)
                    return switch (bin_expr.operator) {
                        .add => LLVM.LLVMBuildAdd(self.builder, left, right, "add"),
                        .subtract => LLVM.LLVMBuildSub(self.builder, left, right, "sub"),
                        .multiply => LLVM.LLVMBuildMul(self.builder, left, right, "mul"),
                        .divide => LLVM.LLVMBuildSDiv(self.builder, left, right, "sdiv"),
                    };
                }
            },
            .unary_expression => |unary_expr| {
                const operand = try self.generateExpression(unary_expr.operand.*);

                const type_kind = LLVM.LLVMGetTypeKind(LLVM.LLVMTypeOf(operand));
                const is_float = switch (type_kind) {
                    LLVM.LLVMFloatTypeKind, LLVM.LLVMDoubleTypeKind, LLVM.LLVMHalfTypeKind, LLVM.LLVMBFloatTypeKind, LLVM.LLVMFP128TypeKind => true,
                    else => false,
                };

                return switch (unary_expr.operator) {
                    .negate => if (is_float)
                        LLVM.LLVMBuildFNeg(self.builder, operand, "fneg")
                    else
                        LLVM.LLVMBuildNeg(self.builder, operand, "neg"),
                };
            },
            .call_expression => |call| {
                const callee_name = switch (call.callee.*) {
                    .identifier => |ident| ident.name,
                    else => return error.InvalidCallee,
                };

                if (self.functions.get(callee_name)) |function| {
                    const args = try self.allocator.alloc(LLVM.LLVMValueRef, call.arguments.len);
                    defer self.allocator.free(args);

                    for (call.arguments, 0..) |arg, i| {
                        const arg_value = try self.generateExpression(arg);

                        // For tensor arguments, pass the pointer directly
                        // (no need to load the array value since functions expect pointers now)
                        args[i] = arg_value;
                    }

                    // Get the actual function type from the stored function
                    const function_type = LLVM.LLVMGlobalGetValueType(function);

                    return LLVM.LLVMBuildCall2(self.builder, function_type, function, args.ptr, @intCast(args.len), "call");
                } else {
                    return error.UndefinedFunction;
                }
            },
            .tensor_literal => |tensor_lit| {
                // For now, create a simple array with the specified value
                const element_type = self.toLLVMType(tensor_lit.element_type);
                _ = LLVM.LLVMArrayType(element_type, @intCast(tensor_lit.shape[0])); // Suppress unused warning
                const value = try self.generateExpression(tensor_lit.value.*);

                // Create a constant array with the value repeated
                const values = try self.allocator.alloc(LLVM.LLVMValueRef, tensor_lit.shape[0]);
                defer self.allocator.free(values);

                for (values) |*val| {
                    val.* = value;
                }

                return LLVM.LLVMConstArray(element_type, values.ptr, @intCast(values.len));
            },
            .tensor_slice => |tensor_slice| {
                // Evaluate the base tensor expression. For variables this will be a pointer
                // to the array aggregate.
                const tensor_ptr = try self.generateExpression(tensor_slice.tensor.*);

                const ptr_ty = LLVM.LLVMTypeOf(tensor_ptr);
                if (LLVM.LLVMGetTypeKind(ptr_ty) != LLVM.LLVMPointerTypeKind) {
                    // Fallback – if we somehow have an aggregate constant instead of a pointer,
                    // simply extract the first element.
                    return LLVM.LLVMBuildExtractValue(self.builder, tensor_ptr, 0, "tensor_slice_const");
                }

                if (tensor_slice.tensor.* != .identifier) return error.CodeGenError;

                const name_slice = tensor_slice.tensor.*.identifier.name;
                const vi_slice = self.variables.get(name_slice) orelse return error.CodeGenError;
                if (vi_slice.ty != .tensor) return error.CodeGenError;

                const array_ty = self.toLLVMType(vi_slice.ty);
                const elem_ty = self.toLLVMType(vi_slice.ty.tensor.element_type.*);

                // Currently only support 1-dimensional explicit indices.
                if (tensor_slice.indices.len != 1) {
                    return error.CodeGenError;
                }

                // Evaluate the index expression (should yield an integer value).
                const index_val = try self.generateExpression(tensor_slice.indices[0]);

                // Build GEP indices [0, index_val] to address the desired element.
                const i64_type = LLVM.LLVMInt64TypeInContext(self.context);
                const zero_const = LLVM.LLVMConstInt(i64_type, 0, 0);
                var gep_indices = [_]LLVM.LLVMValueRef{ zero_const, index_val };

                const elem_ptr = LLVM.LLVMBuildGEP2(self.builder, array_ty, tensor_ptr, &gep_indices[0], 2, "tensor_slice_elem_ptr");
                self.printIR();

                return LLVM.LLVMBuildLoad2(self.builder, elem_ty, elem_ptr, "tensor_slice_load");
            },
            else => return error.InvalidExpression,
        }
    }

    /// Similar to generateExpression but aware of the induction variable of a
    /// parallel tensor loop.  Any implicit_tensor_index nodes will load the
    /// element at the current loop index provided by `idx_val`.
    fn generateTensorExpression(self: *CodeGen, node: parser.ASTNode, idx_val: LLVM.LLVMValueRef) CodeGenError!LLVM.LLVMValueRef {
        switch (node) {
            // Handle the special case first
            .implicit_tensor_index => |tensor_index| {
                const tensor_val = try self.generateExpression(tensor_index.tensor.*);

                if (tensor_index.tensor.* != .identifier) {
                    return error.CodeGenError;
                }

                const name = tensor_index.tensor.*.identifier.name;
                const vi = self.variables.get(name) orelse return error.CodeGenError;
                if (vi.ty != .tensor) return error.CodeGenError;

                const array_ty = self.toLLVMType(vi.ty);
                const elem_ty = self.toLLVMType(vi.ty.tensor.element_type.*);

                // Build GEP indices [0, idx_val]
                const i64_type = LLVM.LLVMInt64TypeInContext(self.context);
                const zero_const = LLVM.LLVMConstInt(i64_type, 0, 0);
                var gep_indices = [_]LLVM.LLVMValueRef{ zero_const, idx_val };

                const elem_ptr = LLVM.LLVMBuildGEP2(self.builder, array_ty, tensor_val, &gep_indices[0], 2, "tensor_elem_ptr");
                return LLVM.LLVMBuildLoad2(self.builder, elem_ty, elem_ptr, "tensor_element");
            },
            .binary_expression => |bin_expr| {
                const left = try self.generateTensorExpression(bin_expr.left.*, idx_val);
                const right = try self.generateTensorExpression(bin_expr.right.*, idx_val);

                const type_kind = LLVM.LLVMGetTypeKind(LLVM.LLVMTypeOf(left));
                const is_float = switch (type_kind) {
                    LLVM.LLVMFloatTypeKind, LLVM.LLVMDoubleTypeKind, LLVM.LLVMHalfTypeKind, LLVM.LLVMBFloatTypeKind, LLVM.LLVMFP128TypeKind => true,
                    else => false,
                };

                if (is_float) {
                    return switch (bin_expr.operator) {
                        .add => LLVM.LLVMBuildFAdd(self.builder, left, right, "fadd"),
                        .subtract => LLVM.LLVMBuildFSub(self.builder, left, right, "fsub"),
                        .multiply => LLVM.LLVMBuildFMul(self.builder, left, right, "fmul"),
                        .divide => LLVM.LLVMBuildFDiv(self.builder, left, right, "fdiv"),
                    };
                } else {
                    return switch (bin_expr.operator) {
                        .add => LLVM.LLVMBuildAdd(self.builder, left, right, "add"),
                        .subtract => LLVM.LLVMBuildSub(self.builder, left, right, "sub"),
                        .multiply => LLVM.LLVMBuildMul(self.builder, left, right, "mul"),
                        .divide => LLVM.LLVMBuildSDiv(self.builder, left, right, "sdiv"),
                    };
                }
            },
            .unary_expression => |unary_expr| {
                const operand = try self.generateTensorExpression(unary_expr.operand.*, idx_val);

                const type_kind = LLVM.LLVMGetTypeKind(LLVM.LLVMTypeOf(operand));
                const is_float = switch (type_kind) {
                    LLVM.LLVMFloatTypeKind, LLVM.LLVMDoubleTypeKind, LLVM.LLVMHalfTypeKind, LLVM.LLVMBFloatTypeKind, LLVM.LLVMFP128TypeKind => true,
                    else => false,
                };

                return switch (unary_expr.operator) {
                    .negate => if (is_float)
                        LLVM.LLVMBuildFNeg(self.builder, operand, "fneg")
                    else
                        LLVM.LLVMBuildNeg(self.builder, operand, "neg"),
                };
            },
            // For all other node kinds, fall back to the regular expression generator
            else => return self.generateExpression(node),
        }
    }

    pub fn generateExecutable(self: *CodeGen, output_path: []const u8, target: std.Target) CodeGenError!void {
        const llvm = try getLLVMTarget(self.allocator, target);
        defer LLVM.LLVMDisposeTargetMachine(llvm.machine);

        if (target.os.tag == .linux) {
            try self.generateStartFunction();
        }

        var error_message: [*c]u8 = undefined;

        // Generate object file
        const obj_path = try std.fmt.allocPrint(self.allocator, "{s}.o", .{output_path});
        defer self.allocator.free(obj_path);

        const obj_path_z = try self.allocator.dupeZ(u8, obj_path);
        defer self.allocator.free(obj_path_z);

        if (LLVM.LLVMTargetMachineEmitToFile(llvm.machine, self.module, obj_path_z.ptr, LLVM.LLVMObjectFile, &error_message) != 0) {
            defer LLVM.LLVMDisposeMessage(error_message);
            std.debug.print("Error generating object file: {s}\n", .{error_message});
            return error.CodeGenError;
        }

        if (self.verbose) {
            std.debug.print("Generated object file: {s}\n", .{obj_path});
        }

        // Link object file into executable
        try self.linkExecutable(obj_path, output_path, target);

        // Clean up object file
        std.fs.cwd().deleteFile(obj_path) catch {};
    }

    fn linkExecutable(self: *CodeGen, obj_path: []const u8, output_path: []const u8, target: std.Target) CodeGenError!void {
        // Use lld directly without external process calls
        var args = std.ArrayList([*:0]const u8).init(self.allocator);
        defer args.deinit();

        // Defer cleanup of allocated strings
        defer for (args.items) |arg| {
            self.allocator.free(std.mem.span(arg));
        };

        switch (target.os.tag) {
            .macos => {
                // Mach-O executable arguments
                try args.append(try self.allocator.dupeZ(u8, "ld64.lld"));
                try args.append(try self.allocator.dupeZ(u8, "-arch"));
                switch (target.cpu.arch) {
                    .aarch64 => {
                        try args.append(try self.allocator.dupeZ(u8, "arm64"));
                    },
                    else => std.debug.panic("Unhandled arch on macos", .{}),
                }
                try args.append(try self.allocator.dupeZ(u8, "-platform_version"));
                try args.append(try self.allocator.dupeZ(u8, "macos"));
                try args.append(try self.allocator.dupeZ(u8, "10.15"));
                try args.append(try self.allocator.dupeZ(u8, "10.15"));
                try args.append(try self.allocator.dupeZ(u8, "-o"));
                try args.append(try self.allocator.dupeZ(u8, output_path));
            },
            .windows => {
                // COFF/PE executable arguments (note: no /dll flag for executables)
                try args.append(try self.allocator.dupeZ(u8, "lld-link"));
                try args.append(try self.allocator.dupeZ(u8, "/subsystem:console"));
                try args.append(try std.fmt.allocPrintZ(self.allocator, "/out:{s}", .{output_path}));
            },
            else => {
                // ELF executable arguments
                try args.append(try self.allocator.dupeZ(u8, "ld.lld"));
                try args.append(try self.allocator.dupeZ(u8, "--entry=_start"));
                // Only set dynamic linker if we're linking against dynamic libraries (i.e., when mlir_codegen is present)
                if (self.mlir_codegen != null) {
                    try args.append(try self.allocator.dupeZ(u8, "--dynamic-linker=/lib64/ld-linux-x86-64.so.2"));
                }
                try args.append(try self.allocator.dupeZ(u8, "-o"));
                try args.append(try self.allocator.dupeZ(u8, output_path));
            },
        }

        // Add object file
        try args.append(try self.allocator.dupeZ(u8, obj_path));

        // No libc needed - using Linux syscall for exit() in cross-compilation

        // Add CUDA library linking for Linux targets if GPU code is present
        if (target.os.tag == .linux and self.mlir_codegen != null) {
            // Check if we have CUDA stub libraries available
            if (self.cuda_stub_mgr) |*stub_mgr| {
                // Use CUDA stub library for linking, but set RPATH to system library
                const stub_lib_path = stub_mgr.getLibCudaPath() catch |err| {
                    if (self.verbose) {
                        std.debug.print("⚠️  Warning: Failed to get CUDA stub library path: {}\n", .{err});
                    }
                    // Fall back to system libraries
                    try args.append(try self.allocator.dupeZ(u8, "-lcuda"));

                    if (self.verbose) {
                        std.debug.print("🔗 Added system CUDA library linking (fallback)\n", .{});
                    }
                    return;
                };
                defer self.allocator.free(stub_lib_path);

                // Add the stub library directory for linking
                const stub_lib_dir = std.fs.path.dirname(stub_lib_path) orelse ".";
                const lib_dir_flag = try std.fmt.allocPrintZ(self.allocator, "-L{s}", .{stub_lib_dir});
                try args.append(lib_dir_flag);

                // Set RPATH to standard Linux CUDA library paths for cross-compilation
                // For cross-compilation, use the most common Linux CUDA library paths
                const linux_cuda_paths = [_][]const u8{
                    "/run/opengl-driver/lib", // NixOS
                    "/usr/local/cuda/lib64", // Standard CUDA installation
                    "/usr/lib/x86_64-linux-gnu", // Ubuntu/Debian
                    "/usr/lib64", // RHEL/CentOS
                };

                // Add multiple RPATH entries for better compatibility
                for (linux_cuda_paths) |cuda_path| {
                    const rpath_flag = try std.fmt.allocPrintZ(self.allocator, "--rpath={s}", .{cuda_path});
                    try args.append(rpath_flag);
                }

                if (self.verbose) {
                    std.debug.print("🔗 Using standard Linux CUDA library paths in RPATH for cross-compilation\n", .{});
                }

                try args.append(try self.allocator.dupeZ(u8, "-lcuda"));

                if (self.verbose) {
                    std.debug.print("🔗 Added CUDA stub library for linking: {s}\n", .{stub_lib_path});
                }
            } else {
                // No stub manager, use system libraries
                try args.append(try self.allocator.dupeZ(u8, "-lcuda"));

                if (self.verbose) {
                    std.debug.print("🔗 Added system CUDA library linking (no stub manager)\n", .{});
                }
            }
        }

        // Call lld_main with all arguments
        if (self.verbose) {
            std.debug.print("LLD Arguments: ", .{});
            for (args.items, 0..) |arg, i| {
                std.debug.print("{s}", .{std.mem.span(arg)});
                if (i < args.items.len - 1) {
                    std.debug.print(" ", .{});
                }
            }
            std.debug.print("\n", .{});
        }
        const result = lld_main(args.items.ptr, @intCast(args.items.len));

        if (result != 0) {
            std.debug.print("lld executable linking failed with code: {}\n", .{result});
            return error.LinkingFailed;
        }

        if (self.verbose) {
            std.debug.print("Generated executable: {s}\n", .{output_path});
        }
    }

    pub fn generateSharedLibrary(self: *CodeGen, output_path: []const u8, target: std.Target) CodeGenError!void {
        const llvm = try getLLVMTarget(self.allocator, target);
        defer LLVM.LLVMDisposeTargetMachine(llvm.machine);

        // Generate object file
        const obj_path = try std.fmt.allocPrint(self.allocator, "{s}.o", .{output_path});
        defer self.allocator.free(obj_path);

        const obj_path_z = try self.allocator.dupeZ(u8, obj_path);
        defer self.allocator.free(obj_path_z);

        var error_message: [*c]u8 = undefined;
        if (LLVM.LLVMTargetMachineEmitToFile(llvm.machine, self.module, obj_path_z.ptr, LLVM.LLVMObjectFile, &error_message) != 0) {
            defer LLVM.LLVMDisposeMessage(error_message);
            std.debug.print("Error generating object file: {s}\n", .{error_message});
            return error.CodeGenError;
        }

        if (self.verbose) {
            std.debug.print("Generated object file: {s}\n", .{obj_path});
        }

        const lib_extension = switch (target.os.tag) {
            .macos => ".dylib",
            .windows => ".dll",
            else => ".so",
        };
        const lib_file = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ output_path, lib_extension });
        defer self.allocator.free(lib_file);

        // Link object file into shared library
        try self.linkSharedLibrary(obj_path, lib_file, target);

        // Clean up object file
        std.fs.cwd().deleteFile(obj_path) catch {};
    }

    fn linkSharedLibrary(self: *CodeGen, obj_path: []const u8, output_path: []const u8, target: std.Target) CodeGenError!void {
        // Use lld directly without external process calls
        var args = std.ArrayList([*:0]const u8).init(self.allocator);
        defer args.deinit();

        // Defer cleanup of allocated strings
        defer for (args.items) |arg| {
            self.allocator.free(std.mem.span(arg));
        };

        switch (target.os.tag) {
            .macos => {
                // Mach-O shared library (dylib) arguments
                try args.append(try self.allocator.dupeZ(u8, "ld64.lld"));
                try args.append(try self.allocator.dupeZ(u8, "-arch"));
                switch (target.cpu.arch) {
                    .aarch64 => {
                        try args.append(try self.allocator.dupeZ(u8, "arm64"));
                    },
                    else => std.debug.panic("Unhandled arch on macos", .{}),
                }

                try args.append(try self.allocator.dupeZ(u8, "-platform_version"));
                try args.append(try self.allocator.dupeZ(u8, "macos"));
                try args.append(try self.allocator.dupeZ(u8, "10.15"));
                try args.append(try self.allocator.dupeZ(u8, "10.15"));
                try args.append(try self.allocator.dupeZ(u8, "-dylib"));
                try args.append(try self.allocator.dupeZ(u8, "-o"));
                try args.append(try self.allocator.dupeZ(u8, output_path));
            },
            .windows => {
                try args.append(try self.allocator.dupeZ(u8, "/dll"));
                try args.append(try std.fmt.allocPrintZ(self.allocator, "/out:{s}", .{output_path}));
            },
            else => {
                try args.append(try self.allocator.dupeZ(u8, "ld.lld"));
                try args.append(try self.allocator.dupeZ(u8, "--shared"));
                try args.append(try self.allocator.dupeZ(u8, "-o"));
                try args.append(try self.allocator.dupeZ(u8, output_path));
            },
        }

        // Add object file
        try args.append(try self.allocator.dupeZ(u8, obj_path));

        // Call lld_main with all arguments
        const result = lld_main(args.items.ptr, @intCast(args.items.len));

        if (result != 0) {
            std.debug.print("lld shared library linking failed with code: {}\n", .{result});
            return error.LinkingFailed;
        }

        if (self.verbose) {
            std.debug.print("Generated shared library: {s}\n", .{output_path});
        }
    }

    // Executable format creation functions
    fn createELFExecutable(self: *CodeGen, obj_data: []const u8, output_path: []const u8) !void {
        _ = self;
        var elf_file = try std.fs.cwd().createFile(output_path, .{ .mode = 0o755 });
        defer elf_file.close();

        var writer = elf_file.writer();

        // ELF Header
        const ehdr = std.elf.Elf64_Ehdr{
            .e_ident = .{ 0x7f, 'E', 'L', 'F', std.elf.ELFCLASS64, std.elf.ELFDATA2LSB, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            .e_type = std.elf.ET.EXEC,
            .e_machine = std.elf.EM.X86_64,
            .e_version = 1,
            .e_entry = 0x401000, // Entry point
            .e_phoff = @sizeOf(std.elf.Elf64_Ehdr),
            .e_shoff = 0, // Will be set later
            .e_flags = 0,
            .e_ehsize = @sizeOf(std.elf.Elf64_Ehdr),
            .e_phentsize = @sizeOf(std.elf.Elf64_Phdr),
            .e_phnum = 1,
            .e_shentsize = @sizeOf(std.elf.Elf64_Shdr),
            .e_shnum = 0, // Will be set later
            .e_shstrndx = 0, // Will be set later
        };
        try writer.writeAll(std.mem.asBytes(&ehdr));

        // Program Header
        const phdr = std.elf.Elf64_Phdr{
            .p_type = std.elf.PT_LOAD,
            .p_flags = std.elf.PF_R | std.elf.PF_X,
            .p_offset = 0x1000,
            .p_vaddr = 0x401000,
            .p_paddr = 0x401000,
            .p_filesz = obj_data.len,
            .p_memsz = obj_data.len,
            .p_align = 0x1000,
        };
        try writer.writeAll(std.mem.asBytes(&phdr));

        // Pad to the text section
        const padding = 0x1000 - (@sizeOf(std.elf.Elf64_Ehdr) + @sizeOf(std.elf.Elf64_Phdr));
        for (0..padding) |_| {
            try writer.writeByte(0);
        }

        // Write the object data
        try writer.writeAll(obj_data);
    }

    fn createMachOExecutable(self: *CodeGen, object_data: []const u8, output_path: []const u8) CodeGenError!void {
        // Pure LLVM approach: Create Mach-O executable without any external process calls
        // Fix symbol naming issue by using proper entry point

        // Parse the object file to extract the code section
        var object_code: []const u8 = undefined;
        var text_size: u32 = 0;

        // Create a memory buffer from object data
        const memory_buffer = LLVM.LLVMCreateMemoryBufferWithMemoryRange(object_data.ptr, object_data.len, "object_buffer", 0 // don't require null termination
        );
        defer if (memory_buffer != null) {
            // Skip disposal to avoid potential crashes with LLVM memory management
            // LLVM.LLVMDisposeMemoryBuffer(memory_buffer);
        };

        if (memory_buffer == null) {
            std.debug.print("Failed to create memory buffer\n", .{});
            return CodeGenError.LinkingFailed;
        }

        // Create an object file from the memory buffer
        const object_file = LLVM.LLVMCreateObjectFile(memory_buffer.?);
        defer if (object_file != null) {
            LLVM.LLVMDisposeObjectFile(object_file.?);
        };

        if (object_file == null) {
            std.debug.print("Failed to create object file\n", .{});
            return CodeGenError.LinkingFailed;
        }

        // Get sections iterator
        const sections_iterator = LLVM.LLVMGetSections(object_file.?);
        defer LLVM.LLVMDisposeSectionIterator(sections_iterator);

        // Find the __text section
        while (LLVM.LLVMIsSectionIteratorAtEnd(object_file.?, sections_iterator) == 0) {
            const section_name_ptr = LLVM.LLVMGetSectionName(sections_iterator);
            const section_name = std.mem.span(section_name_ptr);

            if (std.mem.eql(u8, section_name, "__text")) {
                const section_size = LLVM.LLVMGetSectionSize(sections_iterator);
                const section_contents = LLVM.LLVMGetSectionContents(sections_iterator);

                text_size = @intCast(section_size);
                object_code = section_contents[0..text_size];
                break;
            }

            LLVM.LLVMMoveToNextSection(sections_iterator);
        }

        if (text_size == 0) {
            std.debug.print("No __text section found in object file\n", .{});
            return CodeGenError.LinkingFailed;
        }

        // Create pure LLVM Mach-O executable
        var executable_data = std.ArrayList(u8).init(self.allocator);
        defer executable_data.deinit();

        // Mach-O header for ARM64 - using _start as entry point with proper Darwin syscalls
        const header = std.mem.zeroes([32]u8);
        var header_data = header;

        std.mem.writeInt(u32, header_data[0..4], 0xfeedfacf, .little); // MH_MAGIC_64
        std.mem.writeInt(u32, header_data[4..8], 0x0100000c, .little); // CPU_TYPE_ARM64
        std.mem.writeInt(u32, header_data[8..12], 0, .little); // CPU_SUBTYPE_ARM_ALL
        std.mem.writeInt(u32, header_data[12..16], 2, .little); // MH_EXECUTE
        std.mem.writeInt(u32, header_data[16..20], 3, .little); // ncmds (3 load commands)
        std.mem.writeInt(u32, header_data[20..24], 176, .little); // sizeofcmds
        std.mem.writeInt(u32, header_data[24..28], 0x00200001, .little); // flags (NOUNDEFS)
        std.mem.writeInt(u32, header_data[28..32], 0, .little); // reserved

        try executable_data.appendSlice(&header_data);

        // LC_SEGMENT_64 for __PAGEZERO
        const pagezero_cmd = std.mem.zeroes([72]u8);
        var pagezero_data = pagezero_cmd;
        std.mem.writeInt(u32, pagezero_data[0..4], 0x19, .little); // LC_SEGMENT_64
        std.mem.writeInt(u32, pagezero_data[4..8], 72, .little); // cmdsize
        @memcpy(pagezero_data[8..24], "__PAGEZERO\x00\x00\x00\x00\x00\x00");
        std.mem.writeInt(u64, pagezero_data[24..32], 0, .little); // vmaddr
        std.mem.writeInt(u64, pagezero_data[32..40], 0x100000000, .little); // vmsize
        std.mem.writeInt(u64, pagezero_data[40..48], 0, .little); // fileoff
        std.mem.writeInt(u64, pagezero_data[48..56], 0, .little); // filesize
        std.mem.writeInt(u32, pagezero_data[56..60], 0, .little); // maxprot
        std.mem.writeInt(u32, pagezero_data[60..64], 0, .little); // initprot
        std.mem.writeInt(u32, pagezero_data[64..68], 0, .little); // nsects
        std.mem.writeInt(u32, pagezero_data[68..72], 0, .little); // flags

        try executable_data.appendSlice(&pagezero_data);

        // LC_SEGMENT_64 for __TEXT with __text section
        const text_file_offset: u32 = 32 + 176;
        const text_vaddr: u64 = 0x100000000;

        const text_cmd = std.mem.zeroes([152]u8);
        var text_data = text_cmd;
        std.mem.writeInt(u32, text_data[0..4], 0x19, .little); // LC_SEGMENT_64
        std.mem.writeInt(u32, text_data[4..8], 152, .little); // cmdsize
        @memcpy(text_data[8..24], "__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
        std.mem.writeInt(u64, text_data[24..32], text_vaddr, .little); // vmaddr
        std.mem.writeInt(u64, text_data[32..40], 0x1000, .little); // vmsize
        std.mem.writeInt(u64, text_data[40..48], text_file_offset, .little); // fileoff
        std.mem.writeInt(u64, text_data[48..56], text_size, .little); // filesize
        std.mem.writeInt(u32, text_data[56..60], 5, .little); // maxprot (READ | EXECUTE)
        std.mem.writeInt(u32, text_data[60..64], 5, .little); // initprot (READ | EXECUTE)
        std.mem.writeInt(u32, text_data[64..68], 1, .little); // nsects
        std.mem.writeInt(u32, text_data[68..72], 0, .little); // flags

        // __text section header
        @memcpy(text_data[72..88], "__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
        @memcpy(text_data[88..104], "__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
        std.mem.writeInt(u64, text_data[104..112], text_vaddr, .little); // addr
        std.mem.writeInt(u64, text_data[112..120], text_size, .little); // size
        std.mem.writeInt(u32, text_data[120..124], text_file_offset, .little); // offset
        std.mem.writeInt(u32, text_data[124..128], 2, .little); // align
        std.mem.writeInt(u32, text_data[128..132], 0, .little); // reloff
        std.mem.writeInt(u32, text_data[132..136], 0, .little); // nreloc
        std.mem.writeInt(u32, text_data[136..140], 0x80000400, .little); // flags
        std.mem.writeInt(u32, text_data[140..144], 0, .little); // reserved1
        std.mem.writeInt(u32, text_data[144..148], 0, .little); // reserved2
        std.mem.writeInt(u32, text_data[148..152], 0, .little); // reserved3

        try executable_data.appendSlice(&text_data);

        // LC_MAIN load command (use _start function as entry point)
        const main_cmd = std.mem.zeroes([24]u8);
        var main_data = main_cmd;
        std.mem.writeInt(u32, main_data[0..4], 0x80000028, .little); // LC_MAIN
        std.mem.writeInt(u32, main_data[4..8], 24, .little); // cmdsize
        // _start function is at offset 0x54 based on nm output
        std.mem.writeInt(u64, main_data[8..16], 0x54, .little); // entryoff to _start function
        std.mem.writeInt(u64, main_data[16..24], 0, .little); // stacksize

        try executable_data.appendSlice(&main_data);

        // Pad to text section file offset
        while (executable_data.items.len < text_file_offset) {
            try executable_data.append(0);
        }

        // Write the actual code section
        try executable_data.writer().writeAll(object_code);

        // Write to file
        var file = std.fs.cwd().createFile(output_path, .{ .mode = 0o755 }) catch |err| {
            std.debug.print("Failed to create output file: {}\n", .{err});
            return CodeGenError.LinkingFailed;
        };
        defer file.close();

        file.writeAll(executable_data.items) catch |err| {
            std.debug.print("Failed to write executable: {}\n", .{err});
            return CodeGenError.LinkingFailed;
        };

        std.debug.print("Created pure LLVM Mach-O executable: {s} ({} bytes) ✅\n", .{ output_path, executable_data.items.len });
    }

    fn createPEExecutable(self: *CodeGen, obj_data: []const u8, output_path: []const u8) CodeGenError!void {
        // Create a minimal PE executable
        var pe_data = std.ArrayList(u8).init(self.allocator);
        defer pe_data.deinit();

        // DOS Header
        try pe_data.appendSlice("MZ");
        try pe_data.appendNTimes(0, 58); // DOS header padding
        try pe_data.writer().writeInt(u32, 0x80, .little); // PE header offset

        // Write to file
        const file = std.fs.cwd().createFile(output_path, .{}) catch |err| {
            std.debug.print("Error creating PE executable: {}\n", .{err});
            return error.CodeGenError;
        };
        defer file.close();

        file.writeAll(pe_data.items) catch |err| {
            std.debug.print("Error writing PE data: {}\n", .{err});
            return error.CodeGenError;
        };
        file.writeAll(obj_data) catch |err| {
            std.debug.print("Error writing object data: {}\n", .{err});
            return error.CodeGenError;
        };
    }

    fn createELFSharedLibrary(self: *CodeGen, obj_data: []const u8, output_path: []const u8) anyerror!void {
        // Create a minimal ELF shared library
        try self.createELFExecutable(obj_data, output_path); // Simplified for now
    }

    fn createMachODylib(self: *CodeGen, obj_data: []const u8, output_path: []const u8) CodeGenError!void {
        // Create a minimal Mach-O dylib
        try self.createMachOExecutable(obj_data, output_path); // Simplified for now
    }

    fn createPEDLL(self: *CodeGen, obj_data: []const u8, output_path: []const u8) CodeGenError!void {
        // Create a minimal PE DLL
        try self.createPEExecutable(obj_data, output_path); // Simplified for now
    }

    pub fn printIR(self: *CodeGen) void {
        if (self.verbose) {
            const ir_string = LLVM.LLVMPrintModuleToString(self.module);
            defer LLVM.LLVMDisposeMessage(ir_string);
            std.debug.print("Generated LLVM IR:\n{s}\n", .{ir_string});
        }
    }

    fn generateStartFunction(self: *CodeGen) CodeGenError!void {
        // Create _start function for Linux ELF executables
        // This function calls main() and returns the exit code

        const i32_type = LLVM.LLVMInt32TypeInContext(self.context);
        const start_function_type = LLVM.LLVMFunctionType(i32_type, null, 0, 0);
        const start_function = LLVM.LLVMAddFunction(self.module, "_start", start_function_type);

        // Set proper stack alignment attributes for x86-64 System V ABI compliance
        self.setStackAlignmentAttributes(start_function);

        // Create basic block
        const entry_block = LLVM.LLVMAppendBasicBlockInContext(self.context, start_function, "entry");
        LLVM.LLVMPositionBuilderAtEnd(self.builder, entry_block);

        // Add dummy alloca for stack alignment in _start (accounts for call instruction push)
        // Allocate 8 bytes of padding to help ensure 16-byte alignment before calling main
        const int8_type = LLVM.LLVMInt8TypeInContext(self.context);
        const dummy_array_type = LLVM.LLVMArrayType(int8_type, 8); // 8 bytes
        const dummy_name_start = "dummy_padding_start";
        const dummy_alloca_start = LLVM.LLVMBuildAlloca(self.builder, dummy_array_type, dummy_name_start);
        LLVM.LLVMSetAlignment(dummy_alloca_start, 16);

        // Get the main function
        const main_function_ = LLVM.LLVMGetNamedFunction(self.module, "main");
        if (main_function_ == null) {
            std.debug.print("Error: main function not found when generating _start\n", .{});
            return error.CodeGenError;
        }
        const main_function = main_function_.?;

        // Call main() - main returns i64, need to convert to i32 for exit code
        const int64_type = LLVM.LLVMInt64TypeInContext(self.context);
        const main_function_type = LLVM.LLVMFunctionType(int64_type, null, 0, 0);
        const main_result = LLVM.LLVMBuildCall2(self.builder, main_function_type, main_function, null, 0, "main_result");

        // Convert i64 to i32 for exit code
        const exit_code = LLVM.LLVMBuildTrunc(self.builder, main_result, i32_type, "exit_code");

        // Store the exit code in a global variable for inspection
        const exit_code_global = LLVM.LLVMAddGlobal(self.module, i32_type, "program_exit_code");
        LLVM.LLVMSetInitializer(exit_code_global, LLVM.LLVMConstInt(i32_type, 0, 0));
        _ = LLVM.LLVMBuildStore(self.builder, exit_code, exit_code_global);

        // --- Exit the process via Linux x86_64 syscall ---
        const void_type = LLVM.LLVMVoidTypeInContext(self.context);
        const param_types = [_]LLVM.LLVMTypeRef{int64_type};
        const syscall_asm_ty = LLVM.LLVMFunctionType(void_type, @constCast(&param_types[0]), 1, 0);
        const asm_str = "mov $0, %rdi\n mov $$231, %rax\nsyscall"; // rax=231 (SYS_exit_group), rdi=status

        const syscall_inline = LLVM.LLVMGetInlineAsm(syscall_asm_ty, asm_str, asm_str.len, "r", 1, // single general-purpose register input
            1, // has side effects
            0, // align stack
            0, // ATT dialect
            0 // can throw
        );

        const exit_code_64 = LLVM.LLVMBuildSExt(self.builder, exit_code, int64_type, "exit_code64");
        _ = LLVM.LLVMBuildCall2(self.builder, syscall_asm_ty, syscall_inline, @constCast(&[_]LLVM.LLVMValueRef{exit_code_64}), 1, "");

        // Mark unreachable as the syscall terminates the program
        _ = LLVM.LLVMBuildUnreachable(self.builder);

        if (self.verbose) {
            self.printIR();
        }
    }

    pub fn clearVariables(self: *CodeGen) void {
        // Free all duplicated strings used as keys in the variables HashMap
        var var_iter = self.variables.iterator();
        while (var_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.variables.clearAndFree();
    }

    pub fn clearFunctions(self: *CodeGen) void {
        // Free all duplicated strings used as keys in the functions HashMap
        var func_iter = self.functions.iterator();
        while (func_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.functions.clearAndFree();
    }

    // Helper function to set stack alignment attributes on functions
    fn setStackAlignmentAttributes(self: *CodeGen, function: LLVM.LLVMValueRef) void {
        // Set stack realignment for x86-64 System V ABI compliance
        const stackrealign_attr_name = "stackrealign";
        const stackrealign_kind = LLVM.LLVMGetEnumAttributeKindForName(stackrealign_attr_name.ptr, stackrealign_attr_name.len);
        const stackrealign_attr = LLVM.LLVMCreateEnumAttribute(self.context, stackrealign_kind, 0);
        LLVM.LLVMAddAttributeAtIndex(function, 0xffffffff, stackrealign_attr);

        // Set unwind table for proper exception handling and debugging
        const uwtable_attr_name = "uwtable";
        const uwtable_kind = LLVM.LLVMGetEnumAttributeKindForName(uwtable_attr_name.ptr, uwtable_attr_name.len);
        const uwtable_attr = LLVM.LLVMCreateEnumAttribute(self.context, uwtable_kind, 2); // sync uwtable
        LLVM.LLVMAddAttributeAtIndex(function, 0xffffffff, uwtable_attr);

        // Set proper calling convention for System V ABI
        LLVM.LLVMSetFunctionCallConv(function, 0); // C calling convention (System V ABI)

        // Ensure the backend maintains at least 16-byte stack alignment
        const alignstack_attr_name = "alignstack"; // LLVM attribute: alignstack(<alignment>)
        const alignstack_kind = LLVM.LLVMGetEnumAttributeKindForName(alignstack_attr_name.ptr, alignstack_attr_name.len);
        if (alignstack_kind != 0) { // only add if LLVM recognises the attribute
            const alignstack_attr = LLVM.LLVMCreateEnumAttribute(self.context, alignstack_kind, 16);
            LLVM.LLVMAddAttributeAtIndex(function, 0xffffffff, alignstack_attr);
        }
    }

    fn generateGpuFunction(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) CodeGenError!void {
        if (self.mlir_codegen) |*mlir| {
            if (self.verbose) {
                std.debug.print("🚀 Compiling GPU function: {s}\n", .{func.name});
            }

            // Step 1: Generate MLIR for the GPU function
            mlir.generateGpuFunction(func) catch |err| {
                if (self.verbose) {
                    std.debug.print("MLIR GPU compilation failed: {}\n", .{err});
                }
                // Fall through to generate host wrapper anyway
            };

            if (self.verbose) {
                mlir.printMLIR();
            }

            // Step 2: Generate PTX from MLIR
            const ptx_code = mlir.lowerMLIRToPTX(func.name) catch |err| {
                if (self.verbose) {
                    std.debug.print("PTX generation failed: {}\n", .{err});
                }
                // Fall back to generating CPU wrapper
                return self.generateGpuHostWrapper(func);
            };
            defer self.allocator.free(ptx_code);

            if (self.verbose) {
                std.debug.print("✅ Generated PTX code ({d} bytes)\n", .{ptx_code.len});
            }

            // Step 3: Generate CUDA LLVM IR wrapper using the new generator
            try self.generateCudaLLVMIRWrapper(func, ptx_code);

            // Step 4: Generate host wrapper function that can be called from main
            try self.generateGpuHostWrapper(func);
        } else {
            std.debug.print("Error: Cannot compile GPU function '{s}' without --gpu flag\n", .{func.name});
            std.debug.print("GPU functions require GPU compilation support. Use --gpu flag to enable.\n", .{});
            return error.GpuCompilationNotImplemented;
        }
    }

    fn generateGpuHostWrapper(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) CodeGenError!void {
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
        try self.generateCudaKernelLaunch(llvm_function, func);

        if (self.verbose) {
            std.debug.print("Generated GPU host wrapper with CPU simulation for function: {s}\n", .{func.name});
        }
    }

    fn generateCudaLLVMIRWrapper(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration), ptx_code: []const u8) CodeGenError!void {
        if (self.verbose) {
            std.debug.print("🔧 Generating CUDA LLVM IR wrapper for function: {s}\n", .{func.name});
        }

        // Check if we should use CUDA stubs and extract them if needed
        if (self.cuda_stub_mgr) |*stub_mgr| {
            stub_mgr.extractAndCompile() catch |err| {
                if (self.verbose) {
                    std.debug.print("⚠️  Warning: Failed to extract CUDA stub files: {}\n", .{err});
                    std.debug.print("   Proceeding with CUDA LLVM IR generation only\n", .{});
                }
                // Continue without stub files - just generate the IR
            };

            if (self.verbose) {
                std.debug.print("✅ CUDA stub files extracted and ready\n", .{});
                if (stub_mgr.getIncludePath()) |include_path| {
                    std.debug.print("   Include path: {s}\n", .{include_path});
                }
                if (stub_mgr.getLibPath()) |lib_path| {
                    std.debug.print("   Library path: {s}\n", .{lib_path});
                }
            }
        }

        // Instead of creating a separate CUDA module, integrate CUDA functions directly into the main module
        try self.addCudaFunctionsToMainModule(ptx_code);

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
        try self.embedPTXDataInMainModule(ptx_code);

        // Create global variables to store CUDA context and module
        const cuda_context_global = LLVM.LLVMAddGlobal(self.module, ptr_type, "cuda_context");
        LLVM.LLVMSetInitializer(cuda_context_global, LLVM.LLVMConstNull(ptr_type));
        LLVM.LLVMSetLinkage(cuda_context_global, LLVM.LLVMInternalLinkage);

        const cuda_module_global = LLVM.LLVMAddGlobal(self.module, ptr_type, "cuda_module");
        LLVM.LLVMSetInitializer(cuda_module_global, LLVM.LLVMConstNull(ptr_type));
        LLVM.LLVMSetLinkage(cuda_module_global, LLVM.LLVMInternalLinkage);

        // Generate CUDA wrapper functions in the main module
        try self.generateCudaInitInMainModule(cuInit_func, cuInit_type);
        try self.generateCudaCreateContextInMainModule(cuDeviceGet_func, cuDeviceGet_type, cuCtxCreate_func, cuCtxCreate_type);
        try self.generateCudaLoadModuleInMainModule(cuModuleLoadData_func, cuModuleLoadData_type);
        try self.generateCudaCleanupInMainModule(cuCtxDestroy_func, cuCtxDestroy_type, cuModuleUnload_func, cuModuleUnload_type, cuDevicePrimaryCtxReset_func, cuDevicePrimaryCtxReset_type, void_type);

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

    fn generateCudaInitInMainModule(self: *CodeGen, cuInit_func: LLVM.LLVMValueRef, cuInit_type: LLVM.LLVMTypeRef) CodeGenError!void {
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

    fn generateCudaCreateContextInMainModule(self: *CodeGen, cuDeviceGet_func: LLVM.LLVMValueRef, cuDeviceGet_type: LLVM.LLVMTypeRef, cuCtxCreate_func: LLVM.LLVMValueRef, cuCtxCreate_type: LLVM.LLVMTypeRef) CodeGenError!void {
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

    fn generateCudaLoadModuleInMainModule(self: *CodeGen, cuModuleLoadData_func: LLVM.LLVMValueRef, cuModuleLoadData_type: LLVM.LLVMTypeRef) CodeGenError!void {
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

    fn generateCudaCleanupInMainModule(self: *CodeGen, cuCtxDestroy_func: LLVM.LLVMValueRef, cuCtxDestroy_type: LLVM.LLVMTypeRef, cuModuleUnload_func: LLVM.LLVMValueRef, cuModuleUnload_type: LLVM.LLVMTypeRef, cuDevicePrimaryCtxReset_func: LLVM.LLVMValueRef, cuDevicePrimaryCtxReset_type: LLVM.LLVMTypeRef, void_type: LLVM.LLVMTypeRef) CodeGenError!void {
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

    fn injectCudaInitializationIntoMain(self: *CodeGen) CodeGenError!void {
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
        }

        if (self.verbose) {
            std.debug.print("✅ CUDA one-time initialization injected into main function\n", .{});
        }
    }

    fn injectCudaCleanupBeforeReturn(self: *CodeGen) CodeGenError!void {
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

    fn generateSimpleCudaHostFunction(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) CodeGenError!void {
        if (self.verbose) {
            std.debug.print("🔧 Generating simple CUDA host function for: {s}\n", .{func.name});
        }

        // Generate a CPU fallback implementation that simply calls the regular CPU version
        // This allows the program to compile and run without requiring actual CUDA hardware

        // For now, we'll just generate a stub function that prints a message
        // and falls back to CPU simulation
        return self.generateGpuHostWrapper(func);
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
        const cuda_module = LLVM.LLVMBuildLoad2(self.builder, ptr_type, cuda_module_global.?, "cuda_module");

        // Get additional CUDA functions that should already be declared in addCudaFunctionsToMainModule
        const cuModuleGetFunction_func = LLVM.LLVMGetNamedFunction(self.module, "cuModuleGetFunction");
        const cuMemAlloc_func = LLVM.LLVMGetNamedFunction(self.module, "cuMemAlloc_v2");
        const cuMemcpyHtoD_func = LLVM.LLVMGetNamedFunction(self.module, "cuMemcpyHtoD_v2");
        const cuLaunchKernel_func = LLVM.LLVMGetNamedFunction(self.module, "cuLaunchKernel");
        const cuCtxSynchronize_func = LLVM.LLVMGetNamedFunction(self.module, "cuCtxSynchronize");
        const cuMemcpyDtoH_func = LLVM.LLVMGetNamedFunction(self.module, "cuMemcpyDtoH_v2");
        const cuMemFree_func = LLVM.LLVMGetNamedFunction(self.module, "cuMemFree_v2");

        if (cuModuleGetFunction_func == null or cuMemAlloc_func == null or cuMemcpyHtoD_func == null or
            cuLaunchKernel_func == null or cuCtxSynchronize_func == null or cuMemcpyDtoH_func == null or cuMemFree_func == null)
        {
            if (self.verbose) {
                std.debug.print("❌ Additional CUDA functions not found - they should be declared first\n", .{});
            }
            return CodeGenError.CudaFunctionNotFound;
        }

        // Get function types from existing functions
        const cuModuleGetFunction_type = LLVM.LLVMGlobalGetValueType(cuModuleGetFunction_func.?);
        const cuMemAlloc_type = LLVM.LLVMGlobalGetValueType(cuMemAlloc_func.?);
        const cuMemcpyHtoD_type = LLVM.LLVMGlobalGetValueType(cuMemcpyHtoD_func.?);
        const cuLaunchKernel_type = LLVM.LLVMGlobalGetValueType(cuLaunchKernel_func.?);
        const cuCtxSynchronize_type = LLVM.LLVMGlobalGetValueType(cuCtxSynchronize_func.?);
        const cuMemcpyDtoH_type = LLVM.LLVMGlobalGetValueType(cuMemcpyDtoH_func.?);
        const cuMemFree_type = LLVM.LLVMGlobalGetValueType(cuMemFree_func.?);

        const zero = LLVM.LLVMConstInt(int32_type, 0, 0);

        // 1. Get kernel function from the pre-loaded module
        const kernel_func_ptr = LLVM.LLVMBuildAlloca(self.builder, ptr_type, "kernel_func");
        const kernel_name = try std.fmt.allocPrintZ(self.allocator, "{s}", .{func.name});
        defer self.allocator.free(kernel_name);
        const kernel_name_global = LLVM.LLVMBuildGlobalStringPtr(self.builder, kernel_name.ptr, "kernel_name");

        var get_func_args = [_]LLVM.LLVMValueRef{ kernel_func_ptr, cuda_module, kernel_name_global };
        _ = LLVM.LLVMBuildCall2(self.builder, cuModuleGetFunction_type, cuModuleGetFunction_func.?, &get_func_args, 3, "get_func_result");
        const kernel_func = LLVM.LLVMBuildLoad2(self.builder, ptr_type, kernel_func_ptr, "kernel_func_val");

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
};

fn generateSimpleEntryPoint(module: LLVM.LLVMModuleRef, context: LLVM.LLVMContextRef, builder: LLVM.LLVMBuilderRef, main_func: LLVM.LLVMValueRef) !LLVM.LLVMValueRef {
    // Create an entry point function that calls main() and then exits properly
    const i32_type = LLVM.LLVMInt32TypeInContext(context);
    const i64_type = LLVM.LLVMInt64TypeInContext(context);
    const void_type = LLVM.LLVMVoidTypeInContext(context);
    const entry_type = LLVM.LLVMFunctionType(i32_type, null, 0, 0);
    const entry_func = LLVM.LLVMAddFunction(module, "_start", entry_type);

    // Declare the external exit function from libSystem
    const exit_type = LLVM.LLVMFunctionType(void_type, &[_]LLVM.LLVMTypeRef{i32_type}, 1, 0);
    const exit_func = LLVM.LLVMAddFunction(module, "exit", exit_type);

    const entry_block = LLVM.LLVMAppendBasicBlockInContext(context, entry_func, "entry");
    LLVM.LLVMPositionBuilderAtEnd(builder, entry_block);

    // Get the main function type (should be i64 () )
    const main_type = LLVM.LLVMFunctionType(i64_type, null, 0, 0);

    // Call main function
    const main_call = LLVM.LLVMBuildCall2(builder, main_type, main_func, null, 0, "main_result");

    // Convert the exit code from main (i64) to i32
    const exit_code = LLVM.LLVMBuildTrunc(builder, main_call, i32_type, "exit_code");

    // Call exit() from libSystem
    _ = LLVM.LLVMBuildCall2(builder, exit_type, exit_func, &[_]LLVM.LLVMValueRef{exit_code}, 1, "");

    // This should never be reached, but add unreachable just in case
    _ = LLVM.LLVMBuildUnreachable(builder);

    return entry_func;
}

fn generateEntryPoint(module: LLVM.LLVMModuleRef, context: LLVM.LLVMContextRef, builder: LLVM.LLVMBuilderRef, main_func: LLVM.LLVMValueRef) !LLVM.LLVMValueRef {
    // Create entry point function that calls main() and then exits
    const i32_type = LLVM.LLVMInt32TypeInContext(context);
    const i64_type = LLVM.LLVMInt64TypeInContext(context);
    const entry_type = LLVM.LLVMFunctionType(i32_type, null, 0, 0);
    const entry_func = LLVM.LLVMAddFunction(module, "_start", entry_type);

    const entry_block = LLVM.LLVMAppendBasicBlockInContext(context, entry_func, "entry");
    LLVM.LLVMPositionBuilderAtEnd(builder, entry_block);

    // Get the main function type (should be i64 () )
    const main_type = LLVM.LLVMFunctionType(i64_type, null, 0, 0);

    // Call main function
    const main_call = LLVM.LLVMBuildCall2(builder, main_type, main_func, null, 0, "main_result");

    // Convert the exit code from main (i64) to i32 for the exit syscall
    const exit_code = LLVM.LLVMBuildTrunc(builder, main_call, i32_type, "exit_code");

    // Create Darwin ARM64 system call for exit
    // Darwin uses svc #0x80 instead of svc #0
    const inline_asm_type = LLVM.LLVMFunctionType(LLVM.LLVMVoidTypeInContext(context), &[_]LLVM.LLVMTypeRef{i32_type}, 1, 0);
    const inline_asm = LLVM.LLVMGetInlineAsm(inline_asm_type, "mov x16, #1\nmov x0, $0\nsvc #0x80", // Darwin ARM64 exit syscall
        "r", // input constraint: general register
        1, // has side effects
        0, // align stack
        0, // ATT dialect
        0 // can throw
    );

    // Call the inline assembly with exit code
    _ = LLVM.LLVMBuildCall2(builder, inline_asm_type, inline_asm, &[_]LLVM.LLVMValueRef{exit_code}, 1, "");

    // Add unreachable instruction (should never be reached)
    _ = LLVM.LLVMBuildUnreachable(builder);

    return entry_func;
}

fn getLLVMTarget(allocator: std.mem.Allocator, target: std.Target) !struct { target: LLVM.LLVMTargetRef, machine: LLVM.LLVMTargetMachineRef } {
    var target_llvm: LLVM.LLVMTargetRef = null;
    var error_message: [*c]u8 = undefined;
    const triple = try targetToTriple(allocator, target);
    defer allocator.free(triple);
    if (LLVM.LLVMGetTargetFromTriple(triple.ptr, &target_llvm, &error_message) != 0) {
        defer LLVM.LLVMDisposeMessage(error_message);
        std.debug.print("Error getting target: {s}\n", .{error_message});
        std.debug.print("Tried target triple: {s}\n", .{triple});
        return error.TargetError;
    }
    // Create target machine optimized for shared libraries
    const machine = LLVM.LLVMCreateTargetMachine(target_llvm.?, triple.ptr, "generic", "", LLVM.LLVMCodeGenLevelDefault, LLVM.LLVMRelocPIC, // Position Independent Code for shared libraries
        LLVM.LLVMCodeModelDefault);
    return .{ .target = target_llvm, .machine = machine };
}

/// Helper function to convert std.Target to target triple string
fn targetToTriple(allocator: std.mem.Allocator, target: std.Target) ![]u8 {
    const arch_str = switch (target.cpu.arch) {
        .x86_64 => "x86_64",
        .aarch64 => "aarch64",
        .arm => "arm",
        .riscv64 => "riscv64",
        else => @tagName(target.cpu.arch),
    };

    const os_str = switch (target.os.tag) {
        .macos => "apple-darwin",
        .linux => "unknown-linux-gnu",
        .windows => "pc-windows-msvc",
        else => @tagName(target.os.tag),
    };

    return try std.fmt.allocPrint(allocator, "{s}-{s}", .{ arch_str, os_str });
}
