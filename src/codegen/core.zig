const std = @import("std");
const parser = @import("../parser.zig");
const typechecker = @import("../typechecker.zig");
const mlir_codegen = @import("mlir.zig");
const cuda_stub_manager = @import("cuda_stub_manager.zig");
const gpu_memory_tracker = @import("gpu_memory_tracker.zig");
const gpu_memory_ops = @import("gpu_memory_ops.zig");
const gpu = @import("gpu.zig");
const linking = @import("linking.zig");
pub const LLVM = @cImport({
    @cInclude("llvm-c/Core.h");
    @cInclude("llvm-c/TargetMachine.h");
    @cInclude("llvm-c/Target.h");
    @cInclude("llvm-c/Support.h");
    @cInclude("llvm-c/BitReader.h");
    @cInclude("llvm-c/Object.h");
});

pub const CodeGenError = error{ InvalidTopLevelNode, InvalidStatement, InvalidExpression, InvalidCallee, UndefinedVariable, UndefinedFunction, TargetError, CodeGenError, MainFunctionNotFound, MissingMainFunction, LinkingFailed, GpuCompilationNotImplemented, InvalidGpuTriplet, InvalidTargetTriple, InvalidCharacter, Overflow, CudaFunctionNotFound, CudaStubInitFailed, UnsupportedOperation } || std.mem.Allocator.Error;

const VarInfo = struct {
    alloca: LLVM.LLVMValueRef,
    ty: parser.Type,
};

pub const Accelerator = struct {
    codegen: mlir_codegen.MLIRCodeGen,
    stub: cuda_stub_manager.CudaStubManager,

    pub fn init(allocator: std.mem.Allocator, gpu_target: std.Target, verbose: bool) !Accelerator {
        const mlir_gen = try mlir_codegen.MLIRCodeGen.init(allocator, gpu_target, verbose);
        const cuda_stub_mgr = cuda_stub_manager.CudaStubManager.init(allocator, verbose) catch |err| {
            if (verbose) {
                std.debug.print("⚠️  Warning: Failed to initialize CUDA stub manager: {}\n", .{err});
            }
            // If CUDA stub manager fails, we still want to continue with MLIR codegen
            // Create a minimal stub manager or handle this case appropriately
            return error.CudaStubInitFailed;
        };

        return Accelerator{
            .codegen = mlir_gen,
            .stub = cuda_stub_mgr,
        };
    }

    pub fn deinit(self: *Accelerator) void {
        self.codegen.deinit();
        self.stub.deinit();
    }
};

pub const CodeGen = struct {
    context: LLVM.LLVMContextRef,
    module: LLVM.LLVMModuleRef,
    builder: LLVM.LLVMBuilderRef,
    allocator: std.mem.Allocator,
    verbose: bool,
    variables: std.HashMap([]const u8, VarInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    functions: std.HashMap([]const u8, LLVM.LLVMValueRef, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    accelerator: ?Accelerator,
    gpu_function_names: ?std.ArrayList([]const u8),
    gpu_memory_tracker: ?gpu_memory_tracker.GpuMemoryTracker,
    gpu_memory_ops: ?gpu_memory_ops.GpuMemoryOps,
    reduction_info: ?std.AutoHashMap(*parser.ASTNode, typechecker.ReductionInfo),

    pub fn init(allocator: std.mem.Allocator, module_name: []const u8, verbose: bool, target: std.Target, gpu_target: ?std.Target) !CodeGen {
        _ = target; // Target is passed for completeness but not used in init
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
        var accelerator: ?Accelerator = null;
        if (gpu_target) |gpu_| {
            accelerator = Accelerator.init(allocator, gpu_, verbose) catch |err| blk: {
                if (verbose) {
                    std.debug.print("⚠️  Warning: Failed to initialize accelerator: {}\n", .{err});
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
            .accelerator = accelerator,
            .gpu_function_names = null,
            .gpu_memory_tracker = if (accelerator != null) gpu_memory_tracker.GpuMemoryTracker.init(allocator, verbose) else null,
            .gpu_memory_ops = null, // Will be initialized after CUDA functions are declared
            .reduction_info = null,
        };
    }

    pub fn setReductionInfo(self: *CodeGen, info: std.AutoHashMap(*parser.ASTNode, typechecker.ReductionInfo)) void {
        self.reduction_info = info;
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

        // Free GPU function names if allocated
        if (self.gpu_function_names) |gpu_names| {
            for (gpu_names.items) |name| {
                self.allocator.free(name);
            }
            gpu_names.deinit();
        }

        // Clean up GPU memory tracker
        if (self.gpu_memory_tracker) |*tracker| {
            tracker.deinit();
        }

        // Clean up accelerator resources
        if (self.accelerator) |*accel| {
            accel.deinit();
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
        return self.accelerator != null;
    }

    fn collectGpuFunctionNames(self: *CodeGen, ast: parser.ASTNode, allocator: std.mem.Allocator) !std.ArrayList([]const u8) {
        _ = self;
        var gpu_function_names = std.ArrayList([]const u8).init(allocator);

        switch (ast) {
            .program => |prog| {
                for (prog.statements) |stmt| {
                    if (stmt == .function_declaration) {
                        const func = stmt.function_declaration;
                        if (std.mem.startsWith(u8, func.name, "gpu_")) {
                            try gpu_function_names.append(try allocator.dupe(u8, func.name));
                        }
                    }
                }
            },
            else => {},
        }

        return gpu_function_names;
    }

    pub fn generate(self: *CodeGen, ast: parser.ASTNode, mode: enum { executable, library }) CodeGenError!void {
        // Generate LLVM IR for the program AST node
        switch (ast) {
            .program => |prog| {
                // Always collect GPU function names to check if --gpu is needed
                var temp_gpu_names = try self.collectGpuFunctionNames(ast, self.allocator);
                const has_gpu_functions = temp_gpu_names.items.len > 0;

                // If we found GPU functions but don't have GPU support enabled, error out
                if (has_gpu_functions and self.accelerator == null) {
                    defer temp_gpu_names.deinit();
                    std.debug.print("Error: Cannot compile GPU function '{s}' without --gpu flag\n", .{temp_gpu_names.items[0]});
                    std.debug.print("GPU functions require GPU compilation support. Use --gpu flag to enable.\n", .{});
                    return error.GpuCompilationNotImplemented;
                }

                // Store GPU function names if we have GPU support
                if (self.accelerator != null and has_gpu_functions) {
                    self.gpu_function_names = temp_gpu_names;
                    if (self.verbose) {
                        std.debug.print("🔧 Collected {} GPU functions: ", .{temp_gpu_names.items.len});
                        for (temp_gpu_names.items) |name| {
                            std.debug.print("{s} ", .{name});
                        }
                        std.debug.print("\n", .{});
                    }
                } else if (self.accelerator != null) {
                    // Have GPU support but no GPU functions
                    self.gpu_function_names = temp_gpu_names;
                } else {
                    // No GPU support, clean up
                    temp_gpu_names.deinit();
                }

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

                // First, collect and process all GPU functions together in a single module
                if (self.accelerator != null and self.gpu_function_names != null and self.gpu_function_names.?.items.len > 0) {
                    try gpu.generateAllGpuFunctions(self, prog.statements);
                }

                // Then, process non-GPU functions
                for (prog.statements) |stmt| {
                    if (stmt == .function_declaration) {
                        const func = stmt.function_declaration;
                        if (std.mem.eql(u8, func.name, "main")) {
                            main_func_decl = func;
                        } else if (!std.mem.startsWith(u8, func.name, "gpu_")) {
                            // Skip GPU functions as they're processed separately above
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
        linking.setStackAlignmentAttributes(self, llvm_function);

        try self.functions.put(try self.allocator.dupe(u8, func.name), llvm_function);

        if (self.verbose) {
            std.debug.print("🔧 Declared function signature: {s}\n", .{func.name});
        }
    }

    fn generateFunctionBody(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) CodeGenError!void {
        // GPU functions are processed separately in generateAllGpuFunctions
        if (std.mem.startsWith(u8, func.name, "gpu_")) {
            // This should not be reached since GPU functions are filtered out in generateWithMode
            if (self.verbose) {
                std.debug.print("⚠️  GPU function {s} reached generateFunctionBody - this should not happen\n", .{func.name});
            }
            return;
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
                std.debug.print("   self.accelerator != null: {}\n", .{self.accelerator != null});
                std.debug.print("   hasGpuFunctions(): {}\n", .{self.hasGpuFunctions()});
            }
            if (self.hasGpuFunctions()) {
                if (self.verbose) {
                    std.debug.print("🔧 Injecting CUDA initialization into main function\n", .{});
                }
                try gpu.injectCudaInitializationIntoMain(self);
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
        // Check if we're in the main function with GPU functions
        const current_function = LLVM.LLVMGetBasicBlockParent(LLVM.LLVMGetInsertBlock(self.builder));
        const function_name = LLVM.LLVMGetValueName(current_function);
        const is_main = std.mem.eql(u8, std.mem.span(function_name), "main");

        if (ret.value) |value| {
            // Generate the return expression FIRST (this may trigger GPU sync)
            const llvm_value = try self.generateExpression(value.*);

            // NOW inject CUDA cleanup after expression evaluation but before return
            if (is_main and self.hasGpuFunctions()) {
                if (self.verbose) {
                    std.debug.print("🧹 Injecting CUDA cleanup after expression evaluation\n", .{});
                }
                // Free any allocated GPU memory first
                try gpu.freeAllocatedGpuMemory(self);
                try gpu.injectCudaCleanupBeforeReturn(self);
            }

            _ = LLVM.LLVMBuildRet(self.builder, llvm_value);
        } else {
            // Return statement has no value - must be a void function
            // Inject cleanup before void return
            if (is_main and self.hasGpuFunctions()) {
                if (self.verbose) {
                    std.debug.print("🧹 Injecting CUDA cleanup before void return\n", .{});
                }
                // Free any allocated GPU memory first
                try gpu.freeAllocatedGpuMemory(self);
                try gpu.injectCudaCleanupBeforeReturn(self);
            }
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

        // For reduce expressions, we need special handling
        const rhs_val = blk: {
            if (pa.value.* == .reduce_expression) {
                // Get the reduction info if available
                const reduce_node = &pa.value.*.reduce_expression;
                if (self.verbose) {
                    std.debug.print("DEBUG: Looking for reduction info at node ptr: {*}\n", .{pa.value});
                }
                const reduction_info = if (self.reduction_info) |info| info.get(pa.value) else null;

                if (reduction_info) |red_info| {
                    // Generate multi-dimensional reduction based on free/bound indices
                    if (self.verbose) {
                        std.debug.print("DEBUG: Found reduction info! Using multi-dimensional reduction with {} free indices and {} bound indices\n", .{ red_info.free_indices.len, red_info.bound_indices.len });
                    }
                    break :blk try self.generateMultiDimensionalReduce(reduce_node, red_info, idx_val);
                } else {
                    // Fallback to simple reduction
                    if (self.verbose) {
                        std.debug.print("DEBUG: No reduction info found, using simple reduction\n", .{});
                    }
                    break :blk try self.generateExpression(pa.value.*);
                }
            } else {
                break :blk try self.generateTensorExpression(pa.value.*, idx_val);
            }
        };
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
                        // Check if we need to sync GPU data back to CPU
                        if (self.gpu_memory_tracker) |*tracker| {
                            if (self.verbose) {
                                std.debug.print("  🔍 Checking GPU sync for tensor identifier '{s}'\n", .{ident.name});
                                tracker.printState();
                            }
                            if (tracker.needsTransferToCpu(ident.name)) {
                                if (self.gpu_memory_ops) |*ops| {
                                    if (self.verbose) {
                                        std.debug.print("  🔄 Synchronizing GPU data for '{s}' before CPU access\n", .{ident.name});
                                    }

                                    // Synchronize GPU operations
                                    ops.synchronize();

                                    // Copy data back from GPU
                                    if (tracker.getGpuPtr(ident.name)) |gpu_ptr_opt| {
                                        const gpu_ptr = gpu_ptr_opt;
                                        const size_t_type = LLVM.LLVMInt64TypeInContext(self.context);
                                        const tensor_size = LLVM.LLVMConstInt(size_t_type, 1024 * 4, 0); // TODO: get actual size
                                        const ptr_type = LLVM.LLVMPointerType(LLVM.LLVMInt8TypeInContext(self.context), 0);
                                        const host_ptr = LLVM.LLVMBuildBitCast(self.builder, info.alloca, ptr_type, "host_ptr");

                                        ops.copyDeviceToHost(host_ptr, gpu_ptr, tensor_size);
                                        try tracker.markCopiedToCpu(ident.name);

                                        if (self.verbose) {
                                            std.debug.print("  ⬇️  Copied '{s}' back to CPU\n", .{ident.name});
                                        }
                                    }
                                }
                            }
                        }
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

                // Check if this is a GPU function call with optimized memory management
                if (self.gpu_memory_tracker != null and
                    self.gpu_memory_ops != null and
                    std.mem.startsWith(u8, callee_name, "gpu_"))
                {

                    // Use the optimized GPU launcher
                    const launcher_name = try std.fmt.allocPrint(self.allocator, "{s}_gpu_launch", .{callee_name});
                    defer self.allocator.free(launcher_name);

                    if (self.functions.get(launcher_name)) |launcher_function| {
                        return try self.generateOptimizedGpuCall(call, callee_name, launcher_function);
                    }
                }

                // Regular function call
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

                    const result = LLVM.LLVMBuildCall2(self.builder, function_type, function, args.ptr, @intCast(args.len), "call");

                    // Mark tensor arguments as modified on CPU (for non-GPU functions)
                    if (self.gpu_memory_tracker) |*tracker| {
                        for (call.arguments) |arg_node| {
                            if (arg_node == .identifier) {
                                const var_name = arg_node.identifier.name;
                                if (self.variables.get(var_name)) |var_info| {
                                    if (var_info.ty == .tensor) {
                                        try tracker.markModifiedOnCpu(var_name);
                                        if (self.verbose) {
                                            std.debug.print("  ✏️  Marked '{s}' as modified by CPU function '{s}'\n", .{ var_name, callee_name });
                                        }
                                    }
                                }
                            }
                        }
                    }

                    return result;
                } else {
                    return error.UndefinedFunction;
                }
            },
            .tensor_literal => |tensor_lit| {
                // Create a flattened array for multi-dimensional tensors
                const element_type = self.toLLVMType(tensor_lit.element_type);

                // Calculate total number of elements
                var total_elements: usize = 1;
                for (tensor_lit.shape) |dim| {
                    total_elements *= dim;
                }

                const value = try self.generateExpression(tensor_lit.value.*);

                // Create a constant array with the value repeated for all elements
                const values = try self.allocator.alloc(LLVM.LLVMValueRef, total_elements);
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

                // Check if we need to sync GPU data back to CPU
                if (self.gpu_memory_tracker) |*tracker| {
                    if (self.verbose) {
                        std.debug.print("  🔍 Checking GPU sync for tensor_slice of '{s}'\n", .{name_slice});
                        tracker.printState();
                    }
                    if (tracker.needsTransferToCpu(name_slice)) {
                        if (self.gpu_memory_ops) |*ops| {
                            if (self.verbose) {
                                std.debug.print("  🔄 Synchronizing GPU data for '{s}' before CPU access\n", .{name_slice});
                            }

                            // Synchronize GPU operations
                            ops.synchronize();

                            // Copy data back from GPU
                            if (tracker.getGpuPtr(name_slice)) |gpu_ptr_opt| {
                                const gpu_ptr = gpu_ptr_opt;
                                const size_t_type = LLVM.LLVMInt64TypeInContext(self.context);
                                const tensor_size = LLVM.LLVMConstInt(size_t_type, 1024 * 4, 0); // TODO: get actual size
                                const ptr_type = LLVM.LLVMPointerType(LLVM.LLVMInt8TypeInContext(self.context), 0);
                                const host_ptr = LLVM.LLVMBuildBitCast(self.builder, tensor_ptr, ptr_type, "host_ptr");

                                ops.copyDeviceToHost(host_ptr, gpu_ptr, tensor_size);
                                try tracker.markCopiedToCpu(name_slice);

                                if (self.verbose) {
                                    std.debug.print("  ⬇️  Copied '{s}' back to CPU\n", .{name_slice});
                                }
                            }
                        }
                    }
                }

                const array_ty = self.toLLVMType(vi_slice.ty);
                const elem_ty = self.toLLVMType(vi_slice.ty.tensor.element_type.*);

                // Handle multi-dimensional indexing by computing linear index
                const i64_type = LLVM.LLVMInt64TypeInContext(self.context);
                var linear_index = LLVM.LLVMConstInt(i64_type, 0, 0);

                // Calculate linear index from multi-dimensional indices
                // For tensor with shape [d0, d1, ..., dn], index [i0, i1, ..., in] maps to:
                // i0 * (d1 * d2 * ... * dn) + i1 * (d2 * ... * dn) + ... + in
                const tensor_shape = vi_slice.ty.tensor.shape;

                for (tensor_slice.indices, 0..) |*idx_expr, dim| {
                    const idx_val = try self.generateExpression(idx_expr.*);

                    // Calculate stride for this dimension
                    var stride: u64 = 1;
                    for (tensor_shape[dim + 1 ..]) |d| {
                        stride *= d;
                    }

                    // Add idx_val * stride to linear_index
                    const stride_val = LLVM.LLVMConstInt(i64_type, stride, 0);
                    const contribution = LLVM.LLVMBuildMul(self.builder, idx_val, stride_val, "idx_contribution");
                    linear_index = LLVM.LLVMBuildAdd(self.builder, linear_index, contribution, "linear_idx");
                }

                // Build GEP indices [0, linear_index] to address the desired element.
                const zero_const = LLVM.LLVMConstInt(i64_type, 0, 0);
                var gep_indices = [_]LLVM.LLVMValueRef{ zero_const, linear_index };

                const elem_ptr = LLVM.LLVMBuildGEP2(self.builder, array_ty, tensor_ptr, &gep_indices[0], 2, "tensor_slice_elem_ptr");
                self.printIR();

                return LLVM.LLVMBuildLoad2(self.builder, elem_ty, elem_ptr, "tensor_slice_load");
            },
            .reduce_expression => |reduce| {
                // For now, implement a simple reduction that sums all elements
                // This is a very basic implementation - in practice, you'd want to:
                // 1. Analyze the tensor expression to determine which dimensions to reduce
                // 2. Generate appropriate loops for the reduction
                // 3. Support different reduction operators

                // The tensor expression must be an implicit_tensor_index
                const tensor_expr = reduce.tensor_expr.*;

                if (tensor_expr != .implicit_tensor_index) {
                    return error.InvalidExpression;
                }

                const implicit_index = tensor_expr.implicit_tensor_index;
                const base_tensor = implicit_index.tensor.*;

                // For simplicity, assume the base is a simple tensor identifier
                if (base_tensor != .identifier) {
                    return error.InvalidExpression;
                }

                const tensor_name = base_tensor.identifier.name;
                const var_info = self.variables.get(tensor_name) orelse return error.UndefinedVariable;

                if (var_info.ty != .tensor) {
                    return error.InvalidExpression;
                }

                const tensor_type = var_info.ty.tensor;
                const element_type = self.toLLVMType(tensor_type.element_type.*);
                const array_type = self.toLLVMType(var_info.ty);

                // Initialize accumulator based on the operator
                var init_value: LLVM.LLVMValueRef = undefined;
                switch (reduce.operator) {
                    .add => init_value = LLVM.LLVMConstInt(element_type, 0, 0),
                    .multiply => init_value = LLVM.LLVMConstInt(element_type, 1, 0),
                    else => return error.UnsupportedOperation,
                }

                // For proper multi-dimensional reduction, we need context about which indices are free
                // For now, let's implement a simpler version that still reduces all elements
                // but demonstrates the structure for future enhancement
                const total_elements = tensor_type.total_elements();

                // TODO: Analyze the assignment context to determine which dimensions to reduce
                // For b[i] = reduce(a[i,j], +), we should:
                // 1. Loop over free indices (i)
                // 2. For each i, reduce over bound indices (j)
                // This requires passing context from the assignment

                // Create a basic block for the loop
                const current_function = LLVM.LLVMGetBasicBlockParent(LLVM.LLVMGetInsertBlock(self.builder));
                const loop_block = LLVM.LLVMAppendBasicBlockInContext(self.context, current_function, "reduce_loop");
                const after_loop = LLVM.LLVMAppendBasicBlockInContext(self.context, current_function, "reduce_after");

                // Create accumulator and index variables
                const acc_alloca = LLVM.LLVMBuildAlloca(self.builder, element_type, "reduce_acc");
                const idx_alloca = LLVM.LLVMBuildAlloca(self.builder, LLVM.LLVMInt64TypeInContext(self.context), "reduce_idx");

                _ = LLVM.LLVMBuildStore(self.builder, init_value, acc_alloca);
                _ = LLVM.LLVMBuildStore(self.builder, LLVM.LLVMConstInt(LLVM.LLVMInt64TypeInContext(self.context), 0, 0), idx_alloca);

                // Jump to loop
                _ = LLVM.LLVMBuildBr(self.builder, loop_block);

                // Loop body
                LLVM.LLVMPositionBuilderAtEnd(self.builder, loop_block);

                const current_idx = LLVM.LLVMBuildLoad2(self.builder, LLVM.LLVMInt64TypeInContext(self.context), idx_alloca, "idx");
                const current_acc = LLVM.LLVMBuildLoad2(self.builder, element_type, acc_alloca, "acc");

                // Get element at current index
                const zero = LLVM.LLVMConstInt(LLVM.LLVMInt64TypeInContext(self.context), 0, 0);
                var indices = [_]LLVM.LLVMValueRef{ zero, current_idx };
                const elem_ptr = LLVM.LLVMBuildGEP2(self.builder, array_type, var_info.alloca, &indices[0], 2, "elem_ptr");
                const elem_value = LLVM.LLVMBuildLoad2(self.builder, element_type, elem_ptr, "elem");

                // Apply reduction operator
                const new_acc = switch (reduce.operator) {
                    .add => LLVM.LLVMBuildAdd(self.builder, current_acc, elem_value, "new_acc"),
                    .multiply => LLVM.LLVMBuildMul(self.builder, current_acc, elem_value, "new_acc"),
                    else => unreachable,
                };

                _ = LLVM.LLVMBuildStore(self.builder, new_acc, acc_alloca);

                // Increment index
                const next_idx = LLVM.LLVMBuildAdd(self.builder, current_idx, LLVM.LLVMConstInt(LLVM.LLVMInt64TypeInContext(self.context), 1, 0), "next_idx");
                _ = LLVM.LLVMBuildStore(self.builder, next_idx, idx_alloca);

                // Check loop condition
                const cond = LLVM.LLVMBuildICmp(self.builder, LLVM.LLVMIntULT, next_idx, LLVM.LLVMConstInt(LLVM.LLVMInt64TypeInContext(self.context), total_elements, 0), "loop_cond");
                _ = LLVM.LLVMBuildCondBr(self.builder, cond, loop_block, after_loop);

                // After loop
                LLVM.LLVMPositionBuilderAtEnd(self.builder, after_loop);

                // Return the final accumulator value
                return LLVM.LLVMBuildLoad2(self.builder, element_type, acc_alloca, "reduce_result");
            },
            else => return error.InvalidExpression,
        }
    }

    /// Generate an optimized GPU function call with explicit memory management
    fn generateOptimizedGpuCall(self: *CodeGen, call: @TypeOf(@as(parser.ASTNode, undefined).call_expression), func_name: []const u8, launcher_function: LLVM.LLVMValueRef) CodeGenError!LLVM.LLVMValueRef {
        var tracker = &self.gpu_memory_tracker.?;
        var ops = &self.gpu_memory_ops.?;

        if (self.verbose) {
            std.debug.print("🚀 Generating optimized GPU call for: {s}\n", .{func_name});
        }

        // First, evaluate all arguments
        const args = try self.allocator.alloc(LLVM.LLVMValueRef, call.arguments.len);
        defer self.allocator.free(args);

        for (call.arguments, 0..) |arg, i| {
            args[i] = try self.generateExpression(arg);
        }

        // Get function info to know parameter types
        // In a real implementation, we'd look up the function declaration
        // For now, assume all parameters are tensors of i32[1024]

        // Prepare GPU memory for each tensor argument
        const gpu_args = try self.allocator.alloc(LLVM.LLVMValueRef, args.len);
        defer self.allocator.free(gpu_args);

        const ptr_type = LLVM.LLVMPointerType(LLVM.LLVMInt8TypeInContext(self.context), 0);
        const size_t_type = LLVM.LLVMInt64TypeInContext(self.context);

        for (args, 0..) |arg, i| {
            // For this simplified version, assume all args are tensors
            // Extract variable name from argument (if it's an identifier)
            var var_name_buf: []u8 = undefined;
            var var_name: []const u8 = undefined;
            var should_free = false;
            if (call.arguments[i] == .identifier) {
                var_name = call.arguments[i].identifier.name;
                if (self.verbose) {
                    std.debug.print("  📝 Processing variable '{s}' (arg {})\n", .{ var_name, i });
                }
            } else {
                var_name_buf = try std.fmt.allocPrint(self.allocator, "tmp_arg_{}", .{i});
                var_name = var_name_buf;
                should_free = true;
            }
            defer if (should_free) self.allocator.free(var_name_buf);

            // Check if already on GPU and if transfer is needed
            if (tracker.getGpuPtr(var_name)) |existing_gpu_ptr| {
                gpu_args[i] = existing_gpu_ptr;

                // Check if we need to update GPU data from CPU
                if (tracker.needsTransferToGpu(var_name)) {
                    const tensor_size = LLVM.LLVMConstInt(size_t_type, 1024 * 4, 0); // TODO: get actual size
                    const host_ptr = LLVM.LLVMBuildBitCast(self.builder, arg, ptr_type, "host_ptr");
                    ops.copyHostToDevice(existing_gpu_ptr, host_ptr, tensor_size);
                    try tracker.markCopiedToGpu(var_name);

                    if (self.verbose) {
                        std.debug.print("  🔄 Updated GPU memory for '{s}' from CPU\n", .{var_name});
                    }
                } else {
                    if (self.verbose) {
                        std.debug.print("  ♻️  Reusing GPU memory for '{s}'\n", .{var_name});
                    }
                }
            } else {
                // Need to allocate and transfer
                const tensor_size = LLVM.LLVMConstInt(size_t_type, 1024 * 4, 0); // 1024 * sizeof(i32)

                // Allocate GPU memory
                const gpu_ptr = ops.allocateGpuMemory(tensor_size, var_name);
                gpu_args[i] = gpu_ptr;

                // Register with tracker (only if not already registered)
                if (tracker.getCpuPtr(var_name) == null) {
                    try tracker.registerVariable(var_name, arg, 1024 * 4);
                }
                try tracker.markGpuAllocated(var_name, gpu_ptr);

                // Copy to GPU
                const host_ptr = LLVM.LLVMBuildBitCast(self.builder, arg, ptr_type, "host_ptr");
                ops.copyHostToDevice(gpu_ptr, host_ptr, tensor_size);
                try tracker.markCopiedToGpu(var_name);

                if (self.verbose) {
                    std.debug.print("  ⬆️  Allocated and copied '{s}' to GPU\n", .{var_name});
                }
            }
        }

        // Call the GPU launcher with GPU pointers
        const launcher_type = LLVM.LLVMGlobalGetValueType(launcher_function);
        const result = LLVM.LLVMBuildCall2(self.builder, launcher_type, launcher_function, gpu_args.ptr, @intCast(gpu_args.len), "gpu_call");

        // Mark all tensor arguments as modified on GPU
        for (call.arguments) |arg_node| {
            if (arg_node == .identifier) {
                const var_name = arg_node.identifier.name;
                try tracker.markModifiedOnGpu(var_name);
            }
        }

        // Note: We don't synchronize here! That will be done when needed
        if (self.verbose) {
            std.debug.print("  ✅ GPU kernel launched (no sync)\n", .{});
        }

        return result;
    }

    /// Similar to generateExpression but aware of the induction variable of a
    /// parallel tensor loop.  Any implicit_tensor_index nodes will load the
    /// element at the current loop index provided by `idx_val`.
    fn generateMultiDimensionalReduce(self: *CodeGen, reduce: *@TypeOf(@as(parser.ASTNode, undefined).reduce_expression), red_info: typechecker.ReductionInfo, free_idx: LLVM.LLVMValueRef) CodeGenError!LLVM.LLVMValueRef {
        if (self.verbose) {
            std.debug.print("DEBUG: generateMultiDimensionalReduce called\n", .{});
            std.debug.print("DEBUG: Free indices: ", .{});
            for (red_info.free_indices) |idx| {
                std.debug.print("{s} ", .{idx});
            }
            std.debug.print("\nDEBUG: Bound indices: ", .{});
            for (red_info.bound_indices) |idx| {
                std.debug.print("{s} ", .{idx});
            }
            std.debug.print("\n", .{});
        }
        // Extract tensor information
        const tensor_expr = reduce.tensor_expr.*;
        if (tensor_expr != .implicit_tensor_index) {
            return error.InvalidExpression;
        }

        const implicit_index = tensor_expr.implicit_tensor_index;
        const base_tensor = implicit_index.tensor.*;

        if (base_tensor != .identifier) {
            return error.InvalidExpression;
        }

        const tensor_name = base_tensor.identifier.name;
        const var_info = self.variables.get(tensor_name) orelse return error.UndefinedVariable;

        if (var_info.ty != .tensor) {
            return error.InvalidExpression;
        }

        const tensor_type = var_info.ty.tensor;
        const element_type = self.toLLVMType(tensor_type.element_type.*);
        const array_type = self.toLLVMType(var_info.ty);

        // Initialize accumulator based on the operator
        var init_value: LLVM.LLVMValueRef = undefined;
        const type_kind = LLVM.LLVMGetTypeKind(element_type);
        const is_float = switch (type_kind) {
            LLVM.LLVMFloatTypeKind, LLVM.LLVMDoubleTypeKind => true,
            else => false,
        };

        switch (reduce.operator) {
            .add => init_value = if (is_float) LLVM.LLVMConstReal(element_type, 0.0) else LLVM.LLVMConstInt(element_type, 0, 0),
            .multiply => init_value = if (is_float) LLVM.LLVMConstReal(element_type, 1.0) else LLVM.LLVMConstInt(element_type, 1, 0),
            else => return error.UnsupportedOperation,
        }

        // For multi-dimensional reduction, we need to:
        // 1. Use the free_idx for the free dimension (provided by outer loop)
        // 2. Loop over the bound dimensions

        // For now, let's assume 2D tensor with one bound dimension
        // TODO: Generalize to arbitrary dimensions

        if (tensor_type.rank() != 2 or red_info.bound_indices.len != 1) {
            // Fall back to simple full reduction for now
            return self.generateExpression(.{ .reduce_expression = reduce.* });
        }

        // Determine which dimension is being reduced
        // If the bound index is "i", we're reducing the first dimension
        // If the bound index is "j", we're reducing the second dimension
        const bound_idx_name = red_info.bound_indices[0];
        const reducing_first_dim = std.mem.eql(u8, bound_idx_name, "i");

        const bound_dim_size = if (reducing_first_dim) tensor_type.shape[0] else tensor_type.shape[1];
        const i64_type = LLVM.LLVMInt64TypeInContext(self.context);

        // Create accumulator
        const acc_alloca = LLVM.LLVMBuildAlloca(self.builder, element_type, "md_reduce_acc");
        _ = LLVM.LLVMBuildStore(self.builder, init_value, acc_alloca);

        // Create loop for bound dimension
        const current_function = LLVM.LLVMGetBasicBlockParent(LLVM.LLVMGetInsertBlock(self.builder));
        const loop_block = LLVM.LLVMAppendBasicBlockInContext(self.context, current_function, "md_reduce_loop");
        const after_loop = LLVM.LLVMAppendBasicBlockInContext(self.context, current_function, "md_reduce_after");

        // Initialize bound index
        const bound_idx_alloca = LLVM.LLVMBuildAlloca(self.builder, i64_type, "bound_idx");
        _ = LLVM.LLVMBuildStore(self.builder, LLVM.LLVMConstInt(i64_type, 0, 0), bound_idx_alloca);

        // Jump to loop
        _ = LLVM.LLVMBuildBr(self.builder, loop_block);

        // Loop body
        LLVM.LLVMPositionBuilderAtEnd(self.builder, loop_block);

        const bound_idx = LLVM.LLVMBuildLoad2(self.builder, i64_type, bound_idx_alloca, "bound_idx_val");
        const current_acc = LLVM.LLVMBuildLoad2(self.builder, element_type, acc_alloca, "acc_val");

        // Calculate linear index for 2D array
        // If reducing first dimension: bound_idx * shape[1] + free_idx
        // If reducing second dimension: free_idx * shape[1] + bound_idx
        const linear_idx = if (reducing_first_dim) blk: {
            const offset = LLVM.LLVMBuildMul(self.builder, bound_idx, LLVM.LLVMConstInt(i64_type, tensor_type.shape[1], 0), "offset");
            break :blk LLVM.LLVMBuildAdd(self.builder, offset, free_idx, "linear_idx");
        } else blk: {
            const offset = LLVM.LLVMBuildMul(self.builder, free_idx, LLVM.LLVMConstInt(i64_type, tensor_type.shape[1], 0), "offset");
            break :blk LLVM.LLVMBuildAdd(self.builder, offset, bound_idx, "linear_idx");
        };

        // Get element at [free_idx, bound_idx]
        const zero = LLVM.LLVMConstInt(i64_type, 0, 0);
        var indices = [_]LLVM.LLVMValueRef{ zero, linear_idx };
        const elem_ptr = LLVM.LLVMBuildGEP2(self.builder, array_type, var_info.alloca, &indices[0], 2, "md_elem_ptr");
        const elem_value = LLVM.LLVMBuildLoad2(self.builder, element_type, elem_ptr, "md_elem");

        // Apply reduction operator
        const new_acc = switch (reduce.operator) {
            .add => if (is_float) LLVM.LLVMBuildFAdd(self.builder, current_acc, elem_value, "new_acc") else LLVM.LLVMBuildAdd(self.builder, current_acc, elem_value, "new_acc"),
            .multiply => if (is_float) LLVM.LLVMBuildFMul(self.builder, current_acc, elem_value, "new_acc") else LLVM.LLVMBuildMul(self.builder, current_acc, elem_value, "new_acc"),
            else => unreachable,
        };

        _ = LLVM.LLVMBuildStore(self.builder, new_acc, acc_alloca);

        // Increment bound index
        const next_bound_idx = LLVM.LLVMBuildAdd(self.builder, bound_idx, LLVM.LLVMConstInt(i64_type, 1, 0), "next_bound_idx");
        _ = LLVM.LLVMBuildStore(self.builder, next_bound_idx, bound_idx_alloca);

        // Check loop condition
        const cond = LLVM.LLVMBuildICmp(self.builder, LLVM.LLVMIntULT, next_bound_idx, LLVM.LLVMConstInt(i64_type, bound_dim_size, 0), "bound_loop_cond");
        _ = LLVM.LLVMBuildCondBr(self.builder, cond, loop_block, after_loop);

        // After loop
        LLVM.LLVMPositionBuilderAtEnd(self.builder, after_loop);

        // Return the accumulated value
        return LLVM.LLVMBuildLoad2(self.builder, element_type, acc_alloca, "final_acc");
    }

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
        return linking.generateExecutable(self, output_path, target);
    }

    pub fn generateSharedLibrary(self: *CodeGen, output_path: []const u8, target: std.Target) CodeGenError!void {
        return linking.generateSharedLibrary(self, output_path, target);
    }

    pub fn printIR(self: *CodeGen) void {
        if (self.verbose) {
            const ir_string = LLVM.LLVMPrintModuleToString(self.module);
            defer LLVM.LLVMDisposeMessage(ir_string);
            std.debug.print("Generated LLVM IR:\n{s}\n", .{ir_string});
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
