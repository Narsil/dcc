const std = @import("std");
const parser = @import("parser.zig");
const mlir_codegen = @import("mlir_codegen.zig");
const macho = std.macho;
const LLVM = @cImport({
    @cInclude("llvm-c/Core.h");
    @cInclude("llvm-c/TargetMachine.h");
    @cInclude("llvm-c/Target.h");
    @cInclude("llvm-c/Support.h");
    @cInclude("llvm-c/BitReader.h");
    @cInclude("llvm-c/Object.h");
});

pub const CodeGenError = error{ InvalidTopLevelNode, InvalidStatement, InvalidExpression, InvalidCallee, UndefinedVariable, UndefinedFunction, TargetError, CodeGenError, MainFunctionNotFound, MissingMainFunction, LinkingFailed, GpuCompilationNotImplemented, InvalidCharacter, Overflow } || std.mem.Allocator.Error;

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

    pub fn init(allocator: std.mem.Allocator, module_name: []const u8, verbose: bool) !CodeGen {
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

        return CodeGen{
            .context = context,
            .module = module,
            .builder = builder,
            .allocator = allocator,
            .verbose = verbose,
            .variables = std.HashMap([]const u8, VarInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .functions = std.HashMap([]const u8, LLVM.LLVMValueRef, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .mlir_codegen = mlir_codegen.MLIRCodeGen.init(allocator, 50, verbose) catch null,
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

        // Clean up LLVM resources
        LLVM.LLVMDisposeBuilder(self.builder);
        LLVM.LLVMDisposeModule(self.module);
        LLVM.LLVMContextDispose(self.context);
    }

    pub fn generate(self: *CodeGen, ast: parser.ASTNode) CodeGenError!void {
        // Generate LLVM IR for the program AST node
        switch (ast) {
            .program => |prog| {
                // Generate LLVM IR for all statements in the program
                for (prog.statements) |stmt| {
                    try self.generateNode(stmt);
                }
                // No need for _start function - using main directly as entry point
                // This avoids symbol naming issues with double underscores on macOS
            },
            else => return error.InvalidTopLevelNode,
        }
    }

    fn generateNode(self: *CodeGen, node: parser.ASTNode) CodeGenError!void {
        switch (node) {
            .function_declaration => try self.generateFunction(node.function_declaration),
            .variable_declaration => try self.generateVariableDeclaration(node.variable_declaration),
            .return_statement => try self.generateReturn(node.return_statement),
            .expression_statement => _ = try self.generateExpression(node.expression_statement.expression.*),
            .parallel_assignment => try self.generateParallelAssignment(node.parallel_assignment),
            else => return error.InvalidStatement,
        }
    }

    fn generateFunction(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) CodeGenError!void {
        // Detect GPU functions by naming convention and bail out for now
        if (std.mem.startsWith(u8, func.name, "gpu_")) {
            return self.generateGpuFunction(func);
        }
        // Use i32 return type for main function to match C runtime expectations
        const is_main = std.mem.eql(u8, func.name, "main");

        // Create parameter types array
        const param_types = try self.allocator.alloc(LLVM.LLVMTypeRef, func.parameters.len);
        defer self.allocator.free(param_types);

        for (func.parameters, 0..) |param, i| {
            param_types[i] = param.type.toLLVMType(self.context);
        }

        // Create function type
        const actual_return_type = func.return_type.toLLVMType(self.context);
        const function_type = LLVM.LLVMFunctionType(actual_return_type, param_types.ptr, @intCast(param_types.len), 0);

        // Create function
        const name_z = try self.allocator.dupeZ(u8, func.name);
        defer self.allocator.free(name_z);

        const llvm_function = LLVM.LLVMAddFunction(self.module, name_z.ptr, function_type);

        // Set proper stack alignment attributes for x86-64 System V ABI compliance
        self.setStackAlignmentAttributes(llvm_function);

        try self.functions.put(try self.allocator.dupe(u8, func.name), llvm_function);

        // Create entry basic block
        const entry_block = LLVM.LLVMAppendBasicBlockInContext(self.context, llvm_function, "entry");
        LLVM.LLVMPositionBuilderAtEnd(self.builder, entry_block);

        // Create allocas for parameters
        for (func.parameters, 0..) |param, i| {
            const param_value = LLVM.LLVMGetParam(llvm_function, @intCast(i));
            const param_name_z = try self.allocator.dupeZ(u8, param.name);
            defer self.allocator.free(param_name_z);

            const param_type = param.type.toLLVMType(self.context);
            const alloca = LLVM.LLVMBuildAlloca(self.builder, param_type, param_name_z.ptr);
            LLVM.LLVMSetAlignment(alloca, 16); // 16-byte alignment for x86-64 System V ABI
            const store_inst = LLVM.LLVMBuildStore(self.builder, param_value, alloca);
            LLVM.LLVMSetAlignment(store_inst, 16); // Match alloca alignment
            try self.variables.put(try self.allocator.dupe(u8, param.name), VarInfo{ .alloca = alloca, .ty = param.type });
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
        self.clearVariables();
    }

    fn generateVariableDeclaration(self: *CodeGen, var_decl: @TypeOf(@as(parser.ASTNode, undefined).variable_declaration)) CodeGenError!void {
        const var_type = var_decl.type.toLLVMType(self.context);
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
        const elem_type = tensor_info.ty.toLLVMType(self.context);

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
                const llvm_type = num.type.toLLVMType(self.context);

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

                    const elem_type = info.ty.toLLVMType(self.context);
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
                        args[i] = try self.generateExpression(arg);
                    }

                    const int64_type = LLVM.LLVMInt64TypeInContext(self.context);
                    const return_type = call.return_type.?.toLLVMType(self.context);
                    const param_types = try self.allocator.alloc(LLVM.LLVMTypeRef, call.arguments.len);
                    defer self.allocator.free(param_types);

                    for (param_types) |*param_type| {
                        param_type.* = int64_type;
                    }

                    const function_type = LLVM.LLVMFunctionType(return_type, param_types.ptr, @intCast(param_types.len), 0);

                    return LLVM.LLVMBuildCall2(self.builder, function_type, function, args.ptr, @intCast(args.len), "call");
                } else {
                    return error.UndefinedFunction;
                }
            },
            .tensor_literal => |tensor_lit| {
                // For now, create a simple array with the specified value
                const element_type = tensor_lit.element_type.toLLVMType(self.context);
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

                const array_ty = vi_slice.ty.toLLVMType(self.context);
                const elem_ty = vi_slice.ty.tensor.element_type.toLLVMType(self.context);

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

                const array_ty = vi.ty.toLLVMType(self.context);
                const elem_ty = vi.ty.tensor.element_type.toLLVMType(self.context);

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

    pub fn generateExecutable(self: *CodeGen, output_path: []const u8, target_triple: ?[]const u8) CodeGenError!void {
        // Use provided target triple or default to host triple
        const final_triple = if (target_triple) |triple| blk: {
            break :blk triple;
        } else blk: {
            const default_triple = LLVM.LLVMGetDefaultTargetTriple();
            // defer LLVM.LLVMDisposeMessage(default_triple);

            // Normalize the target triple for known platforms
            const triple_str = std.mem.span(default_triple);
            const normalized_triple = if (std.mem.startsWith(u8, triple_str, "arm64-apple-darwin"))
                "arm64-apple-darwin"
            else if (std.mem.startsWith(u8, triple_str, "x86_64-apple-darwin"))
                "x86_64-apple-darwin"
            else if (std.mem.startsWith(u8, triple_str, "x86_64-pc-linux"))
                "x86_64-pc-linux-gnu"
            else
                triple_str;

            break :blk normalized_triple;
        };

        std.debug.print("Target triple: {s}\n", .{final_triple});

        const triple_z = try self.allocator.dupeZ(u8, final_triple);
        defer self.allocator.free(triple_z);

        var target: LLVM.LLVMTargetRef = null;
        var error_message: [*c]u8 = undefined;

        if (LLVM.LLVMGetTargetFromTriple(triple_z.ptr, &target, &error_message) != 0) {
            defer LLVM.LLVMDisposeMessage(error_message);
            std.debug.print("Error getting target: {s}\n", .{error_message});
            std.debug.print("Tried target triple: {s}\n", .{final_triple});
            return error.TargetError;
        }

        // Temporarily disable _start function to test main computation alignment
        const is_linux = std.mem.indexOf(u8, final_triple, "linux") != null;
        if (is_linux) {
            try self.generateStartFunction();
        }

        // Create target machine optimized for executables
        const target_machine = LLVM.LLVMCreateTargetMachine(target.?, triple_z.ptr, "generic", "", LLVM.LLVMCodeGenLevelDefault, LLVM.LLVMRelocDefault, // Static relocation for executables
            LLVM.LLVMCodeModelDefault);
        defer LLVM.LLVMDisposeTargetMachine(target_machine);

        // Generate object file
        const obj_path = try std.fmt.allocPrint(self.allocator, "{s}.o", .{output_path});
        defer self.allocator.free(obj_path);

        const obj_path_z = try self.allocator.dupeZ(u8, obj_path);
        defer self.allocator.free(obj_path_z);

        if (LLVM.LLVMTargetMachineEmitToFile(target_machine, self.module, obj_path_z.ptr, LLVM.LLVMObjectFile, &error_message) != 0) {
            defer LLVM.LLVMDisposeMessage(error_message);
            std.debug.print("Error generating object file: {s}\n", .{error_message});
            return error.CodeGenError;
        }

        std.debug.print("Generated object file: {s}\n", .{obj_path});

        // Link object file into executable
        try self.linkExecutable(obj_path, output_path, final_triple);

        // Clean up object file
        std.fs.cwd().deleteFile(obj_path) catch {};
    }

    fn linkExecutable(self: *CodeGen, obj_path: []const u8, output_path: []const u8, target_triple: []const u8) CodeGenError!void {
        // Use lld directly without external process calls
        const is_darwin = std.mem.indexOf(u8, target_triple, "darwin") != null;
        const is_windows = std.mem.indexOf(u8, target_triple, "windows") != null;

        var args = std.ArrayList([*:0]const u8).init(self.allocator);
        defer args.deinit();

        // Defer cleanup of allocated strings
        defer for (args.items) |arg| {
            self.allocator.free(std.mem.span(arg));
        };

        // Build arguments for lld
        const linker_name = if (is_darwin) "ld64.lld" else if (is_windows) "lld-link" else "ld.lld";
        try args.append(try self.allocator.dupeZ(u8, linker_name));

        if (is_darwin) {
            // Mach-O executable arguments
            try args.append(try self.allocator.dupeZ(u8, "-arch"));
            // Extract architecture from target_triple (e.g., "aarch64-apple-darwin" -> "arm64")
            var arch_end: usize = 0;
            while (arch_end < target_triple.len and target_triple[arch_end] != '-') : (arch_end += 1) {}
            var arch = target_triple[0..arch_end];
            if (std.mem.eql(u8, arch, "aarch64")) {
                arch = "arm64";
            }
            try args.append(try self.allocator.dupeZ(u8, arch));
            try args.append(try self.allocator.dupeZ(u8, "-platform_version"));
            try args.append(try self.allocator.dupeZ(u8, "macos"));
            try args.append(try self.allocator.dupeZ(u8, "10.15"));
            try args.append(try self.allocator.dupeZ(u8, "10.15"));
        } else if (is_windows) {
            // COFF/PE executable arguments (note: no /dll flag for executables)
            try args.append(try self.allocator.dupeZ(u8, "/subsystem:console"));
        } else {
            // ELF executable arguments
            try args.append(try self.allocator.dupeZ(u8, "--entry=_start"));
        }

        // Add output file
        if (is_windows) {
            try args.append(try std.fmt.allocPrintZ(self.allocator, "/out:{s}", .{output_path}));
        } else {
            try args.append(try self.allocator.dupeZ(u8, "-o"));
            try args.append(try self.allocator.dupeZ(u8, output_path));
        }

        // Add object file
        try args.append(try self.allocator.dupeZ(u8, obj_path));

        // Call lld_main with all arguments
        std.debug.print("Arguments {s}", .{args.items});
        const result = lld_main(args.items.ptr, @intCast(args.items.len));

        if (result != 0) {
            std.debug.print("lld executable linking failed with code: {}\n", .{result});
            return error.LinkingFailed;
        }

        std.debug.print("Generated executable: {s}\n", .{output_path});
    }

    pub fn generateSharedLibrary(self: *CodeGen, output_path: []const u8, target_triple: ?[]const u8) CodeGenError!void {
        // Use provided target triple or default to host triple
        const final_triple = if (target_triple) |triple| blk: {
            break :blk triple;
        } else blk: {
            const default_triple = LLVM.LLVMGetDefaultTargetTriple();
            defer LLVM.LLVMDisposeMessage(default_triple);

            // Normalize the target triple for known platforms
            const triple_str = std.mem.span(default_triple);
            const normalized_triple = if (std.mem.startsWith(u8, triple_str, "arm64-apple-darwin"))
                "arm64-apple-darwin"
            else if (std.mem.startsWith(u8, triple_str, "x86_64-apple-darwin"))
                "x86_64-apple-darwin"
            else if (std.mem.startsWith(u8, triple_str, "x86_64-pc-linux"))
                "x86_64-pc-linux-gnu"
            else
                triple_str;

            break :blk normalized_triple;
        };

        std.debug.print("Target triple: {s}\n", .{final_triple});

        const triple_z = try self.allocator.dupeZ(u8, final_triple);
        defer self.allocator.free(triple_z);

        var target: LLVM.LLVMTargetRef = null;
        var error_message: [*c]u8 = undefined;

        if (LLVM.LLVMGetTargetFromTriple(triple_z.ptr, &target, &error_message) != 0) {
            defer LLVM.LLVMDisposeMessage(error_message);
            std.debug.print("Error getting target: {s}\n", .{error_message});
            std.debug.print("Tried target triple: {s}\n", .{final_triple});
            return error.TargetError;
        }

        // Create target machine optimized for shared libraries
        const target_machine = LLVM.LLVMCreateTargetMachine(target.?, triple_z.ptr, "generic", "", LLVM.LLVMCodeGenLevelDefault, LLVM.LLVMRelocPIC, // Position Independent Code for shared libraries
            LLVM.LLVMCodeModelDefault);
        defer LLVM.LLVMDisposeTargetMachine(target_machine);

        // Generate object file
        const obj_path = try std.fmt.allocPrint(self.allocator, "{s}.o", .{output_path});
        defer self.allocator.free(obj_path);

        const obj_path_z = try self.allocator.dupeZ(u8, obj_path);
        defer self.allocator.free(obj_path_z);

        if (LLVM.LLVMTargetMachineEmitToFile(target_machine, self.module, obj_path_z.ptr, LLVM.LLVMObjectFile, &error_message) != 0) {
            defer LLVM.LLVMDisposeMessage(error_message);
            std.debug.print("Error generating object file: {s}\n", .{error_message});
            return error.CodeGenError;
        }

        std.debug.print("Generated object file: {s}\n", .{obj_path});

        // Determine the appropriate shared library extension
        const is_darwin = std.mem.indexOf(u8, final_triple, "darwin") != null;
        const is_windows = std.mem.indexOf(u8, final_triple, "windows") != null;

        const lib_extension = if (is_darwin) ".dylib" else if (is_windows) ".dll" else ".so";
        const lib_file = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ output_path, lib_extension });
        defer self.allocator.free(lib_file);

        // Link object file into shared library
        try self.linkSharedLibrary(obj_path, lib_file, final_triple);

        // Clean up object file
        std.fs.cwd().deleteFile(obj_path) catch {};
    }

    fn linkSharedLibrary(self: *CodeGen, obj_path: []const u8, output_path: []const u8, target_triple: []const u8) CodeGenError!void {
        // Use lld directly without external process calls
        const is_darwin = std.mem.indexOf(u8, target_triple, "darwin") != null;
        const is_windows = std.mem.indexOf(u8, target_triple, "windows") != null;

        var args = std.ArrayList([*:0]const u8).init(self.allocator);
        defer args.deinit();

        // Defer cleanup of allocated strings
        defer for (args.items) |arg| {
            self.allocator.free(std.mem.span(arg));
        };

        // Build arguments for lld
        const linker_name = if (is_darwin) "ld64.lld" else if (is_windows) "lld-link" else "ld.lld";
        try args.append(try self.allocator.dupeZ(u8, linker_name));

        if (is_darwin) {
            // Mach-O shared library (dylib) arguments
            try args.append(try self.allocator.dupeZ(u8, "-arch"));
            // Extract architecture from target_triple (e.g., "arm64-apple-darwin" -> "arm64")
            var arch_end: usize = 0;
            while (arch_end < target_triple.len and target_triple[arch_end] != '-') : (arch_end += 1) {}
            var arch = target_triple[0..arch_end];
            if (std.mem.eql(u8, arch, "aarch64")) {
                arch = "arm64";
            }
            try args.append(try self.allocator.dupeZ(u8, arch));
            try args.append(try self.allocator.dupeZ(u8, "-platform_version"));
            try args.append(try self.allocator.dupeZ(u8, "macos"));
            try args.append(try self.allocator.dupeZ(u8, "10.15"));
            try args.append(try self.allocator.dupeZ(u8, "10.15"));
            try args.append(try self.allocator.dupeZ(u8, "-dylib"));
        } else if (is_windows) {
            // COFF/PE DLL arguments
            try args.append(try self.allocator.dupeZ(u8, "/dll"));
        } else {
            // ELF shared library (.so) arguments
            try args.append(try self.allocator.dupeZ(u8, "--shared"));
        }

        // Add output file
        if (is_windows) {
            try args.append(try std.fmt.allocPrintZ(self.allocator, "/out:{s}", .{output_path}));
        } else {
            try args.append(try self.allocator.dupeZ(u8, "-o"));
            try args.append(try self.allocator.dupeZ(u8, output_path));
        }

        // Add object file
        try args.append(try self.allocator.dupeZ(u8, obj_path));

        // Call lld_main with all arguments
        const result = lld_main(args.items.ptr, @intCast(args.items.len));

        if (result != 0) {
            std.debug.print("lld shared library linking failed with code: {}\n", .{result});
            return error.LinkingFailed;
        }

        std.debug.print("Generated shared library: {s}\n", .{output_path});
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

    pub fn generateObjectFile(self: *CodeGen, output_path: []const u8, target_triple: ?[]const u8) CodeGenError!void {
        // Use provided target triple or default to host triple
        const final_triple = if (target_triple) |triple| blk: {
            break :blk triple;
        } else blk: {
            const default_triple = LLVM.LLVMGetDefaultTargetTriple();
            defer LLVM.LLVMDisposeMessage(default_triple);

            // Normalize the target triple for known platforms
            const triple_str = std.mem.span(default_triple);
            const normalized_triple = if (std.mem.startsWith(u8, triple_str, "arm64-apple-darwin"))
                "arm64-apple-darwin"
            else if (std.mem.startsWith(u8, triple_str, "x86_64-apple-darwin"))
                "x86_64-apple-darwin"
            else if (std.mem.startsWith(u8, triple_str, "x86_64-pc-linux"))
                "x86_64-pc-linux-gnu"
            else
                triple_str;

            break :blk normalized_triple;
        };

        std.debug.print("Target triple: {s}\n", .{final_triple});

        const triple_z = try self.allocator.dupeZ(u8, final_triple);
        defer self.allocator.free(triple_z);

        var target: LLVM.LLVMTargetRef = null;
        var error_message: [*:0]u8 = undefined;

        if (LLVM.LLVMGetTargetFromTriple(triple_z.ptr, &target, &error_message) != 0) {
            defer LLVM.LLVMDisposeMessage(error_message);
            std.debug.print("Error getting target: {s}\n", .{error_message});
            std.debug.print("Tried target triple: {s}\n", .{final_triple});
            return error.TargetError;
        }

        const target_machine = LLVM.LLVMCreateTargetMachine(target.?, triple_z.ptr, "generic", "", @intFromEnum(LLVM.LLVMOptLevel.default), @intFromEnum(LLVM.LLVMRelocMode.pic), @intFromEnum(LLVM.LLVMCodeModel.default));
        defer LLVM.LLVMDisposeTargetMachine(target_machine);

        const output_path_z = try self.allocator.dupeZ(u8, output_path);
        defer self.allocator.free(output_path_z);

        if (LLVM.LLVMTargetMachineEmitToFile(target_machine, self.module, output_path_z.ptr, @intFromEnum(LLVM.LLVMCodeGenFileType.object), &error_message) != 0) {
            defer LLVM.LLVMDisposeMessage(error_message);
            std.debug.print("Error generating object file: {s}\n", .{error_message});
            return error.CodeGenError;
        }
    }

    pub fn printIR(self: *CodeGen) void {
        const ir_string = LLVM.LLVMPrintModuleToString(self.module);
        defer LLVM.LLVMDisposeMessage(ir_string);
        std.debug.print("Generated LLVM IR:\n{s}\n", .{ir_string});
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

        // Call main() - main now returns i32
        const main_function_type = LLVM.LLVMFunctionType(i32_type, null, 0, 0);
        const exit_code = LLVM.LLVMBuildCall2(self.builder, main_function_type, main_function, null, 0, "main_result");

        // Store the exit code in a global variable for inspection
        const exit_code_global = LLVM.LLVMAddGlobal(self.module, i32_type, "program_exit_code");
        LLVM.LLVMSetInitializer(exit_code_global, LLVM.LLVMConstInt(i32_type, 0, 0));
        _ = LLVM.LLVMBuildStore(self.builder, exit_code, exit_code_global);

        // --- Exit the process via Linux x86_64 syscall ---
        const void_type = LLVM.LLVMVoidTypeInContext(self.context);
        const int64_type = LLVM.LLVMInt64TypeInContext(self.context);
        const param_types = [_]LLVM.LLVMTypeRef{int64_type};
        const syscall_asm_ty = LLVM.LLVMFunctionType(void_type, @constCast(&param_types[0]), 1, 0);
        const asm_str = "mov $0, %rdi\n mov $$60, %rax\nsyscall"; // rax=60 (SYS_exit), rdi=status

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
                std.debug.print("Compiling GPU function: {s}\n", .{func.name});
            }
            
            // Generate MLIR for the GPU function
            mlir.generateGpuFunction(func) catch |err| {
                if (self.verbose) {
                    std.debug.print("MLIR GPU compilation failed: {}\n", .{err});
                }
                // Fall through to generate host wrapper anyway
            };
            
            if (self.verbose) {
                mlir.printMLIR();
            }
            
            // Generate host wrapper function that calls the GPU kernel
            try self.generateGpuHostWrapper(func);
        } else {
            if (self.verbose) {
                std.debug.print("MLIR not available for GPU compilation\n", .{});
            }
            // Generate a simple host wrapper for now
            try self.generateGpuHostWrapper(func);
        }
    }

    fn generateGpuHostWrapper(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) CodeGenError!void {
        // Create a host function that sets up and launches the GPU kernel
        const param_types = try self.allocator.alloc(LLVM.LLVMTypeRef, func.parameters.len);
        defer self.allocator.free(param_types);

        for (func.parameters, 0..) |param, i| {
            param_types[i] = param.type.toLLVMType(self.context);
        }

        // Create function type
        const actual_return_type = func.return_type.toLLVMType(self.context);
        const function_type = LLVM.LLVMFunctionType(actual_return_type, param_types.ptr, @intCast(param_types.len), 0);

        // Create function
        const name_z = try self.allocator.dupeZ(u8, func.name);
        defer self.allocator.free(name_z);

        const llvm_function = LLVM.LLVMAddFunction(self.module, name_z.ptr, function_type);
        try self.functions.put(try self.allocator.dupe(u8, func.name), llvm_function);

        // Create entry basic block
        const entry_block = LLVM.LLVMAppendBasicBlockInContext(self.context, llvm_function, "entry");
        LLVM.LLVMPositionBuilderAtEnd(self.builder, entry_block);

        // For now, just return a dummy value
        const return_value = switch (func.return_type) {
            .i32 => LLVM.LLVMConstInt(LLVM.LLVMInt32TypeInContext(self.context), 0, 0),
            .i64 => LLVM.LLVMConstInt(LLVM.LLVMInt64TypeInContext(self.context), 0, 0),
            .f32 => LLVM.LLVMConstReal(LLVM.LLVMFloatTypeInContext(self.context), 0.0),
            .f64 => LLVM.LLVMConstReal(LLVM.LLVMDoubleTypeInContext(self.context), 0.0),
            else => LLVM.LLVMConstInt(LLVM.LLVMInt32TypeInContext(self.context), 0, 0),
        };
        
        _ = LLVM.LLVMBuildRet(self.builder, return_value);

        if (self.verbose) {
            std.debug.print("Generated host wrapper for GPU function: {s}\n", .{func.name});
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
