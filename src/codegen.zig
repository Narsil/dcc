const std = @import("std");
const parser = @import("parser.zig");
const macho = std.macho;

pub const CodeGenError = error{ InvalidTopLevelNode, InvalidStatement, InvalidExpression, InvalidCallee, UndefinedVariable, UndefinedFunction, TargetError, CodeGenError, MainFunctionNotFound, MissingMainFunction, LinkingFailed } || std.mem.Allocator.Error;

// LLVM opaque types
const LLVMContextRef = *opaque {};
const LLVMModuleRef = *opaque {};
const LLVMBuilderRef = *opaque {};
const LLVMTypeRef = *opaque {};
const LLVMValueRef = *opaque {};
const LLVMBasicBlockRef = *opaque {};
const LLVMTargetRef = *opaque {};
const LLVMTargetMachineRef = *opaque {};

// LLVM C API bindings - simplified for this toy compiler
extern fn LLVMContextCreate() LLVMContextRef;
extern fn LLVMContextDispose(ctx: LLVMContextRef) void;
extern fn LLVMModuleCreateWithNameInContext(name: [*:0]const u8, ctx: LLVMContextRef) LLVMModuleRef;
extern fn LLVMDisposeModule(module: LLVMModuleRef) void;
extern fn LLVMCreateBuilderInContext(ctx: LLVMContextRef) LLVMBuilderRef;
extern fn LLVMDisposeBuilder(builder: LLVMBuilderRef) void;
extern fn LLVMInt64TypeInContext(ctx: LLVMContextRef) LLVMTypeRef;
extern fn LLVMInt32TypeInContext(ctx: LLVMContextRef) LLVMTypeRef;
extern fn LLVMVoidTypeInContext(ctx: LLVMContextRef) LLVMTypeRef;
extern fn LLVMInt8TypeInContext(ctx: LLVMContextRef) LLVMTypeRef;
extern fn LLVMArrayType(element_type: LLVMTypeRef, element_count: c_uint) LLVMTypeRef;
extern fn LLVMFunctionType(return_type: LLVMTypeRef, param_types: ?[*]const LLVMTypeRef, param_count: c_uint, is_var_arg: c_int) LLVMTypeRef;
extern fn LLVMAddFunction(module: LLVMModuleRef, name: [*:0]const u8, function_type: LLVMTypeRef) LLVMValueRef;
extern fn LLVMAppendBasicBlockInContext(ctx: LLVMContextRef, function: LLVMValueRef, name: [*:0]const u8) LLVMBasicBlockRef;
extern fn LLVMPositionBuilderAtEnd(builder: LLVMBuilderRef, block: LLVMBasicBlockRef) void;
extern fn LLVMConstInt(int_type: LLVMTypeRef, value: c_ulonglong, sign_extend: c_int) LLVMValueRef;
extern fn LLVMBuildAdd(builder: LLVMBuilderRef, lhs: LLVMValueRef, rhs: LLVMValueRef, name: [*:0]const u8) LLVMValueRef;
extern fn LLVMBuildSub(builder: LLVMBuilderRef, lhs: LLVMValueRef, rhs: LLVMValueRef, name: [*:0]const u8) LLVMValueRef;
extern fn LLVMBuildMul(builder: LLVMBuilderRef, lhs: LLVMValueRef, rhs: LLVMValueRef, name: [*:0]const u8) LLVMValueRef;
extern fn LLVMBuildSDiv(builder: LLVMBuilderRef, lhs: LLVMValueRef, rhs: LLVMValueRef, name: [*:0]const u8) LLVMValueRef;
extern fn LLVMBuildRet(builder: LLVMBuilderRef, value: LLVMValueRef) LLVMValueRef;
extern fn LLVMBuildRetVoid(builder: LLVMBuilderRef) LLVMValueRef;
extern fn LLVMBuildCall2(builder: LLVMBuilderRef, function_type: LLVMTypeRef, function: LLVMValueRef, args: ?[*]const LLVMValueRef, num_args: c_uint, name: [*:0]const u8) LLVMValueRef;
extern fn LLVMTypeOf(val: LLVMValueRef) LLVMTypeRef;
extern fn LLVMGetElementType(ty: LLVMTypeRef) LLVMTypeRef;
extern fn LLVMBuildTrunc(builder: LLVMBuilderRef, val: LLVMValueRef, dest_ty: LLVMTypeRef, name: [*:0]const u8) LLVMValueRef;
extern fn LLVMBuildUnreachable(builder: LLVMBuilderRef) LLVMValueRef;
extern fn LLVMBuildBr(builder: LLVMBuilderRef, dest: LLVMBasicBlockRef) LLVMValueRef;
extern fn LLVMSetFunctionCallConv(fn_: LLVMValueRef, cc: c_uint) void;
extern fn LLVMAddGlobal(module: LLVMModuleRef, ty: LLVMTypeRef, name: [*:0]const u8) LLVMValueRef;
extern fn LLVMSetInitializer(global_var: LLVMValueRef, constant_val: LLVMValueRef) void;
extern fn LLVMBuildAlloca(builder: LLVMBuilderRef, type_: LLVMTypeRef, name: [*:0]const u8) LLVMValueRef;
extern fn LLVMBuildStore(builder: LLVMBuilderRef, value: LLVMValueRef, ptr: LLVMValueRef) LLVMValueRef;
extern fn LLVMBuildLoad2(builder: LLVMBuilderRef, type_: LLVMTypeRef, ptr: LLVMValueRef, name: [*:0]const u8) LLVMValueRef;
extern fn LLVMSetAlignment(V: LLVMValueRef, Bytes: c_uint) void;
extern fn LLVMGetParam(function: LLVMValueRef, index: c_uint) LLVMValueRef;
extern fn LLVMPrintModuleToString(module: LLVMModuleRef) [*:0]u8;
extern fn LLVMDisposeMessage(message: [*:0]u8) void;
extern fn LLVMGetBasicBlockParent(BB: LLVMBasicBlockRef) LLVMValueRef;
extern fn LLVMGetInsertBlock(Builder: LLVMBuilderRef) LLVMBasicBlockRef;
extern fn LLVMGetValueName(Val: LLVMValueRef) [*:0]const u8;
extern fn LLVMBuildSExt(builder: LLVMBuilderRef, val: LLVMValueRef, dest_ty: LLVMTypeRef, name: [*:0]const u8) LLVMValueRef;

// Function attribute management for stack alignment
extern fn LLVMCreateEnumAttribute(ctx: LLVMContextRef, kind_id: c_uint, val: u64) LLVMAttributeRef;
extern fn LLVMAddAttributeAtIndex(fn_: LLVMValueRef, idx: c_uint, attr: LLVMAttributeRef) void;
extern fn LLVMGetEnumAttributeKindForName(name: [*:0]const u8, s_len: usize) c_uint;

// LLVM attribute types
const LLVMAttributeRef = *opaque {};

// Function attribute indices
const LLVMAttributeFunctionIndex = 0xffffffff;

// Target and code generation - use target-specific functions
extern fn LLVMInitializeX86TargetInfo() void;
extern fn LLVMInitializeX86Target() void;
extern fn LLVMInitializeX86TargetMC() void;
extern fn LLVMInitializeX86AsmParser() void;
extern fn LLVMInitializeX86AsmPrinter() void;

// AArch64 target initialization functions
extern fn LLVMInitializeAArch64TargetInfo() void;
extern fn LLVMInitializeAArch64Target() void;
extern fn LLVMInitializeAArch64TargetMC() void;
extern fn LLVMInitializeAArch64AsmParser() void;
extern fn LLVMInitializeAArch64AsmPrinter() void;

extern fn LLVMGetDefaultTargetTriple() [*:0]u8;
extern fn LLVMGetTargetFromTriple(triple: [*:0]const u8, target: *?LLVMTargetRef, error_message: *[*:0]u8) c_int;
extern fn LLVMCreateTargetMachine(target: LLVMTargetRef, triple: [*:0]const u8, cpu: [*:0]const u8, features: [*:0]const u8, level: c_int, reloc: c_int, code_model: c_int) LLVMTargetMachineRef;
extern fn LLVMDisposeTargetMachine(tm: LLVMTargetMachineRef) void;
extern fn LLVMTargetMachineEmitToFile(tm: LLVMTargetMachineRef, module: LLVMModuleRef, filename: [*:0]const u8, codegen: c_int, error_message: *[*:0]u8) c_int;

// Memory buffer and module reading/writing functions
extern fn LLVMCreateMemoryBufferWithContentsOfFile(path: [*:0]const u8, out_buf: *?LLVMMemoryBufferRef, out_message: *[*:0]u8) c_int;
extern fn LLVMDisposeMemoryBuffer(buf: LLVMMemoryBufferRef) void;
extern fn LLVMWriteBitcodeToFile(module: LLVMModuleRef, path: [*:0]const u8) c_int;
extern fn LLVMWriteBitcodeToMemoryBuffer(module: LLVMModuleRef) LLVMMemoryBufferRef;

// Module linking functions
extern fn LLVMLinkModules2(dest: LLVMModuleRef, src: LLVMModuleRef) c_int;

// LLVM Object File and Binary Generation APIs
extern fn LLVMCreateObjectFile(mem_buf: LLVMMemoryBufferRef) ?LLVMObjectFileRef;
extern fn LLVMDisposeObjectFile(obj_file: LLVMObjectFileRef) void;
extern fn LLVMGetSections(obj_file: LLVMObjectFileRef) LLVMSectionIteratorRef;
extern fn LLVMDisposeSectionIterator(si: LLVMSectionIteratorRef) void;
extern fn LLVMIsSectionIteratorAtEnd(obj_file: LLVMObjectFileRef, si: LLVMSectionIteratorRef) c_int;
extern fn LLVMMoveToNextSection(si: LLVMSectionIteratorRef) void;
extern fn LLVMGetSectionName(si: LLVMSectionIteratorRef) [*:0]const u8;
extern fn LLVMGetSectionContents(si: LLVMSectionIteratorRef) [*:0]const u8;
extern fn LLVMGetSectionSize(si: LLVMSectionIteratorRef) u64;
extern fn LLVMGetNamedFunction(M: LLVMModuleRef, Name: [*:0]const u8) ?LLVMValueRef;
extern fn LLVMCreateMemoryBufferWithMemoryRange(InputData: [*]const u8, InputDataLength: usize, BufferName: [*:0]const u8, RequiresNullTerminator: c_int) ?LLVMMemoryBufferRef;
extern fn LLVMGlobalGetValueType(Global: LLVMValueRef) LLVMTypeRef;

// LLVM Inline Assembly APIs
pub extern "c" fn LLVMGetInlineAsm(
    Ty: LLVMTypeRef, // Function type of the inline assembly
    AsmString: [*c]const u8, // The assembly code string
    AsmStringSize: usize, // Length of the assembly string
    Constraints: [*c]const u8, // Constraints string for operands
    ConstraintsSize: usize, // Length of the constraints string
    HasSideEffects: c_int, // Does the assembly have side effects? (0 for false, 1 for true)
    IsAlignStack: c_int, // Does the assembly need stack alignment? (0 for false, 1 for true)
    Dialect: c_int, // Assembly dialect (e.g., LLVMInlineAsmDialectATT)
    CanThrow: c_int, // Can the assembly throw an exception? (0 for false, 1 for true)
) LLVMValueRef;

// LLD (LLVM Linker) C wrapper API
extern fn lld_main(args: [*]const [*:0]const u8, argc: c_int) c_int;

// Additional LLVM types for memory buffers and object files
const LLVMMemoryBufferRef = *opaque {};
const LLVMObjectFileRef = *opaque {};
const LLVMSectionIteratorRef = *opaque {};

const LLVMCodeGenFileType = enum(c_int) {
    assembly = 0,
    object = 1,
};

const LLVMOptLevel = enum(c_int) {
    none = 0,
    less = 1,
    default = 2,
    aggressive = 3,
};

const LLVMRelocMode = enum(c_int) {
    default = 0,
    static = 1,
    pic = 2,
    dynamic_no_pic = 3,
    ropi = 4,
    rwpi = 5,
    ropi_rwpi = 6,
};

const LLVMCodeModel = enum(c_int) {
    default = 0,
    jit_default = 1,
    tiny = 2,
    small = 3,
    kernel = 4,
    medium = 5,
    large = 6,
};

// Darwin ARM64 system call numbers
const SYS_EXIT_DARWIN = 1; // exit() system call on Darwin

pub const CodeGen = struct {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    builder: LLVMBuilderRef,
    allocator: std.mem.Allocator,
    variables: std.HashMap([]const u8, LLVMValueRef, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    functions: std.HashMap([]const u8, LLVMValueRef, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),

    pub fn init(allocator: std.mem.Allocator, module_name: []const u8) !CodeGen {
        // Initialize LLVM X86 target (for x86_64 support)
        LLVMInitializeX86TargetInfo();
        LLVMInitializeX86Target();
        LLVMInitializeX86TargetMC();
        LLVMInitializeX86AsmParser();
        LLVMInitializeX86AsmPrinter();

        // Initialize LLVM AArch64 target (for arm64 support)
        LLVMInitializeAArch64TargetInfo();
        LLVMInitializeAArch64Target();
        LLVMInitializeAArch64TargetMC();
        LLVMInitializeAArch64AsmParser();
        LLVMInitializeAArch64AsmPrinter();

        const context = LLVMContextCreate();
        const module_name_z = try allocator.dupeZ(u8, module_name);
        defer allocator.free(module_name_z);

        const module = LLVMModuleCreateWithNameInContext(module_name_z.ptr, context);
        const builder = LLVMCreateBuilderInContext(context);

        return CodeGen{
            .context = context,
            .module = module,
            .builder = builder,
            .allocator = allocator,
            .variables = std.HashMap([]const u8, LLVMValueRef, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .functions = std.HashMap([]const u8, LLVMValueRef, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
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

        // Clean up LLVM resources
        LLVMDisposeBuilder(self.builder);
        LLVMDisposeModule(self.module);
        LLVMContextDispose(self.context);
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
            else => return error.InvalidStatement,
        }
    }

    fn generateFunction(self: *CodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) CodeGenError!void {
        const int64_type = LLVMInt64TypeInContext(self.context);
        const int32_type = LLVMInt32TypeInContext(self.context);

        // Use i32 return type for main function to match C runtime expectations
        const is_main = std.mem.eql(u8, func.name, "main");
        const return_type = if (is_main) int32_type else int64_type;

        // Create parameter types array
        const param_types = try self.allocator.alloc(LLVMTypeRef, func.parameters.len);
        defer self.allocator.free(param_types);

        for (param_types) |*param_type| {
            param_type.* = int64_type;
        }

        // Create function type
        const function_type = LLVMFunctionType(return_type, param_types.ptr, @intCast(param_types.len), 0);

        // Create function
        const name_z = try self.allocator.dupeZ(u8, func.name);
        defer self.allocator.free(name_z);

        const llvm_function = LLVMAddFunction(self.module, name_z.ptr, function_type);

        // Set proper stack alignment attributes for x86-64 System V ABI compliance
        self.setStackAlignmentAttributes(llvm_function);

        try self.functions.put(try self.allocator.dupe(u8, func.name), llvm_function);

        // Create entry basic block
        const entry_block = LLVMAppendBasicBlockInContext(self.context, llvm_function, "entry");
        LLVMPositionBuilderAtEnd(self.builder, entry_block);

        // Create allocas for parameters
        for (func.parameters, 0..) |param_name, i| {
            const param_value = LLVMGetParam(llvm_function, @intCast(i));
            const param_name_z = try self.allocator.dupeZ(u8, param_name);
            defer self.allocator.free(param_name_z);

            const alloca = LLVMBuildAlloca(self.builder, int64_type, param_name_z.ptr);
            LLVMSetAlignment(alloca, 16); // 16-byte alignment for x86-64 System V ABI
            const store_inst = LLVMBuildStore(self.builder, param_value, alloca);
            LLVMSetAlignment(store_inst, 16); // Match alloca alignment
            try self.variables.put(try self.allocator.dupe(u8, param_name), alloca);
        }

        // Add dummy alloca for stack alignment in main
        if (is_main) {
            // Allocate 16 bytes of padding so total local size is multiple of 16 (2x i64 = 16 + padding = 32)
            const int8_type = LLVMInt8TypeInContext(self.context);
            const dummy_array_type = LLVMArrayType(int8_type, 16); // 16 bytes
            const dummy_name = "dummy_padding";
            const dummy_alloca = LLVMBuildAlloca(self.builder, dummy_array_type, dummy_name);
            LLVMSetAlignment(dummy_alloca, 16);
        }

        // Generate function body
        for (func.body) |stmt| {
            try self.generateNode(stmt);
        }

        // Clear variables after function generation to avoid conflicts
        self.clearVariables();
    }

    fn generateVariableDeclaration(self: *CodeGen, var_decl: @TypeOf(@as(parser.ASTNode, undefined).variable_declaration)) CodeGenError!void {
        const int64_type = LLVMInt64TypeInContext(self.context);
        const name_z = try self.allocator.dupeZ(u8, var_decl.name);
        defer self.allocator.free(name_z);

        const alloca = LLVMBuildAlloca(self.builder, int64_type, name_z.ptr);
        LLVMSetAlignment(alloca, 16); // 16-byte alignment for x86-64 System V ABI
        const value = try self.generateExpression(var_decl.value.*);
        const store_inst = LLVMBuildStore(self.builder, value, alloca);
        LLVMSetAlignment(store_inst, 16); // Match alloca alignment

        try self.variables.put(try self.allocator.dupe(u8, var_decl.name), alloca);
    }

    fn generateReturn(self: *CodeGen, ret: @TypeOf(@as(parser.ASTNode, undefined).return_statement)) CodeGenError!void {
        if (ret.value) |value| {
            const llvm_value = try self.generateExpression(value.*);

            // Check if we're in the main function and need to convert i64 to i32
            const current_function = LLVMGetBasicBlockParent(LLVMGetInsertBlock(self.builder));
            const function_name = std.mem.span(LLVMGetValueName(current_function));

            if (std.mem.eql(u8, function_name, "main")) {
                // Convert i64 to i32 for main function return
                const int32_type = LLVMInt32TypeInContext(self.context);
                const truncated_value = LLVMBuildTrunc(self.builder, llvm_value, int32_type, "main_return");
                _ = LLVMBuildRet(self.builder, truncated_value);
            } else {
                _ = LLVMBuildRet(self.builder, llvm_value);
            }
        } else {
            _ = LLVMBuildRetVoid(self.builder);
        }
    }

    fn generateExpression(self: *CodeGen, node: parser.ASTNode) CodeGenError!LLVMValueRef {
        switch (node) {
            .number_literal => |num| {
                const int64_type = LLVMInt64TypeInContext(self.context);
                return LLVMConstInt(int64_type, @intCast(num.value), 0);
            },
            .identifier => |ident| {
                if (self.variables.get(ident.name)) |alloca| {
                    const int64_type = LLVMInt64TypeInContext(self.context);
                    const name_z = try self.allocator.dupeZ(u8, ident.name);
                    defer self.allocator.free(name_z);
                    const load_inst = LLVMBuildLoad2(self.builder, int64_type, alloca, name_z.ptr);
                    LLVMSetAlignment(load_inst, 16); // Match alloca alignment
                    return load_inst;
                } else {
                    return error.UndefinedVariable;
                }
            },
            .binary_expression => |bin_expr| {
                const left = try self.generateExpression(bin_expr.left.*);
                const right = try self.generateExpression(bin_expr.right.*);

                return switch (bin_expr.operator) {
                    .add => LLVMBuildAdd(self.builder, left, right, "add"),
                    .subtract => LLVMBuildSub(self.builder, left, right, "sub"),
                    .multiply => LLVMBuildMul(self.builder, left, right, "mul"),
                    .divide => LLVMBuildSDiv(self.builder, left, right, "div"),
                };
            },
            .call_expression => |call| {
                const callee_name = switch (call.callee.*) {
                    .identifier => |ident| ident.name,
                    else => return error.InvalidCallee,
                };

                if (self.functions.get(callee_name)) |function| {
                    const args = try self.allocator.alloc(LLVMValueRef, call.arguments.len);
                    defer self.allocator.free(args);

                    for (call.arguments, 0..) |arg, i| {
                        args[i] = try self.generateExpression(arg);
                    }

                    const int64_type = LLVMInt64TypeInContext(self.context);
                    const param_types = try self.allocator.alloc(LLVMTypeRef, call.arguments.len);
                    defer self.allocator.free(param_types);

                    for (param_types) |*param_type| {
                        param_type.* = int64_type;
                    }

                    const function_type = LLVMFunctionType(int64_type, param_types.ptr, @intCast(param_types.len), 0);

                    return LLVMBuildCall2(self.builder, function_type, function, args.ptr, @intCast(args.len), "call");
                } else {
                    return error.UndefinedFunction;
                }
            },
            else => return error.InvalidExpression,
        }
    }

    pub fn generateExecutable(self: *CodeGen, output_path: []const u8, target_triple: ?[]const u8) CodeGenError!void {
        // Use provided target triple or default to host triple
        const final_triple = if (target_triple) |triple| blk: {
            break :blk triple;
        } else blk: {
            const default_triple = LLVMGetDefaultTargetTriple();
            // defer LLVMDisposeMessage(default_triple);

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

        var target: ?LLVMTargetRef = null;
        var error_message: [*:0]u8 = undefined;

        if (LLVMGetTargetFromTriple(triple_z.ptr, &target, &error_message) != 0) {
            defer LLVMDisposeMessage(error_message);
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
        const target_machine = LLVMCreateTargetMachine(target.?, triple_z.ptr, "generic", "", @intFromEnum(LLVMOptLevel.default), @intFromEnum(LLVMRelocMode.static), // Static relocation for executables
            @intFromEnum(LLVMCodeModel.default));
        defer LLVMDisposeTargetMachine(target_machine);

        // Generate object file
        const obj_path = try std.fmt.allocPrint(self.allocator, "{s}.o", .{output_path});
        defer self.allocator.free(obj_path);

        const obj_path_z = try self.allocator.dupeZ(u8, obj_path);
        defer self.allocator.free(obj_path_z);

        if (LLVMTargetMachineEmitToFile(target_machine, self.module, obj_path_z.ptr, @intFromEnum(LLVMCodeGenFileType.object), &error_message) != 0) {
            defer LLVMDisposeMessage(error_message);
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
            try args.append(try self.allocator.dupeZ(u8, if (std.mem.indexOf(u8, target_triple, "arm64") != null) "arm64" else "x86_64"));
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
            const default_triple = LLVMGetDefaultTargetTriple();
            defer LLVMDisposeMessage(default_triple);

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

        var target: ?LLVMTargetRef = null;
        var error_message: [*:0]u8 = undefined;

        if (LLVMGetTargetFromTriple(triple_z.ptr, &target, &error_message) != 0) {
            defer LLVMDisposeMessage(error_message);
            std.debug.print("Error getting target: {s}\n", .{error_message});
            std.debug.print("Tried target triple: {s}\n", .{final_triple});
            return error.TargetError;
        }

        // Create target machine optimized for shared libraries
        const target_machine = LLVMCreateTargetMachine(target.?, triple_z.ptr, "generic", "", @intFromEnum(LLVMOptLevel.default), @intFromEnum(LLVMRelocMode.pic), // Position Independent Code for shared libraries
            @intFromEnum(LLVMCodeModel.default));
        defer LLVMDisposeTargetMachine(target_machine);

        // Generate object file
        const obj_path = try std.fmt.allocPrint(self.allocator, "{s}.o", .{output_path});
        defer self.allocator.free(obj_path);

        const obj_path_z = try self.allocator.dupeZ(u8, obj_path);
        defer self.allocator.free(obj_path_z);

        if (LLVMTargetMachineEmitToFile(target_machine, self.module, obj_path_z.ptr, @intFromEnum(LLVMCodeGenFileType.object), &error_message) != 0) {
            defer LLVMDisposeMessage(error_message);
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
            try args.append(try self.allocator.dupeZ(u8, if (std.mem.indexOf(u8, target_triple, "arm64") != null) "arm64" else "x86_64"));
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
        const memory_buffer = LLVMCreateMemoryBufferWithMemoryRange(object_data.ptr, object_data.len, "object_buffer", 0 // don't require null termination
        );
        defer if (memory_buffer != null) {
            // Skip disposal to avoid potential crashes with LLVM memory management
            // LLVMDisposeMemoryBuffer(memory_buffer);
        };

        if (memory_buffer == null) {
            std.debug.print("Failed to create memory buffer\n", .{});
            return CodeGenError.LinkingFailed;
        }

        // Create an object file from the memory buffer
        const object_file = LLVMCreateObjectFile(memory_buffer.?);
        defer if (object_file != null) {
            LLVMDisposeObjectFile(object_file.?);
        };

        if (object_file == null) {
            std.debug.print("Failed to create object file\n", .{});
            return CodeGenError.LinkingFailed;
        }

        // Get sections iterator
        const sections_iterator = LLVMGetSections(object_file.?);
        defer LLVMDisposeSectionIterator(sections_iterator);

        // Find the __text section
        while (LLVMIsSectionIteratorAtEnd(object_file.?, sections_iterator) == 0) {
            const section_name_ptr = LLVMGetSectionName(sections_iterator);
            const section_name = std.mem.span(section_name_ptr);

            if (std.mem.eql(u8, section_name, "__text")) {
                const section_size = LLVMGetSectionSize(sections_iterator);
                const section_contents = LLVMGetSectionContents(sections_iterator);

                text_size = @intCast(section_size);
                object_code = section_contents[0..text_size];
                break;
            }

            LLVMMoveToNextSection(sections_iterator);
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
            const default_triple = LLVMGetDefaultTargetTriple();
            defer LLVMDisposeMessage(default_triple);

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

        var target: ?LLVMTargetRef = null;
        var error_message: [*:0]u8 = undefined;

        if (LLVMGetTargetFromTriple(triple_z.ptr, &target, &error_message) != 0) {
            defer LLVMDisposeMessage(error_message);
            std.debug.print("Error getting target: {s}\n", .{error_message});
            std.debug.print("Tried target triple: {s}\n", .{final_triple});
            return error.TargetError;
        }

        const target_machine = LLVMCreateTargetMachine(target.?, triple_z.ptr, "generic", "", @intFromEnum(LLVMOptLevel.default), @intFromEnum(LLVMRelocMode.pic), @intFromEnum(LLVMCodeModel.default));
        defer LLVMDisposeTargetMachine(target_machine);

        const output_path_z = try self.allocator.dupeZ(u8, output_path);
        defer self.allocator.free(output_path_z);

        if (LLVMTargetMachineEmitToFile(target_machine, self.module, output_path_z.ptr, @intFromEnum(LLVMCodeGenFileType.object), &error_message) != 0) {
            defer LLVMDisposeMessage(error_message);
            std.debug.print("Error generating object file: {s}\n", .{error_message});
            return error.CodeGenError;
        }
    }

    pub fn printIR(self: *CodeGen) void {
        const ir_string = LLVMPrintModuleToString(self.module);
        defer LLVMDisposeMessage(ir_string);
        std.debug.print("Generated LLVM IR:\n{s}\n", .{ir_string});
    }

    fn generateStartFunction(self: *CodeGen) CodeGenError!void {
        // Create _start function for Linux ELF executables
        // This function calls main() and returns the exit code

        const i32_type = LLVMInt32TypeInContext(self.context);
        const start_function_type = LLVMFunctionType(i32_type, null, 0, 0);
        const start_function = LLVMAddFunction(self.module, "_start", start_function_type);

        // Set proper stack alignment attributes for x86-64 System V ABI compliance
        self.setStackAlignmentAttributes(start_function);

        // Create basic block
        const entry_block = LLVMAppendBasicBlockInContext(self.context, start_function, "entry");
        LLVMPositionBuilderAtEnd(self.builder, entry_block);

        // Add dummy alloca for stack alignment in _start (accounts for call instruction push)
        // Allocate 8 bytes of padding to help ensure 16-byte alignment before calling main
        const int8_type = LLVMInt8TypeInContext(self.context);
        const dummy_array_type = LLVMArrayType(int8_type, 8); // 8 bytes
        const dummy_name_start = "dummy_padding_start";
        const dummy_alloca_start = LLVMBuildAlloca(self.builder, dummy_array_type, dummy_name_start);
        LLVMSetAlignment(dummy_alloca_start, 16);

        // Get the main function
        const main_function_ = LLVMGetNamedFunction(self.module, "main");
        if (main_function_ == null) {
            std.debug.print("Error: main function not found when generating _start\n", .{});
            return error.CodeGenError;
        }
        const main_function = main_function_.?;

        // Call main() - main now returns i32
        const main_function_type = LLVMFunctionType(i32_type, null, 0, 0);
        const exit_code = LLVMBuildCall2(self.builder, main_function_type, main_function, null, 0, "main_result");

        // Store the exit code in a global variable for inspection
        const exit_code_global = LLVMAddGlobal(self.module, i32_type, "program_exit_code");
        LLVMSetInitializer(exit_code_global, LLVMConstInt(i32_type, 0, 0));
        _ = LLVMBuildStore(self.builder, exit_code, exit_code_global);

        // --- Exit the process via Linux x86_64 syscall ---
        const void_type = LLVMVoidTypeInContext(self.context);
        const int64_type = LLVMInt64TypeInContext(self.context);
        const syscall_asm_ty = LLVMFunctionType(void_type, &[_]LLVMTypeRef{int64_type}, 1, 0);
        const asm_str = "mov $$60, %rax\nsyscall"; // rax=60 (SYS_exit), rdi=status

        const syscall_inline = LLVMGetInlineAsm(syscall_asm_ty, asm_str, asm_str.len, "r", 1, // single general-purpose register input
            1, // has side effects
            0, // align stack
            0, // ATT dialect
            0 // can throw
        );

        const exit_code_64 = LLVMBuildSExt(self.builder, exit_code, int64_type, "exit_code64");
        _ = LLVMBuildCall2(self.builder, syscall_asm_ty, syscall_inline, &[_]LLVMValueRef{exit_code_64}, 1, "");

        // Mark unreachable as the syscall terminates the program
        _ = LLVMBuildUnreachable(self.builder);

        self.printIR();
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
    fn setStackAlignmentAttributes(self: *CodeGen, function: LLVMValueRef) void {
        // Set stack realignment for x86-64 System V ABI compliance
        const stackrealign_attr_name = "stackrealign";
        const stackrealign_kind = LLVMGetEnumAttributeKindForName(stackrealign_attr_name.ptr, stackrealign_attr_name.len);
        const stackrealign_attr = LLVMCreateEnumAttribute(self.context, stackrealign_kind, 0);
        LLVMAddAttributeAtIndex(function, LLVMAttributeFunctionIndex, stackrealign_attr);

        // Set unwind table for proper exception handling and debugging
        const uwtable_attr_name = "uwtable";
        const uwtable_kind = LLVMGetEnumAttributeKindForName(uwtable_attr_name.ptr, uwtable_attr_name.len);
        const uwtable_attr = LLVMCreateEnumAttribute(self.context, uwtable_kind, 2); // sync uwtable
        LLVMAddAttributeAtIndex(function, LLVMAttributeFunctionIndex, uwtable_attr);

        // Set proper calling convention for System V ABI
        LLVMSetFunctionCallConv(function, 0); // C calling convention (System V ABI)

        // Ensure the backend maintains at least 16-byte stack alignment
        const alignstack_attr_name = "alignstack"; // LLVM attribute: alignstack(<alignment>)
        const alignstack_kind = LLVMGetEnumAttributeKindForName(alignstack_attr_name.ptr, alignstack_attr_name.len);
        if (alignstack_kind != 0) { // only add if LLVM recognises the attribute
            const alignstack_attr = LLVMCreateEnumAttribute(self.context, alignstack_kind, 16);
            LLVMAddAttributeAtIndex(function, LLVMAttributeFunctionIndex, alignstack_attr);
        }
    }
};

fn generateSimpleEntryPoint(module: LLVMModuleRef, context: LLVMContextRef, builder: LLVMBuilderRef, main_func: LLVMValueRef) !LLVMValueRef {
    // Create an entry point function that calls main() and then exits properly
    const i32_type = LLVMInt32TypeInContext(context);
    const i64_type = LLVMInt64TypeInContext(context);
    const void_type = LLVMVoidTypeInContext(context);
    const entry_type = LLVMFunctionType(i32_type, null, 0, 0);
    const entry_func = LLVMAddFunction(module, "_start", entry_type);

    // Declare the external exit function from libSystem
    const exit_type = LLVMFunctionType(void_type, &[_]LLVMTypeRef{i32_type}, 1, 0);
    const exit_func = LLVMAddFunction(module, "exit", exit_type);

    const entry_block = LLVMAppendBasicBlockInContext(context, entry_func, "entry");
    LLVMPositionBuilderAtEnd(builder, entry_block);

    // Get the main function type (should be i64 () )
    const main_type = LLVMFunctionType(i64_type, null, 0, 0);

    // Call main function
    const main_call = LLVMBuildCall2(builder, main_type, main_func, null, 0, "main_result");

    // Convert the exit code from main (i64) to i32
    const exit_code = LLVMBuildTrunc(builder, main_call, i32_type, "exit_code");

    // Call exit() from libSystem
    _ = LLVMBuildCall2(builder, exit_type, exit_func, &[_]LLVMValueRef{exit_code}, 1, "");

    // This should never be reached, but add unreachable just in case
    _ = LLVMBuildUnreachable(builder);

    return entry_func;
}

fn generateEntryPoint(module: LLVMModuleRef, context: LLVMContextRef, builder: LLVMBuilderRef, main_func: LLVMValueRef) !LLVMValueRef {
    // Create entry point function that calls main() and then exits
    const i32_type = LLVMInt32TypeInContext(context);
    const i64_type = LLVMInt64TypeInContext(context);
    const entry_type = LLVMFunctionType(i32_type, null, 0, 0);
    const entry_func = LLVMAddFunction(module, "_start", entry_type);

    const entry_block = LLVMAppendBasicBlockInContext(context, entry_func, "entry");
    LLVMPositionBuilderAtEnd(builder, entry_block);

    // Get the main function type (should be i64 () )
    const main_type = LLVMFunctionType(i64_type, null, 0, 0);

    // Call main function
    const main_call = LLVMBuildCall2(builder, main_type, main_func, null, 0, "main_result");

    // Convert the exit code from main (i64) to i32 for the exit syscall
    const exit_code = LLVMBuildTrunc(builder, main_call, i32_type, "exit_code");

    // Create Darwin ARM64 system call for exit
    // Darwin uses svc #0x80 instead of svc #0
    const inline_asm_type = LLVMFunctionType(LLVMVoidTypeInContext(context), &[_]LLVMTypeRef{i32_type}, 1, 0);
    const inline_asm = LLVMGetInlineAsm(inline_asm_type, "mov x16, #1\nmov x0, $0\nsvc #0x80", // Darwin ARM64 exit syscall
        "r", // input constraint: general register
        1, // has side effects
        0, // align stack
        0, // ATT dialect
        0 // can throw
    );

    // Call the inline assembly with exit code
    _ = LLVMBuildCall2(builder, inline_asm_type, inline_asm, &[_]LLVMValueRef{exit_code}, 1, "");

    // Add unreachable instruction (should never be reached)
    _ = LLVMBuildUnreachable(builder);

    return entry_func;
}
