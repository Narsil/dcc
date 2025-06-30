const std = @import("std");
const parser = @import("parser.zig");

pub const CodeGenError = error{ InvalidTopLevelNode, InvalidStatement, InvalidExpression, InvalidCallee, UndefinedVariable, UndefinedFunction, TargetError, CodeGenError } || std.mem.Allocator.Error;

// LLVM opaque types
const LLVMContextRef = *opaque{};
const LLVMModuleRef = *opaque{};
const LLVMBuilderRef = *opaque{};
const LLVMTypeRef = *opaque{};
const LLVMValueRef = *opaque{};
const LLVMBasicBlockRef = *opaque{};
const LLVMTargetRef = *opaque{};
const LLVMTargetMachineRef = *opaque{};

// LLVM C API bindings - simplified for this toy compiler
extern fn LLVMContextCreate() LLVMContextRef;
extern fn LLVMContextDispose(ctx: LLVMContextRef) void;
extern fn LLVMModuleCreateWithNameInContext(name: [*:0]const u8, ctx: LLVMContextRef) LLVMModuleRef;
extern fn LLVMDisposeModule(module: LLVMModuleRef) void;
extern fn LLVMCreateBuilderInContext(ctx: LLVMContextRef) LLVMBuilderRef;
extern fn LLVMDisposeBuilder(builder: LLVMBuilderRef) void;
extern fn LLVMInt64TypeInContext(ctx: LLVMContextRef) LLVMTypeRef;
extern fn LLVMFunctionType(return_type: LLVMTypeRef, param_types: [*]const LLVMTypeRef, param_count: c_uint, is_var_arg: c_int) LLVMTypeRef;
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
extern fn LLVMBuildCall2(builder: LLVMBuilderRef, function_type: LLVMTypeRef, function: LLVMValueRef, args: [*]const LLVMValueRef, num_args: c_uint, name: [*:0]const u8) LLVMValueRef;
extern fn LLVMBuildAlloca(builder: LLVMBuilderRef, type_: LLVMTypeRef, name: [*:0]const u8) LLVMValueRef;
extern fn LLVMBuildStore(builder: LLVMBuilderRef, value: LLVMValueRef, ptr: LLVMValueRef) LLVMValueRef;
extern fn LLVMBuildLoad2(builder: LLVMBuilderRef, type_: LLVMTypeRef, ptr: LLVMValueRef, name: [*:0]const u8) LLVMValueRef;
extern fn LLVMGetParam(function: LLVMValueRef, index: c_uint) LLVMValueRef;
extern fn LLVMPrintModuleToString(module: LLVMModuleRef) [*:0]u8;
extern fn LLVMDisposeMessage(message: [*:0]u8) void;

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
        switch (ast) {
            .program => |prog| {
                for (prog.statements) |stmt| {
                    try self.generateStatement(stmt);
                }
            },
            else => return error.InvalidTopLevelNode,
        }
    }
    
    fn generateStatement(self: *CodeGen, node: parser.ASTNode) CodeGenError!void {
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
        
        // Create parameter types array
        const param_types = try self.allocator.alloc(LLVMTypeRef, func.parameters.len);
        defer self.allocator.free(param_types);
        
        for (param_types) |*param_type| {
            param_type.* = int64_type;
        }
        
        // Create function type
        const function_type = LLVMFunctionType(
            int64_type,
            param_types.ptr,
            @intCast(param_types.len),
            0
        );
        
        // Create function
        const name_z = try self.allocator.dupeZ(u8, func.name);
        defer self.allocator.free(name_z);
        
        const llvm_function = LLVMAddFunction(self.module, name_z.ptr, function_type);
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
            _ = LLVMBuildStore(self.builder, param_value, alloca);
            try self.variables.put(try self.allocator.dupe(u8, param_name), alloca);
        }
        
        // Generate function body
        for (func.body) |stmt| {
            try self.generateStatement(stmt);
        }
        
        // Clear variables after function generation to avoid conflicts
        self.clearVariables();
    }
    
    fn generateVariableDeclaration(self: *CodeGen, var_decl: @TypeOf(@as(parser.ASTNode, undefined).variable_declaration)) CodeGenError!void {
        const int64_type = LLVMInt64TypeInContext(self.context);
        const name_z = try self.allocator.dupeZ(u8, var_decl.name);
        defer self.allocator.free(name_z);
        
        const alloca = LLVMBuildAlloca(self.builder, int64_type, name_z.ptr);
        const value = try self.generateExpression(var_decl.value.*);
        _ = LLVMBuildStore(self.builder, value, alloca);
        
        try self.variables.put(try self.allocator.dupe(u8, var_decl.name), alloca);
    }
    
    fn generateReturn(self: *CodeGen, ret: @TypeOf(@as(parser.ASTNode, undefined).return_statement)) CodeGenError!void {
        if (ret.value) |value| {
            const llvm_value = try self.generateExpression(value.*);
            _ = LLVMBuildRet(self.builder, llvm_value);
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
                    return LLVMBuildLoad2(self.builder, int64_type, alloca, name_z.ptr);
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
                    
                    const function_type = LLVMFunctionType(
                        int64_type,
                        param_types.ptr,
                        @intCast(param_types.len),
                        0
                    );
                    
                    return LLVMBuildCall2(self.builder, function_type, function, args.ptr, @intCast(args.len), "call");
                } else {
                    return error.UndefinedFunction;
                }
            },
            else => return error.InvalidExpression,
        }
    }
    
    pub fn generateObjectFile(self: *CodeGen, output_path: []const u8) CodeGenError!void {
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
        
        const triple_z = try self.allocator.dupeZ(u8, normalized_triple);
        defer self.allocator.free(triple_z);
        
        var target: ?LLVMTargetRef = null;
        var error_message: [*:0]u8 = undefined;
        
        if (LLVMGetTargetFromTriple(triple_z.ptr, &target, &error_message) != 0) {
            defer LLVMDisposeMessage(error_message);
            std.debug.print("Error getting target: {s}\n", .{error_message});
            std.debug.print("Tried target triple: {s}\n", .{normalized_triple});
            return error.TargetError;
        }
        
        const target_machine = LLVMCreateTargetMachine(
            target.?,
            triple_z.ptr,
            "generic",
            "",
            @intFromEnum(LLVMOptLevel.default),
            @intFromEnum(LLVMRelocMode.pic),
            @intFromEnum(LLVMCodeModel.default)
        );
        defer LLVMDisposeTargetMachine(target_machine);
        
        const output_path_z = try self.allocator.dupeZ(u8, output_path);
        defer self.allocator.free(output_path_z);
        
        if (LLVMTargetMachineEmitToFile(
            target_machine,
            self.module,
            output_path_z.ptr,
            @intFromEnum(LLVMCodeGenFileType.object),
            &error_message
        ) != 0) {
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
    
    pub fn clearVariables(self: *CodeGen) void {
        // Free all duplicated strings used as keys in the variables HashMap
        var var_iter = self.variables.iterator();
        while (var_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.variables.clearAndFree();
    }
}; 