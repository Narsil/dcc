const std = @import("std");
const parser = @import("parser.zig");
const lexer = @import("lexer.zig");

pub const TypeCheckError = error{
    TypeMismatch,
    UndefinedVariable,
    UndefinedFunction,
    InvalidReturnType,
    InvalidExpression,
    DuplicateFunction,
    InvalidFunctionCall,
    TargetMustBeTensor,
    ImplicitIndexConflictsWithVariable,
    IndexCountMismatch,
    InvalidIndexType,
    IndexOutOfBounds,
    InvalidBinaryOperation,
    InvalidUnaryOperation,
    InvalidVariableType,
    InvalidMainFunctionReturnType,
    UnusedVariable,
    UnusedFunction,
    MainMustBePublic,
} || std.mem.Allocator.Error;

pub const FunctionInfo = struct {
    params: []parser.Parameter,
    return_type: parser.Type,
    is_public: bool,
    declaration_offset: usize,
    used: bool,
};

pub const ReductionInfo = struct {
    free_indices: [][]const u8,     // Indices that appear on both sides (not reduced)
    bound_indices: [][]const u8,    // Indices that only appear in reduce (reduced)
    operator: parser.BinaryOperator,
};

pub const VariableInfo = struct {
    type: parser.Type,
    declaration_offset: usize,
    used: bool,
};

pub const TypeChecker = struct {
    allocator: std.mem.Allocator,
    source: []const u8,
    verbose: bool,
    variables: std.HashMap([]const u8, VariableInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    functions: std.HashMap([]const u8, FunctionInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    current_function_return_type: ?parser.Type,
    allocated_types: std.ArrayList(parser.Type),
    reduction_info: std.AutoHashMap(*parser.ASTNode, ReductionInfo),

    pub fn init(allocator: std.mem.Allocator, source: []const u8, verbose: bool) TypeChecker {
        return TypeChecker{
            .allocator = allocator,
            .source = source,
            .verbose = verbose,
            .variables = std.HashMap([]const u8, VariableInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .functions = std.HashMap([]const u8, FunctionInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .current_function_return_type = null,
            .allocated_types = std.ArrayList(parser.Type).init(allocator),
            .reduction_info = std.AutoHashMap(*parser.ASTNode, ReductionInfo).init(allocator),
        };
    }

    pub fn deinit(self: *TypeChecker) void {
        // Free all dynamically allocated types created during type checking
        for (self.allocated_types.items) |allocated_type| {
            parser.freeType(self.allocator, allocated_type);
        }
        self.allocated_types.deinit();

        // Free all duplicated strings used as keys
        var var_iter = self.variables.iterator();
        while (var_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.variables.deinit();

        var func_iter = self.functions.iterator();
        while (func_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.functions.deinit();
        
        // Free reduction info
        var red_iter = self.reduction_info.iterator();
        while (red_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.free_indices);
            self.allocator.free(entry.value_ptr.bound_indices);
        }
        self.reduction_info.deinit();
    }

    pub fn typeCheck(self: *TypeChecker, ast: parser.ASTNode) TypeCheckError!void {
        if (self.verbose) {
            std.debug.print("DEBUG: Entering typeCheck\n", .{});
        }
        switch (ast) {
            .program => |prog| {
                // First pass: collect function signatures
                for (prog.statements) |stmt| {
                    if (stmt == .function_declaration) {
                        if (self.verbose) {
                            std.debug.print("DEBUG: Collecting signature for function '{s}'\n", .{stmt.function_declaration.name});
                        }
                        try self.collectFunctionSignature(stmt.function_declaration);
                    }
                }

                // Second pass: type check all statements
                for (prog.statements) |stmt| {
                    if (self.verbose) {
                        std.debug.print("DEBUG: Type checking statement of type {}\n", .{stmt});
                    }
                    try self.typeCheckStatement(stmt);
                }
                
                // Check that main is public
                if (self.functions.get("main")) |main_func| {
                    if (!main_func.is_public) {
                        const pos = self.getPositionFromOffset(main_func.declaration_offset);
                        std.debug.print("Error at line {}, column {}: Function 'main' must be declared public\n", .{ pos.line, pos.column });
                        self.printSourceContext(main_func.declaration_offset);
                        return error.MainMustBePublic;
                    }
                }
                
                // Check for unused functions
                var func_iter = self.functions.iterator();
                while (func_iter.next()) |entry| {
                    if (!entry.value_ptr.used and !entry.value_ptr.is_public) {
                        const pos = self.getPositionFromOffset(entry.value_ptr.declaration_offset);
                        std.debug.print("Error at line {}, column {}: Function '{s}' is declared but never used\n", .{ pos.line, pos.column, entry.key_ptr.* });
                        self.printSourceContext(entry.value_ptr.declaration_offset);
                        return error.UnusedFunction;
                    }
                }
            },
            else => return error.TypeMismatch,
        }
    }

    fn collectFunctionSignature(self: *TypeChecker, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) TypeCheckError!void {
        const name = try self.allocator.dupe(u8, func.name);
        const is_main = std.mem.eql(u8, func.name, "main");
        try self.functions.put(name, .{
            .params = func.parameters,
            .return_type = func.return_type,
            .is_public = func.is_public,
            .declaration_offset = func.offset,
            .used = is_main or func.is_public, // main and public functions are considered used
        });
        if (self.verbose) {
            std.debug.print("DEBUG: Collected function signature '{s}' -> return type {}, public: {}\n", .{ func.name, func.return_type, func.is_public });
        }
    }

    fn typeCheckStatement(self: *TypeChecker, stmt: parser.ASTNode) TypeCheckError!void {
        switch (stmt) {
            .function_declaration => |func| {
                // Only type check the function, do not type check its body as a statement
                try self.typeCheckFunction(func);
                return;
            },
            .variable_declaration => |var_decl| {
                try self.typeCheckVariableDeclaration(var_decl);
            },
            .return_statement => |ret| {
                try self.typeCheckReturnStatement(ret);
            },
            .expression_statement => |expr_stmt| {
                _ = try self.typeCheckExpression(expr_stmt.expression);
            },
            .parallel_assignment => |parallel_assign| {
                try self.typeCheckParallelAssignment(parallel_assign);
            },
            else => return error.TypeMismatch,
        }
    }

    fn typeCheckFunction(self: *TypeChecker, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) TypeCheckError!void {
        // Create new scope for function parameters and body
        const old_variables = self.variables;
        const old_return_type = self.current_function_return_type;
        self.variables = std.HashMap([]const u8, VariableInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(self.allocator);
        self.current_function_return_type = func.return_type;

        if (self.verbose) {
            std.debug.print("DEBUG: Entering function '{s}' with return type {}\n", .{ func.name, func.return_type });
        }

        // Add parameters to scope
        for (func.parameters) |param| {
            const name = try self.allocator.dupe(u8, param.name);
            try self.variables.put(name, .{
                .type = param.type,
                .declaration_offset = func.offset, // Use function offset since parameters don't have individual offsets
                .used = true, // Parameters are considered used by default
            });
        }

        // Type check function body, skipping nested function declarations
        for (func.body) |stmt| {
            if (stmt == .function_declaration) continue;
            try self.typeCheckStatement(stmt);
        }

        // Check that non-void functions have return statements
        if (func.return_type != .void) {
            // For a simple implementation, check that the last statement is a return statement
            // This covers the most common case where functions end with a return
            var has_return = false;
            
            // Look for any return statement in the function body
            for (func.body) |stmt| {
                if (stmt == .return_statement) {
                    has_return = true;
                    break;
                }
            }
            
            if (!has_return) {
                // Get the actual position of the function declaration
                const func_pos = self.getPositionFromOffset(func.offset);
                std.debug.print("Error at line {}, column {}: Must return value from non-void function '{s}'\n", .{ func_pos.line, func_pos.column, func.name });
                self.printSourceContext(func.offset);
                return error.InvalidReturnType;
            }
        }

        // Check for invalid main function return type
        if (std.mem.eql(u8, func.name, "main")) {
            if (func.return_type == .f32) {
                const pos = self.getPositionFromOffset(func.offset);
                std.debug.print("Error at line {}, column {}: main function cannot return f32\n", .{ pos.line, pos.column });
                self.printSourceContext(func.offset);
                return error.InvalidMainFunctionReturnType;
            }
        }

        // Check for unused variables before restoring scope
        var unused_iter = self.variables.iterator();
        while (unused_iter.next()) |entry| {
            if (!entry.value_ptr.used) {
                const pos = self.getPositionFromOffset(entry.value_ptr.declaration_offset);
                std.debug.print("Error at line {}, column {}: Variable '{s}' is declared but never used\n", .{ pos.line, pos.column, entry.key_ptr.* });
                self.printSourceContext(entry.value_ptr.declaration_offset);
                return error.UnusedVariable;
            }
        }

        // Restore old scope
        var var_iter = self.variables.iterator();
        while (var_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.variables.deinit();
        self.variables = old_variables;
        self.current_function_return_type = old_return_type;

        if (self.verbose) {
            std.debug.print("DEBUG: Exiting function '{s}'\n", .{func.name});
        }
    }

    fn typeCheckVariableDeclaration(self: *TypeChecker, var_decl: @TypeOf(@as(parser.ASTNode, undefined).variable_declaration)) TypeCheckError!void {
        const declared_type = var_decl.type;
        const value_type = try self.typeCheckExpression(var_decl.value);
        if (self.verbose) {
            std.debug.print("DEBUG: typeCheckNode - variable_declaration: declared_type={}, value_type={} (tag: {s})\n", .{ declared_type, value_type, @tagName(value_type) });
            if (declared_type == .tensor and value_type == .tensor) {
                std.debug.print("DEBUG: declared_type.element_type ptr: {any}, value: {}\n", .{ declared_type.tensor.element_type, declared_type.tensor.element_type.* });
                std.debug.print("DEBUG: value_type.element_type ptr: {any}, value: {}\n", .{ value_type.tensor.element_type, value_type.tensor.element_type.* });
            }
        }
        if (!self.typesCompatible(declared_type, value_type)) {
            const pos = self.getNodePosition(var_decl.value.*);
            std.debug.print("Error at line {}, column {}: Cannot assign value of type {} to variable of type {}\n", .{ pos.line, pos.column, value_type, declared_type });
            self.printSourceContext(self.getNodeOffset(var_decl.value.*));
            return error.TypeMismatch;
        }

        // Add variable to scope
        const name = try self.allocator.dupe(u8, var_decl.name);
        try self.variables.put(name, .{
            .type = var_decl.type,
            .declaration_offset = var_decl.offset,
            .used = false,
        });
    }

    fn typeCheckReturnStatement(self: *TypeChecker, ret: @TypeOf(@as(parser.ASTNode, undefined).return_statement)) TypeCheckError!void {
        const expected_type = self.current_function_return_type orelse {
            const pos = if (ret.value) |val| self.getNodePosition(val.*) else self.getNodePosition(parser.ASTNode{ .return_statement = ret });
            std.debug.print("Error at line {}, column {}: Return statement outside of function\n", .{ pos.line, pos.column });
            const offset = if (ret.value) |val| self.getNodeOffset(val.*) else ret.offset;
            self.printSourceContext(offset);
            return error.InvalidReturnType;
        };

        if (self.verbose) {
            std.debug.print("DEBUG: Return statement - expected type: {}\n", .{expected_type});
        }

        if (ret.value) |value| {
            // Return statement has a value
            if (expected_type == .void) {
                // Function returns void but return statement has a value
                const pos = self.getNodePosition(value.*);
                std.debug.print("Error at line {}, column {}: Cannot return value from void function\n", .{ pos.line, pos.column });
                self.printSourceContext(self.getNodeOffset(value.*));
                return error.InvalidReturnType;
            }
            
            const actual_type = try self.typeCheckExpression(value);
            if (self.verbose) {
                std.debug.print("DEBUG: Return statement - actual type: {}\n", .{actual_type});
            }

            if (!self.typesCompatible(expected_type, actual_type)) {
                const pos = self.getNodePosition(value.*);
                std.debug.print("Error at line {}, column {}: Cannot return value of type {} from function returning {}\n", .{ pos.line, pos.column, actual_type, expected_type });
                self.printSourceContext(self.getNodeOffset(value.*));
                return error.InvalidReturnType;
            }
        } else {
            // Return statement has no value
            if (expected_type != .void) {
                // Function expects a return value but none provided
                const pos = self.getNodePosition(parser.ASTNode{ .return_statement = ret });
                std.debug.print("Error at line {}, column {}: Must return value of type {} from non-void function\n", .{ pos.line, pos.column, expected_type });
                self.printSourceContext(ret.offset);
                return error.InvalidReturnType;
            }
        }
    }

    fn typeCheckExpression(self: *TypeChecker, node: *parser.ASTNode) TypeCheckError!parser.Type {
        const result_type = switch (node.*) {
            .tensor_literal => |tensor_lit| try self.typeCheckTensorLiteral(tensor_lit),
            .implicit_tensor_index => |tensor_index| try self.typeCheckImplicitTensorIndex(tensor_index),
            .tensor_slice => |tensor_slice| try self.typeCheckTensorSlice(tensor_slice),
            .parallel_assignment => |parallel_assign| {
                try self.typeCheckParallelAssignment(parallel_assign);
                return parser.Type.i64;
            },
            .number_literal => |num| num.type,
            .identifier => |ident| blk: {
                if (self.variables.getPtr(ident.name)) |var_info| {
                    var_info.used = true;
                    break :blk var_info.type;
                } else {
                    const pos = self.getNodePosition(node.*);
                    std.debug.print("Error at line {}, column {}: Undefined variable '{s}'\n", .{ pos.line, pos.column, ident.name });
                    self.printSourceContext(self.getNodeOffset(node.*));
                    return error.UndefinedVariable;
                }
            },
            .binary_expression => |bin_expr| try self.typeCheckBinaryOperation(bin_expr.operator, try self.typeCheckExpression(bin_expr.left), try self.typeCheckExpression(bin_expr.right), bin_expr),
            .unary_expression => |unary_expr| try self.typeCheckUnaryOperation(unary_expr.operator, try self.typeCheckExpression(unary_expr.operand), unary_expr),
            .call_expression => |*call| {
                const return_type = try self.typeCheckFunctionCall(call.*);
                call.*.return_type = return_type;
                return return_type;
            },
            .reduce_expression => |reduce| try self.typeCheckReduceExpression(reduce),
            .string_literal => |str| blk: {
                // String literals are typed as [n]u8
                const tensor_type = try self.allocator.create(parser.Type);
                tensor_type.* = parser.Type{ .u8 = {} };
                
                const shape = try self.allocator.alloc(u32, 1);
                shape[0] = str.length;
                
                const result_type = parser.Type{
                    .tensor = parser.Type.TensorType{
                        .shape = shape,
                        .element_type = tensor_type,
                    },
                };
                try self.allocated_types.append(result_type);
                break :blk result_type;
            },
            .namespace_access => |ns| blk: {
                // For now, we'll treat io.stdout, io.stderr, io.stdin as special IO handles
                if (std.mem.eql(u8, ns.namespace, "io")) {
                    if (std.mem.eql(u8, ns.member, "stdout") or 
                        std.mem.eql(u8, ns.member, "stderr") or 
                        std.mem.eql(u8, ns.member, "stdin")) {
                        // Return a special IO handle type (for now, treat as void pointer)
                        break :blk parser.Type{ .void = {} };
                    }
                }
                const pos = self.getNodePosition(node.*);
                std.debug.print("Error at line {}, column {}: Unknown namespace member '{s}.{s}'\n", .{ pos.line, pos.column, ns.namespace, ns.member });
                self.printSourceContext(ns.offset);
                return error.UndefinedVariable;
            },
            .write_expression => |write| blk: {
                // Type check the handle
                const handle_type = try self.typeCheckExpression(write.handle);
                // Accept void type (for io.stdout/stderr) or i32 (file descriptors)
                if (handle_type != .void and handle_type != .i32) {
                    const pos = self.getNodePosition(node.*);
                    std.debug.print("Error at line {}, column {}: First argument to write must be an IO handle (void or i32)\n", .{ pos.line, pos.column });
                    self.printSourceContext(write.offset);
                    return error.TypeMismatch;
                }
                
                // Type check the data - for now, accept any type
                _ = try self.typeCheckExpression(write.data);
                
                // write returns void
                break :blk parser.Type{ .void = {} };
            },
            .read_expression => |read| blk: {
                // Type check the handle
                const handle_type = try self.typeCheckExpression(read.handle);
                // For now, accept void type as IO handle
                if (handle_type != .void) {
                    const pos = self.getNodePosition(node.*);
                    std.debug.print("Error at line {}, column {}: First argument to read must be an IO handle\n", .{ pos.line, pos.column });
                    self.printSourceContext(read.offset);
                    return error.TypeMismatch;
                }
                
                // Return the requested type
                break :blk read.read_type;
            },
            else => return error.InvalidExpression,
        };
        if (self.verbose) {
            std.debug.print("DEBUG: typeCheckExpression - node tag: {any}, result_type: {}\n", .{ @tagName(node.*), result_type });
        }
        return result_type;
    }

    fn typeCheckBinaryOperation(self: *TypeChecker, op: parser.BinaryOperator, left_type: parser.Type, right_type: parser.Type, expr: @TypeOf(@as(parser.ASTNode, undefined).binary_expression)) TypeCheckError!parser.Type {
        // Check if operands are compatible for the operation
        if (!self.typesCompatible(left_type, right_type)) {
            const pos = self.getPositionFromOffset(expr.offset);
            std.debug.print("Error at line {}, column {}: Cannot perform {} operation on types ", .{ pos.line, pos.column, op });
            self.printType(left_type);
            std.debug.print(" and ", .{});
            self.printType(right_type);
            std.debug.print("\n", .{});
            self.printSourceContext(expr.offset);
            return error.InvalidBinaryOperation;
        }

        // For arithmetic operations, both operands must be numeric
        switch (op) {
            .add, .subtract, .multiply, .divide => {
                if (!self.isNumericType(left_type) or !self.isNumericType(right_type)) {
                    const pos = self.getNodePosition(expr.left.*);
                    std.debug.print("Error at line {}, column {}: Arithmetic operation {} requires numeric types, got {} and {}\n", .{ pos.line, pos.column, op, left_type, right_type });
                    self.printSourceContext(self.getNodeOffset(expr.left.*));
                    return error.InvalidBinaryOperation;
                }
                return left_type; // Result type is same as left operand
            },
        }
    }

    fn typeCheckUnaryOperation(self: *TypeChecker, op: parser.UnaryOperator, operand_type: parser.Type, expr: @TypeOf(@as(parser.ASTNode, undefined).unary_expression)) TypeCheckError!parser.Type {
        switch (op) {
            .negate => {
                if (!self.isNumericType(operand_type)) {
                    const pos = self.getNodePosition(expr.operand.*);
                    std.debug.print("Error at line {}, column {}: Negation requires numeric type, got {}\n", .{ pos.line, pos.column, operand_type });
                    self.printSourceContext(self.getNodeOffset(expr.operand.*));
                    return error.InvalidUnaryOperation;
                }
                return operand_type;
            },
        }
    }

    fn typeCheckFunctionCall(self: *TypeChecker, call: @TypeOf(@as(parser.ASTNode, undefined).call_expression)) TypeCheckError!parser.Type {
        const callee_name = switch (call.callee.*) {
            .identifier => |ident| ident.name,
            .namespace_access => |ns| {
                // Handle namespace functions like io.file()
                if (std.mem.eql(u8, ns.namespace, "io") and std.mem.eql(u8, ns.member, "file")) {
                    // io.file(filename) returns a file descriptor (i32)
                    if (call.arguments.len != 1) {
                        const pos = self.getNodePosition(call.callee.*);
                        std.debug.print("Error at line {}, column {}: io.file() expects exactly 1 argument\n", .{ pos.line, pos.column });
                        self.printSourceContext(ns.offset);
                        return error.InvalidFunctionCall;
                    }
                    
                    // Type check the filename argument (should be string)
                    _ = try self.typeCheckExpression(&call.arguments[0]);
                    if (call.arguments[0] != .string_literal) {
                        const pos = self.getNodePosition(call.arguments[0]);
                        std.debug.print("Error at line {}, column {}: io.file() expects a string literal argument\n", .{ pos.line, pos.column });
                        self.printSourceContext(self.getNodeOffset(call.arguments[0]));
                        return error.InvalidFunctionCall;
                    }
                    
                    // Return i32 (file descriptor)
                    return parser.Type.i32;
                }
                
                const pos = self.getNodePosition(call.callee.*);
                std.debug.print("Error at line {}, column {}: Unknown namespace function\n", .{ pos.line, pos.column });
                self.printSourceContext(ns.offset);
                return error.InvalidFunctionCall;
            },
            else => {
                const pos = self.getNodePosition(call.callee.*);
                std.debug.print("Error at line {}, column {}: Cannot call non-function\n", .{ pos.line, pos.column });
                self.printSourceContext(self.getNodeOffset(call.callee.*));
                return error.InvalidFunctionCall;
            },
        };

        if (self.functions.getPtr(callee_name)) |func_info| {
            // Mark function as used
            func_info.used = true;
            // Check argument count
            if (call.arguments.len != func_info.params.len) {
                const pos = self.getNodePosition(call.callee.*);
                std.debug.print("Error at line {}, column {}: Function '{s}' expects {} arguments, got {}\n", .{ pos.line, pos.column, callee_name, func_info.params.len, call.arguments.len });
                self.printSourceContext(self.getNodeOffset(call.callee.*));
                return error.InvalidFunctionCall;
            }

            // Check argument types
            for (call.arguments, 0..) |*arg, i| {
                const arg_type = try self.typeCheckExpression(arg);
                const param_type = func_info.params[i].type;

                if (!self.typesCompatible(param_type, arg_type)) {
                    const pos = self.getNodePosition(arg.*);
                    std.debug.print("Error at line {}, column {}: Argument {} of function '{s}' expects type {}, got {}\n", .{ pos.line, pos.column, i + 1, callee_name, param_type, arg_type });
                    self.printSourceContext(self.getNodeOffset(arg.*));
                    return error.InvalidFunctionCall;
                }
            }
            return func_info.return_type;
        } else {
            const pos = self.getNodePosition(call.callee.*);
            std.debug.print("Error at line {}, column {}: Undefined function '{s}'\n", .{ pos.line, pos.column, callee_name });
            self.printSourceContext(self.getNodeOffset(call.callee.*));
            return error.UndefinedFunction;
        }
    }

    fn typesCompatible(_: *TypeChecker, expected: parser.Type, actual: parser.Type) bool {
        switch (expected) {
            .tensor => |expected_| {
                switch (actual) {
                    .tensor => |actual_| {
                        if (std.mem.eql(u32, expected_.shape, actual_.shape) and std.meta.eql(expected_.element_type.*, actual_.element_type.*)) {
                            return true;
                        }
                    },
                    else => {},
                }
            },
            else => {},
        }
        return std.meta.eql(expected, actual);
    }

    fn isNumericType(_: *TypeChecker, t: parser.Type) bool {
        return switch (t) {
            .u8, .u16, .u32, .u64, .i8, .i16, .i32, .i64, .f32, .f64 => true,
            .void => false,
            .tensor => |tensor_type| {
                // A tensor is numeric if its element type is numeric
                return switch (tensor_type.element_type.*) {
                    .u8, .u16, .u32, .u64, .i8, .i16, .i32, .i64, .f32, .f64 => true,
                    .void => false,
                    .tensor => false, // Nested tensors are not considered numeric for operations
                };
            },
        };
    }

    fn getNodePosition(self: *TypeChecker, node: parser.ASTNode) struct { line: usize, column: usize } {
        const offset: usize = switch (node) {
            .program => |n| n.offset,
            .function_declaration => |n| n.offset,
            .parameter => |n| n.offset,
            .variable_declaration => |n| n.offset,
            .return_statement => |n| n.offset,
            .expression_statement => |n| n.offset,
            .number_literal => |n| n.offset,
            .identifier => |n| n.offset,
            .binary_expression => |n| n.offset,
            .unary_expression => |n| n.offset,
            .call_expression => |n| n.offset,
            .tensor_literal => |n| n.offset,
            .implicit_tensor_index => |n| n.offset,
            .tensor_slice => |n| n.offset,
            .parallel_assignment => |n| n.offset,
            .reduce_expression => |n| n.offset,
            .string_literal => |n| n.offset,
            .namespace_access => |n| n.offset,
            .write_expression => |n| n.offset,
            .read_expression => |n| n.offset,
        };
        const pos = lexer.Lexer.offsetToLineColumn(self.source, offset);
        return .{ .line = @as(usize, pos.line), .column = @as(usize, pos.column) };
    }

    fn typeCheckTensorLiteral(self: *TypeChecker, tensor_lit: @TypeOf(@as(parser.ASTNode, undefined).tensor_literal)) TypeCheckError!parser.Type {
        // Check that the value type matches the tensor element type
        const value_type = try self.typeCheckExpression(tensor_lit.value);

        if (self.verbose) {
            std.debug.print("DEBUG: typeCheckTensorLiteral - shape={any}, element_type={}, value_type={}\n", .{ tensor_lit.shape, tensor_lit.element_type, value_type });
        }
        if (!self.typesCompatible(tensor_lit.element_type, value_type)) {
            const pos = self.getNodePosition(tensor_lit.value.*);
            std.debug.print("Error at line {}, column {}: Tensor literal value type {} does not match element type {}\n", .{ pos.line, pos.column, value_type, tensor_lit.element_type });
            self.printSourceContext(self.getNodeOffset(tensor_lit.value.*));
            return error.TypeMismatch;
        }

        // Allocate a new copy of the element type to avoid pointer sharing issues
        const element_type_ptr = try self.allocator.create(parser.Type);
        element_type_ptr.* = tensor_lit.element_type;

        // Return the tensor type
        const result = parser.Type{
            .tensor = try parser.Type.TensorType.init(self.allocator, tensor_lit.shape, element_type_ptr, false),
        };

        // Track the allocated type for cleanup
        try self.allocated_types.append(result);

        if (self.verbose) {
            std.debug.print("DEBUG: typeCheckTensorLiteral - created tensor type: {}\n", .{result});
        }

        return result;
    }

    fn typeCheckImplicitTensorIndex(self: *TypeChecker, tensor_index: @TypeOf(@as(parser.ASTNode, undefined).implicit_tensor_index)) TypeCheckError!parser.Type {
        // Check that base is a tensor
        const tensor_type = try self.typeCheckExpression(tensor_index.tensor);
        if (tensor_type != .tensor) {
            const pos = self.getNodePosition(tensor_index.tensor.*);
            std.debug.print("Error at line {}, column {}: Cannot index non-tensor type {}\n", .{ pos.line, pos.column, tensor_type });
            self.printSourceContext(self.getNodeOffset(tensor_index.tensor.*));
            return error.TargetMustBeTensor;
        }

        // Check for scope conflicts
        for (tensor_index.implicit_indices) |index_name| {
            if (self.verbose) {
                std.debug.print("DEBUG: Checking implicit index '{s}' for conflicts\n", .{index_name});
            }
            if (self.variables.get(index_name) != null) {
                const pos = self.getNodePosition(tensor_index.tensor.*);
                std.debug.print("Error at line {}, column {}: Implicit index '{s}' conflicts with variable in scope\n", .{ pos.line, pos.column, index_name });
                self.printSourceContext(self.getNodeOffset(tensor_index.tensor.*));
                return error.ImplicitIndexConflictsWithVariable;
            }
        }

        // Check that the number of indices matches the tensor rank
        const expected_indices = tensor_type.tensor.rank();
        const provided_indices = tensor_index.implicit_indices.len;
        
        if (provided_indices != expected_indices) {
            const pos = self.getNodePosition(tensor_index.tensor.*);
            std.debug.print("Error at line {}, column {}: Index count {} does not match tensor rank {}\n", .{ pos.line, pos.column, provided_indices, expected_indices });
            self.printSourceContext(self.getNodeOffset(tensor_index.tensor.*));
            return error.IndexCountMismatch;
        }

        return tensor_type.tensor.element_type.*;
    }

    fn typeCheckTensorSlice(self: *TypeChecker, tensor_slice: @TypeOf(@as(parser.ASTNode, undefined).tensor_slice)) TypeCheckError!parser.Type {
        // Check that base is a tensor
        const tensor_type = try self.typeCheckExpression(tensor_slice.tensor);
        if (tensor_type != .tensor) {
            const pos = self.getNodePosition(tensor_slice.tensor.*);
            std.debug.print("Error at line {}, column {}: Cannot slice non-tensor type {}\n", .{ pos.line, pos.column, tensor_type });
            self.printSourceContext(self.getNodeOffset(tensor_slice.tensor.*));
            return error.TargetMustBeTensor;
        }

        // Check that index count matches tensor rank
        if (tensor_slice.indices.len != tensor_type.tensor.rank()) {
            const pos = self.getNodePosition(tensor_slice.tensor.*);
            std.debug.print("Error at line {}, column {}: Index count {} does not match tensor rank {}\n", .{ pos.line, pos.column, tensor_slice.indices.len, tensor_type.tensor.rank() });
            self.printSourceContext(self.getNodeOffset(tensor_slice.tensor.*));
            return error.IndexCountMismatch;
        }

        // Validate each index expression
        for (tensor_slice.indices, 0..) |*index_ptr, dim| {
            // Type check the index expression to ensure it is numeric
            const idx_type = try self.typeCheckExpression(index_ptr);

            if (!self.isNumericType(idx_type)) {
                const pos = self.getNodePosition(index_ptr.*);
                std.debug.print("Error at line {}, column {}: Tensor slice indices must be numeric, got {}\n", .{ pos.line, pos.column, idx_type });
                self.printSourceContext(self.getNodeOffset(index_ptr.*));
                return error.InvalidIndexType;
            }

            // Perform static bounds check when the index is a compile-time numeric literal
            if (index_ptr.* == .number_literal) {
                const idx_val_str = index_ptr.*.number_literal.value;

                // Strip any type suffix such as 'u32', 'i64', etc.
                var end: usize = 0;
                while (end < idx_val_str.len and idx_val_str[end] >= '0' and idx_val_str[end] <= '9') : (end += 1) {}
                const digits = idx_val_str[0..end];

                const idx_val = std.fmt.parseInt(u64, digits, 10) catch 0;
                const dim_size = tensor_type.tensor.shape[dim];

                if (idx_val >= dim_size) {
                    const pos = self.getNodePosition(index_ptr.*);
                    std.debug.print("Error at line {}, column {}: Index {} out of bounds for dimension size {}\n", .{ pos.line, pos.column, idx_val, dim_size });
                    self.printSourceContext(self.getNodeOffset(index_ptr.*));
                    return error.IndexOutOfBounds;
                }
            }
        }

        // Calculate reduced tensor type
        const reduced_type = tensor_type.tensor.reduceRank(@intCast(tensor_slice.indices.len));

        if (reduced_type) |reduced| {
            // Return reduced tensor type
            return parser.Type{
                .tensor = reduced,
            };
        } else {
            // All dimensions indexed, return element type
            return tensor_type.tensor.element_type.*;
        }
    }

    fn typeCheckReduceExpression(self: *TypeChecker, reduce: @TypeOf(@as(parser.ASTNode, undefined).reduce_expression)) TypeCheckError!parser.Type {
        // Check for valid reduction operator
        switch (reduce.operator) {
            .add, .multiply => {}, // Valid
            .subtract, .divide => {
                const pos = self.getPositionFromOffset(reduce.offset);
                std.debug.print("Error at line {}, column {}: Invalid reduction operator '{}' - only '+' and '*' are supported\n", .{ pos.line, pos.column, reduce.operator });
                self.printSourceContext(reduce.offset);
                return error.InvalidBinaryOperation;
            },
        }
        
        // Ensure the expression is an implicit tensor index
        if (reduce.tensor_expr.* != .implicit_tensor_index) {
            const pos = self.getNodePosition(reduce.tensor_expr.*);
            std.debug.print("Error at line {}, column {}: first argument of reduce must be an implicit tensor expression\n", .{ pos.line, pos.column });
            self.printSourceContext(self.getNodeOffset(reduce.tensor_expr.*));
            return error.TypeMismatch;
        }
        
        // Type check the entire implicit tensor expression - this will check for variable conflicts
        _ = try self.typeCheckExpression(reduce.tensor_expr);
        
        // Get the implicit index info
        const implicit_index = reduce.tensor_expr.*.implicit_tensor_index;
        
        // Check for duplicate indices in the implicit tensor expression
        const indices = implicit_index.implicit_indices;
        for (indices, 0..) |idx1, i| {
            for (indices[i + 1..]) |idx2| {
                if (std.mem.eql(u8, idx1, idx2)) {
                    const pos = self.getPositionFromOffset(reduce.offset);
                    std.debug.print("Error at line {}, column {}: Duplicate implicit index '{s}' in reduce expression\n", .{ pos.line, pos.column, idx1 });
                    self.printSourceContext(reduce.offset);
                    return error.InvalidExpression;
                }
            }
        }
        
        // Get the base tensor type
        const base_tensor_type = try self.typeCheckExpression(implicit_index.tensor);
        
        // Ensure the base is a tensor
        if (base_tensor_type != .tensor) {
            const pos = self.getNodePosition(implicit_index.tensor.*);
            std.debug.print("Error at line {}, column {}: reduce expects a tensor expression, got {}\n", .{ pos.line, pos.column, base_tensor_type });
            self.printSourceContext(self.getNodeOffset(implicit_index.tensor.*));
            return error.TypeMismatch;
        }
        
        const tensor_type = base_tensor_type.tensor;
        
        // For now, we support reducing tensors of rank >= 1
        if (tensor_type.rank() < 1) {
            const pos = self.getNodePosition(reduce.tensor_expr.*);
            std.debug.print("Error at line {}, column {}: cannot reduce a scalar\n", .{ pos.line, pos.column });
            self.printSourceContext(self.getNodeOffset(reduce.tensor_expr.*));
            return error.TypeMismatch;
        }
        
        // The result type depends on the tensor expression
        // If the expression has implicit indices, we need to determine which dimensions are reduced
        // For now, return the element type for simplicity
        // TODO: Properly handle rank reduction based on implicit indices
        
        // Check if the operator is valid for the element type
        if (!self.isNumericType(tensor_type.element_type.*)) {
            const pos = self.getNodePosition(reduce.tensor_expr.*);
            std.debug.print("Error at line {}, column {}: reduce operator {} requires numeric element type, got {}\n", .{ pos.line, pos.column, reduce.operator, tensor_type.element_type.* });
            self.printSourceContext(self.getNodeOffset(reduce.tensor_expr.*));
            return error.InvalidBinaryOperation;
        }
        
        // Determine which dimensions are being reduced
        // The dimensions that appear in the reduce expression but not in the assignment target
        // are the ones being reduced
        
        // For now, if we have a[i,j] on the right in reduce, we need to know what's on the left
        // This is tricky without more context. For simplicity, let's return the element type
        // if all indices are present (full reduction)
        
        if (implicit_index.implicit_indices.len == tensor_type.rank()) {
            // All dimensions are indexed, so we're reducing to a scalar
            return tensor_type.element_type.*;
        }
        
        // Otherwise, we'd need to know which indices are free vs bound
        // This requires more sophisticated analysis
        return tensor_type.element_type.*;
    }

    fn typeCheckParallelAssignment(self: *TypeChecker, parallel_assign: @TypeOf(@as(parser.ASTNode, undefined).parallel_assignment)) TypeCheckError!void {
        // Check that target is an implicit tensor index
        if (parallel_assign.target.* != .implicit_tensor_index) {
            const pos = self.getNodePosition(parallel_assign.target.*);
            std.debug.print("Error at line {}, column {}: Parallel assignment target must be an implicit tensor index\n", .{ pos.line, pos.column });
            self.printSourceContext(self.getNodeOffset(parallel_assign.target.*));
            return error.TypeMismatch;
        }
        
        const target_indices = parallel_assign.target.*.implicit_tensor_index.implicit_indices;
        const target_type = try self.typeCheckExpression(parallel_assign.target);

        // Special handling for reduce expressions
        if (parallel_assign.value.* == .reduce_expression) {
            const reduce_expr = &parallel_assign.value.*.reduce_expression;
            
            // Analyze the reduction and store with the value ASTNode pointer
            try self.analyzeReduction(target_indices, reduce_expr, parallel_assign.value);
        }
        
        // Check that value is compatible with tensor element type
        const value_type = try self.typeCheckExpression(parallel_assign.value);

        if (!self.typesCompatible(target_type, value_type)) {
            const pos = self.getNodePosition(parallel_assign.value.*);
            std.debug.print("Error at line {}, column {}: Cannot assign value of type {} to tensor element of type {}\n", .{ pos.line, pos.column, value_type, target_type });
            self.printSourceContext(self.getNodeOffset(parallel_assign.value.*));
            return error.TypeMismatch;
        }

        // Parallel assignment is valid
    }

    fn analyzeReduction(self: *TypeChecker, target_indices: [][]const u8, reduce_expr: *@TypeOf(@as(parser.ASTNode, undefined).reduce_expression), reduce_node: *parser.ASTNode) TypeCheckError!void {
        // The reduce expression must contain an implicit tensor index
        if (reduce_expr.tensor_expr.* != .implicit_tensor_index) {
            return; // This error is handled in typeCheckReduceExpression
        }
        
        const reduce_indices = reduce_expr.tensor_expr.*.implicit_tensor_index.implicit_indices;
        
        // Check for valid reduction operator
        switch (reduce_expr.operator) {
            .add, .multiply => {}, // Valid
            .subtract, .divide => {
                const pos = self.getPositionFromOffset(reduce_node.*.reduce_expression.offset);
                std.debug.print("Error at line {}, column {}: Invalid reduction operator '{}' - only '+' and '*' are supported\n", .{ pos.line, pos.column, reduce_expr.operator });
                self.printSourceContext(reduce_node.*.reduce_expression.offset);
                return error.InvalidBinaryOperation;
            },
        }
        
        // Check for duplicate indices in reduce expression
        for (reduce_indices, 0..) |idx1, i| {
            for (reduce_indices[i + 1..]) |idx2| {
                if (std.mem.eql(u8, idx1, idx2)) {
                    const pos = self.getPositionFromOffset(reduce_node.*.reduce_expression.offset);
                    std.debug.print("Error at line {}, column {}: Duplicate implicit index '{s}' in reduce expression\n", .{ pos.line, pos.column, idx1 });
                    self.printSourceContext(reduce_node.*.reduce_expression.offset);
                    return error.InvalidExpression;
                }
            }
        }
        
        // Determine which indices are free (appear in target) vs bound (only in reduce)
        var free_indices = std.ArrayList([]const u8).init(self.allocator);
        var bound_indices = std.ArrayList([]const u8).init(self.allocator);
        defer free_indices.deinit();
        defer bound_indices.deinit();
        
        // Check each reduce index to see if it appears in the target
        for (reduce_indices) |reduce_idx| {
            var found_in_target = false;
            for (target_indices) |target_idx| {
                if (std.mem.eql(u8, reduce_idx, target_idx)) {
                    found_in_target = true;
                    break;
                }
            }
            
            if (found_in_target) {
                try free_indices.append(reduce_idx);
            } else {
                try bound_indices.append(reduce_idx);
            }
        }
        
        // Validate that we're actually reducing dimensions (bound indices must be non-empty)
        if (bound_indices.items.len == 0) {
            const pos = self.getPositionFromOffset(reduce_node.*.reduce_expression.offset);
            std.debug.print("Error at line {}, column {}: Cannot reduce to higher or equal rank - reduce expression must have more indices than target\n", .{ pos.line, pos.column });
            self.printSourceContext(reduce_node.*.reduce_expression.offset);
            return error.InvalidExpression;
        }
        
        // Validate that free indices match target indices
        if (free_indices.items.len != target_indices.len) {
            const pos = self.getPositionFromOffset(reduce_node.*.reduce_expression.offset);
            std.debug.print("Error at line {}, column {}: Free indices must match between target and reduce expression\n", .{ pos.line, pos.column });
            self.printSourceContext(reduce_node.*.reduce_expression.offset);
            return error.InvalidExpression;
        }
        
        // Check that all free indices appear in target
        for (free_indices.items) |free_idx| {
            var found = false;
            for (target_indices) |target_idx| {
                if (std.mem.eql(u8, free_idx, target_idx)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                const pos = self.getPositionFromOffset(reduce_node.*.reduce_expression.offset);
                std.debug.print("Error at line {}, column {}: Free index '{s}' in reduce expression does not match any target index\n", .{ pos.line, pos.column, free_idx });
                self.printSourceContext(reduce_node.*.reduce_expression.offset);
                return error.InvalidExpression;
            }
        }
        
        // Store the reduction info
        const info = ReductionInfo{
            .free_indices = try self.allocator.dupe([]const u8, free_indices.items),
            .bound_indices = try self.allocator.dupe([]const u8, bound_indices.items),
            .operator = reduce_expr.operator,
        };
        
        try self.reduction_info.put(reduce_node, info);
        
        if (self.verbose) {
            std.debug.print("DEBUG: Storing reduction info at node ptr: {*}\n", .{reduce_node});
            std.debug.print("DEBUG: Reduction analysis - free indices: ", .{});
            for (free_indices.items) |idx| {
                std.debug.print("{s} ", .{idx});
            }
            std.debug.print(", bound indices: ", .{});
            for (bound_indices.items) |idx| {
                std.debug.print("{s} ", .{idx});
            }
            std.debug.print("\n", .{});
        }
    }

    fn getNodeOffset(_: *TypeChecker, node: parser.ASTNode) usize {
        return switch (node) {
            .program => |n| n.offset,
            .function_declaration => |n| n.offset,
            .parameter => |n| n.offset,
            .variable_declaration => |n| n.offset,
            .return_statement => |n| n.offset,
            .expression_statement => |n| n.offset,
            .number_literal => |n| n.offset,
            .identifier => |n| n.offset,
            .binary_expression => |n| n.offset,
            .unary_expression => |n| n.offset,
            .call_expression => |n| n.offset,
            .tensor_literal => |n| n.offset,
            .implicit_tensor_index => |n| n.offset,
            .tensor_slice => |n| n.offset,
            .parallel_assignment => |n| n.offset,
            .reduce_expression => |n| n.offset,
            .string_literal => |n| n.offset,
            .namespace_access => |n| n.offset,
            .write_expression => |n| n.offset,
            .read_expression => |n| n.offset,
        };
    }

    fn getSourceLine(self: *TypeChecker, offset: usize) []const u8 {
        var line_start: usize = offset;
        while (line_start > 0 and self.source[line_start - 1] != '\n') : (line_start -= 1) {}
        var line_end: usize = offset;
        while (line_end < self.source.len and self.source[line_end] != '\n') : (line_end += 1) {}
        return self.source[line_start..line_end];
    }

    fn printSourceContext(self: *TypeChecker, offset: usize) void {
        const source_line = self.getSourceLine(offset);
        std.debug.print("  {s}\n", .{source_line});
        // caret line
        var caret_line = std.ArrayList(u8).init(self.allocator);
        defer caret_line.deinit();
        var line_start: usize = offset;
        while (line_start > 0 and self.source[line_start - 1] != '\n') : (line_start -= 1) {}
        const column_in_line = offset - line_start;
        for (0..column_in_line) |_| {
            caret_line.append(' ') catch {};
        }
        caret_line.append('^') catch {};
        std.debug.print("  {s}\n", .{caret_line.items});
    }

    fn getPositionFromOffset(self: *TypeChecker, offset: usize) struct { line: usize, column: usize } {
        const pos = lexer.Lexer.offsetToLineColumn(self.source, offset);
        return .{ .line = @as(usize, pos.line), .column = @as(usize, pos.column) };
    }
    
    fn printType(_: *TypeChecker, t: parser.Type) void {
        switch (t) {
            .u8 => std.debug.print("u8", .{}),
            .u16 => std.debug.print("u16", .{}),
            .u32 => std.debug.print("u32", .{}),
            .u64 => std.debug.print("u64", .{}),
            .i8 => std.debug.print("i8", .{}),
            .i16 => std.debug.print("i16", .{}),
            .i32 => std.debug.print("i32", .{}),
            .i64 => std.debug.print("i64", .{}),
            .f32 => std.debug.print("f32", .{}),
            .f64 => std.debug.print("f64", .{}),
            .void => std.debug.print("void", .{}),
            .tensor => |tensor_type| {
                std.debug.print("[", .{});
                for (tensor_type.shape, 0..) |dim, i| {
                    if (i > 0) std.debug.print(", ", .{});
                    std.debug.print("{}", .{dim});
                }
                std.debug.print("]", .{});
                switch (tensor_type.element_type.*) {
                    .u8 => std.debug.print("u8", .{}),
                    .u16 => std.debug.print("u16", .{}),
                    .u32 => std.debug.print("u32", .{}),
                    .u64 => std.debug.print("u64", .{}),
                    .i8 => std.debug.print("i8", .{}),
                    .i16 => std.debug.print("i16", .{}),
                    .i32 => std.debug.print("i32", .{}),
                    .i64 => std.debug.print("i64", .{}),
                    .f32 => std.debug.print("f32", .{}),
                    .f64 => std.debug.print("f64", .{}),
                    .void => std.debug.print("void", .{}),
                    .tensor => std.debug.print("tensor", .{}),
                }
            },
        }
    }
};
