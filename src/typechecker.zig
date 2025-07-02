const std = @import("std");
const parser = @import("parser.zig");

pub const TypeCheckError = error{
    TypeMismatch,
    UndefinedVariable,
    UndefinedFunction,
    InvalidBinaryOperation,
    InvalidUnaryOperation,
    InvalidFunctionCall,
    InvalidReturnType,
    InvalidVariableType,
    OutOfMemory,
} || std.mem.Allocator.Error;

const FunctionInfo = struct {
    params: []parser.Parameter,
    return_type: parser.Type,
};

pub const TypeChecker = struct {
    allocator: std.mem.Allocator,
    source: []const u8,
    variables: std.HashMap([]const u8, parser.Type, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    functions: std.HashMap([]const u8, FunctionInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    current_function_return_type: ?parser.Type,

    pub fn init(allocator: std.mem.Allocator, source: []const u8) TypeChecker {
        return TypeChecker{
            .allocator = allocator,
            .source = source,
            .variables = std.HashMap([]const u8, parser.Type, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .functions = std.HashMap([]const u8, FunctionInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .current_function_return_type = null,
        };
    }

    pub fn deinit(self: *TypeChecker) void {
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
    }

    pub fn typeCheck(self: *TypeChecker, ast: parser.ASTNode) TypeCheckError!void {
        std.debug.print("DEBUG: Entering typeCheck\n", .{});
        switch (ast) {
            .program => |prog| {
                // First pass: collect function signatures
                for (prog.statements) |stmt| {
                    if (stmt == .function_declaration) {
                        std.debug.print("DEBUG: Collecting signature for function '{s}'\n", .{stmt.function_declaration.name});
                        try self.collectFunctionSignature(stmt.function_declaration);
                    }
                }

                // Second pass: type check all statements
                for (prog.statements) |stmt| {
                    std.debug.print("DEBUG: Type checking statement of type {}\n", .{stmt});
                    try self.typeCheckStatement(stmt);
                }
            },
            else => return error.TypeMismatch,
        }
    }

    fn collectFunctionSignature(self: *TypeChecker, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) TypeCheckError!void {
        const name = try self.allocator.dupe(u8, func.name);
        try self.functions.put(name, .{
            .params = func.parameters,
            .return_type = func.return_type,
        });
        std.debug.print("DEBUG: Collected function signature '{s}' -> return type {}\n", .{ func.name, func.return_type });
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
                _ = try self.typeCheckExpression(expr_stmt.expression.*);
            },
            else => return error.TypeMismatch,
        }
    }

    fn typeCheckFunction(self: *TypeChecker, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) TypeCheckError!void {
        // Create new scope for function parameters and body
        const old_variables = self.variables;
        const old_return_type = self.current_function_return_type;
        self.variables = std.HashMap([]const u8, parser.Type, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(self.allocator);
        self.current_function_return_type = func.return_type;

        std.debug.print("DEBUG: Entering function '{s}' with return type {}\n", .{ func.name, func.return_type });

        // Add parameters to scope
        for (func.parameters) |param| {
            const name = try self.allocator.dupe(u8, param.name);
            try self.variables.put(name, param.type);
        }

        // Type check function body, skipping nested function declarations
        for (func.body) |stmt| {
            if (stmt == .function_declaration) continue;
            try self.typeCheckStatement(stmt);
        }

        // Restore old scope
        var var_iter = self.variables.iterator();
        while (var_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.variables.deinit();
        self.variables = old_variables;
        self.current_function_return_type = old_return_type;

        std.debug.print("DEBUG: Exiting function '{s}'\n", .{func.name});
    }

    fn typeCheckVariableDeclaration(self: *TypeChecker, var_decl: @TypeOf(@as(parser.ASTNode, undefined).variable_declaration)) TypeCheckError!void {
        const value_type = try self.typeCheckExpression(var_decl.value.*);

        if (!self.typesCompatible(var_decl.type, value_type)) {
            const pos = self.getNodePosition(var_decl.value.*);
            std.debug.print("Error at line {}, column {}: Cannot assign value of type {} to variable of type {}\n", .{ pos.line, pos.column, value_type, var_decl.type });
            return error.TypeMismatch;
        }

        // Add variable to scope
        const name = try self.allocator.dupe(u8, var_decl.name);
        try self.variables.put(name, var_decl.type);
    }

    fn typeCheckReturnStatement(self: *TypeChecker, ret: @TypeOf(@as(parser.ASTNode, undefined).return_statement)) TypeCheckError!void {
        const expected_type = self.current_function_return_type orelse {
            const pos = self.getNodePosition(ret.value.?.*);
            std.debug.print("Error at line {}, column {}: Return statement outside of function\n", .{ pos.line, pos.column });
            return error.InvalidReturnType;
        };

        std.debug.print("DEBUG: Return statement - expected type: {}\n", .{expected_type});

        if (ret.value) |value| {
            const actual_type = try self.typeCheckExpression(value.*);

            std.debug.print("DEBUG: Return statement - actual type: {}\n", .{actual_type});

            if (!self.typesCompatible(expected_type, actual_type)) {
                const pos = self.getNodePosition(value.*);
                std.debug.print("Error at line {}, column {}: Cannot return value of type {} from function returning {}\n", .{ pos.line, pos.column, actual_type, expected_type });
                return error.InvalidReturnType;
            }
        }
    }

    fn typeCheckExpression(self: *TypeChecker, expr: parser.ASTNode) TypeCheckError!parser.Type {
        switch (expr) {
            .number_literal => |num| {
                return num.type;
            },
            .identifier => |ident| {
                if (self.variables.get(ident.name)) |var_type| {
                    return var_type;
                } else {
                    const pos = self.getNodePosition(expr);
                    std.debug.print("Error at line {}, column {}: Undefined variable '{s}'\n", .{ pos.line, pos.column, ident.name });
                    return error.UndefinedVariable;
                }
            },
            .binary_expression => |bin_expr| {
                const left_type = try self.typeCheckExpression(bin_expr.left.*);
                const right_type = try self.typeCheckExpression(bin_expr.right.*);

                return try self.typeCheckBinaryOperation(bin_expr.operator, left_type, right_type, bin_expr);
            },
            .unary_expression => |unary_expr| {
                const operand_type = try self.typeCheckExpression(unary_expr.operand.*);

                return try self.typeCheckUnaryOperation(unary_expr.operator, operand_type, unary_expr);
            },
            .call_expression => |call| {
                return try self.typeCheckFunctionCall(call);
            },
            else => return error.TypeMismatch,
        }
    }

    fn typeCheckBinaryOperation(self: *TypeChecker, op: parser.BinaryOperator, left_type: parser.Type, right_type: parser.Type, expr: @TypeOf(@as(parser.ASTNode, undefined).binary_expression)) TypeCheckError!parser.Type {
        // Check if operands are compatible for the operation
        if (!self.typesCompatible(left_type, right_type)) {
            const pos = self.getNodePosition(expr.left.*);
            std.debug.print("Error at line {}, column {}: Cannot perform {} operation on types {} and {}\n", .{ pos.line, pos.column, op, left_type, right_type });
            return error.InvalidBinaryOperation;
        }

        // For arithmetic operations, both operands must be numeric
        switch (op) {
            .add, .subtract, .multiply, .divide => {
                if (!self.isNumericType(left_type) or !self.isNumericType(right_type)) {
                    const pos = self.getNodePosition(expr.left.*);
                    std.debug.print("Error at line {}, column {}: Arithmetic operation {} requires numeric types, got {} and {}\n", .{ pos.line, pos.column, op, left_type, right_type });
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
                    return error.InvalidUnaryOperation;
                }
                return operand_type;
            },
        }
    }

    fn typeCheckFunctionCall(self: *TypeChecker, call: @TypeOf(@as(parser.ASTNode, undefined).call_expression)) TypeCheckError!parser.Type {
        const callee_name = switch (call.callee.*) {
            .identifier => |ident| ident.name,
            else => {
                const pos = self.getNodePosition(call.callee.*);
                std.debug.print("Error at line {}, column {}: Cannot call non-function\n", .{ pos.line, pos.column });
                return error.InvalidFunctionCall;
            },
        };

        if (self.functions.get(callee_name)) |func_info| {
            // Check argument count
            if (call.arguments.len != func_info.params.len) {
                const pos = self.getNodePosition(call.callee.*);
                std.debug.print("Error at line {}, column {}: Function '{s}' expects {} arguments, got {}\n", .{ pos.line, pos.column, callee_name, func_info.params.len, call.arguments.len });
                return error.InvalidFunctionCall;
            }

            // Check argument types
            for (call.arguments, 0..) |arg, i| {
                const arg_type = try self.typeCheckExpression(arg);
                const param_type = func_info.params[i].type;

                if (!self.typesCompatible(param_type, arg_type)) {
                    const pos = self.getNodePosition(arg);
                    std.debug.print("Error at line {}, column {}: Argument {} of function '{s}' expects type {}, got {}\n", .{ pos.line, pos.column, i + 1, callee_name, param_type, arg_type });
                    return error.InvalidFunctionCall;
                }
            }

            return func_info.return_type;
        } else {
            const pos = self.getNodePosition(call.callee.*);
            std.debug.print("Error at line {}, column {}: Undefined function '{s}'\n", .{ pos.line, pos.column, callee_name });
            return error.UndefinedFunction;
        }
    }

    fn typesCompatible(_: *TypeChecker, expected: parser.Type, actual: parser.Type) bool {
        return std.meta.eql(expected, actual);
    }

    fn isNumericType(_: *TypeChecker, t: parser.Type) bool {
        return switch (t) {
            .u8, .u16, .u32, .u64, .i8, .i16, .i32, .i64, .f32, .f64 => true,
        };
    }

    fn getNodePosition(_: *TypeChecker, _: parser.ASTNode) struct { line: usize, column: usize } {
        // For now, return a default position since we don't have direct access to node positions
        // In a more complete implementation, you'd want to store position info in AST nodes
        return .{ .line = 1, .column = 1 };
    }
};

