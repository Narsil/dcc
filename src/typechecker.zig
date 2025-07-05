const std = @import("std");
const parser = @import("parser.zig");
const lexer = @import("lexer.zig");

pub const TypeCheckError = error{
    TypeMismatch,
    UndefinedVariable,
    UndefinedFunction,
    InvalidBinaryOperation,
    InvalidUnaryOperation,
    InvalidFunctionCall,
    InvalidReturnType,
    InvalidVariableType,
    TargetMustBeTensor,
    ImplicitIndexConflictsWithVariable,
    IndexCountMismatch,
    InvalidIndexType,
    InvalidExpression,
    OutOfMemory,
    IndexOutOfBounds,
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
        const declared_type = var_decl.type;
        const value_type = try self.typeCheckExpression(var_decl.value);
        std.debug.print("DEBUG: typeCheckNode - variable_declaration: declared_type={}, value_type={} (tag: {s})\n", .{ declared_type, value_type, @tagName(value_type) });
        if (declared_type == .tensor and value_type == .tensor) {
            std.debug.print("DEBUG: declared_type.element_type ptr: {any}, value: {}\n", .{ declared_type.tensor.element_type, declared_type.tensor.element_type.* });
            std.debug.print("DEBUG: value_type.element_type ptr: {any}, value: {}\n", .{ value_type.tensor.element_type, value_type.tensor.element_type.* });
        }
        std.debug.print("Somethingsomething\n", .{});
        if (!self.typesCompatible(declared_type, value_type)) {
            const pos = self.getNodePosition(var_decl.value.*);
            std.debug.print("Error at line {}, column {}: Cannot assign value of type {} to variable of type {}\n", .{ pos.line, pos.column, value_type, declared_type });
            self.printSourceContext(self.getNodeOffset(var_decl.value.*));
            return error.TypeMismatch;
        }

        // Add variable to scope
        const name = try self.allocator.dupe(u8, var_decl.name);
        try self.variables.put(name, var_decl.type);
    }

    fn typeCheckReturnStatement(self: *TypeChecker, ret: @TypeOf(@as(parser.ASTNode, undefined).return_statement)) TypeCheckError!void {
        const expected_type = self.current_function_return_type orelse {
            const pos = if (ret.value) |val| self.getNodePosition(val.*) else self.getNodePosition(parser.ASTNode{ .return_statement = ret });
            std.debug.print("Error at line {}, column {}: Return statement outside of function\n", .{ pos.line, pos.column });
            const offset = if (ret.value) |val| self.getNodeOffset(val.*) else ret.offset;
            self.printSourceContext(offset);
            return error.InvalidReturnType;
        };

        std.debug.print("DEBUG: Return statement - expected type: {}\n", .{expected_type});

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
            std.debug.print("DEBUG: Return statement - actual type: {}\n", .{actual_type});

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
            .identifier => |ident| self.variables.get(ident.name) orelse return error.UndefinedVariable,
            .binary_expression => |bin_expr| try self.typeCheckBinaryOperation(bin_expr.operator, try self.typeCheckExpression(bin_expr.left), try self.typeCheckExpression(bin_expr.right), bin_expr),
            .unary_expression => |unary_expr| try self.typeCheckUnaryOperation(unary_expr.operator, try self.typeCheckExpression(unary_expr.operand), unary_expr),
            .call_expression => |*call| {
                const return_type = try self.typeCheckFunctionCall(call.*);
                call.*.return_type = return_type;
                return return_type;
            },
            else => return error.InvalidExpression,
        };
        std.debug.print("DEBUG: typeCheckExpression - node tag: {any}, result_type: {}\n", .{ @tagName(node.*), result_type });
        return result_type;
    }

    fn typeCheckBinaryOperation(self: *TypeChecker, op: parser.BinaryOperator, left_type: parser.Type, right_type: parser.Type, expr: @TypeOf(@as(parser.ASTNode, undefined).binary_expression)) TypeCheckError!parser.Type {
        // Check if operands are compatible for the operation
        if (!self.typesCompatible(left_type, right_type)) {
            const pos = self.getNodePosition(expr.left.*);
            std.debug.print("Error at line {}, column {}: Cannot perform {} operation on types {} and {}\n", .{ pos.line, pos.column, op, left_type, right_type });
            self.printSourceContext(self.getNodeOffset(expr.left.*));
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
            else => {
                const pos = self.getNodePosition(call.callee.*);
                std.debug.print("Error at line {}, column {}: Cannot call non-function\n", .{ pos.line, pos.column });
                self.printSourceContext(self.getNodeOffset(call.callee.*));
                return error.InvalidFunctionCall;
            },
        };

        if (self.functions.get(callee_name)) |func_info| {
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
        };
        const pos = lexer.Lexer.offsetToLineColumn(self.source, offset);
        return .{ .line = @as(usize, pos.line), .column = @as(usize, pos.column) };
    }

    fn typeCheckTensorLiteral(self: *TypeChecker, tensor_lit: @TypeOf(@as(parser.ASTNode, undefined).tensor_literal)) TypeCheckError!parser.Type {
        // Check that the value type matches the tensor element type
        const value_type = try self.typeCheckExpression(tensor_lit.value);

        std.debug.print("DEBUG: typeCheckTensorLiteral - shape={any}, element_type={}, value_type={}\n", .{ tensor_lit.shape, tensor_lit.element_type, value_type });
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

        std.debug.print("DEBUG: typeCheckTensorLiteral - created tensor type: {}\n", .{result});

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
        if (self.variables.contains(tensor_index.implicit_index)) {
            const pos = self.getNodePosition(tensor_index.tensor.*);
            std.debug.print("Error at line {}, column {}: Implicit index '{s}' conflicts with variable in scope\n", .{ pos.line, pos.column, tensor_index.implicit_index });
            self.printSourceContext(self.getNodeOffset(tensor_index.tensor.*));
            return error.ImplicitIndexConflictsWithVariable;
        }

        // For now, assume 1D tensors and return element type
        // TODO: Handle multi-dimensional tensors properly
        if (tensor_type.tensor.rank() != 1) {
            const pos = self.getNodePosition(tensor_index.tensor.*);
            std.debug.print("Error at line {}, column {}: Multi-dimensional tensor indexing not yet supported\n", .{ pos.line, pos.column });
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

    fn typeCheckParallelAssignment(self: *TypeChecker, parallel_assign: @TypeOf(@as(parser.ASTNode, undefined).parallel_assignment)) TypeCheckError!void {
        // Check that target is an implicit tensor index
        const target_type = try self.typeCheckExpression(parallel_assign.target);

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
};
