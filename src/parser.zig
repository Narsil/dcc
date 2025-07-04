const std = @import("std");
const lexer = @import("lexer.zig");
const LLVM = @cImport({
    @cInclude("llvm-c/Core.h");
    @cInclude("llvm-c/TargetMachine.h");
    @cInclude("llvm-c/Target.h");
    @cInclude("llvm-c/Support.h");
    @cInclude("llvm-c/BitReader.h");
    @cInclude("llvm-c/Object.h");
});

pub const ParseError = error{ ParseError, OutOfMemory, InvalidCharacter, Overflow, InvalidUtf8 } || std.mem.Allocator.Error;

pub const Type = union(enum) {
    // Basic types
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,
    
    // Special types
    void,

    // Tensor types
    tensor: TensorType,

    pub const TensorType = struct {
        shape: []const u32,
        element_type: *Type,

        pub fn init(allocator: std.mem.Allocator, shape: []const u32, element_type: *const Type) !TensorType {
            std.debug.print("DEBUG: TensorType.init - element_type ptr: {any}, value: {}\n", .{ element_type, element_type.* });
            return TensorType{
                .shape = try allocator.dupe(u32, shape),
                .element_type = @constCast(element_type),
            };
        }

        pub fn rank(self: *const TensorType) usize {
            return self.shape.len;
        }

        pub fn total_elements(self: *const TensorType) u64 {
            var total: u64 = 1;
            for (self.shape) |dim| {
                total *= dim;
            }
            return total;
        }

        pub fn deinit(self: *TensorType, allocator: std.mem.Allocator) void {
            allocator.free(self.shape);
        }

        /// Calculate the reduced tensor type after applying explicit indices
        /// This reduces the rank by the number of indices provided
        pub fn reduceRank(self: *const TensorType, num_indices: u32) ?TensorType {
            if (num_indices >= self.rank()) {
                // If we index all dimensions, return the element type
                return null; // This will be handled as element access
            }

            // Create new shape with reduced dimensions
            const new_shape = self.shape[num_indices..];

            return TensorType{
                .shape = new_shape,
                .element_type = self.element_type,
            };
        }
    };

    pub fn fromString(type_str: []const u8) ?Type {
        // Handle basic types by checking each enum variant
        if (std.mem.eql(u8, type_str, "u8")) return Type.u8;
        if (std.mem.eql(u8, type_str, "u16")) return Type.u16;
        if (std.mem.eql(u8, type_str, "u32")) return Type.u32;
        if (std.mem.eql(u8, type_str, "u64")) return Type.u64;
        if (std.mem.eql(u8, type_str, "i8")) return Type.i8;
        if (std.mem.eql(u8, type_str, "i16")) return Type.i16;
        if (std.mem.eql(u8, type_str, "i32")) return Type.i32;
        if (std.mem.eql(u8, type_str, "i64")) return Type.i64;
        if (std.mem.eql(u8, type_str, "f32")) return Type.f32;
        if (std.mem.eql(u8, type_str, "f64")) return Type.f64;
        if (std.mem.eql(u8, type_str, "void")) return Type.void;
        return null;
    }

    pub fn toString(self: Type) []const u8 {
        return switch (self) {
            .u8 => "u8",
            .u16 => "u16",
            .u32 => "u32",
            .u64 => "u64",
            .i8 => "i8",
            .i16 => "i16",
            .i32 => "i32",
            .i64 => "i64",
            .f32 => "f32",
            .f64 => "f64",
            .void => "void",
            .tensor => |tensor_type| {
                // Format as [dim1, dim2, ...]element_type
                var result = std.ArrayList(u8).init(std.heap.page_allocator);
                defer result.deinit();

                result.append('[') catch {};
                for (tensor_type.shape, 0..) |dim, i| {
                    if (i > 0) result.append(',') catch {};
                    result.writer().print("{d}", .{dim}) catch {};
                }
                result.append(']') catch {};
                result.appendSlice(tensor_type.element_type.toString()) catch {};

                return result.toOwnedSlice();
            },
        };
    }

    /// Get the LLVM type for this type
    pub fn toLLVMType(self: Type, context: anytype) LLVM.LLVMTypeRef {
        return switch (self) {
            .u8, .i8 => LLVM.LLVMInt8TypeInContext(context),
            .u16, .i16 => LLVM.LLVMInt16TypeInContext(context),
            .u32, .i32 => LLVM.LLVMInt32TypeInContext(context),
            .u64, .i64 => LLVM.LLVMInt64TypeInContext(context),
            .f32 => LLVM.LLVMFloatTypeInContext(context),
            .f64 => LLVM.LLVMDoubleTypeInContext(context),
            .void => LLVM.LLVMVoidTypeInContext(context),
            .tensor => |tensor_type| {
                // For now, create a simple array type
                // In a full implementation, we'd want to handle multi-dimensional arrays properly
                const element_type = tensor_type.element_type.toLLVMType(context);
                return LLVM.LLVMArrayType(element_type, @intCast(tensor_type.total_elements()));
            },
        };
    }

    /// Check if this is a signed integer type
    pub fn isSignedInt(self: Type) bool {
        return switch (self) {
            .i8, .i16, .i32, .i64 => true,
            .tensor => |tensor_type| tensor_type.element_type.isSignedInt(),
            else => false,
        };
    }

    /// Check if this is an unsigned integer type
    pub fn isUnsignedInt(self: Type) bool {
        return switch (self) {
            .u8, .u16, .u32, .u64 => true,
            .tensor => |tensor_type| tensor_type.element_type.isUnsignedInt(),
            else => false,
        };
    }

    /// Check if this is a float type
    pub fn isFloat(self: Type) bool {
        return switch (self) {
            .f32, .f64 => true,
            .tensor => |tensor_type| tensor_type.element_type.isFloat(),
            else => false,
        };
    }
};

pub const BinaryOperator = enum {
    add,
    subtract,
    multiply,
    divide,
};

pub const UnaryOperator = enum {
    negate,
};

pub const ASTNode = union(enum) {
    program: struct {
        offset: usize,
        statements: []ASTNode,
    },
    function_declaration: struct {
        offset: usize,
        name: []const u8,
        parameters: []Parameter,
        return_type: Type,
        body: []ASTNode,
    },
    parameter: struct {
        offset: usize,
        name: []const u8,
        type: Type,
    },
    variable_declaration: struct {
        offset: usize,
        name: []const u8,
        type: Type,
        value: *ASTNode,
    },
    return_statement: struct {
        offset: usize,
        value: ?*ASTNode,
    },
    expression_statement: struct {
        offset: usize,
        expression: *ASTNode,
    },
    number_literal: struct {
        offset: usize,
        value: []const u8, // Keep as string to preserve type suffix
        type: Type,
    },
    identifier: struct {
        offset: usize,
        name: []const u8,
    },
    binary_expression: struct {
        offset: usize,
        left: *ASTNode,
        operator: BinaryOperator,
        right: *ASTNode,
    },
    unary_expression: struct {
        offset: usize,
        operator: UnaryOperator,
        operand: *ASTNode,
    },
    call_expression: struct {
        offset: usize,
        return_type: ?Type,
        callee: *ASTNode,
        arguments: []ASTNode,
    },
    tensor_literal: struct {
        offset: usize,
        shape: []const u32,
        element_type: Type,
        value: *ASTNode, // The fill value
    },
    // Implicit tensor indexing: vector[i]
    implicit_tensor_index: struct {
        offset: usize,
        tensor: *ASTNode, // The tensor being indexed
        implicit_index: []const u8, // The implicit index name
    },
    // Explicit tensor indexing: vector[0] - reduces rank
    tensor_slice: struct {
        offset: usize,
        tensor: *ASTNode, // The tensor being sliced
        indices: []ASTNode, // The explicit indices (numbers)
    },
    // Parallel assignment: vector[i] = expression
    parallel_assignment: struct {
        offset: usize,
        target: *ASTNode, // The target (implicit_tensor_index)
        value: *ASTNode, // The value expression
    },
};

pub const Parameter = struct {
    name: []const u8,
    type: Type,
};

pub const Parser = struct {
    tokens: []lexer.Token,
    current: usize,
    allocator: std.mem.Allocator,
    source: []const u8,

    pub fn init(allocator: std.mem.Allocator, tokens: []lexer.Token, source: []const u8) Parser {
        return Parser{
            .tokens = tokens,
            .current = 0,
            .allocator = allocator,
            .source = source,
        };
    }

    /// Get the source line at a given offset
    fn getSourceLine(self: *Parser, offset: usize) []const u8 {
        var line_start: usize = offset;
        while (line_start > 0 and self.source[line_start - 1] != '\n') : (line_start -= 1) {}

        var line_end: usize = offset;
        while (line_end < self.source.len and self.source[line_end] != '\n') : (line_end += 1) {}

        return self.source[line_start..line_end];
    }

    /// Report error with source line context
    fn reportError(self: *Parser, offset: usize, message: []const u8) void {
        const pos = lexer.Lexer.offsetToLineColumn(self.source, offset);
        const source_line = self.getSourceLine(offset);

        std.debug.print("Error at line {}, column {}: {s}\n", .{ pos.line, pos.column, message });
        std.debug.print("  {s}\n", .{source_line});

        // Add caret pointing to the exact position
        var caret_line = std.ArrayList(u8).init(self.allocator);
        defer caret_line.deinit();

        // Calculate the column position within the line
        var line_start: usize = offset;
        while (line_start > 0 and self.source[line_start - 1] != '\n') : (line_start -= 1) {}
        const column_in_line = offset - line_start;

        // Build the caret line
        for (0..column_in_line) |_| {
            caret_line.append(' ') catch {};
        }
        caret_line.append('^') catch {};

        std.debug.print("  {s}\n", .{caret_line.items});
    }

    /// Clean up a single AST node and its children (const version)
    fn cleanupASTNode(self: *Parser, node: *const ASTNode) void {
        switch (node.*) {
            .binary_expression => |bin_expr| {
                self.cleanupASTNode(bin_expr.left);
                self.allocator.destroy(bin_expr.left);
                self.cleanupASTNode(bin_expr.right);
                self.allocator.destroy(bin_expr.right);
            },
            .unary_expression => |unary_expr| {
                self.cleanupASTNode(unary_expr.operand);
                self.allocator.destroy(unary_expr.operand);
            },
            .call_expression => |call| {
                self.cleanupASTNode(call.callee);
                self.allocator.destroy(call.callee);
                for (call.arguments) |arg| {
                    self.cleanupASTNode(&arg);
                }
                self.allocator.free(call.arguments);
            },
            .variable_declaration => |var_decl| {
                self.cleanupASTNode(var_decl.value);
                self.allocator.destroy(var_decl.value);
            },
            .return_statement => |ret| {
                if (ret.value) |val| {
                    self.cleanupASTNode(val);
                    self.allocator.destroy(val);
                }
            },
            .expression_statement => |expr_stmt| {
                self.cleanupASTNode(expr_stmt.expression);
                self.allocator.destroy(expr_stmt.expression);
            },
            .function_declaration => |func| {
                self.allocator.free(func.parameters);
                for (func.body) |stmt| {
                    self.cleanupASTNode(&stmt);
                }
                self.allocator.free(func.body);
            },
            .program => |prog| {
                for (prog.statements) |stmt| {
                    self.cleanupASTNode(&stmt);
                }
                self.allocator.free(prog.statements);
            },
            .identifier, .number_literal, .parameter, .tensor_literal, .implicit_tensor_index, .tensor_slice, .parallel_assignment => {},
        }
    }

    /// Clean up a single AST node and its children (non-const version)
    fn cleanupASTNodeMut(self: *Parser, node: *ASTNode) void {
        switch (node.*) {
            .binary_expression => |bin_expr| {
                self.cleanupASTNode(bin_expr.left);
                self.allocator.destroy(bin_expr.left);
                self.cleanupASTNode(bin_expr.right);
                self.allocator.destroy(bin_expr.right);
            },
            .unary_expression => |unary_expr| {
                self.cleanupASTNode(unary_expr.operand);
                self.allocator.destroy(unary_expr.operand);
            },
            .call_expression => |call| {
                self.cleanupASTNode(call.callee);
                self.allocator.destroy(call.callee);
                for (call.arguments) |arg| {
                    self.cleanupASTNode(&arg);
                }
                self.allocator.free(call.arguments);
            },
            .variable_declaration => |var_decl| {
                self.cleanupASTNode(var_decl.value);
                self.allocator.destroy(var_decl.value);
            },
            .return_statement => |ret| {
                if (ret.value) |val| {
                    self.cleanupASTNode(val);
                    self.allocator.destroy(val);
                }
            },
            .expression_statement => |expr_stmt| {
                self.cleanupASTNode(expr_stmt.expression);
                self.allocator.destroy(expr_stmt.expression);
            },
            .function_declaration => |func| {
                self.allocator.free(func.parameters);
                for (func.body) |stmt| {
                    self.cleanupASTNode(&stmt);
                }
                self.allocator.free(func.body);
            },
            .program => |prog| {
                for (prog.statements) |stmt| {
                    self.cleanupASTNode(&stmt);
                }
                self.allocator.free(prog.statements);
            },
            .identifier, .number_literal, .parameter, .tensor_literal, .implicit_tensor_index, .tensor_slice, .parallel_assignment => {},
        }
    }

    pub fn parse(self: *Parser) ParseError!ASTNode {
        var statements = std.ArrayList(ASTNode).init(self.allocator);
        defer statements.deinit();

        while (!self.isAtEnd()) {
            const stmt = try self.parseStatement();
            try statements.append(stmt);
        }

        return ASTNode{
            .program = .{
                .offset = if (statements.items.len > 0) blk: {
                    // Use first statement's offset as program offset
                    const first = statements.items[0];
                    const off: usize = switch (first) {
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
                    break :blk off;
                } else 0,
                .statements = try statements.toOwnedSlice(),
            },
        };
    }

    fn parseStatement(self: *Parser) ParseError!ASTNode {
        std.debug.print("DEBUG: parseStatement - checking for let, current token: {}\n", .{self.peek().type});

        if (self.match(.let)) {
            std.debug.print("DEBUG: parseStatement - match(.let) succeeded, current token: {}\n", .{self.peek().type});
            return try self.parseVariableDeclaration();
        }

        if (self.match(.fn_)) {
            return try self.parseFunctionDeclaration();
        }

        if (self.match(.return_)) {
            return try self.parseReturnStatement();
        }

        // Expression statement
        const expr = try self.parseAssignment();

        std.debug.print("DEBUG: parseStatement - current token type: {}\n", .{self.peek().type});

        if (!self.match(.semicolon)) {
            self.cleanupASTNode(&expr);
            self.reportError(self.peek().offset, "Expected ';' after expression");
            return error.ParseError;
        }

        // If the parsed expression itself is a parallel_assignment, treat it as a statement directly
        switch (expr) {
            .parallel_assignment => {
                return expr; // already a full statement
            },
            else => {
                const expr_ptr = try self.allocator.create(ASTNode);
                expr_ptr.* = expr;
                const expr_offset: usize = switch (expr) {
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
                return ASTNode{
                    .expression_statement = .{
                        .offset = expr_offset,
                        .expression = expr_ptr,
                    },
                };
            },
        }
    }

    fn parseFunctionDeclaration(self: *Parser) ParseError!ASTNode {
        const name = try self.consume(.identifier, "Expected function name");

        if (!self.match(.left_paren)) {
            self.reportError(self.peek().offset, "Expected '(' after function name");
            return error.ParseError;
        }

        var params = std.ArrayList(Parameter).init(self.allocator);
        defer params.deinit();

        if (!self.check(.right_paren)) {
            repeat: while (true) {
                const param = try self.parseParameter();
                try params.append(param);

                if (!self.match(.comma)) break :repeat;
            }
        }

        if (!self.match(.right_paren)) {
            self.reportError(self.peek().offset, "Expected ')' after function parameters");
            return error.ParseError;
        }

        if (!self.match(.colon)) {
            self.reportError(self.peek().offset, "Expected ':' before return type");
            return error.ParseError;
        }

        const return_type = try self.parseType();

        if (!self.match(.left_brace)) {
            self.reportError(self.peek().offset, "Expected '{' before function body");
            return error.ParseError;
        }

        var body = std.ArrayList(ASTNode).init(self.allocator);
        defer body.deinit();

        while (!self.check(.right_brace) and !self.isAtEnd()) {
            const stmt = try self.parseStatement();
            try body.append(stmt);
        }

        if (!self.match(.right_brace)) {
            // Clean up all the statements that were successfully parsed
            for (body.items) |stmt| {
                self.cleanupASTNode(&stmt);
            }
            self.reportError(self.peek().offset, "Expected '}' after function body");
            return error.ParseError;
        }

        const name_lexeme = self.source[name.offset .. name.offset + name.getLength()];
        return ASTNode{
            .function_declaration = .{
                .offset = name.offset,
                .name = name_lexeme,
                .parameters = try params.toOwnedSlice(),
                .return_type = return_type,
                .body = try body.toOwnedSlice(),
            },
        };
    }

    fn parseVariableDeclaration(self: *Parser) ParseError!ASTNode {
        std.debug.print("DEBUG: parseVariableDeclaration - starting\n", .{});

        // Note: 'let' token has already been consumed by match(.let) in parseStatement

        const name = try self.consume(.identifier, "Expected variable name");
        const name_lexeme = self.source[name.offset .. name.offset + name.getLength()];

        std.debug.print("DEBUG: parseVariableDeclaration - variable name: {s}\n", .{name_lexeme});

        _ = try self.consume(.colon, "Expected ':' after variable name");

        std.debug.print("DEBUG: parseVariableDeclaration - before parseType\n", .{});

        const var_type = try self.parseType();

        std.debug.print("DEBUG: Parsed variable type: {}\n", .{var_type});

        _ = try self.consume(.assign, "Expected '=' after type");

        const value = try self.parseExpression();
        const value_ptr = try self.allocator.create(ASTNode);
        errdefer self.allocator.destroy(value_ptr);
        value_ptr.* = value;
        // Expect semicolon at the end of variable declaration
        if (!self.match(.semicolon)) {
            self.cleanupASTNode(&value);
            self.reportError(self.peek().offset, "Expected ';' after variable declaration");
            return error.ParseError;
        }
        return ASTNode{
            .variable_declaration = .{
                .offset = name.offset,
                .name = name_lexeme,
                .type = var_type,
                .value = value_ptr,
            },
        };
    }

    fn parseReturnStatement(self: *Parser) ParseError!ASTNode {
        var value: ?*ASTNode = null;

        if (!self.check(.semicolon)) {
            const expr = try self.allocator.create(ASTNode);
            errdefer self.allocator.destroy(expr);
            expr.* = try self.parseExpression();
            value = expr;
        }

        if (!self.match(.semicolon)) {
            if (value) |val| {
                self.cleanupASTNodeMut(val);
                self.allocator.destroy(val);
            }
            self.reportError(self.peek().offset, "Expected ';' after return statement");
            return error.ParseError;
        }

        const return_offset = self.previous().offset; // 'return' token
        return ASTNode{
            .return_statement = .{
                .offset = return_offset,
                .value = value,
            },
        };
    }

    fn parseExpression(self: *Parser) ParseError!ASTNode {
        // Check for tensor literal: [shape]type{value}
        if (self.check(.left_bracket)) {
            const start_offset = self.peek().offset; // '[' position
            // Parse shape
            _ = try self.consume(.left_bracket, "Expected '['");
            var dimensions = std.ArrayList(u32).init(self.allocator);
            defer dimensions.deinit();
            while (true) {
                const dim_token = try self.consume(.number, "Expected dimension");
                const dim_str = self.source[dim_token.offset .. dim_token.offset + dim_token.getLength()];
                const dim = try std.fmt.parseInt(u32, dim_str, 10);
                try dimensions.append(dim);
                if (self.check(.comma)) {
                    _ = self.advance();
                } else {
                    break;
                }
            }
            _ = try self.consume(.right_bracket, "Expected ']' after dimensions");
            // Parse element type
            const element_type = try self.parseType();
            // Parse tensor literal value
            _ = try self.consume(.left_brace, "Expected '{' for tensor literal");
            const value = try self.parseExpression();
            _ = try self.consume(.right_brace, "Expected '}' after tensor literal value");
            // Build AST node
            const value_ptr = try self.allocator.create(ASTNode);
            value_ptr.* = value;
            return ASTNode{
                .tensor_literal = .{
                    .offset = start_offset,
                    .shape = try self.allocator.dupe(u32, dimensions.items),
                    .element_type = element_type,
                    .value = value_ptr,
                },
            };
        }
        return try self.parseAddition();
    }

    fn parseAddition(self: *Parser) ParseError!ASTNode {
        var expr = try self.parseMultiplication();

        while (self.match(.plus) or self.match(.minus)) {
            const operator_token = self.previous();
            const operator = switch (operator_token.type) {
                .plus => BinaryOperator.add,
                .minus => BinaryOperator.subtract,
                else => unreachable,
            };

            const right = try self.allocator.create(ASTNode);
            errdefer self.allocator.destroy(right);
            right.* = try self.parseMultiplication();

            const left = try self.allocator.create(ASTNode);
            errdefer self.allocator.destroy(left);
            left.* = expr;

            expr = ASTNode{
                .binary_expression = .{
                    .offset = operator_token.offset,
                    .left = left,
                    .operator = operator,
                    .right = right,
                },
            };
        }

        return expr;
    }

    fn parseMultiplication(self: *Parser) ParseError!ASTNode {
        var expr = try self.parseCall();

        while (self.match(.multiply) or self.match(.divide)) {
            const operator_token = self.previous();
            const operator = switch (operator_token.type) {
                .multiply => BinaryOperator.multiply,
                .divide => BinaryOperator.divide,
                else => unreachable,
            };

            const right = try self.allocator.create(ASTNode);
            errdefer self.allocator.destroy(right);
            right.* = try self.parseCall();

            const left = try self.allocator.create(ASTNode);
            errdefer self.allocator.destroy(left);
            left.* = expr;

            expr = ASTNode{
                .binary_expression = .{
                    .offset = operator_token.offset,
                    .left = left,
                    .operator = operator,
                    .right = right,
                },
            };
        }

        return expr;
    }

    fn parseCall(self: *Parser) ParseError!ASTNode {
        var expr = try self.parseUnary();

        while (self.match(.left_paren)) {
            var args = std.ArrayList(ASTNode).init(self.allocator);
            defer args.deinit();

            if (!self.check(.right_paren)) {
                repeat: while (true) {
                    const arg = try self.parseExpression();
                    try args.append(arg);

                    if (!self.match(.comma)) break :repeat;
                }
            }

            _ = try self.consume(.right_paren, "Expected ')' after arguments");

            const callee = try self.allocator.create(ASTNode);
            errdefer self.allocator.destroy(callee);
            callee.* = expr;

            switch (callee.*) {
                .identifier => {},
                .call_expression => {},
                else => {
                    self.cleanupASTNodeMut(&expr);
                    self.reportError(self.peek().offset, "Cannot call non-function");
                    return error.ParseError;
                },
            }

            const callee_offset: usize = switch (callee.*) {
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
            expr = ASTNode{
                .call_expression = .{
                    .offset = callee_offset,
                    .callee = callee,
                    .return_type = null,
                    .arguments = try args.toOwnedSlice(),
                },
            };
        }

        return expr;
    }

    fn parseUnary(self: *Parser) ParseError!ASTNode {
        if (self.match(.minus)) {
            const minus_token = self.previous();
            const operand = try self.allocator.create(ASTNode);
            errdefer self.allocator.destroy(operand);
            operand.* = try self.parseUnary();

            return ASTNode{
                .unary_expression = .{
                    .offset = minus_token.offset,
                    .operator = UnaryOperator.negate,
                    .operand = operand,
                },
            };
        }

        return try self.parsePrimary();
    }

    fn parsePrimary(self: *Parser) ParseError!ASTNode {
        if (self.match(.number)) {
            const token = self.previous();
            const lexeme = self.source[token.offset .. token.offset + token.getLength()];

            // Parse the number and extract type
            const number_info = try self.parseNumberWithType(lexeme);

            return ASTNode{
                .number_literal = .{
                    .offset = token.offset,
                    .value = lexeme,
                    .type = number_info.type,
                },
            };
        }

        if (self.match(.identifier)) {
            const token = self.previous();
            const lexeme = self.source[token.offset .. token.offset + token.getLength()];
            var expr = ASTNode{
                .identifier = .{
                    .offset = token.offset,
                    .name = lexeme,
                },
            };

            // Check for tensor indexing: identifier[index]
            while (self.check(.left_bracket)) {
                expr = try self.parseTensorIndex(expr);
            }

            return expr;
        }

        if (self.match(.left_paren)) {
            const expr = try self.parseExpression();

            if (!self.match(.right_paren)) {
                self.cleanupASTNode(&expr);
                self.reportError(self.peek().offset, "Expected ')' after expression");
                return error.ParseError;
            }

            return expr;
        }

        self.reportError(self.peek().offset, "Unexpected token");
        return error.ParseError;
    }

    fn parseNumberWithType(self: *Parser, lexeme: []const u8) ParseError!struct { value: []const u8, type: Type } {
        std.debug.print("DEBUG: parseNumberWithType - parsing lexeme: '{s}'\n", .{lexeme});

        // Find the type suffix
        var type_start: ?usize = null;
        for (lexeme, 0..) |c, i| {
            if (c == 'u' or c == 'i' or c == 'f') {
                type_start = i;
                break;
            }
        }

        if (type_start) |start| {
            const number_part = lexeme[0..start];
            const type_part = lexeme[start..];

            std.debug.print("DEBUG: parseNumberWithType - number_part: '{s}', type_part: '{s}'\n", .{ number_part, type_part });

            const parsed_type = Type.fromString(type_part) orelse {
                self.reportError(0, "Invalid type suffix");
                return error.ParseError;
            };

            std.debug.print("DEBUG: parseNumberWithType - parsed type: {}\n", .{parsed_type});

            return .{ .value = number_part, .type = parsed_type };
        } else {
            // Default to i64 if no type suffix
            std.debug.print("DEBUG: parseNumberWithType - no type suffix, defaulting to i64\n", .{});
            return .{ .value = lexeme, .type = .i64 };
        }
    }

    fn parseParameter(self: *Parser) ParseError!Parameter {
        const name = try self.consume(.identifier, "Expected parameter name");

        if (!self.match(.colon)) {
            self.reportError(self.peek().offset, "Expected ':' after parameter name");
            return error.ParseError;
        }

        const param_type = try self.parseType();
        const name_lexeme = self.source[name.offset .. name.offset + name.getLength()];

        return Parameter{
            .name = name_lexeme,
            .type = param_type,
        };
    }

    fn parseType(self: *Parser) ParseError!Type {
        // Check if this is a tensor type (starts with '[')
        if (self.check(.left_bracket)) {
            return self.parseTensorType();
        }

        // Otherwise, parse as a basic type
        const type_token = try self.consume(.type, "Expected type");
        const type_str = self.source[type_token.offset .. type_token.offset + type_token.getLength()];

        std.debug.print("DEBUG: parseType - parsing type string: '{s}'\n", .{type_str});

        const parsed_type = Type.fromString(type_str) orelse {
            self.reportError(type_token.offset, "Invalid type");
            return error.ParseError;
        };

        std.debug.print("DEBUG: parseType - parsed type: {}\n", .{parsed_type});

        return parsed_type;
    }

    fn parseTensorType(self: *Parser) ParseError!Type {
        // Parse [dim1, dim2, ...]element_type syntax

        _ = try self.consume(.left_bracket, "Expected '['");

        var dimensions = std.ArrayList(u32).init(self.allocator);
        defer dimensions.deinit();

        // Parse dimensions
        while (true) {
            const dim_token = try self.consume(.number, "Expected dimension");
            const dim_str = self.source[dim_token.offset .. dim_token.offset + dim_token.getLength()];
            const dim = try std.fmt.parseInt(u32, dim_str, 10);
            try dimensions.append(dim);

            if (self.check(.comma)) {
                _ = self.advance(); // consume comma
            } else {
                break;
            }
        }

        _ = try self.consume(.right_bracket, "Expected ']'");

        // Parse element type
        const element_type = try self.parseType();

        // Allocate a new copy of the element type to avoid pointer sharing
        const element_type_ptr = try self.allocator.create(Type);
        element_type_ptr.* = element_type;

        // Create tensor type
        const tensor_type = try Type.TensorType.init(self.allocator, dimensions.items, element_type_ptr);
        const result = Type{ .tensor = tensor_type };

        std.debug.print("DEBUG: parseTensorType - created tensor type: {}\n", .{result});

        return result;
    }

    fn parseTensorIndex(self: *Parser, base: ASTNode) ParseError!ASTNode {
        // Parse tensor[index] syntax
        _ = try self.consume(.left_bracket, "Expected '['");

        var indices = std.ArrayList(ASTNode).init(self.allocator);
        defer indices.deinit();

        // Parse indices (can be numbers or identifiers)
        while (true) {
            if (self.check(.number)) {
                // Explicit numeric index
                const index_token = try self.consume(.number, "Expected index");
                const index_str = self.source[index_token.offset .. index_token.offset + index_token.getLength()];

                // Parse the number and extract type
                const number_info = try self.parseNumberWithType(index_str);

                try indices.append(ASTNode{
                    .number_literal = .{
                        .offset = index_token.offset,
                        .value = index_str,
                        .type = number_info.type,
                    },
                });
            } else if (self.check(.identifier)) {
                // Implicit index (identifier)
                const index_token = try self.consume(.identifier, "Expected index");
                const index_name = self.source[index_token.offset .. index_token.offset + index_token.getLength()];

                // For now, treat any identifier as implicit index
                // Scope checking will happen in type checker
                const tensor_ptr = try self.allocator.create(ASTNode);
                tensor_ptr.* = base;

                _ = try self.consume(.right_bracket, "Expected ']'");

                return ASTNode{
                    .implicit_tensor_index = .{
                        .offset = index_token.offset,
                        .tensor = tensor_ptr,
                        .implicit_index = try self.allocator.dupe(u8, index_name),
                    },
                };
            } else {
                self.reportError(self.peek().offset, "Expected number or identifier for index");
                return error.ParseError;
            }

            if (self.check(.comma)) {
                _ = self.advance(); // consume comma
            } else {
                break;
            }
        }

        _ = try self.consume(.right_bracket, "Expected ']'");

        // Create tensor slice for explicit numeric indices
        const tensor_ptr = try self.allocator.create(ASTNode);
        tensor_ptr.* = base;

        return ASTNode{
            .tensor_slice = .{
                .offset = 0,
                .tensor = tensor_ptr,
                .indices = try indices.toOwnedSlice(),
            },
        };
    }

    fn match(self: *Parser, token_type: lexer.TokenType) bool {
        if (self.check(token_type)) {
            _ = self.advance();
            return true;
        }
        return false;
    }

    fn check(self: *Parser, token_type: lexer.TokenType) bool {
        if (self.isAtEnd()) return false;
        return self.peek().type == token_type;
    }

    fn advance(self: *Parser) lexer.Token {
        if (!self.isAtEnd()) self.current += 1;
        return self.previous();
    }

    fn isAtEnd(self: *Parser) bool {
        return self.peek().type == .eof;
    }

    fn peek(self: *Parser) lexer.Token {
        return self.tokens[self.current];
    }

    fn previous(self: *Parser) lexer.Token {
        return self.tokens[self.current - 1];
    }

    fn consume(self: *Parser, token_type: lexer.TokenType, message: []const u8) ParseError!lexer.Token {
        if (self.check(token_type)) return self.advance();

        const current_token = self.peek();
        self.reportError(current_token.offset, message);
        return error.ParseError;
    }

    /// Check if an identifier should be treated as an implicit index
    /// For now, any identifier can be an implicit index
    /// Scope checking will happen in type checking
    fn isImplicitIndex(_: *Parser, _: []const u8) bool {
        // Any identifier can be an implicit index
        // We'll check for scope conflicts later in type checking
        return true; // For now, accept any identifier
    }

    fn parseAssignment(self: *Parser) ParseError!ASTNode {
        const left = try self.parseExpression();

        if (self.check(.assign)) {
            _ = self.advance(); // consume '='
            const right = try self.parseExpression();

            // Check if this is a parallel assignment (target is implicit_tensor_index)
            if (left == .implicit_tensor_index) {
                const target_ptr = try self.allocator.create(ASTNode);
                target_ptr.* = left;
                const value_ptr = try self.allocator.create(ASTNode);
                value_ptr.* = right;

                const target_offset = switch (left) {
                    .implicit_tensor_index => |n| n.offset,
                    else => 0,
                };
                return ASTNode{
                    .parallel_assignment = .{
                        .offset = target_offset,
                        .target = target_ptr,
                        .value = value_ptr,
                    },
                };
            } else {
                // For now, only support parallel assignments
                // Regular assignments will be added later
                self.cleanupASTNode(&left);
                self.cleanupASTNode(&right);
                self.reportError(self.peek().offset, "Only parallel assignments are supported");
                return error.ParseError;
            }
        }

        return left;
    }
};

test "parser simple variable" {
    const allocator = std.testing.allocator;
    const source = "let x: i64 = 42i64;";

    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);

    var parser = Parser.init(allocator, tokens, source);
    const ast = try parser.parse();
    defer freeAST(allocator, ast);

    try std.testing.expect(ast == .program);
    try std.testing.expect(ast.program.statements.len == 1);
    try std.testing.expect(ast.program.statements[0] == .variable_declaration);
}

test "parser error - calling non-function" {
    const allocator = std.testing.allocator;
    const source = "fn main(): i64 { 42i64(); }";

    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);

    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - missing function parentheses" {
    const allocator = std.testing.allocator;
    const source = "fn main: i64 { return 42i64; }";

    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);

    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - missing closing parenthesis" {
    const allocator = std.testing.allocator;
    const source = "fn main(x: i64 { return 42i64; }";

    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);

    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - missing opening brace" {
    const allocator = std.testing.allocator;
    const source = "fn main(): i64 return 42i64; }";

    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);

    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - missing closing brace" {
    const allocator = std.testing.allocator;
    const source = "fn main(): i64 { return 42i64;";

    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);

    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - missing assignment operator" {
    const allocator = std.testing.allocator;
    const source = "let x: i64 42i64;";

    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);

    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - missing semicolon after variable" {
    const allocator = std.testing.allocator;
    const source = "let x: i64 = 42i64";

    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);

    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - missing closing parenthesis in expression" {
    const allocator = std.testing.allocator;
    const source = "let x: i64 = (5i64 + 3i64;";

    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);

    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - calling expression" {
    const allocator = std.testing.allocator;
    const source = "fn main(): i64 { (5i64 + 3i64)(); }";

    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);

    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

fn freeType(allocator: std.mem.Allocator, type_info: Type) void {
    switch (type_info) {
        .tensor => |tensor_type| {
            // Free the shape array
            allocator.free(tensor_type.shape);
            // Recursively free the element type
            freeType(allocator, tensor_type.element_type.*);
            // Free the element type pointer
            allocator.destroy(tensor_type.element_type);
        },
        else => {
            // Basic types don't need cleanup
        },
    }
}

pub fn freeAST(allocator: std.mem.Allocator, node: ASTNode) void {
    switch (node) {
        .program => |prog| {
            for (prog.statements) |stmt| {
                freeAST(allocator, stmt);
            }
            allocator.free(prog.statements);
        },
        .function_declaration => |func| {
            // Free parameter types
            for (func.parameters) |param| {
                freeType(allocator, param.type);
            }
            allocator.free(func.parameters);
            
            // Free return type
            freeType(allocator, func.return_type);
            
            for (func.body) |stmt| {
                freeAST(allocator, stmt);
            }
            allocator.free(func.body);
        },
        .variable_declaration => |var_decl| {
            // Free the variable type
            freeType(allocator, var_decl.type);
            freeAST(allocator, var_decl.value.*);
            allocator.destroy(var_decl.value);
        },
        .return_statement => |ret| {
            if (ret.value) |val| {
                freeAST(allocator, val.*);
                allocator.destroy(val);
            }
        },
        .expression_statement => |expr_stmt| {
            freeAST(allocator, expr_stmt.expression.*);
            allocator.destroy(expr_stmt.expression);
        },
        .binary_expression => |bin_expr| {
            freeAST(allocator, bin_expr.left.*);
            freeAST(allocator, bin_expr.right.*);
            allocator.destroy(bin_expr.left);
            allocator.destroy(bin_expr.right);
        },
        .unary_expression => |unary_expr| {
            freeAST(allocator, unary_expr.operand.*);
            allocator.destroy(unary_expr.operand);
        },
        .call_expression => |call| {
            freeAST(allocator, call.callee.*);
            allocator.destroy(call.callee);
            for (call.arguments) |arg| {
                freeAST(allocator, arg);
            }
            allocator.free(call.arguments);
        },
        .tensor_literal => |tensor_lit| {
            // Free the tensor literal type
            freeType(allocator, tensor_lit.element_type);
            freeAST(allocator, tensor_lit.value.*);
            allocator.destroy(tensor_lit.value);
        },
        .tensor_slice => |tensor_slice| {
            freeAST(allocator, tensor_slice.tensor.*);
            allocator.destroy(tensor_slice.tensor);
            for (tensor_slice.indices) |index| {
                freeAST(allocator, index);
            }
            allocator.free(tensor_slice.indices);
        },
        .implicit_tensor_index => |tensor_index| {
            // Free the implicit index name string
            allocator.free(tensor_index.implicit_index);
            freeAST(allocator, tensor_index.tensor.*);
            allocator.destroy(tensor_index.tensor);
        },
        .parallel_assignment => |pa| {
            freeAST(allocator, pa.target.*);
            freeAST(allocator, pa.value.*);
            allocator.destroy(pa.target);
            allocator.destroy(pa.value);
        },
        .identifier, .number_literal, .parameter => {},
    }
}
