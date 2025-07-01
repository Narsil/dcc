const std = @import("std");
const lexer = @import("lexer.zig");

pub const ParseError = error{ ParseError, OutOfMemory, InvalidCharacter, Overflow, InvalidUtf8 } || std.mem.Allocator.Error;

pub const NodeType = enum {
    program,
    function_declaration,
    variable_declaration,
    return_statement,
    expression_statement,
    binary_expression,
    call_expression,
    identifier,
    number_literal,
};

pub const BinaryOperator = enum {
    add,
    subtract,
    multiply,
    divide,
};

pub const ASTNode = union(NodeType) {
    program: struct {
        statements: []ASTNode,
    },
    function_declaration: struct {
        name: []const u8,
        parameters: [][]const u8,
        body: []ASTNode,
    },
    variable_declaration: struct {
        name: []const u8,
        value: *ASTNode,
    },
    return_statement: struct {
        value: ?*ASTNode,
    },
    expression_statement: struct {
        expression: *ASTNode,
    },
    binary_expression: struct {
        left: *ASTNode,
        operator: BinaryOperator,
        right: *ASTNode,
    },
    call_expression: struct {
        callee: *ASTNode,
        arguments: []ASTNode,
    },
    identifier: struct {
        name: []const u8,
    },
    number_literal: struct {
        value: i64,
    },
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
            .identifier, .number_literal => {},
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
            .identifier, .number_literal => {},
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
                .statements = try statements.toOwnedSlice(),
            },
        };
    }
    
    fn parseStatement(self: *Parser) ParseError!ASTNode {
        if (self.match(.fn_)) {
            return try self.parseFunctionDeclaration();
        }
        
        if (self.match(.let)) {
            return try self.parseVariableDeclaration();
        }
        
        if (self.match(.return_)) {
            return try self.parseReturnStatement();
        }
        
        return try self.parseExpressionStatement();
    }
    
    fn parseFunctionDeclaration(self: *Parser) ParseError!ASTNode {
        const name = try self.consume(.identifier, "Expected function name");
        
        if (!self.match(.left_paren)) {
            const pos = lexer.Lexer.offsetToLineColumn(self.source, self.peek().offset);
            std.debug.print("Error at line {}, column {}: Expected '(' after function name '{s}'\n", .{ 
                pos.line, pos.column, 
                self.source[name.offset .. name.offset + name.getLength()]
            });
            return error.ParseError;
        }
        
        var params = std.ArrayList([]const u8).init(self.allocator);
        defer params.deinit();
        
        if (!self.check(.right_paren)) {
            repeat: while (true) {
                const param = try self.consume(.identifier, "Expected parameter name");
                const param_lexeme = self.source[param.offset .. param.offset + param.getLength()];
                try params.append(param_lexeme);
                
                if (!self.match(.comma)) break :repeat;
            }
        }
        
        if (!self.match(.right_paren)) {
            self.reportError(self.peek().offset, "Expected ')' after function parameters");
            return error.ParseError;
        }
        
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
                .name = name_lexeme,
                .parameters = try params.toOwnedSlice(),
                .body = try body.toOwnedSlice(),
            },
        };
    }
    
    fn parseVariableDeclaration(self: *Parser) ParseError!ASTNode {
        const name = try self.consume(.identifier, "Expected variable name");
        
        if (!self.match(.assign)) {
            self.reportError(self.peek().offset, "Expected '=' after variable name");
            return error.ParseError;
        }
        
        const value = try self.allocator.create(ASTNode);
        errdefer self.allocator.destroy(value);
        value.* = try self.parseExpression();
        
        if (!self.match(.semicolon)) {
            self.cleanupASTNodeMut(value);
            self.reportError(self.peek().offset, "Expected ';' after variable declaration");
            return error.ParseError;
        }
        
        const name_lexeme = self.source[name.offset .. name.offset + name.getLength()];
        return ASTNode{
            .variable_declaration = .{
                .name = name_lexeme,
                .value = value,
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
        
        return ASTNode{
            .return_statement = .{
                .value = value,
            },
        };
    }
    
    fn parseExpressionStatement(self: *Parser) ParseError!ASTNode {
        const expr = try self.allocator.create(ASTNode);
        errdefer self.allocator.destroy(expr);
        expr.* = try self.parseExpression();
        
        if (!self.match(.semicolon)) {
            self.cleanupASTNodeMut(expr);
            self.reportError(self.peek().offset, "Expected ';' after expression");
            return error.ParseError;
        }
        
        return ASTNode{
            .expression_statement = .{
                .expression = expr,
            },
        };
    }
    
    fn parseExpression(self: *Parser) ParseError!ASTNode {
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
                    .left = left,
                    .operator = operator,
                    .right = right,
                },
            };
        }
        
        return expr;
    }
    
    fn parseCall(self: *Parser) ParseError!ASTNode {
        var expr = try self.parsePrimary();
        
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
                }
            }
            
            expr = ASTNode{
                .call_expression = .{
                    .callee = callee,
                    .arguments = try args.toOwnedSlice(),
                },
            };
        }
        
        return expr;
    }
    
    fn parsePrimary(self: *Parser) ParseError!ASTNode {
        if (self.match(.number)) {
            const token = self.previous();
            const lexeme = self.source[token.offset .. token.offset + token.getLength()];
            const value = try std.fmt.parseInt(i64, lexeme, 10);
            return ASTNode{
                .number_literal = .{
                    .value = value,
                },
            };
        }
        
        if (self.match(.identifier)) {
            const token = self.previous();
            const lexeme = self.source[token.offset .. token.offset + token.getLength()];
            return ASTNode{
                .identifier = .{
                    .name = lexeme,
                },
            };
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
};

test "parser simple variable" {
    const allocator = std.testing.allocator;
    const source = "let x = 42;";
    
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
    const source = "fn main() { 42(); }";
    
    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);
    
    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - missing function parentheses" {
    const allocator = std.testing.allocator;
    const source = "fn main { return 42; }";
    
    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);
    
    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - missing closing parenthesis" {
    const allocator = std.testing.allocator;
    const source = "fn main( { return 42; }";
    
    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);
    
    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - missing opening brace" {
    const allocator = std.testing.allocator;
    const source = "fn main() return 42; }";
    
    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);
    
    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - missing closing brace" {
    const allocator = std.testing.allocator;
    const source = "fn main() { return 42;";
    
    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);
    
    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - missing assignment operator" {
    const allocator = std.testing.allocator;
    const source = "let x 42;";
    
    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);
    
    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - missing semicolon after variable" {
    const allocator = std.testing.allocator;
    const source = "let x = 42";
    
    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);
    
    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - missing closing parenthesis in expression" {
    const allocator = std.testing.allocator;
    const source = "let x = (5 + 3;";
    
    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);
    
    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
}

test "parser error - calling expression" {
    const allocator = std.testing.allocator;
    const source = "fn main() { (5 + 3)(); }";
    
    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);
    
    var parser = Parser.init(allocator, tokens, source);
    const result = parser.parse();
    try std.testing.expectError(error.ParseError, result);
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
            allocator.free(func.parameters);
            for (func.body) |stmt| {
                freeAST(allocator, stmt);
            }
            allocator.free(func.body);
        },
        .variable_declaration => |var_decl| {
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
        .call_expression => |call| {
            freeAST(allocator, call.callee.*);
            allocator.destroy(call.callee);
            for (call.arguments) |arg| {
                freeAST(allocator, arg);
            }
            allocator.free(call.arguments);
        },
        .identifier, .number_literal => {},
    }
} 