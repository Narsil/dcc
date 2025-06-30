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
    
    pub fn init(allocator: std.mem.Allocator, tokens: []lexer.Token) Parser {
        return Parser{
            .tokens = tokens,
            .current = 0,
            .allocator = allocator,
        };
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
        
        _ = try self.consume(.left_paren, "Expected '(' after function name");
        
        var params = std.ArrayList([]const u8).init(self.allocator);
        defer params.deinit();
        
        if (!self.check(.right_paren)) {
            repeat: while (true) {
                const param = try self.consume(.identifier, "Expected parameter name");
                try params.append(param.lexeme);
                
                if (!self.match(.comma)) break :repeat;
            }
        }
        
        _ = try self.consume(.right_paren, "Expected ')' after parameters");
        _ = try self.consume(.left_brace, "Expected '{' before function body");
        
        var body = std.ArrayList(ASTNode).init(self.allocator);
        defer body.deinit();
        
        while (!self.check(.right_brace) and !self.isAtEnd()) {
            const stmt = try self.parseStatement();
            try body.append(stmt);
        }
        
        _ = try self.consume(.right_brace, "Expected '}' after function body");
        
        return ASTNode{
            .function_declaration = .{
                .name = name.lexeme,
                .parameters = try params.toOwnedSlice(),
                .body = try body.toOwnedSlice(),
            },
        };
    }
    
    fn parseVariableDeclaration(self: *Parser) ParseError!ASTNode {
        const name = try self.consume(.identifier, "Expected variable name");
        _ = try self.consume(.assign, "Expected '=' after variable name");
        
        const value = try self.allocator.create(ASTNode);
        value.* = try self.parseExpression();
        
        _ = try self.consume(.semicolon, "Expected ';' after variable declaration");
        
        return ASTNode{
            .variable_declaration = .{
                .name = name.lexeme,
                .value = value,
            },
        };
    }
    
    fn parseReturnStatement(self: *Parser) ParseError!ASTNode {
        var value: ?*ASTNode = null;
        
        if (!self.check(.semicolon)) {
            const expr = try self.allocator.create(ASTNode);
            expr.* = try self.parseExpression();
            value = expr;
        }
        
        _ = try self.consume(.semicolon, "Expected ';' after return statement");
        
        return ASTNode{
            .return_statement = .{
                .value = value,
            },
        };
    }
    
    fn parseExpressionStatement(self: *Parser) ParseError!ASTNode {
        const expr = try self.allocator.create(ASTNode);
        expr.* = try self.parseExpression();
        
        _ = try self.consume(.semicolon, "Expected ';' after expression");
        
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
            right.* = try self.parseMultiplication();
            
            const left = try self.allocator.create(ASTNode);
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
            right.* = try self.parseCall();
            
            const left = try self.allocator.create(ASTNode);
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
            callee.* = expr;
            
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
            const value = try std.fmt.parseInt(i64, token.lexeme, 10);
            return ASTNode{
                .number_literal = .{
                    .value = value,
                },
            };
        }
        
        if (self.match(.identifier)) {
            const token = self.previous();
            return ASTNode{
                .identifier = .{
                    .name = token.lexeme,
                },
            };
        }
        
        if (self.match(.left_paren)) {
            const expr = try self.parseExpression();
            _ = try self.consume(.right_paren, "Expected ')' after expression");
            return expr;
        }
        
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
        std.debug.print("Parse error at line {}, column {}: {s}. Got: {}\n", .{ current_token.line, current_token.column, message, current_token.type });
        return error.ParseError;
    }
};

test "parser simple variable" {
    const allocator = std.testing.allocator;
    const source = "let x = 42;";
    
    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);
    
    var parser = Parser.init(allocator, tokens);
    const ast = try parser.parse();
    defer freeAST(allocator, ast);
    
    try std.testing.expect(ast == .program);
    try std.testing.expect(ast.program.statements.len == 1);
    try std.testing.expect(ast.program.statements[0] == .variable_declaration);
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