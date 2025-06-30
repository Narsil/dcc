const std = @import("std");

pub const TokenType = enum {
    // Literals
    number,
    identifier,
    
    // Keywords
    let,
    fn_,
    return_,
    
    // Operators
    plus,
    minus,
    multiply,
    divide,
    assign,
    
    // Delimiters
    left_paren,
    right_paren,
    left_brace,
    right_brace,
    comma,
    semicolon,
    
    // Special
    eof,
    invalid,
};

pub const Token = struct {
    type: TokenType,
    lexeme: []const u8,
    line: u32,
    column: u32,
};

pub const Lexer = struct {
    source: []const u8,
    current: usize,
    line: u32,
    column: u32,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, source: []const u8) Lexer {
        return Lexer{
            .source = source,
            .current = 0,
            .line = 1,
            .column = 1,
            .allocator = allocator,
        };
    }
    
    pub fn tokenize(self: *Lexer) ![]Token {
        var tokens = std.ArrayList(Token).init(self.allocator);
        defer tokens.deinit();
        
        while (!self.isAtEnd()) {
            const token = self.nextToken();
            try tokens.append(token);
            if (token.type == .invalid) {
                std.debug.print("Invalid token at line {}, column {}: '{s}'\n", .{ token.line, token.column, token.lexeme });
            }
        }
        
        try tokens.append(Token{
            .type = .eof,
            .lexeme = "",
            .line = self.line,
            .column = self.column,
        });
        
        return tokens.toOwnedSlice();
    }
    
    fn nextToken(self: *Lexer) Token {
        self.skipWhitespace();
        
        if (self.isAtEnd()) {
            return Token{
                .type = .eof,
                .lexeme = "",
                .line = self.line,
                .column = self.column,
            };
        }
        
        const start_line = self.line;
        const start_column = self.column;
        const start = self.current;
        
        const c = self.advance();
        
        return switch (c) {
            '+' => self.makeToken(.plus, start, start_line, start_column),
            '-' => self.makeToken(.minus, start, start_line, start_column),
            '*' => self.makeToken(.multiply, start, start_line, start_column),
            '/' => self.makeToken(.divide, start, start_line, start_column),
            '=' => self.makeToken(.assign, start, start_line, start_column),
            '(' => self.makeToken(.left_paren, start, start_line, start_column),
            ')' => self.makeToken(.right_paren, start, start_line, start_column),
            '{' => self.makeToken(.left_brace, start, start_line, start_column),
            '}' => self.makeToken(.right_brace, start, start_line, start_column),
            ',' => self.makeToken(.comma, start, start_line, start_column),
            ';' => self.makeToken(.semicolon, start, start_line, start_column),
            else => {
                if (std.ascii.isDigit(c)) {
                    return self.number(start, start_line, start_column);
                } else if (std.ascii.isAlphabetic(c) or c == '_') {
                    return self.identifier(start, start_line, start_column);
                } else {
                    return Token{
                        .type = .invalid,
                        .lexeme = self.source[start..self.current],
                        .line = start_line,
                        .column = start_column,
                    };
                }
            },
        };
    }
    
    fn number(self: *Lexer, start: usize, start_line: u32, start_column: u32) Token {
        while (!self.isAtEnd() and std.ascii.isDigit(self.peek())) {
            _ = self.advance();
        }
        return self.makeToken(.number, start, start_line, start_column);
    }
    
    fn identifier(self: *Lexer, start: usize, start_line: u32, start_column: u32) Token {
        while (!self.isAtEnd() and (std.ascii.isAlphanumeric(self.peek()) or self.peek() == '_')) {
            _ = self.advance();
        }
        
        const text = self.source[start..self.current];
        const token_type = self.identifierType(text);
        return Token{
            .type = token_type,
            .lexeme = text,
            .line = start_line,
            .column = start_column,
        };
    }
    
    fn identifierType(self: *Lexer, text: []const u8) TokenType {
        _ = self;
        if (std.mem.eql(u8, text, "let")) return .let;
        if (std.mem.eql(u8, text, "fn")) return .fn_;
        if (std.mem.eql(u8, text, "return")) return .return_;
        return .identifier;
    }
    
    fn makeToken(self: *Lexer, token_type: TokenType, start: usize, start_line: u32, start_column: u32) Token {
        return Token{
            .type = token_type,
            .lexeme = self.source[start..self.current],
            .line = start_line,
            .column = start_column,
        };
    }
    
    fn skipWhitespace(self: *Lexer) void {
        while (!self.isAtEnd()) {
            const c = self.peek();
            switch (c) {
                ' ', '\r', '\t' => {
                    _ = self.advance();
                },
                '\n' => {
                    self.line += 1;
                    self.column = 1;
                    _ = self.advance();
                },
                else => break,
            }
        }
    }
    
    fn isAtEnd(self: *Lexer) bool {
        return self.current >= self.source.len;
    }
    
    fn advance(self: *Lexer) u8 {
        if (self.isAtEnd()) return 0;
        const c = self.source[self.current];
        self.current += 1;
        self.column += 1;
        return c;
    }
    
    fn peek(self: *Lexer) u8 {
        if (self.isAtEnd()) return 0;
        return self.source[self.current];
    }
};

test "lexer basic tokens" {
    const allocator = std.testing.allocator;
    const source = "let x = 42;";
    
    var lexer = Lexer.init(allocator, source);
    const tokens = try lexer.tokenize();
    defer allocator.free(tokens);
    
    try std.testing.expect(tokens.len == 6); // let, x, =, 42, ;, eof
    try std.testing.expect(tokens[0].type == .let);
    try std.testing.expect(tokens[1].type == .identifier);
    try std.testing.expect(tokens[2].type == .assign);
    try std.testing.expect(tokens[3].type == .number);
    try std.testing.expect(tokens[4].type == .semicolon);
    try std.testing.expect(tokens[5].type == .eof);
} 