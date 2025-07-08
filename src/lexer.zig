const std = @import("std");

pub const TokenType = enum {
    // Literals
    number,
    identifier,
    type, // For type identifiers like u32, i64, f32, etc.

    // Keywords
    let,
    fn_,
    return_,
    reduce,

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
    left_bracket,
    right_bracket,
    comma,
    semicolon,
    colon, // For type annotations

    // Comparison operators
    greater_than,

    // Special
    eof,
    invalid,

    /// Returns the fixed length for tokens that have a known length
    /// Returns 0 for variable-length tokens (number, identifier, keywords, type)
    pub fn length(self: TokenType) usize {
        return switch (self) {
            .plus, .minus, .multiply, .divide, .assign, .colon, .greater_than => 1,
            .left_paren, .right_paren, .left_brace, .right_brace, .left_bracket, .right_bracket, .comma, .semicolon => 1,
            .let => 3,
            .fn_ => 2,
            .return_ => 6,
            .reduce => 6,
            .number, .identifier, .type, .eof, .invalid => 0, // Variable length
        };
    }

    /// Returns true if this token type has a fixed length
    pub fn hasFixedLength(self: TokenType) bool {
        return self.length() > 0;
    }
};

pub const Token = struct {
    type: TokenType,
    offset: usize,
    length: usize, // Only used for variable-length tokens

    /// Get the actual length of this token
    pub fn getLength(self: Token) usize {
        if (self.type.hasFixedLength()) {
            return self.type.length();
        } else {
            return self.length;
        }
    }
};

pub const Lexer = struct {
    source: []const u8,
    current: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, source: []const u8) Lexer {
        return Lexer{
            .source = source,
            .current = 0,
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
                const pos = Lexer.offsetToLineColumn(self.source, token.offset);
                const lexeme = self.source[token.offset .. token.offset + token.getLength()];
                std.debug.print("Invalid token at line {}, column {}: '{s}'\n", .{ pos.line, pos.column, lexeme });
            }
        }

        try tokens.append(Token{
            .type = .eof,
            .offset = self.current,
            .length = 0,
        });

        return tokens.toOwnedSlice();
    }

    fn nextToken(self: *Lexer) Token {
        self.skipWhitespace();

        if (self.isAtEnd()) {
            return Token{
                .type = .eof,
                .offset = self.current,
                .length = 0,
            };
        }

        const start = self.current;
        const c = self.advance();

        return switch (c) {
            '+' => self.makeToken(.plus, start),
            '-' => self.makeToken(.minus, start),
            '*' => self.makeToken(.multiply, start),
            '/' => self.makeToken(.divide, start),
            '=' => self.makeToken(.assign, start),
            '(' => self.makeToken(.left_paren, start),
            ')' => self.makeToken(.right_paren, start),
            '{' => self.makeToken(.left_brace, start),
            '}' => self.makeToken(.right_brace, start),
            '[' => self.makeToken(.left_bracket, start),
            ']' => self.makeToken(.right_bracket, start),
            ',' => self.makeToken(.comma, start),
            ';' => self.makeToken(.semicolon, start),
            ':' => self.makeToken(.colon, start),
            '>' => self.makeToken(.greater_than, start),
            else => {
                if (std.ascii.isDigit(c)) {
                    return self.number(start);
                } else if (std.ascii.isAlphabetic(c) or c == '_') {
                    return self.identifier(start);
                } else {
                    return Token{
                        .type = .invalid,
                        .offset = start,
                        .length = self.current - start,
                    };
                }
            },
        };
    }

    fn number(self: *Lexer, start: usize) Token {
        while (!self.isAtEnd() and std.ascii.isDigit(self.peek())) {
            _ = self.advance();
        }

        // Check for decimal point
        if (!self.isAtEnd() and self.peek() == '.') {
            _ = self.advance(); // consume '.'
            while (!self.isAtEnd() and std.ascii.isDigit(self.peek())) {
                _ = self.advance();
            }
        }

        // Check for type suffix
        if (!self.isAtEnd()) {
            const suffix_start = self.current;
            const c = self.peek();

            // Check for unsigned integer types
            if (c == 'u') {
                _ = self.advance(); // consume 'u'
                if (!self.isAtEnd() and std.ascii.isDigit(self.peek())) {
                    while (!self.isAtEnd() and std.ascii.isDigit(self.peek())) {
                        _ = self.advance();
                    }
                } else {
                    // Invalid suffix, backtrack
                    self.current = suffix_start;
                }
            }
            // Check for signed integer types
            else if (c == 'i') {
                _ = self.advance(); // consume 'i'
                if (!self.isAtEnd() and std.ascii.isDigit(self.peek())) {
                    while (!self.isAtEnd() and std.ascii.isDigit(self.peek())) {
                        _ = self.advance();
                    }
                } else {
                    // Invalid suffix, backtrack
                    self.current = suffix_start;
                }
            }
            // Check for float types
            else if (c == 'f') {
                _ = self.advance(); // consume 'f'
                if (!self.isAtEnd() and std.ascii.isDigit(self.peek())) {
                    while (!self.isAtEnd() and std.ascii.isDigit(self.peek())) {
                        _ = self.advance();
                    }
                } else {
                    // Invalid suffix, backtrack
                    self.current = suffix_start;
                }
            }
        }

        return self.makeToken(.number, start);
    }

    fn identifier(self: *Lexer, start: usize) Token {
        while (!self.isAtEnd() and (std.ascii.isAlphanumeric(self.peek()) or self.peek() == '_')) {
            _ = self.advance();
        }

        const text = self.source[start..self.current];
        const token_type = self.identifierType(text);
        return Token{
            .type = token_type,
            .offset = start,
            .length = self.current - start,
        };
    }

    fn identifierType(self: *Lexer, text: []const u8) TokenType {
        if (std.mem.eql(u8, text, "let")) return .let;
        if (std.mem.eql(u8, text, "fn")) return .fn_;
        if (std.mem.eql(u8, text, "return")) return .return_;
        if (std.mem.eql(u8, text, "reduce")) return .reduce;

        // Check if it's a type identifier
        if (self.isTypeIdentifier(text)) return .type;

        return .identifier;
    }

    fn isTypeIdentifier(_: *Lexer, text: []const u8) bool {
        // Check for supported types
        const types = [_][]const u8{ "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64", "void" };
        for (types) |type_str| {
            if (std.mem.eql(u8, text, type_str)) return true;
        }
        return false;
    }

    fn makeToken(self: *Lexer, token_type: TokenType, start: usize) Token {
        return Token{
            .type = token_type,
            .offset = start,
            .length = if (token_type.hasFixedLength()) 0 else self.current - start,
        };
    }

    fn skipWhitespace(self: *Lexer) void {
        while (!self.isAtEnd()) {
            const c = self.peek();
            switch (c) {
                ' ', '\r', '\t', '\n' => {
                    _ = self.advance();
                },
                '/' => {
                    // Check for comment
                    if (!self.isAtEnd() and self.peekNext() == '/') {
                        // Skip single-line comment
                        while (!self.isAtEnd() and self.peek() != '\n') {
                            _ = self.advance();
                        }
                    } else {
                        break;
                    }
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
        return c;
    }

    fn peek(self: *Lexer) u8 {
        if (self.isAtEnd()) return 0;
        return self.source[self.current];
    }

    fn peekNext(self: *Lexer) u8 {
        if (self.current + 1 >= self.source.len) return 0;
        return self.source[self.current + 1];
    }

    /// Convert an offset back to line and column numbers
    /// This is useful for error reporting when you need human-readable positions
    pub fn offsetToLineColumn(source: []const u8, offset: usize) struct { line: u32, column: u32 } {
        var line: u32 = 1;
        var column: u32 = 1;

        for (source[0..@min(offset, source.len)]) |c| {
            if (c == '\n') {
                line += 1;
                column = 1;
            } else {
                column += 1;
            }
        }

        return .{ .line = line, .column = column };
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

test "offset to line column conversion" {
    const source = "hello\nworld\ntest";

    // Test offset 0 (start of file)
    var pos = Lexer.offsetToLineColumn(source, 0);
    try std.testing.expect(pos.line == 1);
    try std.testing.expect(pos.column == 1);

    // Test offset 3 (middle of first line)
    pos = Lexer.offsetToLineColumn(source, 3);
    try std.testing.expect(pos.line == 1);
    try std.testing.expect(pos.column == 4);

    // Test offset 5 (newline character)
    pos = Lexer.offsetToLineColumn(source, 5);
    try std.testing.expect(pos.line == 1);
    try std.testing.expect(pos.column == 6);

    // Test offset 6 (start of second line)
    pos = Lexer.offsetToLineColumn(source, 6);
    try std.testing.expect(pos.line == 2);
    try std.testing.expect(pos.column == 1);

    // Test offset 10 (middle of second line)
    pos = Lexer.offsetToLineColumn(source, 10);
    try std.testing.expect(pos.line == 2);
    try std.testing.expect(pos.column == 5);

    // Test offset 11 (newline character)
    pos = Lexer.offsetToLineColumn(source, 11);
    try std.testing.expect(pos.line == 2);
    try std.testing.expect(pos.column == 6);

    // Test offset 12 (start of third line)
    pos = Lexer.offsetToLineColumn(source, 12);
    try std.testing.expect(pos.line == 3);
    try std.testing.expect(pos.column == 1);

    // Test offset 15 (end of file)
    pos = Lexer.offsetToLineColumn(source, 15);
    try std.testing.expect(pos.line == 3);
    try std.testing.expect(pos.column == 4);
}

