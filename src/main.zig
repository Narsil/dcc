//! By convention, main.zig is where your main function lives in the case that
//! you are building an executable. If you are making a library, the convention
//! is to delete this file and start with root.zig instead.

const std = @import("std");
const lexer = @import("lexer.zig");
const parser = @import("parser.zig");
const codegen = @import("codegen.zig");

fn hasMainFunction(ast: parser.ASTNode) bool {
    switch (ast) {
        .program => |prog| {
            for (prog.statements) |stmt| {
                if (stmt == .function_declaration) {
                    if (std.mem.eql(u8, stmt.function_declaration.name, "main")) {
                        return true;
                    }
                }
            }
        },
        else => return false,
    }
    return false;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    if (args.len < 2) {
        std.debug.print("Usage: {s} <source_file>\n", .{args[0]});
        return;
    }
    
    const source_file = args[1];
    std.debug.print("Compiling: {s}\n", .{source_file});
    
    // Read source file
    const source = try std.fs.cwd().readFileAlloc(allocator, source_file, 1024 * 1024);
    defer allocator.free(source);
    
    std.debug.print("Source code:\n{s}\n", .{source});
    
    // Tokenize
    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);
    
    std.debug.print("Tokens:\n", .{});
    for (tokens) |token| {
        std.debug.print("  {} '{s}' ({}:{})\n", .{ token.type, token.lexeme, token.line, token.column });
    }
    
    // Parse
    var parse = parser.Parser.init(allocator, tokens);
    const ast = try parse.parse();
    defer parser.freeAST(allocator, ast);
    
    std.debug.print("AST parsed successfully!\n", .{});
    
    // Check if there's a main function
    const has_main = hasMainFunction(ast);
    
    // Generate LLVM IR using C bindings
    var code_gen = try codegen.CodeGen.init(allocator, "toy_program");
    defer code_gen.deinit();
    
    try code_gen.generate(ast);
    code_gen.printIR();
    
    // Generate object file
    const obj_file = "output.o";
    try code_gen.generateObjectFile(obj_file);
    std.debug.print("Generated object file: {s}\n", .{obj_file});
    
    if (has_main) {
        // Create executable binary
        const bin_file = "output";
        const result = try std.process.Child.run(.{
            .allocator = allocator,
            .argv = &[_][]const u8{ "gcc", obj_file, "-o", bin_file },
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        
        if (result.term.Exited == 0) {
            std.debug.print("Generated executable binary: {s}\n", .{bin_file});
        } else {
            std.debug.print("Error creating executable:\n{s}\n", .{result.stderr});
        }
    } else {
        // Create shared library
        const so_file = "output.so";
        const result = try std.process.Child.run(.{
            .allocator = allocator,
            .argv = &[_][]const u8{ "gcc", "-shared", "-fPIC", obj_file, "-o", so_file },
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        
        if (result.term.Exited == 0) {
            std.debug.print("Generated shared library: {s}\n", .{so_file});
        } else {
            std.debug.print("Error creating shared library:\n{s}\n", .{result.stderr});
        }
    }
    
    // Clean up object file
    std.fs.cwd().deleteFile(obj_file) catch {};
    
    std.debug.print("Compilation complete!\n", .{});
}

test "integration test" {
    const allocator = std.testing.allocator;
    const source = 
        \\fn add(a, b) {
        \\    return a + b;
        \\}
        \\
        \\fn main() {
        \\    let result = add(5, 3);
        \\    return result;
        \\}
    ;
    
    // Tokenize
    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);
    
    // Parse
    var parse = parser.Parser.init(allocator, tokens);
    const ast = try parse.parse();
    defer parser.freeAST(allocator, ast);
    
    // Should have two function declarations
    try std.testing.expect(ast == .program);
    try std.testing.expect(ast.program.statements.len == 2);
    try std.testing.expect(ast.program.statements[0] == .function_declaration);
    try std.testing.expect(ast.program.statements[1] == .function_declaration);
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // Try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "use other module" {
    try std.testing.expectEqual(@as(i32, 150), lib.add(100, 50));
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}

/// This imports the separate module containing `root.zig`. Take a look in `build.zig` for details.
const lib = @import("dcc_lib");
