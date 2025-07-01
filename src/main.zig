//! By convention, main.zig is where your main function lives when building 
//! an executable. This is the main entry point for the toy compiler.

const std = @import("std");
const lexer = @import("lexer.zig");
const parser = @import("parser.zig");
const codegen = @import("codegen.zig");

/// Target information parsed from target triplet
const TargetInfo = struct {
    arch: []const u8,
    os: []const u8,
    abi: ?[]const u8,
    
    fn parseFromTriple(allocator: std.mem.Allocator, triple: []const u8) !TargetInfo {
        var parts = std.mem.splitSequence(u8, triple, "-");
        
        const arch = parts.next() orelse return error.InvalidTargetTriple;
        const vendor_or_os = parts.next() orelse return error.InvalidTargetTriple;
        
        // Handle different triplet formats:
        // arm64-apple-darwin, x86_64-pc-linux-gnu, x86_64-linux-gnu, etc.
        var os: []const u8 = undefined;
        var abi: ?[]const u8 = null;
        
        if (std.mem.eql(u8, vendor_or_os, "apple")) {
            os = parts.next() orelse return error.InvalidTargetTriple;
        } else if (std.mem.eql(u8, vendor_or_os, "pc")) {
            os = parts.next() orelse return error.InvalidTargetTriple;
            abi = parts.next(); // Optional ABI like "gnu"
        } else {
            // Assume vendor_or_os is actually the OS (like x86_64-linux-gnu)
            os = vendor_or_os;
            abi = parts.next(); // Optional ABI
        }
        
        return TargetInfo{
            .arch = try allocator.dupe(u8, arch),
            .os = try allocator.dupe(u8, os),
            .abi = if (abi) |a| try allocator.dupe(u8, a) else null,
        };
    }
    
    fn deinit(self: *const TargetInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.arch);
        allocator.free(self.os);
        if (self.abi) |abi| {
            allocator.free(abi);
        }
    }
    
    fn getOsTag(self: *const TargetInfo) std.Target.Os.Tag {
        if (std.mem.eql(u8, self.os, "darwin") or std.mem.eql(u8, self.os, "macos")) {
            return .macos;
        } else if (std.mem.eql(u8, self.os, "linux")) {
            return .linux;
        } else if (std.mem.eql(u8, self.os, "windows")) {
            return .windows;
        } else {
            return .other;
        }
    }
};

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
        std.debug.print("Usage: {s} <source_file> [--target <target_triplet>]\n", .{args[0]});
        std.debug.print("Example: {s} program.toy --target x86_64-pc-linux-gnu\n", .{args[0]});
        return;
    }
    
    var source_file: []const u8 = undefined;
    var target_triple: ?[]const u8 = null;
    
    // Parse arguments
    var i: usize = 1;
    while (i < args.len) {
        const arg = args[i];
        
        if (std.mem.eql(u8, arg, "--target")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: --target requires a target triplet\n", .{});
                return;
            }
            target_triple = args[i];
        } else if (std.mem.startsWith(u8, arg, "--")) {
            std.debug.print("Error: Unknown option: {s}\n", .{arg});
            return;
        } else {
            source_file = arg;
        }
        i += 1;
    }
    
    // Parse target information if provided
    var target_info: ?TargetInfo = null;
    defer if (target_info) |*info| info.deinit(allocator);
    
    if (target_triple) |triple| {
        target_info = TargetInfo.parseFromTriple(allocator, triple) catch |err| {
            std.debug.print("Error: Invalid target triplet '{s}': {}\n", .{ triple, err });
            return;
        };
        std.debug.print("Cross-compiling for target: {s} ({s}-{s})\n", .{ triple, target_info.?.arch, target_info.?.os });
    }
    
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
        const lexeme = source[token.offset .. token.offset + token.getLength()];
        std.debug.print("  {} '{s}' (offset: {})\n", .{ token.type, lexeme, token.offset });
    }
    
    // Parse
    var parse = parser.Parser.init(allocator, tokens, source);
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
    
    // Use LLVM directly for linking (no need for external linker detection)
    const actual_target_triple = if (target_triple) |triple| triple else null;
    
    if (has_main) {
        // Generate executable binary
        const bin_file = "output";
        try code_gen.generateExecutable(bin_file, actual_target_triple);
    } else {
        // Generate shared library
        const lib_file = "output";
        try code_gen.generateSharedLibrary(lib_file, actual_target_triple);
    }
    
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
    var parse = parser.Parser.init(allocator, tokens, source);
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


