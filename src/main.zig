//! By convention, main.zig is where your main function lives when building
//! an executable. This is the main entry point for the toy compiler.

const std = @import("std");
const lexer = @import("lexer.zig");
const parser = @import("parser.zig");
const typechecker = @import("typechecker.zig");
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
        std.debug.print("Usage: {s} <source_file> [--target <target_triplet>] [--verbose] [--gpu <gpu_triplet>]\n", .{args[0]});
        std.debug.print("Example: {s} program.toy --target x86_64-pc-linux-gnu --verbose --gpu nvidia-ptx-sm50\n", .{args[0]});
        return;
    }

    var source_file: []const u8 = undefined;
    var target_triple: ?[]const u8 = null;
    var verbose: bool = false;
    var gpu_triplet: ?[]const u8 = null;

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
        } else if (std.mem.eql(u8, arg, "--verbose")) {
            verbose = true;
        } else if (std.mem.eql(u8, arg, "--gpu")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: --gpu requires a GPU triplet (e.g., nvidia-ptx-sm50)\n", .{});
                return;
            }
            gpu_triplet = args[i];
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
        if (verbose) {
            std.debug.print("Cross-compiling for target: {s} ({s}-{s})\n", .{ triple, target_info.?.arch, target_info.?.os });
        }
    }

    // Wrap compilation in catch block to handle normal errors gracefully
    const result = compile(allocator, source_file, target_triple, verbose, gpu_triplet);
    if (result) |_| {
        // Compilation successful
    } else |err| {
        // Handle normal compilation errors without stack trace
        switch (err) {
            error.FileNotFound => {
                std.debug.print("Error: File '{s}' not found\n", .{source_file});
                std.process.exit(1);
            },
            error.AccessDenied => {
                std.debug.print("Error: Access denied reading file '{s}'\n", .{source_file});
                std.process.exit(1);
            },
            error.OutOfMemory => {
                std.debug.print("Error: Out of memory\n", .{});
                std.process.exit(1);
            },
            // Parser errors (normal compilation errors)
            parser.ParseError.ParseError => {
                // Parser already prints the error message, just exit
                std.process.exit(1);
            },
            parser.ParseError.InvalidCharacter => {
                std.debug.print("Error: Invalid character in source\n", .{});
                std.process.exit(1);
            },
            parser.ParseError.Overflow => {
                std.debug.print("Error: Integer overflow\n", .{});
                std.process.exit(1);
            },
            parser.ParseError.InvalidUtf8 => {
                std.debug.print("Error: Invalid UTF-8 in source file\n", .{});
                std.process.exit(1);
            },
            // Code generator errors (normal compilation errors)
            codegen.CodeGenError.InvalidTopLevelNode => {
                std.debug.print("Error: Invalid top-level node\n", .{});
                std.process.exit(1);
            },
            codegen.CodeGenError.InvalidStatement => {
                std.debug.print("Error: Invalid statement\n", .{});
                std.process.exit(1);
            },
            codegen.CodeGenError.InvalidExpression => {
                std.debug.print("Error: Invalid expression\n", .{});
                std.process.exit(1);
            },
            codegen.CodeGenError.InvalidCallee => {
                std.debug.print("Error: Invalid function call\n", .{});
                std.process.exit(1);
            },
            codegen.CodeGenError.UndefinedVariable => {
                std.debug.print("Error: Undefined variable\n", .{});
                std.process.exit(1);
            },
            codegen.CodeGenError.UndefinedFunction => {
                std.debug.print("Error: Undefined function\n", .{});
                std.process.exit(1);
            },
            codegen.CodeGenError.TargetError => {
                std.debug.print("Error: Target machine error\n", .{});
                std.process.exit(1);
            },
            codegen.CodeGenError.CodeGenError => {
                std.debug.print("Error: Code generation error\n", .{});
                std.process.exit(1);
            },
            codegen.CodeGenError.MainFunctionNotFound => {
                std.debug.print("Error: Main function not found\n", .{});
                std.process.exit(1);
            },
            codegen.CodeGenError.LinkingFailed => {
                std.debug.print("Error: Linking failed\n", .{});
                std.process.exit(1);
            },
            codegen.CodeGenError.GpuCompilationNotImplemented => {
                std.debug.print("Error: GPU compilation not implemented\n", .{});
                std.process.exit(1);
            },
            codegen.CodeGenError.InvalidGpuTriplet => {
                std.debug.print("Error: Invalid GPU triplet format\n", .{});
                std.process.exit(1);
            },
            codegen.CodeGenError.CudaFunctionNotFound => {
                std.debug.print("Error: CUDA function not found in module\n", .{});
                std.process.exit(1);
            },
            // Type checker errors (normal compilation errors)
            typechecker.TypeCheckError.TypeMismatch => {
                // Type checker already prints the error message, just exit
                std.process.exit(1);
            },

            typechecker.TypeCheckError.InvalidBinaryOperation => {
                // Type checker already prints the error message, just exit
                std.process.exit(1);
            },
            typechecker.TypeCheckError.InvalidUnaryOperation => {
                // Type checker already prints the error message, just exit
                std.process.exit(1);
            },
            typechecker.TypeCheckError.InvalidFunctionCall => {
                // Type checker already prints the error message, just exit
                std.process.exit(1);
            },
            typechecker.TypeCheckError.InvalidReturnType => {
                // Type checker already prints the error message, just exit
                std.process.exit(1);
            },
            typechecker.TypeCheckError.InvalidVariableType => {
                // Type checker already prints the error message, just exit
                std.process.exit(1);
            },
            typechecker.TypeCheckError.InvalidMainFunctionReturnType => {
                // Type checker already prints the error message, just exit
                std.process.exit(1);
            },
            else => {
                // For any other errors, re-raise to get stack trace (unhandled errors)
                return err;
            },
        }
    }
}

fn compile(allocator: std.mem.Allocator, source_file: []const u8, target_triple: ?[]const u8, verbose: bool, gpu_triplet: ?[]const u8) !void {
    if (verbose) {
        std.debug.print("Compiling: {s}\n", .{source_file});
    }

    // Read source file
    const source = try std.fs.cwd().readFileAlloc(allocator, source_file, 1024 * 1024);
    defer allocator.free(source);

    if (verbose) {
        std.debug.print("Source code:\n{s}\n", .{source});
    }

    // Tokenize
    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);

    if (verbose) {
        std.debug.print("Tokens:\n", .{});
        for (tokens) |token| {
            const lexeme = source[token.offset .. token.offset + token.getLength()];
            std.debug.print("  {} '{s}' (offset: {})\n", .{ token.type, lexeme, token.offset });
        }
    }

    // Parse
    var parse = parser.Parser.init(allocator, tokens, source, verbose);
    const ast = try parse.parse();
    defer parser.freeAST(allocator, ast);

    if (verbose) {
        std.debug.print("AST parsed successfully!\n", .{});
    }

    // Type check
    var type_checker = typechecker.TypeChecker.init(allocator, source, verbose);
    defer type_checker.deinit();
    try type_checker.typeCheck(ast);

    if (verbose) {
        std.debug.print("Type checking passed!\n", .{});
    }

    // Check if there's a main function
    const has_main = hasMainFunction(ast);

    // Generate LLVM IR using C bindings
    if (verbose) {
        std.debug.print("CodeGen\n", .{});
    }
    var code_gen = try codegen.CodeGen.init(allocator, "toy_program", verbose, gpu_triplet);
    defer code_gen.deinit();

    // Choose compilation mode based on presence of main function
    if (has_main) {
        try code_gen.generateWithMode(ast, .executable);
    } else {
        try code_gen.generateWithMode(ast, .library);
    }

    if (verbose) {
        code_gen.printIR();
    }

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

    if (verbose) {
        std.debug.print("Compilation complete!\n", .{});
    }
}
