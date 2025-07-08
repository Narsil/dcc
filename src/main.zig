//! By convention, main.zig is where your main function lives when building
//! an executable. This is the main entry point for the toy compiler.

const std = @import("std");
const builtin = @import("builtin");
const lexer = @import("lexer.zig");
const parser = @import("parser.zig");
const typechecker = @import("typechecker.zig");
const codegen = @import("codegen.zig");

/// Helper function to parse target triple string to std.Target
fn parseTargetFromTriple(triple: []const u8) !std.Target {
    var parts = std.mem.splitSequence(u8, triple, "-");

    const arch_str = parts.next() orelse return error.InvalidTargetTriple;
    const vendor_or_os = parts.next() orelse return error.InvalidTargetTriple;

    // Parse architecture
    const arch = if (std.mem.eql(u8, arch_str, "x86_64"))
        std.Target.Cpu.Arch.x86_64
    else if (std.mem.eql(u8, arch_str, "aarch64") or std.mem.eql(u8, arch_str, "arm64"))
        std.Target.Cpu.Arch.aarch64
    else if (std.mem.eql(u8, arch_str, "arm"))
        std.Target.Cpu.Arch.arm
    else if (std.mem.eql(u8, arch_str, "riscv64"))
        std.Target.Cpu.Arch.riscv64
    else
        return error.InvalidTargetTriple;

    // Parse OS
    var os_tag: std.Target.Os.Tag = undefined;
    if (std.mem.eql(u8, vendor_or_os, "apple")) {
        const os_part = parts.next() orelse return error.InvalidTargetTriple;
        if (std.mem.eql(u8, os_part, "darwin") or std.mem.eql(u8, os_part, "macos")) {
            os_tag = .macos;
        } else {
            return error.InvalidTargetTriple;
        }
    } else if (std.mem.eql(u8, vendor_or_os, "pc")) {
        const os_part = parts.next() orelse return error.InvalidTargetTriple;
        if (std.mem.eql(u8, os_part, "windows")) {
            os_tag = .windows;
        } else if (std.mem.eql(u8, os_part, "linux")) {
            os_tag = .linux;
        } else {
            return error.InvalidTargetTriple;
        }
    } else if (std.mem.eql(u8, vendor_or_os, "unknown")) {
        const os_part = parts.next() orelse return error.InvalidTargetTriple;
        if (std.mem.eql(u8, os_part, "linux")) {
            os_tag = .linux;
        } else {
            return error.InvalidTargetTriple;
        }
    } else if (std.mem.eql(u8, vendor_or_os, "linux")) {
        os_tag = .linux;
    } else {
        return error.InvalidTargetTriple;
    }

    // Create a simple target based on parsed components
    var target = builtin.target;
    target.os = std.Target.Os{ .tag = os_tag, .version_range = target.os.version_range };
    target.cpu = std.Target.Cpu.baseline(arch, target.os);
    target.abi = std.Target.Abi.default(arch, target.os);
    return target;
}

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
    const target = if (target_triple) |triple| blk: {
        const parsed_target = parseTargetFromTriple(triple) catch |err| {
            std.debug.print("Error: Invalid target triplet '{s}': {}\n", .{ triple, err });
            return;
        };
        if (verbose) {
            std.debug.print("Cross-compiling for target: {s} ({s}-{s})\n", .{ triple, @tagName(parsed_target.cpu.arch), @tagName(parsed_target.os.tag) });
        }
        break :blk parsed_target;
    } else blk: {
        if (verbose) {
            std.debug.print("Defaulting to target: {s}-{s}\n", .{ @tagName(builtin.target.cpu.arch), @tagName(builtin.target.os.tag) });
        }
        break :blk builtin.target;
    };

    // Wrap compilation in catch block to handle normal errors gracefully
    const result = compile(allocator, source_file, target, verbose, gpu_triplet);
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
            codegen.CodeGenError.InvalidTargetTriple => {
                std.debug.print("Error: Invalid target triplet format\n", .{});
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

fn compile(allocator: std.mem.Allocator, source_file: []const u8, target: std.Target, verbose: bool, gpu_triplet: ?[]const u8) !void {
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

    var code_gen = try codegen.CodeGen.init(allocator, "toy_program", verbose, target, gpu_triplet);
    defer code_gen.deinit();

    // Extract output name from source file (remove .toy extension)
    const basename = std.fs.path.basename(source_file);
    const output_name = if (std.mem.endsWith(u8, basename, ".toy"))
        basename[0 .. basename.len - 4]
    else
        basename;

    // Choose compilation mode based on presence of main function
    if (has_main) {
        try code_gen.generateWithMode(ast, .executable);
        // Generate executable binary
        try code_gen.generateExecutable(output_name, target);
    } else {
        try code_gen.generateWithMode(ast, .library);
        // Generate shared library
        try code_gen.generateSharedLibrary(output_name, target);
    }

    if (verbose) {
        code_gen.printIR();
    }

    if (verbose) {
        std.debug.print("Compilation complete!\n", .{});
    }
}
