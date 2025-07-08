//! By convention, main.zig is where your main function lives when building
//! an executable. This is the main entry point for the toy compiler.

const std = @import("std");
const builtin = @import("builtin");
const lexer = @import("lexer.zig");
const parser = @import("parser.zig");
const typechecker = @import("typechecker.zig");
const codegen = @import("codegen.zig");

/// Helper function to convert std.Target to target triple string using Zig's built-in functionality
fn targetToTriple(allocator: std.mem.Allocator, target: std.Target) ![]u8 {
    const query = std.Target.Query.fromTarget(target);
    return query.zigTriple(allocator);
}

/// Helper function to parse target triple string to std.Target using Zig's built-in parsing
fn parseTargetFromTriple(triple: []const u8) !std.Target {
    const query = std.Target.Query.parse(.{ .arch_os_abi = triple }) catch return error.InvalidTargetTriple;
    return std.zig.system.resolveTargetQuery(query) catch return error.InvalidTargetTriple;
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
        std.debug.print("Example: {s} program.toy --target x86_64-pc-linux-gnu --verbose --gpu nvptx-cuda:sm_50\n", .{args[0]});
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
                std.debug.print("Error: --gpu requires a GPU triplet (e.g., nvptx-cuda:sm_50)\n", .{});
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

    const gpu_target: ?std.Target = if (gpu_triplet) |gpu| blk: {
        // Check target OS first - GPU compilation is only supported on Linux targets
        if (target.os.tag == .macos) {
            std.debug.print("Error: NVIDIA GPU compilation is not supported on macOS targets\n", .{});
            std.debug.print("NVIDIA GPUs are not available on macOS. GPU compilation is only supported on Linux targets.\n", .{});
            return error.GpuNotSupportedOnMacOS;
        }
        
        var iter = std.mem.splitScalar(u8, gpu, ':');
        const arch_os_abi = iter.next() orelse return error.InvalidGpuTriplet;
        const cpu_features = iter.next() orelse {
            std.debug.print("Invalid GPU triplet: {s}. Expected format: nvptx64-cuda:sm_XX", .{gpu});
            return error.InvalidGpuTriplet;
        };

        const query = std.Target.Query.parse(.{ .arch_os_abi = arch_os_abi, .cpu_features = cpu_features }) catch return error.InvalidTargetTriple;
        const parsed_target = std.zig.system.resolveTargetQuery(query) catch return error.InvalidTargetTriple;
        if (parsed_target.os.tag != .cuda) {
            std.debug.print("Only cuda is supported for now not : {s}", .{gpu});
            return error.InvalidGpuTriplet;
        }
        if (verbose) {
            std.debug.print("Cross-compiling for target: {s} ({s}-{s})\n", .{ gpu, @tagName(parsed_target.cpu.arch), @tagName(parsed_target.os.tag) });
        }
        break :blk parsed_target;
    } else null;

    // Wrap compilation in catch block to handle normal errors gracefully
    const result = compile(allocator, source_file, target, verbose, gpu_target);
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
            error.GpuNotSupportedOnMacOS => {
                // Error already printed in main
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

fn compile(allocator: std.mem.Allocator, source_file: []const u8, target: std.Target, verbose: bool, gpu_target: ?std.Target) anyerror!void {
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

    var code_gen = try codegen.CodeGen.init(allocator, "toy_program", verbose, target, gpu_target);
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
