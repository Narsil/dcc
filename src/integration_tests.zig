const std = @import("std");
const builtin = @import("builtin");
const process = std.process;

fn assertCompiles(allocator: std.mem.Allocator, source: []const u8, filename: []const u8) !void {
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";
    {
        const file = try std.fs.cwd().createFile(filename, .{ .truncate = true });
        defer file.close();
        try file.writeAll(source);
    }
    defer std.fs.cwd().deleteFile(filename) catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, filename } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);

    switch (out.term) {
        .Exited => |code| {
            if (code != 0) {
                std.debug.print("{s} failed with exit code {}\n", .{ filename, code });
                std.debug.print("stdout: {s}\n", .{out.stdout});
                std.debug.print("stderr: {s}\n", .{out.stderr});
                return error.CompilationFailed;
            }
        },
        else => {
            std.debug.print("Integer types test terminated abnormally\n", .{});
            return error.CompilationFailed;
        },
    }
}

fn assertReturns(allocator: std.mem.Allocator, expected: usize) !void {
    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{"./output"} });
    switch (out.term) {
        .Exited => |term| {
            if (term != expected) {
                std.debug.print("Expected exit code {}, got {}\n", .{ expected, term });
                std.debug.print("stdout: {s}\n", .{out.stdout});
                std.debug.print("stderr: {s}\n", .{out.stderr});
                return error.UnexpectedExitCode;
            } else {
                std.debug.print("Compiled test_tensor.toy produced correct exit code: {}\n", .{out.term.Exited});
            }
        },
        else => {
            return error.UnexpectedExitCode;
        },
    }
}

// Assert that compilation fails and stderr contains the expected substring (if provided)
fn assertCompileFailure(
    allocator: std.mem.Allocator,
    source: []const u8,
    filename: []const u8,
    expected_substr: []const u8,
    expected_line: []const u8,
) !void {
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Write the temporary source file
    {
        const file = try std.fs.cwd().createFile(filename, .{ .truncate = true });
        defer file.close();
        try file.writeAll(source);
    }
    defer std.fs.cwd().deleteFile(filename) catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, filename } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);

    // Expect non-zero exit status
    switch (out.term) {
        .Exited => |code| {
            if (code == 0) return error.UnexpectedSuccess;
        },
        else => {},
    }

    // Extract and validate "Error at line X, column Y" header
    const header_start = std.mem.indexOf(u8, out.stderr, "Error at line") orelse {
        std.debug.print("Line/column header missing in stderr: {s}\n", .{out.stderr});
        return error.MissingError;
    };

    // Slice header until newline
    const newline_pos_opt = std.mem.indexOfPos(u8, out.stderr, header_start, "\n") orelse out.stderr.len;
    const header_line = out.stderr[header_start..newline_pos_opt];

    // Parse numbers from header
    var it = std.mem.splitSequence(u8, header_line, " ");
    _ = it.next(); // "Error"
    _ = it.next(); // "at"
    _ = it.next(); // "line"
    const line_str = it.next() orelse ""; // "X,"
    const line_num = std.fmt.parseInt(u32, line_str[0 .. line_str.len - 1], 10) catch 0; // remove trailing comma
    _ = it.next(); // "column"
    const col_str = it.next() orelse ""; // "Y:"
    const col_num = std.fmt.parseInt(u32, col_str[0 .. col_str.len - 1], 10) catch 0;

    // Compute expected line number from the provided source
    var expected_line_number: u32 = 0;
    var current: u32 = 1;
    var idx: usize = 0;
    while (idx < source.len) : (idx += 1) {
        // Find start of each line
        const line_start = idx;
        // Move to end of line
        while (idx < source.len and source[idx] != '\n') : (idx += 1) {}
        const line_slice = source[line_start..idx];
        if (std.mem.eql(u8, line_slice, expected_line)) {
            expected_line_number = current;
            break;
        }
        current += 1;
    }

    if (expected_line_number == 0 or expected_line_number != line_num) {
        std.debug.print("Expected error on line {}, got {}. Stderr: {s}\n", .{ expected_line_number, line_num, out.stderr });
        return error.MissingError;
    }

    // Ensure the full source line appears in stderr with leading two spaces
    const formatted_line = try std.fmt.allocPrint(allocator, "  {s}", .{expected_line});
    defer allocator.free(formatted_line);

    if (std.mem.indexOf(u8, out.stderr, formatted_line) == null) {
        std.debug.print("Expected source line not found in stderr. Looking for '{s}'. Stderr: {s}\n", .{ formatted_line, out.stderr });
        return error.MissingError;
    }

    // Verify caret alignment matches column number
    const caret_line_start = std.mem.indexOf(u8, out.stderr, "^") orelse {
        std.debug.print("Caret '^' not found in stderr: {s}\n", .{out.stderr});
        return error.MissingError;
    };
    // Count spaces before caret by scanning backwards to preceding newline
    var tmp_idx: usize = caret_line_start;
    while (tmp_idx > 0 and out.stderr[tmp_idx - 1] != '\n') : (tmp_idx -= 1) {}
    const spaces = caret_line_start - tmp_idx - 2; // exclude the two leading indent spaces
    if (spaces + 1 != col_num) {
        std.debug.print("Caret column {} does not match parsed column {}\n", .{ spaces + 1, col_num });
        return error.MissingError;
    }

    // If caller supplied a substring, ensure it's present in stderr
    if (expected_substr.len > 0 and std.mem.indexOf(u8, out.stderr, expected_substr) == null) {
        std.debug.print("Expected substring '{s}' not found in stderr: {s}\n", .{ expected_substr, out.stderr });
        return error.MissingError;
    }
}

test "compile example.toy and verify exit code" {
    const allocator = std.testing.allocator;

    // Get the path to the dcc binary
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Test that dcc can compile example.toy
    {
        const run = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "example.toy" } });
        defer allocator.free(run.stdout);
        defer allocator.free(run.stderr);
        std.debug.print("dcc compilation successful\n", .{});
    }

    // Test that the compiled binary produces exit code 16
    try assertReturns(allocator, 16);
}

test "compile library.toy and verify it works" {
    const allocator = std.testing.allocator;

    // Get the path to the dcc binary
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Test that dcc can compile library.toy
    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "library.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    switch (out.term) {
        .Exited => |code| {
            if (code != 0) {
                std.debug.print("dcc compilation of library.toy failed with exit code {}\n", .{code});
                std.debug.print("stdout: {s}\n", .{out.stdout});
                std.debug.print("stderr: {s}\n", .{out.stderr});
                return error.CompilationFailed;
            }
        },
        else => {
            std.debug.print("dcc compilation of library.toy terminated abnormally\n", .{});
            return error.CompilationFailed;
        },
    }
    std.debug.print("dcc compilation of library.toy successful\n", .{});
}

test "type system - different integer types" {
    const allocator = std.testing.allocator;
    // Create a test file with different integer types
    const test_source =
        \\fn main(): i64 {
        \\    let x: u8 = 255u8;
        \\    let y: u16 = 65535u16;
        \\    let z: u32 = 4294967295u32;
        \\    let a: i8 = -128i8;
        \\    let b: i16 = -32768i16;
        \\    let c: i32 = -2147483648i32;
        \\    return 42i64;
        \\}
    ;
    try assertCompiles(allocator, test_source, "test_integers.toy");
    try assertReturns(allocator, 42);

    std.debug.print("Integer types test passed\n", .{});
}

test "type system - floating point types" {
    const allocator = std.testing.allocator;

    // Create a test file with floating point types
    const test_source =
        \\fn main(): i64 {
        \\    let x: f32 = 3.14f32;
        \\    let y: f64 = 2.718281828f64;
        \\    return 42i64;
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_floats.toy");
    try assertReturns(allocator, 42);

    std.debug.print("Float types test passed\n", .{});
}

test "type system - function with typed parameters" {
    const allocator = std.testing.allocator;

    // Create a test file with typed function parameters
    const test_source =
        \\fn add_u8(a: u8, b: u8): u8 {
        \\    return a + b;
        \\}
        \\
        \\fn add_i32(a: i32, b: i32): i32 {
        \\    return a + b;
        \\}
        \\
        \\fn add_f64(a: f64, b: f64): f64 {
        \\    return a + b;
        \\}
        \\
        \\fn main(): i64 {
        \\    let x: u8 = add_u8(1u8, 2u8);
        \\    let y: i32 = add_i32(10i32, 20i32);
        \\    let z: f64 = add_f64(1.5f64, 2.5f64);
        \\    return 42i64;
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_functions.toy");
    try assertReturns(allocator, 42);

    std.debug.print("Function types test passed\n", .{});
}

test "type system - missing parameter type annotation" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn add(a, b: i64): i64 {
        \\    return a + b;
        \\}
        \\
        \\fn main(): i64 {
        \\    return 42i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_missing_param_type.toy", "Expected ':' after parameter name", "fn add(a, b: i64): i64 {");

    std.debug.print("Missing parameter type error test passed\n", .{});
}

test "type system - missing return type annotation" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn add(a: i64, b: i64) {
        \\    return a + b;
        \\}
        \\
        \\fn main(): i64 {
        \\    return 42i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_missing_return_type.toy", "Expected ':' before return type", "fn add(a: i64, b: i64) {");

    std.debug.print("Missing return type error test passed\n", .{});
}

test "type system - missing variable type annotation" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn main(): i64 {
        \\    let x = 42i64;
        \\    return x;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_missing_var_type.toy", "Expected ':' after variable name", "    let x = 42i64;");

    std.debug.print("Missing variable type error test passed\n", .{});
}

test "type system - invalid type annotation" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn main(): i64 {
        \\    let x: invalid_type = 42i64;
        \\    return x;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_invalid_type.toy", "Expected type", "    let x: invalid_type = 42i64;");

    std.debug.print("Invalid type error test passed\n", .{});
}

test "type system - type mismatch in function call" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn add(a: i64, b: i64): i64 {
        \\    return a + b;
        \\}
        \\
        \\fn main(): i64 {
        \\    let x: f64 = 3.14f64;
        \\    let result: i64 = add(x, 5i64);
        \\    return result;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_type_mismatch.toy", "Argument 1 of function 'add' expects type parser.Type{ .i64 = void }, got parser.Type{ .f64 = void }", "    let result: i64 = add(x, 5i64);");

    std.debug.print("Type mismatch error test passed\n", .{});
}

test "type system - type mismatch in assignment" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn main(): i64 {
        \\    let x: i64 = 42i64;
        \\    let y: f64 = x;
        \\    return 0i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_assignment_mismatch.toy", "Cannot assign value of type parser.Type{ .i64 = void } to variable of type parser.Type{ .f64 = void }", "    let y: f64 = x;");

    std.debug.print("Assignment type mismatch error test passed\n", .{});
}

test "type system - type mismatch in return statement" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn main(): i64 {
        \\    return 3.14f64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_return_mismatch.toy", "Cannot return value of type parser.Type{ .f64 = void } from function returning parser.Type{ .i64 = void }", "    return 3.14f64;");

    std.debug.print("Return type mismatch error test passed\n", .{});
}

test "type system - type mismatch in binary expression" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn main(): i64 {
        \\    let x: i64 = 42i64;
        \\    let y: f64 = 3.14f64;
        \\    let z: i64 = x + y;
        \\    return z;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_binary_mismatch.toy", "Cannot perform parser.BinaryOperator.add operation on types parser.Type{ .i64 = void } and parser.Type{ .f64 = void }", "    let z: i64 = x + y;");

    std.debug.print("Binary expression type mismatch error test passed\n", .{});
}

test "type system - integer type mismatches" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn add_u8(a: u8, b: u8): u8 {
        \\    return a + b;
        \\}
        \\
        \\fn main(): i64 {
        \\    let x: u8 = 255u8;
        \\    let y: i64 = 42i64;
        \\    let result: u8 = add_u8(x, y);
        \\    return 0i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_integer_mismatch.toy", "Argument 2 of function 'add_u8' expects type parser.Type{ .u8 = void }, got parser.Type{ .i64 = void }", "    let result: u8 = add_u8(x, y);");

    std.debug.print("Integer type mismatch error test passed\n", .{});
}

test "type system - signed vs unsigned mismatch" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn main(): i64 {
        \\    let x: u64 = 42u64;
        \\    let y: i64 = -10i64;
        \\    let z: u64 = x + y;
        \\    return 0i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_signed_unsigned.toy", "Cannot perform parser.BinaryOperator.add operation on types parser.Type{ .u64 = void } and parser.Type{ .i64 = void }", "    let z: u64 = x + y;");

    std.debug.print("Signed/unsigned type mismatch error test passed\n", .{});
}

test "type system - tensor simple" {
    const allocator = std.testing.allocator;

    // Test tensor operations: vector initialization, element-wise addition, and indexing
    const test_source =
        \\ fn main(): u32 {
        \\     // Create a vector of 5 elements initialized to zero
        \\     let vector: [5]u32 = [5]u32{0u32};
        \\     
        \\     // Add 1 to all elements (implicit parallelization)
        \\     vector[i] = vector[i] + 1u32;
        \\     
        \\     // Return the first element (should be 1)
        \\     return vector[0];
        \\ }
    ;

    try assertCompiles(allocator, test_source, "test_tensor.toy");
    try assertReturns(allocator, 1);

    std.debug.print("Tensor basic test passed\n", .{});
}
