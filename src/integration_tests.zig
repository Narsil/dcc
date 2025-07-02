const std = @import("std");
const builtin = @import("builtin");
const process = std.process;

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
    {
        const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{"./output"} });
        if (out.term.Exited != 16) {
            std.debug.print("Expected exit code 16, got {}\n", .{out.term.Exited});
            std.debug.print("stdout: {s}\n", .{out.stdout});
            std.debug.print("stderr: {s}\n", .{out.stderr});
            return error.UnexpectedExitCode;
        }
        std.debug.print("Compiled example.toy produced correct exit code: {}\n", .{out.term.Exited});
    }
}

test "compile library.toy and verify it works" {
    const allocator = std.testing.allocator;

    // Get the path to the dcc binary
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Test that dcc can compile library.toy
    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "library.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    if (out.term.Exited != 0) {
        std.debug.print("dcc compilation of library.toy failed with exit code {}\n", .{out.term.Exited});
        std.debug.print("stdout: {s}\n", .{out.stdout});
        std.debug.print("stderr: {s}\n", .{out.stderr});
        return error.CompilationFailed;
    }
    std.debug.print("dcc compilation of library.toy successful\n", .{});
}

test "type system - different integer types" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

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

    {
        const file = try std.fs.cwd().createFile("test_integers.toy", .{ .truncate = true });
        defer file.close();
        try file.writeAll(test_source);
    }
    defer std.fs.cwd().deleteFile("test_integers.toy") catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "test_integers.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    
    if (out.term.Exited != 0) {
        std.debug.print("Integer types test failed with exit code {}\n", .{out.term.Exited});
        std.debug.print("stdout: {s}\n", .{out.stdout});
        std.debug.print("stderr: {s}\n", .{out.stderr});
        return error.CompilationFailed;
    }
    
    std.debug.print("Integer types test passed\n", .{});
}

test "type system - floating point types" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Create a test file with floating point types
    const test_source = 
        \\fn main(): i64 {
        \\    let x: f32 = 3.14f32;
        \\    let y: f64 = 2.718281828f64;
        \\    return 42i64;
        \\}
    ;

    {
        const file = try std.fs.cwd().createFile("test_floats.toy", .{ .truncate = true });
        defer file.close();
        try file.writeAll(test_source);
    }
    defer std.fs.cwd().deleteFile("test_floats.toy") catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "test_floats.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    
    if (out.term.Exited != 0) {
        std.debug.print("Float types test failed with exit code {}\n", .{out.term.Exited});
        std.debug.print("stdout: {s}\n", .{out.stdout});
        std.debug.print("stderr: {s}\n", .{out.stderr});
        return error.CompilationFailed;
    }
    
    std.debug.print("Float types test passed\n", .{});
}

test "type system - function with typed parameters" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

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

    {
        const file = try std.fs.cwd().createFile("test_functions.toy", .{ .truncate = true });
        defer file.close();
        try file.writeAll(test_source);
    }
    defer std.fs.cwd().deleteFile("test_functions.toy") catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "test_functions.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    
    if (out.term.Exited != 0) {
        std.debug.print("Function types test failed with exit code {}\n", .{out.term.Exited});
        std.debug.print("stdout: {s}\n", .{out.stdout});
        std.debug.print("stderr: {s}\n", .{out.stderr});
        return error.CompilationFailed;
    }
    
    std.debug.print("Function types test passed\n", .{});
}

test "type system - missing parameter type annotation" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Create a test file with missing parameter type
    const test_source = 
        \\fn add(a, b: i64): i64 {
        \\    return a + b;
        \\}
        \\
        \\fn main(): i64 {
        \\    return 42i64;
        \\}
    ;

    {
        const file = try std.fs.cwd().createFile("test_missing_param_type.toy", .{ .truncate = true });
        defer file.close();
        try file.writeAll(test_source);
    }
    defer std.fs.cwd().deleteFile("test_missing_param_type.toy") catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "test_missing_param_type.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    
    if (out.term.Exited == 0) {
        std.debug.print("Expected compilation to fail due to missing parameter type\n", .{});
        return error.UnexpectedSuccess;
    }
    
    // Check that the error message is readable and mentions the missing type
    if (std.mem.indexOf(u8, out.stderr, "Expected ':' after parameter name") == null) {
        std.debug.print("Error message not found in stderr: {s}\n", .{out.stderr});
        return error.MissingError;
    }
    
    std.debug.print("Missing parameter type error test passed\n", .{});
    std.debug.print("Error message: {s}\n", .{out.stderr});
}

test "type system - missing return type annotation" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Create a test file with missing return type
    const test_source = 
        \\fn add(a: i64, b: i64) {
        \\    return a + b;
        \\}
        \\
        \\fn main(): i64 {
        \\    return 42i64;
        \\}
    ;

    {
        const file = try std.fs.cwd().createFile("test_missing_return_type.toy", .{ .truncate = true });
        defer file.close();
        try file.writeAll(test_source);
    }
    defer std.fs.cwd().deleteFile("test_missing_return_type.toy") catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "test_missing_return_type.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    
    if (out.term.Exited == 0) {
        std.debug.print("Expected compilation to fail due to missing return type\n", .{});
        return error.UnexpectedSuccess;
    }
    
    // Check that the error message is readable and mentions the missing return type
    if (std.mem.indexOf(u8, out.stderr, "Expected ':' before return type") == null) {
        std.debug.print("Error message not found in stderr: {s}\n", .{out.stderr});
        return error.MissingError;
    }
    
    std.debug.print("Missing return type error test passed\n", .{});
    std.debug.print("Error message: {s}\n", .{out.stderr});
}

test "type system - missing variable type annotation" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Create a test file with missing variable type
    const test_source = 
        \\fn main(): i64 {
        \\    let x = 42i64;
        \\    return x;
        \\}
    ;

    {
        const file = try std.fs.cwd().createFile("test_missing_var_type.toy", .{ .truncate = true });
        defer file.close();
        try file.writeAll(test_source);
    }
    defer std.fs.cwd().deleteFile("test_missing_var_type.toy") catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "test_missing_var_type.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    
    if (out.term.Exited == 0) {
        std.debug.print("Expected compilation to fail due to missing variable type\n", .{});
        return error.UnexpectedSuccess;
    }
    
    // Check that the error message is readable and mentions the missing type
    if (std.mem.indexOf(u8, out.stderr, "Expected ':' after variable name") == null) {
        std.debug.print("Error message not found in stderr: {s}\n", .{out.stderr});
        return error.MissingError;
    }
    
    std.debug.print("Missing variable type error test passed\n", .{});
    std.debug.print("Error message: {s}\n", .{out.stderr});
}

test "type system - invalid type annotation" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Create a test file with invalid type
    const test_source = 
        \\fn main(): i64 {
        \\    let x: invalid_type = 42i64;
        \\    return x;
        \\}
    ;

    {
        const file = try std.fs.cwd().createFile("test_invalid_type.toy", .{ .truncate = true });
        defer file.close();
        try file.writeAll(test_source);
    }
    defer std.fs.cwd().deleteFile("test_invalid_type.toy") catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "test_invalid_type.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    
    if (out.term.Exited == 0) {
        std.debug.print("Expected compilation to fail due to invalid type\n", .{});
        return error.UnexpectedSuccess;
    }
    
    // Check that the error message is readable and mentions the invalid type
    if (std.mem.indexOf(u8, out.stderr, "Expected type") == null) {
        std.debug.print("Error message not found in stderr: {s}\n", .{out.stderr});
        return error.MissingError;
    }
    
    std.debug.print("Invalid type error test passed\n", .{});
    std.debug.print("Error message: {s}\n", .{out.stderr});
}

test "type system - type mismatch in function call" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Create a test file with type mismatch
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

    {
        const file = try std.fs.cwd().createFile("test_type_mismatch.toy", .{ .truncate = true });
        defer file.close();
        try file.writeAll(test_source);
    }
    defer std.fs.cwd().deleteFile("test_type_mismatch.toy") catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "test_type_mismatch.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    
    if (out.term.Exited == 0) {
        std.debug.print("Expected compilation to fail due to type mismatch\n", .{});
        return error.UnexpectedSuccess;
    }
    
    // Check that the error message mentions type mismatch
    if (std.mem.indexOf(u8, out.stderr, "type mismatch") == null and 
        std.mem.indexOf(u8, out.stderr, "Type") == null) {
        std.debug.print("Type mismatch error message not found in stderr: {s}\n", .{out.stderr});
        return error.MissingError;
    }
    
    std.debug.print("Type mismatch error test passed\n", .{});
    std.debug.print("Error message: {s}\n", .{out.stderr});
}

test "type system - type mismatch in assignment" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Create a test file with type mismatch in assignment
    const test_source = 
        \\fn main(): i64 {
        \\    let x: i64 = 42i64;
        \\    let y: f64 = x;
        \\    return 0i64;
        \\}
    ;

    {
        const file = try std.fs.cwd().createFile("test_assignment_mismatch.toy", .{ .truncate = true });
        defer file.close();
        try file.writeAll(test_source);
    }
    defer std.fs.cwd().deleteFile("test_assignment_mismatch.toy") catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "test_assignment_mismatch.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    
    if (out.term.Exited == 0) {
        std.debug.print("Expected compilation to fail due to assignment type mismatch\n", .{});
        return error.UnexpectedSuccess;
    }
    
    std.debug.print("Assignment type mismatch error test passed\n", .{});
    std.debug.print("Error message: {s}\n", .{out.stderr});
}

test "type system - type mismatch in return statement" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Create a test file with type mismatch in return
    const test_source = 
        \\fn main(): i64 {
        \\    return 3.14f64;
        \\}
    ;

    {
        const file = try std.fs.cwd().createFile("test_return_mismatch.toy", .{ .truncate = true });
        defer file.close();
        try file.writeAll(test_source);
    }
    defer std.fs.cwd().deleteFile("test_return_mismatch.toy") catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "test_return_mismatch.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    
    if (out.term.Exited == 0) {
        std.debug.print("Expected compilation to fail due to return type mismatch\n", .{});
        return error.UnexpectedSuccess;
    }
    
    std.debug.print("Return type mismatch error test passed\n", .{});
    std.debug.print("Error message: {s}\n", .{out.stderr});
}

test "type system - type mismatch in binary expression" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Create a test file with type mismatch in binary expression
    const test_source = 
        \\fn main(): i64 {
        \\    let x: i64 = 42i64;
        \\    let y: f64 = 3.14f64;
        \\    let z: i64 = x + y;
        \\    return z;
        \\}
    ;

    {
        const file = try std.fs.cwd().createFile("test_binary_mismatch.toy", .{ .truncate = true });
        defer file.close();
        try file.writeAll(test_source);
    }
    defer std.fs.cwd().deleteFile("test_binary_mismatch.toy") catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "test_binary_mismatch.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    
    if (out.term.Exited == 0) {
        std.debug.print("Expected compilation to fail due to binary expression type mismatch\n", .{});
        return error.UnexpectedSuccess;
    }
    
    std.debug.print("Binary expression type mismatch error test passed\n", .{});
    std.debug.print("Error message: {s}\n", .{out.stderr});
}

test "type system - integer type mismatches" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Create a test file with integer type mismatches
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

    {
        const file = try std.fs.cwd().createFile("test_integer_mismatch.toy", .{ .truncate = true });
        defer file.close();
        try file.writeAll(test_source);
    }
    defer std.fs.cwd().deleteFile("test_integer_mismatch.toy") catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "test_integer_mismatch.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    
    if (out.term.Exited == 0) {
        std.debug.print("Expected compilation to fail due to integer type mismatch\n", .{});
        return error.UnexpectedSuccess;
    }
    
    std.debug.print("Integer type mismatch error test passed\n", .{});
    std.debug.print("Error message: {s}\n", .{out.stderr});
}

test "type system - signed vs unsigned mismatch" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Create a test file with signed vs unsigned mismatch
    const test_source = 
        \\fn main(): i64 {
        \\    let x: u64 = 42u64;
        \\    let y: i64 = -10i64;
        \\    let z: u64 = x + y;
        \\    return 0i64;
        \\}
    ;

    {
        const file = try std.fs.cwd().createFile("test_signed_unsigned.toy", .{ .truncate = true });
        defer file.close();
        try file.writeAll(test_source);
    }
    defer std.fs.cwd().deleteFile("test_signed_unsigned.toy") catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "test_signed_unsigned.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    
    if (out.term.Exited == 0) {
        std.debug.print("Expected compilation to fail due to signed/unsigned type mismatch\n", .{});
        return error.UnexpectedSuccess;
    }
    
    std.debug.print("Signed/unsigned type mismatch error test passed\n", .{});
    std.debug.print("Error message: {s}\n", .{out.stderr});
}

