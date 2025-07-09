const std = @import("std");
const builtin = @import("builtin");
const process = std.process;

test "type system - unused variable error" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i64 {
        \\    let x: i32 = 42i32;
        \\    return 0i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_unused_variable.toy", "Variable 'x' is declared but never used", "    let x: i32 = 42i32;");

    std.debug.print("Unused variable error test passed\n", .{});
}

test "io stdout - simple" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i32 {
        \\    write(io.stdout, "Hello word!");
        \\    return 0i32;
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_stdout.toy");
    try assertExpectStdout(allocator, "test_stdout.toy", "Hello world!", 0);

    std.debug.print("stdout\n", .{});
}

test "io stdout - escape chars" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i32 {
        \\    write(io.stdout, "Hello word!\n");
        \\    return 0i32;
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_stdout_escape.toy");
    try assertExpectStdout(allocator, "test_stdout_escape.toy", "Hello world!\n", 0);

    std.debug.print("stdout with escaped", .{});
}

test "type system - used variable compiles" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i64 {
        \\    let x: i64 = 42i64;
        \\    let y: i64 = x + 1i64;
        \\    return y;
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_used_variable.toy");
    try assertReturns(allocator, "test_used_variable.toy", 43);

    std.debug.print("Used variable test passed\n", .{});
}

test "function visibility - main must be public" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn main() i64 {
        \\    return 0i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_private_main.toy", "Function 'main' must be declared public", "fn main() i64 {");

    std.debug.print("Main must be public error test passed\n", .{});
}

test "function visibility - unused private function error" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn helper() i64 {
        \\    return 42i64;
        \\}
        \\
        \\pub fn main() i64 {
        \\    return 0i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_unused_function.toy", "Function 'helper' is declared but never used", "fn helper() i64 {");

    std.debug.print("Unused function error test passed\n", .{});
}

test "function visibility - used private function compiles" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn helper() i64 {
        \\    return 42i64;
        \\}
        \\
        \\pub fn main() i64 {
        \\    return helper();
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_used_private_function.toy");
    try assertReturns(allocator, "test_used_private_function.toy", 42);

    std.debug.print("Used private function test passed\n", .{});
}

test "function visibility - public function always compiles" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn helper() i64 {
        \\    return 42i64;
        \\}
        \\
        \\pub fn main() i64 {
        \\    return 0i64;
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_public_function.toy");
    try assertReturns(allocator, "test_public_function.toy", 0);

    std.debug.print("Public function test passed\n", .{});
}

test "type system - different integer types" {
    const allocator = std.testing.allocator;
    // Create a test file with different integer types
    const test_source =
        \\pub fn main() i32 {
        \\    let a: i32 = 255i32;
        \\    return a;
        \\}
    ;
    try assertCompiles(allocator, test_source, "test_integers.toy");
    try assertReturns(allocator, "test_integers.toy", 255);

    std.debug.print("Integer types test passed\n", .{});
}

// Disabled while we don't have a good way to produce a side effect with f32
// test "type system - floating point types" {
//     const allocator = std.testing.allocator;
//
//     // Create a test file with floating point types
//     const test_source =
//         \\pub fn main() i64 {
//         \\    let x: f32 = 3.14f32;
//         \\    let y: f64 = 2.718281828f64;
//         \\    // Use the variables to avoid unused variable error
//         \\    // Convert f32 to f64 before adding
//         \\    let sum: f64 = (x as f64) + y;
//         \\    return 42i64;
//         \\}
//     ;
//
//     try assertCompiles(allocator, test_source, "test_floats.toy");
//     try assertReturns(allocator, "test_floats.toy", 42);
//
//     std.debug.print("Float types test passed\n", .{});
// }

test "type system - function with typed parameters" {
    const allocator = std.testing.allocator;

    // Create a test file with typed function parameters
    const test_source =
        \\fn add_i32(a: i32, b: i32) i32 {
        \\    return a + b;
        \\}
        \\pub fn main() i32 {
        \\    let y: i32 = add_i32(10i32, 20i32);
        \\    return y;
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_functions.toy");
    try assertReturns(allocator, "test_functions.toy", 30);

    std.debug.print("Function types test passed\n", .{});
}

test "type system - missing parameter type annotation" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn add(a, b: i64) i64 {
        \\    return a + b;
        \\}
        \\
        \\pub fn main() i64 {
        \\    return 42i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_missing_param_type.toy", "Expected ':' after parameter name", "fn add(a, b: i64) i64 {");

    std.debug.print("Missing parameter type error test passed\n", .{});
}

test "type system - missing return type annotation" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn add(a: i64, b: i64) {
        \\    return a + b;
        \\}
        \\
        \\pub fn main() i64 {
        \\    return 42i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_missing_return_type.toy", "Error at line 1, column 24: Expected type", "fn add(a: i64, b: i64) {");

    std.debug.print("Missing return type error test passed\n", .{});
}

test "type system - missing variable type annotation" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i64 {
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
        \\pub fn main() i64 {
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
        \\fn add(a: i64, b: i64) i64 {
        \\    return a + b;
        \\}
        \\
        \\pub fn main() i64 {
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
        \\pub fn main() i64 {
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
        \\pub fn main() i64 {
        \\    return 3.14f64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_return_mismatch.toy", "Cannot return value of type parser.Type{ .f64 = void } from function returning parser.Type{ .i64 = void }", "    return 3.14f64;");

    std.debug.print("Return type mismatch error test passed\n", .{});
}

test "type system - type mismatch in binary expression" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i64 {
        \\    let x: i64 = 42i64;
        \\    let y: f64 = 3.14f64;
        \\    let z: i64 = x + y;
        \\    return z;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_binary_mismatch.toy", "Cannot perform parser.BinaryOperator.add operation on types i64 and f64", "    let z: i64 = x + y;");

    std.debug.print("Binary expression type mismatch error test passed\n", .{});
}

test "type system - integer type mismatches" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn add_u8(a: u8, b: u8) u8 {
        \\    return a + b;
        \\}
        \\
        \\pub fn main() i64 {
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
        \\pub fn main() i64 {
        \\    let x: u64 = 42u64;
        \\    let y: i64 = -10i64;
        \\    let z: u64 = x + y;
        \\    return 0i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_signed_unsigned.toy", "Cannot perform parser.BinaryOperator.add operation on types u64 and i64", "    let z: u64 = x + y;");

    std.debug.print("Signed/unsigned type mismatch error test passed\n", .{});
}

test "type system - tensor simple" {
    const allocator = std.testing.allocator;

    // Test tensor operations: vector initialization, element-wise addition, and indexing
    const test_source =
        \\ pub fn main() u32 {
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
    try assertReturns(allocator, "test_tensor.toy", 1);

    std.debug.print("Tensor basic test passed\n", .{});
}
test "type system - tensor float" {
    const allocator = std.testing.allocator;

    // Test tensor operations: vector initialization, element-wise addition, and indexing
    const test_source =
        \\ fn main() f32 {
        \\     // Create a vector of 5 elements initialized to zero
        \\     let vector: [5]f32 = [5]f32{2f32};
        \\     
        \\     // Add 1 to all elements (implicit parallelization)
        \\     vector[i] = vector[i] + 1f32;
        \\     
        \\     // Return the first element (should be 1)
        \\     return vector[0];
        \\ }
    ;

    try assertCompileFailure(allocator, test_source, "test_tensor_float.toy", "main function cannot return f32", " fn main() f32 {");

    std.debug.print("Tensor float test passed\n", .{});
}

test "tensor - out of bounds index" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i64 {
        \\    let vector: [5]u32 = [5]u32{0u32};
        \\    let x: u32 = vector[10];
        \\    return 0i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_tensor_oob.toy", "out of bounds", "    let x: u32 = vector[10];");

    std.debug.print("Tensor out-of-bounds index error test passed\n", .{});
}

test "tensor - incorrect indexing rank (1D accessed with 2 indices)" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i64 {
        \\    let vector: [5]u32 = [5]u32{0u32};
        \\    let x: u32 = vector[1, 2];
        \\    return 0i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_tensor_rank_mismatch1.toy", "does not match tensor rank", "    let x: u32 = vector[1, 2];");

    std.debug.print("Tensor rank mismatch (too many indices) error test passed\n", .{});
}

test "tensor - incorrect indexing rank (2D accessed with 1 index)" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i64 {
        \\    let matrix: [2, 2]u32 = [2, 2]u32{0u32};
        \\    let x: u32 = matrix[1];
        \\    return 0i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_tensor_rank_mismatch2.toy", "does not match tensor rank", "    let x: u32 = matrix[1];");

    std.debug.print("Tensor rank mismatch (too few indices) error test passed\n", .{});
}

test "tensor - mismatching dtype in allocation" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i64 {
        \\    let vector: [5]u32 = [5]u32{0.0f32};
        \\    return 0i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_tensor_dtype_alloc.toy", "Tensor literal value type", "    let vector: [5]u32 = [5]u32{0.0f32};");

    std.debug.print("Tensor dtype mismatch in allocation error test passed\n", .{});
}

test "tensor - mismatching dtype in binary op" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i64 {
        \\    let a: [5]u32 = [5]u32{1u32};
        \\    let b: [5]i64 = [5]i64{1i64};
        \\    a[i] = a[i] + b[i];
        \\    return 0i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_tensor_dtype_binop.toy", "Cannot perform", "    a[i] = a[i] + b[i];");

    std.debug.print("Tensor dtype mismatch in binary op error test passed\n", .{});
}

test "gpu vector addition compilation without --gpu flag" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    const gpu_source =
        \\fn gpu_vector_add(a: [16]i32, b: [16]i32, c: [16]i32) void {
        \\    c[i] = a[i] + b[i];
        \\}
        \\
        \\pub fn main() i32 {
        \\    let a: [16]i32 = [16]i32{1i32};
        \\    let b: [16]i32 = [16]i32{2i32};
        \\    let c: [16]i32 = [16]i32{2i32};
        \\    gpu_vector_add(a, b, c);
        \\    return c[0];
        \\}
    ;

    const filename = "test_gpu_vector_add.toy";

    // Write the temporary source file
    {
        const file = try std.fs.cwd().createFile(filename, .{ .truncate = true });
        defer file.close();
        try file.writeAll(gpu_source);
    }
    defer std.fs.cwd().deleteFile(filename) catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, filename } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);

    switch (out.term) {
        .Exited => |code| {
            // Expect compilation failure because GPU functions require --gpu flag
            if (code == 0) {
                std.debug.print("Expected GPU compilation to fail without --gpu flag\n", .{});
                std.debug.print("stdout: {s}\n", .{out.stdout});
                std.debug.print("stderr: {s}\n", .{out.stderr});
                return error.TestFailed;
            }
            // Check that the error message contains the expected text
            if (!std.mem.containsAtLeast(u8, out.stderr, 1, "Cannot compile GPU function") or
                !std.mem.containsAtLeast(u8, out.stderr, 1, "without --gpu flag"))
            {
                std.debug.print("Expected error message about missing --gpu flag\n", .{});
                std.debug.print("stderr: {s}\n", .{out.stderr});
                return error.TestFailed;
            }
        },
        else => return error.TestFailed,
    }

    std.debug.print("GPU compilation without --gpu flag test passed\n", .{});
}

test "gpu function compilation failure without --gpu flag" {
    const allocator = std.testing.allocator;

    const gpu_function_source =
        \\fn gpu_vector_add(a: [8]i32, b: [8]i32) void {
        \\    a[i] = a[i] + b[i];
        \\}
        \\
        \\pub fn main() i32 {
        \\    let x: [8]i32 = [8]i32{1i32};
        \\    let y: [8]i32 = [8]i32{2i32};
        \\    gpu_vector_add(x, y);
        \\    return x[0];
        \\}
    ;

    try assertGpuCompileFailure(allocator, gpu_function_source, "test_gpu_compile_failure.toy");

    std.debug.print("GPU function compilation failure test passed\n", .{});
}

test "gpu function compilation with valid triplet" {
    const allocator = std.testing.allocator;

    // Skip this test on non-Linux systems since GPU targets only work on Linux (for now)
    if (builtin.target.os.tag != .linux) {
        std.debug.print("GPU compilation test skipped on {s} (GPU targets only supported on Linux)\n", .{@tagName(builtin.target.os.tag)});
        return;
    }

    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    const gpu_source =
        \\fn gpu_vector_add(a: [8]i32, b: [8]i32) void {
        \\    a[i] = a[i] + b[i];
        \\}
        \\
        \\pub fn main() i32 {
        \\    let x: [8]i32 = [8]i32{1i32};
        \\    let y: [8]i32 = [8]i32{2i32};
        \\    gpu_vector_add(x, y);
        \\    return x[0];
        \\}
    ;

    const filename = "test_gpu_triplet_valid.toy";

    // Write the temporary source file
    {
        const file = try std.fs.cwd().createFile(filename, .{ .truncate = true });
        defer file.close();
        try file.writeAll(gpu_source);
    }
    defer std.fs.cwd().deleteFile(filename) catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, filename, "--gpu", "nvptx64-cuda:sm_50" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);

    switch (out.term) {
        .Exited => |code| {
            // Expect successful compilation with valid GPU triplet
            if (code != 0) {
                std.debug.print("GPU compilation with valid triplet failed with exit code {}\n", .{code});
                std.debug.print("stdout: {s}\n", .{out.stdout});
                std.debug.print("stderr: {s}\n", .{out.stderr});
                return error.TestFailed;
            }
        },
        else => return error.TestFailed,
    }

    std.debug.print("GPU compilation with valid triplet test passed\n", .{});
}

test "gpu function compilation with invalid triplet" {
    const allocator = std.testing.allocator;
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    const gpu_source =
        \\fn gpu_vector_add(a: [8]i32, b: [8]i32) void {
        \\    a[i] = a[i] + b[i];
        \\}
        \\
        \\pub fn main() i32 {
        \\    return 0i32;
        \\}
    ;

    const filename = "test_gpu_triplet_invalid.toy";

    // Write the temporary source file
    {
        const file = try std.fs.cwd().createFile(filename, .{ .truncate = true });
        defer file.close();
        try file.writeAll(gpu_source);
    }
    defer std.fs.cwd().deleteFile(filename) catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, filename, "--gpu", "invalid-triplet" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);

    switch (out.term) {
        .Exited => |code| {
            // Expect compilation failure
            if (code == 0) {
                std.debug.print("Expected GPU compilation to fail with invalid triplet\n", .{});
                std.debug.print("stdout: {s}\n", .{out.stdout});
                std.debug.print("stderr: {s}\n", .{out.stderr});
                return error.TestFailed;
            }
            // Check error message based on platform
            if (builtin.target.os.tag == .macos) {
                // On macOS, we should get an error about GPU not being supported
                if (!std.mem.containsAtLeast(u8, out.stderr, 1, "NVIDIA GPU compilation is not supported on macOS") and
                    !std.mem.containsAtLeast(u8, out.stderr, 1, "GPU compilation is only supported on Linux"))
                {
                    std.debug.print("Expected error message about GPU not supported on macOS\n", .{});
                    std.debug.print("stderr: {s}\n", .{out.stderr});
                    return error.TestFailed;
                }
            } else {
                // On other platforms, check for invalid triplet error
                if (!std.mem.containsAtLeast(u8, out.stderr, 1, "Invalid GPU triplet") or
                    !std.mem.containsAtLeast(u8, out.stderr, 1, "Expected format: nvptx64-cuda:sm_XX"))
                {
                    std.debug.print("Expected error message about invalid GPU triplet format\n", .{});
                    std.debug.print("stderr: {s}\n", .{out.stderr});
                    return error.TestFailed;
                }
            }
        },
        else => return error.TestFailed,
    }

    std.debug.print("GPU compilation with invalid triplet test passed\n", .{});
}

test "void functions - basic void function" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn simple_void() void {
        \\    return;
        \\}
        \\
        \\pub fn main() i64 {
        \\    simple_void();
        \\    return 0i64;
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_void_basic.toy");
    try assertReturns(allocator, "test_void_basic.toy", 0);

    std.debug.print("Basic void function test passed\n", .{});
}

test "void functions - float main" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() f32 {
        \\    return 2f32;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_float_main.toy", "main function cannot return f32", "pub fn main() f32 {");

    std.debug.print("Float main function error test passed\n", .{});
}

test "void functions - void functions calling void functions" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn simple_void() void {
        \\    return;
        \\}
        \\
        \\fn void_with_variable() void {
        \\}
        \\
        \\fn void_calling_void() void {
        \\    simple_void();
        \\    void_with_variable();
        \\    return;
        \\}
        \\
        \\pub fn main() i64 {
        \\    void_calling_void();
        \\    return 0i64;
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_void_comprehensive.toy");
    try assertReturns(allocator, "test_void_comprehensive.toy", 0);

    std.debug.print("Comprehensive void function test passed\n", .{});
}

test "void functions - error: return value from void function" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn bad_void() void {
        \\    return 42i32;
        \\}
        \\
        \\pub fn main() i64 {
        \\    bad_void();
        \\    return 0i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_void_error_return_value.toy", "Cannot return value from void function", "    return 42i32;");

    std.debug.print("Void function return value error test passed\n", .{});
}

test "void functions - error: missing return from non-void function" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn missing_return() i32 {
        \\    let x: i32 = 42i32;
        \\}
        \\
        \\pub fn main() i64 {
        \\    let result: i32 = missing_return();
        \\    return 0i64;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_missing_return.toy", "Must return value from non-void function", "fn missing_return() i32 {");

    std.debug.print("Missing return statement error test passed\n", .{});
}

test "vector operations - add then multiply" {
    const allocator = std.testing.allocator;

    const test_source =
        \\fn vector_add(a: [1024]i32, b: [1024]i32) void {
        \\    a[i] = a[i] + b[i];
        \\}
        \\
        \\fn vector_mul(a: [1024]i32, b: [1024]i32) void {
        \\    a[i] = a[i] * b[i];
        \\}
        \\
        \\pub fn main() i32 {
        \\    let a: [1024]i32 = [1024]i32{2i32};
        \\    let b: [1024]i32 = [1024]i32{3i32};
        \\    vector_add(a, b);
        \\    vector_mul(a, b);
        \\    return a[0];
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_vector_ops.toy");
    try assertReturns(allocator, "test_vector_ops.toy", 15);

    std.debug.print("vector operations test passed - program correctly returns 15\n", .{});
}

test "reduce operation - simple sum" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i32 {
        \\    // Create a 1D tensor with 5 elements
        \\    let a: [5]i32 = [5]i32{10i32};
        \\    
        \\    // Reduce by summing all elements using implicit index
        \\    let sum: i32 = reduce(a[i], +);
        \\    
        \\    // Should return 50 (sum of five 10s)
        \\    return sum;
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_reduce_sum.toy");
    try assertReturns(allocator, "test_reduce_sum.toy", 50);

    std.debug.print("reduce sum operation test passed - correctly returns 50\n", .{});
}

test "reduce operation - product" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i32 {
        \\    // Create a 1D tensor with 4 elements
        \\    let a: [4]i32 = [4]i32{3i32};
        \\    
        \\    // Reduce by multiplying all elements using implicit index
        \\    let product: i32 = reduce(a[i], *);
        \\    
        \\    // Should return 81 (3 * 3 * 3 * 3)
        \\    return product;
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_reduce_product.toy");
    try assertReturns(allocator, "test_reduce_product.toy", 81);

    std.debug.print("reduce product operation test passed - correctly returns 81\n", .{});
}

test "reduce operation - error on non-implicit tensor expression" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i32 {
        \\    let a: [5]i32 = [5]i32{10i32};
        \\    
        \\    // This should fail - reduce requires implicit tensor index
        \\    let sum: i32 = reduce(a, +);
        \\    
        \\    return sum;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_reduce_error.toy", "first argument of reduce must be an implicit tensor expression", "    let sum: i32 = reduce(a, +);");

    std.debug.print("reduce error test passed - correctly rejects non-implicit tensor expression\n", .{});
}

test "reduce operation - multiple reductions (second dimension then first)" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i32 {
        \\    // Create a 2D tensor 5x3 filled with 1s
        \\    let a: [5, 3]i32 = [5, 3]i32{1i32};
        \\    
        \\    // First reduce along second dimension (j) - sum across columns
        \\    // Result should be a 1D tensor of size 5, each element = 3
        \\    let b: [5]i32 = [5]i32{0i32};
        \\    b[i] = reduce(a[i, j], +);
        \\    
        \\    // Then reduce the result - multiply all elements
        \\    // Result should be 3 * 3 * 3 * 3 * 3 = 243
        \\    let c: i32 = reduce(b[i], *);
        \\    
        \\    return c;
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_reduce_multi_2d_1d.toy");
    // TODO: Current implementation reduces all elements, not per-dimension
    // Expected: 243 (3^5), Actual: Will compute 15^5 = too large
    // Commenting out until multi-dimensional reduction is properly implemented
    try assertReturns(allocator, "test_reduce_multi_2d_1d.toy", 243);

    std.debug.print("reduce multiple operations test passed - correctly returns 243\n", .{});
}

test "reduce operation - different dimension order" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i32 {
        \\    // Create a 2D tensor 4x2 filled with 2s
        \\    let a: [4, 2]i32 = [4, 2]i32{2i32};
        \\    
        \\    // First reduce along first dimension (i) - sum down rows
        \\    // Result should be a 1D tensor of size 2, each element = 8 (4 * 2)
        \\    let b: [2]i32 = [2]i32{0i32};
        \\    b[j] = reduce(a[i, j], +);
        \\    
        \\    // Then reduce the result - sum all elements
        \\    // Result should be 16 (8 + 8)
        \\    let c: i32 = reduce(b[j], +);
        \\    
        \\    return c;
        \\}
    ;

    try assertCompiles(allocator, test_source, "test_reduce_diff_dims.toy");
    // TODO: Current implementation reduces all elements, not per-dimension
    // Expected: 16, Actual: Different value due to full reduction
    // Commenting out until multi-dimensional reduction is properly implemented
    try assertReturns(allocator, "test_reduce_diff_dims.toy", 16);

    std.debug.print("reduce different dimension order test passed - correctly returns 16\n", .{});
}

test "reduce operation - error: LHS rank greater than RHS rank" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i32 {
        \\    let a: [3, 4]i32 = [3, 4]i32{1i32};
        \\    let b: [3, 4, 5]i32 = [3, 4, 5]i32{0i32};
        \\    // Error: trying to reduce from rank 2 to rank 3
        \\    b[i, j, k] = reduce(a[i, j], +);
        \\    return 0i32;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_reduce_rank_mismatch.toy", "Cannot reduce to higher or equal rank", "    b[i, j, k] = reduce(a[i, j], +);");

    std.debug.print("reduce rank mismatch error test passed\n", .{});
}

test "reduce operation - error: non-matching free indices" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i32 {
        \\    let a: [3]i32 = [3]i32{0i32};
        \\    let b: [3, 4]i32 = [3, 4]i32{1i32};
        \\    // Error: LHS has index 'i' but reduce has indices 'j, k'
        \\    a[i] = reduce(b[j, k], +);
        \\    return 0i32;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_reduce_index_mismatch.toy", "Free indices must match", "    a[i] = reduce(b[j, k], +);");

    std.debug.print("reduce index mismatch error test passed\n", .{});
}

test "reduce operation - error: implicit index conflicts with variable" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i32 {
        \\    let a: i32 = 5i32;
        \\    let b: [3, 4]i32 = [3, 4]i32{1i32};
        \\    let c: [3]i32 = [3]i32{0i32};
        \\    // Error: 'a' is already a variable, can't use as implicit index
        \\    c[i] = reduce(b[i, a], +);
        \\    return 0i32;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_reduce_var_conflict.toy", "Implicit index 'a' conflicts with variable", "    c[i] = reduce(b[i, a], +);");

    std.debug.print("reduce variable conflict error test passed\n", .{});
}

test "reduce operation - error: duplicate indices" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i32 {
        \\    let a: [3, 3]i32 = [3, 3]i32{1i32};
        \\    // Error: using 'i' twice in the same expression
        \\    let b: i32 = reduce(a[i, i], +);
        \\    return b;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_reduce_dup_indices.toy", "Duplicate implicit index", "    let b: i32 = reduce(a[i, i], +);");

    std.debug.print("reduce duplicate indices error test passed\n", .{});
}

test "reduce operation - error: invalid operator" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i32 {
        \\    let a: [5]i32 = [5]i32{10i32};
        \\    // Error: subtraction is not a valid reduction operator
        \\    let b: i32 = reduce(a[i], -);
        \\    return b;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_reduce_invalid_op.toy", "Invalid reduction operator", "    let b: i32 = reduce(a[i], -);");

    std.debug.print("reduce invalid operator error test passed\n", .{});
}

test "reduce operation - error: non-tensor argument" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i32 {
        \\    let a: i32 = 5i32;
        \\    // Error: can't reduce a scalar
        \\    let b: i32 = reduce(a, +);
        \\    return b;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_reduce_non_tensor.toy", "first argument of reduce must be an implicit tensor expression", "    let b: i32 = reduce(a, +);");

    std.debug.print("reduce non-tensor argument error test passed\n", .{});
}

test "reduce operation - error: mismatched index in parallel assignment" {
    const allocator = std.testing.allocator;

    const test_source =
        \\pub fn main() i32 {
        \\    let a: [3, 4]i32 = [3, 4]i32{1i32};
        \\    let b: [3, 5]i32 = [3, 5]i32{0i32};
        \\    // Error: LHS has [i,j] but reduce result would have [i,k]
        \\    b[i, j] = reduce(a[i, j, k], +);
        \\    return 0i32;
        \\}
    ;

    try assertCompileFailure(allocator, test_source, "test_reduce_parallel_mismatch.toy", "Index count", "    b[i, j] = reduce(a[i, j, k], +);");

    std.debug.print("reduce parallel assignment mismatch error test passed\n", .{});
}

test "integration test" {
    const lexer = @import("lexer.zig");
    const parser = @import("parser.zig");
    const allocator = std.testing.allocator;
    const source =
        \\fn add(a: i64, b: i64) i64 {
        \\    return a + b;
        \\}
        \\
        \\pub fn main() i64 {
        \\    let result: i64 = add(5i64, 3i64);
        \\    return result;
        \\}
    ;

    // Tokenize
    var lex = lexer.Lexer.init(allocator, source);
    const tokens = try lex.tokenize();
    defer allocator.free(tokens);

    // Parse
    var parse = parser.Parser.init(allocator, tokens, source, false);
    const ast = try parse.parse();
    defer parser.freeAST(allocator, ast);

    // Should have two function declarations
    try std.testing.expect(ast == .program);
    try std.testing.expect(ast.program.statements.len == 2);
    try std.testing.expect(ast.program.statements[0] == .function_declaration);
    try std.testing.expect(ast.program.statements[1] == .function_declaration);
}

test "compile example.toy and verify exit code" {
    const allocator = std.testing.allocator;

    // Get the path to the dcc binary
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Test that dcc can compile examples/example.toy
    {
        const run = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "examples/example.toy" } });
        defer allocator.free(run.stdout);
        defer allocator.free(run.stderr);
        std.debug.print("dcc compilation successful\n", .{});
    }

    // Test that the compiled binary produces exit code 16
    try assertReturns(allocator, "examples/example.toy", 16);
}

test "compile library.toy and verify it works" {
    const allocator = std.testing.allocator;

    // Get the path to the dcc binary
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Test that dcc can compile examples/library.toy
    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, "examples/library.toy" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);
    switch (out.term) {
        .Exited => |code| {
            if (code != 0) {
                std.debug.print("dcc compilation of examples/library.toy failed with exit code {}\n", .{code});
                std.debug.print("stdout: {s}\n", .{out.stdout});
                std.debug.print("stderr: {s}\n", .{out.stderr});
                return error.CompilationFailed;
            }
        },
        else => {
            std.debug.print("dcc compilation of examples/library.toy terminated abnormally\n", .{});
            return error.CompilationFailed;
        },
    }
    std.debug.print("dcc compilation of examples/library.toy successful\n", .{});
}

test "gpu function compilation failure on macOS target" {
    const allocator = std.testing.allocator;

    const gpu_source =
        \\fn gpu_vector_add(a: [8]i32, b: [8]i32) void {
        \\    a[i] = a[i] + b[i];
        \\}
        \\
        \\pub fn main() i32 {
        \\    let x: [8]i32 = [8]i32{1i32};
        \\    let y: [8]i32 = [8]i32{2i32};
        \\    gpu_vector_add(x, y);
        \\    return x[0];
        \\}
    ;

    try assertMacOsGpuCompileFailure(allocator, gpu_source, "test_macos_gpu_failure.toy");

    std.debug.print("macOS GPU compilation failure test passed\n", .{});
}

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
            std.debug.print("Compilation failed:\n{s}\n{s}\n", .{ out.stdout, out.stderr });
            return error.CompilationFailed;
        },
    }
}

fn assertReturns(allocator: std.mem.Allocator, filename: []const u8, expected: i32) !void {
    return assertExpectStdout(allocator, filename, null, expected);
}

fn assertExpectStdout(allocator: std.mem.Allocator, filename: []const u8, expected_out: ?[]const u8, expected: i32) !void {
    // Extract output name from source file (remove .toy extension)
    const basename = std.fs.path.basename(filename);
    const output_name = if (std.mem.endsWith(u8, basename, ".toy"))
        basename[0 .. basename.len - 4]
    else
        basename;
    defer std.fs.cwd().deleteFile(output_name) catch {};

    const binary_path = try std.fmt.allocPrint(allocator, "./{s}", .{output_name});
    defer allocator.free(binary_path);

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{binary_path} });
    defer {
        allocator.free(out.stdout);
        allocator.free(out.stderr);
    }
    switch (out.term) {
        .Exited => |term| {
            if (term != expected) {
                std.debug.print("Expected exit code {}, got {}\n", .{ expected, term });
                std.debug.print("stdout: {s}\n", .{out.stdout});
                std.debug.print("stderr: {s}\n", .{out.stderr});
                return error.UnexpectedExitCode;
            } else {
                if (expected_out) |eout| {
                    if (std.mem.indexOf(u8, out.stdout, eout) == 0) {
                        std.debug.print("Missing output {s} in full output:\n{s}", .{ eout, out.stdout });
                    }
                }
                std.debug.print("Compiled {s} produced correct exit code: {}\n", .{ filename, out.term.Exited });
            }
        },
        else => {
            return error.UnexpectedExitCode;
        },
    }
}

// Assert that GPU compilation fails without --gpu flag
fn assertGpuCompileFailure(
    allocator: std.mem.Allocator,
    source: []const u8,
    filename: []const u8,
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
            if (code == 0) {
                std.debug.print("Expected GPU compilation to fail without --gpu flag\n", .{});
                std.debug.print("stdout: {s}\n", .{out.stdout});
                std.debug.print("stderr: {s}\n", .{out.stderr});
                return error.UnexpectedSuccess;
            }
        },
        else => return error.UnexpectedTermination,
    }

    // Check that the error message contains the expected GPU error text
    if (!std.mem.containsAtLeast(u8, out.stderr, 1, "Cannot compile GPU function") or
        !std.mem.containsAtLeast(u8, out.stderr, 1, "without --gpu flag"))
    {
        std.debug.print("Expected GPU error message not found in stderr:\n{s}\n", .{out.stderr});
        return error.MissingGpuError;
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

// Assert that GPU compilation fails on macOS with the expected error message
fn assertMacOsGpuCompileFailure(
    allocator: std.mem.Allocator,
    source: []const u8,
    filename: []const u8,
) !void {
    const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";

    // Write the temporary source file
    {
        const file = try std.fs.cwd().createFile(filename, .{ .truncate = true });
        defer file.close();
        try file.writeAll(source);
    }
    defer std.fs.cwd().deleteFile(filename) catch {};

    const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, filename, "--target", "aarch64-macos-none", "--gpu", "nvptx64-cuda:sm_50" } });
    defer allocator.free(out.stdout);
    defer allocator.free(out.stderr);

    // Expect non-zero exit status
    switch (out.term) {
        .Exited => |code| {
            if (code == 0) {
                std.debug.print("Expected macOS GPU compilation to fail\n", .{});
                std.debug.print("stdout: {s}\n", .{out.stdout});
                std.debug.print("stderr: {s}\n", .{out.stderr});
                return error.UnexpectedSuccess;
            }
        },
        else => return error.UnexpectedTermination,
    }

    // Check that the error message contains the expected macOS GPU error text
    if (!std.mem.containsAtLeast(u8, out.stderr, 1, "NVIDIA GPU compilation is not supported on macOS targets")) {
        std.debug.print("Expected macOS GPU error message not found in stderr:\n{s}\n", .{out.stderr});
        return error.MissingMacOsGpuError;
    }
}
