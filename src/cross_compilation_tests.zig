const std = @import("std");
const builtin = @import("builtin");
const process = std.process;

const Target = struct {
    remote: []const u8,
    target: []const u8,
};

const targets = [_]Target{
    Target{
        .target = "x86_64-linux",
        .remote = "home",
    },
    Target{
        .target = "arm64-apple-darwin",
        .remote = "laptop",
    },
};

const GpuTarget = struct {
    remote: []const u8,
    target: []const u8,
    gpu: []const u8,
};

const gpu_targets = [_]GpuTarget{
    GpuTarget{
        .target = "x86_64-linux",
        .remote = "home",
        .gpu = "nvptx64-cuda:sm_50",
    },
};

test "type system - different integer types" {
    const allocator = std.testing.allocator;
    const test_source =
        \\fn add_i32(a: i32, b: i32) i32 {
        \\    return a + b;
        \\}
        \\
        \\pub fn main() i32 {
        \\    let y: i32 = add_i32(10i32, 20i32);
        \\    return y;
        \\}
    ;
    try assertCrossCompiles(allocator, test_source, "test_cross_integers.toy", 30);
}

test "type system - tensor gpu add" {
    const allocator = std.testing.allocator;
    const test_source =
        \\fn gpu_vector_add(a: [1024]i32, b: [1024]i32) void{
        \\    a[i] = a[i] + b[i];
        \\}
        \\pub fn main() i32 {
        \\    let a: [1024]i32 = [1024]i32{2i32};
        \\    let b: [1024]i32 = [1024]i32{4i32};
        \\    gpu_vector_add(a, b);
        \\    return a[0];
        \\} 
    ;
    try assertGpuCrossCompiles(allocator, test_source, "test_cross_tensors_add.toy", 6);
}

test "type system - tensor gpu mul" {
    const allocator = std.testing.allocator;
    const test_source =
        \\fn gpu_vector_mul(a: [1024]i32, b: [1024]i32) void{
        \\    a[i] = a[i] * b[i];
        \\}
        \\pub fn main() i32 {
        \\    let a: [1024]i32 = [1024]i32{2i32};
        \\    let b: [1024]i32 = [1024]i32{4i32};
        \\    gpu_vector_mul(a, b);
        \\    return a[0];
        \\} 
    ;
    try assertGpuCrossCompiles(allocator, test_source, "test_cross_tensors_mul.toy", 8);
}

test "type system - tensor gpu stacked" {
    const allocator = std.testing.allocator;
    const test_source =
        \\fn gpu_vector_add(a: [1024]i32, b: [1024]i32) void{
        \\    a[i] = a[i] + b[i];
        \\}
        \\fn gpu_vector_mul(a: [1024]i32, b: [1024]i32) void{
        \\    a[i] = a[i] * b[i];
        \\}
        \\pub fn main() i32 {
        \\    let a: [1024]i32 = [1024]i32{2i32};
        \\    let b: [1024]i32 = [1024]i32{4i32};
        \\    gpu_vector_add(a, b);
        \\    gpu_vector_mul(a, b);
        \\    return a[0];
        \\} 
    ;
    try assertGpuCrossCompiles(allocator, test_source, "test_cross_tensors_stacked.toy", 24);
}

test "type system - tensor gpu interleaved" {
    const allocator = std.testing.allocator;
    const test_source =
        \\fn gpu_vector_add(a: [1024]i32, b: [1024]i32) void{
        \\    a[i] = a[i] + b[i];
        \\}
        \\fn gpu_vector_mul(a: [1024]i32, b: [1024]i32) void{
        \\    a[i] = a[i] * b[i];
        \\}
        \\fn vector_mul(a: [1024]i32, b: [1024]i32) void{
        \\    a[i] = a[i] * b[i];
        \\}
        \\pub fn main() i32 {
        \\    let a: [1024]i32 = [1024]i32{2i32};
        \\    let b: [1024]i32 = [1024]i32{4i32};
        \\    gpu_vector_add(a, b);
        \\    gpu_vector_mul(a, b);
        \\    vector_mul(a, b);
        \\    gpu_vector_add(a, b);
        \\    return a[0];
        \\} 
    ;
    try assertGpuCrossCompiles(allocator, test_source, "test_cross_tensors_stacked.toy", 100);
}

test "reduce - gpu 2D to 1D sum" {
    const allocator = std.testing.allocator;
    const test_source =
        \\fn gpu_reduce(a: [5, 3]f32, b: [5]f32) void {
        \\    b[i] = reduce(a[i, j], +);
        \\}
        \\
        \\pub fn main() i32 {
        \\    let a: [5, 3]f32 = [5, 3]f32{2.0f32};
        \\    let b: [5]f32 = [5]f32{0.0f32};
        \\    gpu_reduce(a, b);
        \\    return 6i32; // Expected: 2.0 * 3 = 6.0
        \\} 
    ;
    try assertGpuCrossCompiles(allocator, test_source, "test_gpu_reduce_2d_1d.toy", 6);
}

test "reduce - gpu 2D to 1D product" {
    const allocator = std.testing.allocator;
    const test_source =
        \\fn gpu_reduce_rows(a: [3, 3]f32, result: [3]f32) void {
        \\    result[i] = reduce(a[i, j], *);
        \\}
        \\
        \\pub fn main() i32 {
        \\    let a: [3, 3]f32 = [3, 3]f32{3.0f32};
        \\    let result: [3]f32 = [3]f32{1.0f32};
        \\    gpu_reduce_rows(a, result);
        \\    return 27i32; // Expected: 3^3 = 27 (first row product)
        \\} 
    ;
    try assertGpuCrossCompiles(allocator, test_source, "test_gpu_reduce_2d_1d_product.toy", 27);
}

test "write - stdout" {
    const allocator = std.testing.allocator;
    const test_source =
        \\pub fn main() void {
        \\  write(io.stdout, "Hello world!\n");
        \\} 
    ;
    try assertCrossCompilesStdout(allocator, test_source, "test_cross_stdout.toy", 0, "Hello world!\n");
}

test "write - file" {
    const allocator = std.testing.allocator;
    const test_source =
        \\pub fn main() void {
        \\  write(io.file("hello.txt"), "Hello\n");
        \\} 
    ;
    try assertCrossCompiles(allocator, test_source, "test_cross_file.toy", 0);
    try assertRemoteFile(allocator, "hello.txt", "Hello\n");
}

fn assertCrossCompiles(allocator: std.mem.Allocator, source: []const u8, filename: []const u8, expected: u32) !void {
    return assertCrossCompilesStdout(allocator, source, filename, expected, null);
}
fn assertCrossCompilesStdout(allocator: std.mem.Allocator, source: []const u8, filename: []const u8, expected: u32, expected_out: ?[]const u8) !void {
    for (targets) |target| {
        const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";
        {
            const file = try std.fs.cwd().createFile(filename, .{ .truncate = true });
            defer file.close();
            try file.writeAll(source);
        }
        defer std.fs.cwd().deleteFile(filename) catch {};

        {
            const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, filename, "--target", target.target } });
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

        // Extract output name from source file (remove .toy extension)
        const basename = std.fs.path.basename(filename);
        const output_name = if (std.mem.endsWith(u8, basename, ".toy"))
            basename[0 .. basename.len - 4]
        else
            basename;

        const local_binary_path = try std.fmt.allocPrint(allocator, "./{s}", .{output_name});
        defer allocator.free(local_binary_path);
        defer std.fs.cwd().deleteFile(output_name) catch {};

        const remote_binary_path = try std.fmt.allocPrint(allocator, "./{s}", .{output_name});
        defer allocator.free(remote_binary_path);

        {
            const remote = try std.fmt.allocPrint(allocator, "{s}:", .{target.remote});
            defer allocator.free(remote);
            const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ "scp", local_binary_path, remote } });
            defer allocator.free(out.stdout);
            defer allocator.free(out.stderr);
            switch (out.term) {
                .Exited => |term| {
                    if (term == 255) {
                        std.debug.print("Skipping host check, host {s} - {s} is unreachable\n", .{ target.remote, target.target });
                        return;
                    }
                    if (term != 0) {
                        std.debug.print("Failed to SCP\n", .{});
                        std.debug.print("stdout: {s}\n", .{out.stdout});
                        std.debug.print("stderr: {s}\n", .{out.stderr});
                        return error.UnexpectedExitCode;
                    }
                },
                else => {
                    return error.UnexpectedExitCode;
                },
            }
        }
        {
            const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ "ssh", target.remote, remote_binary_path } });
            defer allocator.free(out.stdout);
            defer allocator.free(out.stderr);
            switch (out.term) {
                .Exited => |term| {
                    if (term != expected) {
                        std.debug.print("Expected exit code {}, got {}\n", .{ expected, term });
                        std.debug.print("stdout: {s}\n", .{out.stdout});
                        std.debug.print("stderr: {s}\n", .{out.stderr});
                        return error.UnexpectedExitCode;
                    } else {
                        if (expected_out) |eout| {
                            try std.testing.expectEqualStrings(eout, out.stdout);
                        }
                        std.debug.print("Cross-compiled {s} for target {s} produced correct exit code: {}\n", .{ filename, target.target, out.term.Exited });
                    }
                },
                else => {
                    return error.UnexpectedExitCode;
                },
            }
        }
        // {
        //     const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ "ssh", target.remote, "rm", remote_binary_path } });
        //     defer allocator.free(out.stdout);
        //     defer allocator.free(out.stderr);
        // }
    }
}

fn assertGpuCrossCompiles(allocator: std.mem.Allocator, source: []const u8, filename: []const u8, expected: u32) !void {
    for (gpu_targets) |target| {
        const dcc_path = if (builtin.target.os.tag == .windows) "zig-out/bin/dcc.exe" else "zig-out/bin/dcc";
        {
            const file = try std.fs.cwd().createFile(filename, .{ .truncate = true });
            defer file.close();
            try file.writeAll(source);
        }
        defer std.fs.cwd().deleteFile(filename) catch {};

        {
            const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ dcc_path, filename, "--target", target.target, "--gpu", target.gpu } });
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

        // Extract output name from source file (remove .toy extension)
        const basename = std.fs.path.basename(filename);
        const output_name = if (std.mem.endsWith(u8, basename, ".toy"))
            basename[0 .. basename.len - 4]
        else
            basename;
        defer std.fs.cwd().deleteFile(output_name) catch {};
        std.debug.print("--Should delete {s}\n", .{output_name});

        const local_binary_path = try std.fmt.allocPrint(allocator, "./{s}", .{output_name});
        defer allocator.free(local_binary_path);

        const remote_binary_path = try std.fmt.allocPrint(allocator, "./{s}", .{output_name});
        defer allocator.free(remote_binary_path);

        {
            const remote = try std.fmt.allocPrint(allocator, "{s}:", .{target.remote});
            defer allocator.free(remote);
            const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ "scp", local_binary_path, remote } });
            defer allocator.free(out.stdout);
            defer allocator.free(out.stderr);
            switch (out.term) {
                .Exited => |term| {
                    if (term != 0) {
                        std.debug.print("Expected exit code {}, got {}\n", .{ expected, term });
                        std.debug.print("stdout: {s}\n", .{out.stdout});
                        std.debug.print("stderr: {s}\n", .{out.stderr});
                        return error.UnexpectedExitCode;
                    }
                },
                else => {
                    return error.UnexpectedExitCode;
                },
            }
        }
        {
            const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ "ssh", target.remote, remote_binary_path } });
            defer allocator.free(out.stdout);
            defer allocator.free(out.stderr);
            switch (out.term) {
                .Exited => |term| {
                    if (term != expected) {
                        std.debug.print("Expected exit code {}, got {}\n", .{ expected, term });
                        std.debug.print("stdout: {s}\n", .{out.stdout});
                        std.debug.print("stderr: {s}\n", .{out.stderr});
                        return error.UnexpectedExitCode;
                    } else {
                        std.debug.print("Cross-compiled {s} for target {s} and gpu {s} produced correct exit code: {}\n", .{ filename, target.target, target.gpu, out.term.Exited });
                    }
                },
                else => {
                    return error.UnexpectedExitCode;
                },
            }
        }
        // {
        //     const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ "ssh", target.remote, "rm", remote_binary_path } });
        //     defer allocator.free(out.stdout);
        //     defer allocator.free(out.stderr);
        // }
    }
}
fn assertRemoteFile(allocator: std.mem.Allocator, filename: []const u8, expected: []const u8) !void {
    for (targets) |target| {
        {
            const remote = try std.fmt.allocPrint(allocator, "{s}:", .{target.remote});
            defer allocator.free(remote);
            const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ "ssh", target.remote, "cat", filename } });
            defer allocator.free(out.stdout);
            defer allocator.free(out.stderr);
            switch (out.term) {
                .Exited => |term| {
                    if (term == 255) {
                        std.debug.print("Skipping host check, host {s} - {s} is unreachable\n", .{ target.remote, target.target });
                        return;
                    }
                    if (term != 0) {
                        std.debug.print("Failed to SCP\n", .{});
                        std.debug.print("stdout: {s}\n", .{out.stdout});
                        std.debug.print("stderr: {s}\n", .{out.stderr});
                        return error.UnexpectedExitCode;
                    }
                    try std.testing.expectEqualStrings(expected, out.stdout);
                },
                else => {
                    return error.UnexpectedExitCode;
                },
            }
        }
    }
}
