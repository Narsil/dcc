const std = @import("std");
const builtin = @import("builtin");
const process = std.process;

const Target = struct {
    remote: []const u8,
    target: []const u8,
};

const targets = [_]Target{
    Target{
        .target = "x86_64-unknown-linux-gnu",
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
        .target = "x86_64-unknown-linux-gnu",
        .remote = "home",
        .gpu = "nvidia-ptx-sm50",
    },
};

test "type system - different integer types" {
    const allocator = std.testing.allocator;
    // Create a test file with different integer types
    const test_source =
        \\fn add_u8(a: u8, b: u8) u8 {
        \\    return a + b;
        \\}
        \\
        \\fn add_i32(a: i32, b: i32) i32 {
        \\    return a + b;
        \\}
        \\
        \\fn add_f64(a: f64, b: f64) f64 {
        \\    return a + b;
        \\}
        \\
        \\fn main() i32 {
        \\    let x: u8 = add_u8(1u8, 2u8);
        \\    let y: i32 = add_i32(10i32, 20i32);
        \\    let z: f64 = add_f64(1.5f64, 2.5f64);
        \\    return y;
        \\}
    ;
    try assertCrossCompiles(allocator, test_source, "test_cross_integers.toy", 30);
}

test "type system - tensor gpu" {
    const allocator = std.testing.allocator;
    // Create a test file with different integer types
    const test_source =
        \\fn gpu_vector_add(a: [1024]i32, b: [1024]i32) void{
        \\    a[i] = a[i] + b[i];
        \\}
        \\fn main() i32 {
        \\    let a: [1024]i32 = [1024]i32{2i32};
        \\    let b: [1024]i32 = [1024]i32{4i32};
        \\    gpu_vector_add(a, b);
        \\    return a[0];
        \\} 
    ;
    try assertGpuCrossCompiles(allocator, test_source, "test_cross_tensors.toy", 6);
}

test "type system - tensor gpu stacked" {
    const allocator = std.testing.allocator;
    // Create a test file with different integer types
    const test_source =
        \\fn gpu_vector_mul(a: [1024]i32, b: [1024]i32) void{
        \\    a[i] = a[i] * b[i];
        \\}
        \\fn main() i32 {
        \\    let a: [1024]i32 = [1024]i32{2i32};
        \\    let b: [1024]i32 = [1024]i32{4i32};
        \\    gpu_vector_mul(a, b);
        \\    return a[0];
        \\} 
    ;
    try assertGpuCrossCompiles(allocator, test_source, "test_cross_tensors_stacked.toy", 8);
}

fn assertCrossCompiles(allocator: std.mem.Allocator, source: []const u8, filename: []const u8, expected: u32) !void {
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
                    std.debug.print("Cross-compiled {s} for target {s} produced correct exit code: {}\n", .{ filename, target.target, out.term.Exited });
                }
            },
            else => {
                return error.UnexpectedExitCode;
            },
        }
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

        const local_binary_path = try std.fmt.allocPrint(allocator, "./{s}", .{output_name});
        defer allocator.free(local_binary_path);

        const remote_binary_path = try std.fmt.allocPrint(allocator, "./{s}", .{output_name});
        defer allocator.free(remote_binary_path);

        {
            const remote = try std.fmt.allocPrint(allocator, "{s}:", .{target.remote});
            defer allocator.free(remote);
            const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ "scp", local_binary_path, remote } });
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
        const out = try process.Child.run(.{ .allocator = allocator, .argv = &.{ "ssh", target.remote, remote_binary_path } });
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
}
