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

