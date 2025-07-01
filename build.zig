const std = @import("std");

// Add this function to your build.zig file
fn buildLldWrapper(b: *std.Build) *std.Build.Step {
    // Get environment variables for library paths
    const llvm_include_dir = std.process.getEnvVarOwned(b.allocator, "LLVM_INCLUDE_DIR") catch "/usr/include";
    const lld_include_dir = std.process.getEnvVarOwned(b.allocator, "LLD_INCLUDE_DIR") catch "/usr/include";
    const lld_lib_dir = std.process.getEnvVarOwned(b.allocator, "LLD_LIB_DIR") catch "/usr/lib";
    const libstdcxx_path = std.process.getEnvVarOwned(b.allocator, "LIBSTDCXX_PATH") catch "/nix/store/sa7j7cddyblhcb3ch3ds10w7nw75yjj1-gcc-14.3.0/lib/gcc/x86_64-unknown-linux-gnu/14.3.0/../../../libstdc++.a";
    defer if (!std.mem.eql(u8, llvm_include_dir, "/usr/include")) b.allocator.free(llvm_include_dir);
    defer if (!std.mem.eql(u8, lld_include_dir, "/usr/include")) b.allocator.free(lld_include_dir);
    defer if (!std.mem.eql(u8, lld_lib_dir, "/usr/lib")) b.allocator.free(lld_lib_dir);
    defer if (!std.mem.eql(u8, libstdcxx_path, "/nix/store/sa7j7cddyblhcb3ch3ds10w7nw75yjj1-gcc-14.3.0/lib/gcc/x86_64-unknown-linux-gnu/14.3.0/../../../libstdc++.a")) b.allocator.free(libstdcxx_path);

    // Step 1: Compile lld_wrapper.cpp to object file using g++ directly
    const wrapper_obj_cmd = b.addSystemCommand(&.{ "g++", "-c", "-std=c++17", "-fPIC", "-O2", "-o", "lld_wrapper.o", "src/lld_wrapper.cpp" });

    // Add include paths
    const llvm_include_arg = b.fmt("-I{s}", .{llvm_include_dir});
    const lld_include_arg = b.fmt("-I{s}", .{lld_include_dir});
    wrapper_obj_cmd.addArg(llvm_include_arg);
    wrapper_obj_cmd.addArg(lld_include_arg);

    // Step 2: Create directory for build process
    const mkdir_cmd = b.addSystemCommand(&.{ "mkdir", "-p", "comprehensive_build" });

    // Step 3: Create directory for build process

    // Step 4: Use ld -r to combine all libraries with relocatable linking
    const ld_cmd = b.addSystemCommand(&.{ "ld", "-r", "-o", "comprehensive_build/combined.o" });
    ld_cmd.addArg("lld_wrapper.o"); // Use the object file created by g++
    ld_cmd.addArgs(&.{"--whole-archive"});

    // Add all LLD library paths
    const lld_libs = [_][]const u8{
        "liblldCommon.a",
        "liblldELF.a",
        "liblldMachO.a",
        "liblldCOFF.a",
        "liblldMinGW.a",
        "liblldWasm.a",
    };

    for (lld_libs) |lib| {
        const lib_path = b.fmt("{s}/{s}", .{ lld_lib_dir, lib });
        ld_cmd.addArg(lib_path);
    }

    // Add libstdc++.a directly
    ld_cmd.addArg(libstdcxx_path);

    ld_cmd.addArg("--no-whole-archive");
    ld_cmd.step.dependOn(&mkdir_cmd.step);
    ld_cmd.step.dependOn(&wrapper_obj_cmd.step); // Depend on g++ compilation

    // Step 5: Create the final static library
    const ar_cmd = b.addSystemCommand(&.{ "ar", "rcs", "liblldwrapper.a", "comprehensive_build/combined.o" });
    ar_cmd.step.dependOn(&ld_cmd.step);

    // Step 6: Clean up temporary files
    const cleanup_cmd = b.addSystemCommand(&.{ "rm", "-rf", "comprehensive_build", "lld_wrapper.o" });
    cleanup_cmd.step.dependOn(&ar_cmd.step);

    // Return the final step that represents the complete LLD wrapper build
    return &cleanup_cmd.step;
}

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    // We will create a module for our entry point, 'main.zig'.
    const exe_mod = b.createModule(.{
        // `root_source_file` is the Zig "entry point" of the module. If a module
        // only contains e.g. external object files, you can make this `null`.
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // This creates a `std.Build.Step.Compile`, that builds an executable.
    const exe = b.addExecutable(.{
        .name = "dcc",
        .root_module = exe_mod,
    });

    // Link with LLVM and LLD libraries (statically)
    exe.linkLibC();
    exe.linkLibCpp();

    // Get library directories from environment
    const llvm_include_dir = std.process.getEnvVarOwned(b.allocator, "LLVM_INCLUDE_DIR") catch null;
    const llvm_lib_dir = std.process.getEnvVarOwned(b.allocator, "LLVM_LIB_DIR") catch null;
    const lld_include_dir = std.process.getEnvVarOwned(b.allocator, "LLD_INCLUDE_DIR") catch null;
    const lld_lib_dir = std.process.getEnvVarOwned(b.allocator, "LLD_LIB_DIR") catch null;
    defer if (llvm_include_dir) |dir| b.allocator.free(dir);
    defer if (llvm_lib_dir) |dir| b.allocator.free(dir);
    defer if (lld_include_dir) |dir| b.allocator.free(dir);
    defer if (lld_lib_dir) |dir| b.allocator.free(dir);

    exe.addIncludePath(.{ .cwd_relative = llvm_include_dir orelse "/usr/include" });
    exe.addIncludePath(.{ .cwd_relative = lld_include_dir orelse "/usr/include" });
    exe.addLibraryPath(.{ .cwd_relative = llvm_lib_dir orelse "/usr/lib" });
    exe.addLibraryPath(.{ .cwd_relative = "./" });

    exe.linkSystemLibrary2("LLVM", .{ .preferred_link_mode = .static });
    exe.linkSystemLibrary("z"); // zlib for compression (required by LLD)
    exe.linkSystemLibrary("dl"); // dynamic linking library
    exe.linkSystemLibrary("pthread"); // threading library
    // exe.addCSourceFile(.{ .file = b.path("src/llvm_c_api.h") });
    const lld_wrapper_step = buildLldWrapper(b);

    // Link with our comprehensive static library (depends on lld_wrapper_step)
    exe.addObjectFile(b.path("liblldwrapper.a"));
    exe.step.dependOn(lld_wrapper_step); // Main exe depends on LLD wrapper

    // // Link essential LLD libraries (static linking approach)
    // if (lld_lib_dir) |lib_dir| {
    //     exe.addLibraryPath(.{ .cwd_relative = lib_dir });
    //     exe.linkSystemLibrary2("lldCommon", .{ .preferred_link_mode = .static });
    //     exe.linkSystemLibrary2("lldELF", .{ .preferred_link_mode = .static });
    //     exe.linkSystemLibrary2("lldMachO", .{ .preferred_link_mode = .static });
    //     exe.linkSystemLibrary2("lldCOFF", .{ .preferred_link_mode = .static });
    //     exe.linkSystemLibrary2("lldWasm", .{ .preferred_link_mode = .static });
    //     exe.linkSystemLibrary2("lldMinGW", .{ .preferred_link_mode = .static });
    //     exe.linkSystemLibrary2("z", .{ .preferred_link_mode = .static }); // zlib for compression
    // }

    // Add C++ wrapper for lld
    // exe.addCSourceFile(.{
    //     .file = b.path("src/lld_wrapper.cpp"),
    //     .flags = &.{"-std=c++17"},
    // });

    // Keep dynamic linkage for system libraries (required on macOS)
    // But try to statically embed LLVM/LLD code
    exe.want_lto = false;

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // // Creates a step for unit testing. This only builds the test executable
    // // but does not run it.
    // const exe_unit_tests = b.addTest(.{
    //     .root_module = exe_mod,
    // });

    // // Copy build configuration to tests
    // exe_unit_tests.linkLibC();
    // exe_unit_tests.linkLibCpp();

    // const test_llvm_include_dir = std.process.getEnvVarOwned(b.allocator, "LLVM_INCLUDE_DIR") catch null;
    // const test_llvm_lib_dir = std.process.getEnvVarOwned(b.allocator, "LLVM_LIB_DIR") catch null;
    // const test_lld_include_dir = std.process.getEnvVarOwned(b.allocator, "LLD_INCLUDE_DIR") catch null;
    // const test_lld_lib_dir = std.process.getEnvVarOwned(b.allocator, "LLD_LIB_DIR") catch null;
    // defer if (test_llvm_include_dir) |dir| b.allocator.free(dir);
    // defer if (test_llvm_lib_dir) |dir| b.allocator.free(dir);
    // defer if (test_lld_include_dir) |dir| b.allocator.free(dir);
    // defer if (test_lld_lib_dir) |dir| b.allocator.free(dir);

    // exe_unit_tests.addIncludePath(.{ .cwd_relative = test_llvm_include_dir orelse "/usr/include" });
    // exe_unit_tests.addIncludePath(.{ .cwd_relative = test_lld_include_dir orelse "/usr/include" });
    // exe_unit_tests.addLibraryPath(.{ .cwd_relative = test_llvm_lib_dir orelse "/usr/lib" });

    // // Link essential LLD libraries for tests (static linking)
    // if (test_lld_lib_dir) |lib_dir| {
    //     exe_unit_tests.addLibraryPath(.{ .cwd_relative = lib_dir });
    //     exe_unit_tests.linkSystemLibrary2("lldCommon", .{ .preferred_link_mode = .static });
    //     exe_unit_tests.linkSystemLibrary2("lldELF", .{ .preferred_link_mode = .static });
    //     exe_unit_tests.linkSystemLibrary2("lldMachO", .{ .preferred_link_mode = .static });
    //     exe_unit_tests.linkSystemLibrary2("lldCOFF", .{ .preferred_link_mode = .static });
    //     exe_unit_tests.linkSystemLibrary2("lldWasm", .{ .preferred_link_mode = .static });
    //     exe_unit_tests.linkSystemLibrary2("lldMinGW", .{ .preferred_link_mode = .static });
    //     exe_unit_tests.linkSystemLibrary2("z", .{ .preferred_link_mode = .static });
    // }

    // const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // // Similar to creating the run step earlier, this exposes a `test` step to
    // // the `zig build --help` menu, providing a way for the user to request
    // // running the unit tests.
    // const test_step = b.step("test", "Run unit tests");
    // test_step.dependOn(&run_exe_unit_tests.step);
}
