const std = @import("std");
const builtin = @import("builtin");

// Add this function to your build.zig file
fn buildLldWrapper(b: *std.Build, llvm_include_dir: []u8, lld_include_dir: []u8, lld_lib_dir: []u8) *std.Build.Step {
    const command_args = [_][]const u8{
        "gcc", "-print-file-name=libstdc++.a",
    };
    // Use builder.run() which handles spawning and waiting, and gives you the Child result
    // This is the cleanest way to run a child process and get its output within build.zig.
    const result = b.run(&command_args);
    const libstdcxx_path = std.mem.trim(u8, result, "\n");

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

fn createLinking(b: *std.Build, exe: *std.Build.Step.Compile, llvm_include_dir: []u8, llvm_lib_dir: []u8, lld_include_dir: []u8, lld_lib_dir: []u8, use_wrapper: bool) void {
    // Copy build configuration to tests
    exe.linkLibC();
    exe.linkLibCpp();
    exe.linkSystemLibrary2("LLVM", .{ .preferred_link_mode = .static });
    exe.linkSystemLibrary("z"); // zlib for compression (required by LLD)
    exe.linkSystemLibrary("dl"); // dynamic linking library
    exe.linkSystemLibrary("pthread"); // threading library

    exe.addIncludePath(.{ .cwd_relative = llvm_include_dir });
    exe.addLibraryPath(.{ .cwd_relative = llvm_lib_dir });

    // Try to add MLIR support (optional)
    const mlir_include_dir = std.process.getEnvVarOwned(b.allocator, "MLIR_INCLUDE_DIR") catch null;
    const mlir_lib_dir = std.process.getEnvVarOwned(b.allocator, "MLIR_LIB_DIR") catch null;
    
    if (mlir_include_dir != null and mlir_lib_dir != null) {
        defer b.allocator.free(mlir_include_dir.?);
        defer b.allocator.free(mlir_lib_dir.?);
        
        exe.addIncludePath(.{ .cwd_relative = mlir_include_dir.? });
        exe.addLibraryPath(.{ .cwd_relative = mlir_lib_dir.? });
        
        // Link essential MLIR libraries for C API support
        exe.linkSystemLibrary2("MLIR", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("MLIRIR", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("MLIRSupport", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("MLIRArithDialect", .{ .preferred_link_mode = .static });
        // Core C API libraries
        exe.linkSystemLibrary2("MLIRCAPIIR", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("MLIRCAPIFunc", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("MLIRCAPIGPU", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("MLIRCAPIArith", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("MLIRCAPIMemRef", .{ .preferred_link_mode = .static });
        // NVVM dialect and target libraries
        exe.linkSystemLibrary2("MLIRCAPINVVM", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("MLIRNVVMDialect", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("MLIRGPUToNVVMTransforms", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("MLIRNVVMTarget", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("MLIRNVVMToLLVM", .{ .preferred_link_mode = .static });
        
        std.debug.print("MLIR support enabled\n", .{});
    } else {
        std.debug.print("MLIR support not available (MLIR_INCLUDE_DIR and MLIR_LIB_DIR not set)\n", .{});
    }

    // Link essential LLD libraries for tests (static linking)
    exe.addIncludePath(.{ .cwd_relative = lld_include_dir });
    if (builtin.target.os.tag == .linux) {
        const lld_wrapper_step = buildLldWrapper(b, llvm_include_dir, lld_include_dir, lld_lib_dir);
        // Link with our comprehensive static library (depends on lld_wrapper_step)
        exe.addObjectFile(b.path("liblldwrapper.a"));
        exe.step.dependOn(lld_wrapper_step); // Main exe depends on LLD wrapper
    } else {

        // Link essential LLD libraries (static linking approach)
        exe.addLibraryPath(.{ .cwd_relative = lld_lib_dir });
        exe.linkSystemLibrary2("lldCommon", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("lldELF", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("lldMachO", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("lldCOFF", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("lldWasm", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("lldMinGW", .{ .preferred_link_mode = .static });
        exe.linkSystemLibrary2("z", .{ .preferred_link_mode = .static }); // zlib for compression
        if (use_wrapper) {
            exe.addCSourceFile(.{
                .file = b.path("src/lld_wrapper.cpp"),
                .flags = &.{"-std=c++17"},
            });
        }
    }
}

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) !void {
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
    const llvm_include_dir = try std.process.getEnvVarOwned(b.allocator, "LLVM_INCLUDE_DIR");
    const llvm_lib_dir = try std.process.getEnvVarOwned(b.allocator, "LLVM_LIB_DIR");
    const lld_include_dir = try std.process.getEnvVarOwned(b.allocator, "LLD_INCLUDE_DIR");
    const lld_lib_dir = try std.process.getEnvVarOwned(b.allocator, "LLD_LIB_DIR");
    defer b.allocator.free(llvm_include_dir);
    defer b.allocator.free(llvm_lib_dir);
    defer b.allocator.free(lld_include_dir);
    defer b.allocator.free(lld_lib_dir);

    exe.addIncludePath(.{ .cwd_relative = llvm_include_dir });
    exe.addLibraryPath(.{ .cwd_relative = llvm_lib_dir });
    // exe.addIncludePath(.{ .cwd_relative = lld_include_dir });
    exe.addLibraryPath(.{ .cwd_relative = "./" });

    createLinking(b, exe, llvm_include_dir, llvm_lib_dir, lld_include_dir, lld_lib_dir, true);

    // Add C++ wrapper for lld

    // Keep dynamic linkage for system libraries (required on macOS)
    // But try to statically embed LLVM/LLD code
    exe.want_lto = false;

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // Create emit_ptx executable
    const emit_ptx_mod = b.createModule(.{
        .root_source_file = b.path("src/emit_ptx.zig"),
        .target = target,
        .optimize = optimize,
    });

    const emit_ptx_exe = b.addExecutable(.{
        .name = "emit_ptx",
        .root_module = emit_ptx_mod,
    });

    // Link emit_ptx with the same libraries as main dcc
    createLinking(b, emit_ptx_exe, llvm_include_dir, llvm_lib_dir, lld_include_dir, lld_lib_dir, false);

    // Install emit_ptx executable
    b.installArtifact(emit_ptx_exe);

    // Create run step for emit_ptx
    const run_emit_ptx_cmd = b.addRunArtifact(emit_ptx_exe);
    run_emit_ptx_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_emit_ptx_cmd.addArgs(args);
    }

    const run_emit_ptx_step = b.step("emit-ptx", "Run the emit_ptx tool");
    run_emit_ptx_step.dependOn(&run_emit_ptx_cmd.step);

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

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    createLinking(b, exe_unit_tests, llvm_include_dir, llvm_lib_dir, lld_include_dir, lld_lib_dir, false);

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);

    // Integration tests that require the dcc binary
    const integration_tests = b.addTest(.{
        .name = "integration-tests",
        .root_source_file = b.path("src/integration_tests.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add integration tests step
    const run_integration_tests = b.addRunArtifact(integration_tests);
    run_integration_tests.step.dependOn(b.getInstallStep()); // Ensure dcc binary is built first

    const integration_test_step = b.step("test-integration", "Run integration tests");
    integration_test_step.dependOn(&run_integration_tests.step);

    // Make the main test step depend on both unit and integration tests
    test_step.dependOn(&run_integration_tests.step);

    // Add specific MLIR codegen test step
    const mlir_codegen_tests = b.addTest(.{
        .name = "mlir-codegen-tests",
        .root_source_file = b.path("src/mlir_codegen.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Configure MLIR test with all necessary dependencies
    createLinking(b, mlir_codegen_tests, llvm_include_dir, llvm_lib_dir, lld_include_dir, lld_lib_dir, false);

    const run_mlir_codegen_tests = b.addRunArtifact(mlir_codegen_tests);

    const mlir_test_step = b.step("test-mlir", "Run MLIR codegen tests");
    mlir_test_step.dependOn(&run_mlir_codegen_tests.step);

    // Also add MLIR tests to the main test step
    test_step.dependOn(&run_mlir_codegen_tests.step);
}
