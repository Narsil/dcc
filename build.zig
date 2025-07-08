const std = @import("std");
const builtin = @import("builtin");

// Create a build step to copy CUDA stub files from environment to src directory
fn createCudaStubCopyStep(b: *std.Build) *std.Build.Step {
    const cuda_include_dir = std.process.getEnvVarOwned(b.allocator, "CUDA_INCLUDE_DIR") catch {
        std.debug.print("🔧 CUDA_INCLUDE_DIR not found, skipping CUDA stub file copying\n", .{});
        // Return a no-op step if CUDA is not available
        const noop_step = b.step("cuda-stub-noop", "No-op step for missing CUDA");
        return noop_step;
    };
    defer b.allocator.free(cuda_include_dir);

    const cuda_lib_dir = std.process.getEnvVarOwned(b.allocator, "CUDA_LIB_DIR") catch {
        std.debug.print("🔧 CUDA_LIB_DIR not found, skipping CUDA stub file copying\n", .{});
        // Return a no-op step if CUDA is not available
        const noop_step = b.step("cuda-stub-noop", "No-op step for missing CUDA");
        return noop_step;
    };
    defer b.allocator.free(cuda_lib_dir);

    // Create destination directories
    const mkdir_step = b.addSystemCommand(&.{ "mkdir", "-p", "src/cuda_stub/include", "src/cuda_stub/lib" });

    // Copy CUDA header
    const src_header = std.fmt.allocPrint(b.allocator, "{s}/cuda.h", .{cuda_include_dir}) catch {
        const noop_step = b.step("cuda-stub-noop", "No-op step for CUDA header copy failure");
        return noop_step;
    };
    defer b.allocator.free(src_header);

    // Copy CUDA stub library
    const src_lib = std.fmt.allocPrint(b.allocator, "{s}/stubs/libcuda.so", .{cuda_lib_dir}) catch {
        const noop_step = b.step("cuda-stub-noop", "No-op step for CUDA lib copy failure");
        return noop_step;
    };
    defer b.allocator.free(src_lib);

    // Create a combined step that removes read-only files first, then copies
    const combined_copy_step = b.addSystemCommand(&.{ "sh", "-c" });
    const combined_cmd = std.fmt.allocPrint(b.allocator, "rm -f src/cuda_stub/include/cuda.h src/cuda_stub/lib/libcuda.so && cp '{s}' src/cuda_stub/include/cuda.h && cp '{s}' src/cuda_stub/lib/libcuda.so", .{ src_header, src_lib }) catch {
        const noop_step = b.step("cuda-stub-noop", "No-op step for combined copy failure");
        return noop_step;
    };
    defer b.allocator.free(combined_cmd);
    combined_copy_step.addArg(combined_cmd);
    combined_copy_step.step.dependOn(&mkdir_step.step);

    // Create a final step that depends on the combined copy
    const final_step = b.step("cuda-stub-copy", "Copy CUDA stub files");
    final_step.dependOn(&combined_copy_step.step);

    return final_step;
}

// Add this function to your build.zig file
fn buildLldWrapper(b: *std.Build, llvm_include_dir: []u8, lld_include_dir: []u8, lld_lib_dir: []u8) *std.Build.Step {
    const command_args = [_][]const u8{
        "gcc", "-print-file-name=libstdc++.a",
    };
    // Use builder.run() which handles spawning and waiting, and gives you the Child result
    // This is the cleanest way to run a child process and get its output within build.zig.
    const result = b.run(&command_args);
    const libstdcxx_path = std.mem.trim(u8, result, "\n");

    // Step 2: Create directory for build process
    const mkdir_cmd = b.addSystemCommand(&.{ "mkdir", "-p", "comprehensive_build" });

    // Step 3: Compile lld_wrapper.cpp to object file using g++ directly
    const wrapper_obj_cmd = b.addSystemCommand(&.{ "g++", "-c", "-std=c++17", "-fPIC", "-O2", "-o", "lld_wrapper.o", "src/lld_wrapper.cpp" });
    // Add include paths
    const llvm_include_arg = b.fmt("-I{s}", .{llvm_include_dir});
    const lld_include_arg = b.fmt("-I{s}", .{lld_include_dir});
    wrapper_obj_cmd.addArg(llvm_include_arg);
    wrapper_obj_cmd.addArg(lld_include_arg);
    wrapper_obj_cmd.step.dependOn(&mkdir_cmd.step); // Ensure directory exists before compiling

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

// Auto-link all static libraries in a directory matching a prefix
fn linkAllStaticLibraries(_: *std.Build, exe: *std.Build.Step.Compile, lib_dir: []const u8, prefix: []const u8, exclusion_list: ?[]const []const u8) !void {
    // std.debug.print("🔍 Auto-discovering {s} libraries in: {s}\n", .{ prefix, lib_dir });

    var dir = std.fs.cwd().openDir(lib_dir, .{ .iterate = true }) catch |err| {
        // std.debug.print("❌ Could not open library directory: {}\n", .{err});
        return err;
    };
    defer dir.close();

    var iterator = dir.iterate();
    var count: u32 = 0;
    var excluded_count: u32 = 0;

    while (try iterator.next()) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".a")) {
            // Check if it's a library matching our prefix (e.g., "libMLIR*.a")
            if (std.mem.startsWith(u8, entry.name, "lib") and
                std.mem.indexOf(u8, entry.name, prefix) != null)
            {

                // Extract library name: "libMLIRParser.a" -> "MLIRParser"
                const lib_name = entry.name[3 .. entry.name.len - 2]; // Remove "lib" and ".a"

                // Check if this library is in the exclusion list
                var should_exclude = false;
                if (exclusion_list) |exclusions| {
                    for (exclusions) |excluded_lib| {
                        if (std.mem.eql(u8, lib_name, excluded_lib)) {
                            should_exclude = true;
                            excluded_count += 1;
                            // std.debug.print("  🚫 Excluded: {s}\n", .{lib_name});
                            break;
                        }
                    }
                }

                if (!should_exclude) {
                    exe.linkSystemLibrary2(lib_name, .{ .preferred_link_mode = .static });
                    count += 1;

                    if (count <= 5) { // Show first 5 for debugging
                        // std.debug.print("  ✅ Linked: {s}\n", .{lib_name});
                    } else if (count == 6) {
                        // std.debug.print("  ... (showing first 5, found more)\n", .{});
                    }
                }
            }
        }
    }

    // std.debug.print("📚 Auto-linked {d} {s} libraries ({d} excluded)\n", .{ count, prefix, excluded_count });
}

fn createLinking(b: *std.Build, exe: *std.Build.Step.Compile, llvm_include_dir: []u8, llvm_lib_dir: []u8, lld_include_dir: []u8, lld_lib_dir: []u8, use_wrapper: bool) !void {
    // Copy build configuration to tests
    // exe.linkLibC();  // Removed - using Linux syscalls directly, no libc needed
    exe.linkLibCpp();

    // Auto-link all LLVM static libraries
    try linkAllStaticLibraries(b, exe, llvm_lib_dir, "LLVM", null);
    exe.linkSystemLibrary("z"); // zlib for compression (required by LLD)
    exe.linkSystemLibrary("xml2"); // zlib for compression (required by LLD)
    exe.linkSystemLibrary("Polly"); // zlib for compression (required by LLD)
    exe.linkSystemLibrary("PollyISL"); // zlib for compression (required by LLD)
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

        // Link ALL MLIR static libraries automatically
        const mlir_exclusions = [_][]const u8{
            "MLIRMlirOptMain", // Exclude specific Main library
        };
        try linkAllStaticLibraries(b, exe, mlir_lib_dir.?, "MLIR", &mlir_exclusions);
        // std.debug.print("MLIR support enabled (auto-linked all static libraries)\n", .{});
    } else {
        // std.debug.print("MLIR support not available (MLIR_INCLUDE_DIR and MLIR_LIB_DIR not set)\n", .{});
    }

    // Add CUDA cross-compilation support for --target x86_64-unknown-linux --gpu nvptx-cuda:sm_XX
    // Note: CUDA variables are now declared at the top level of build() function
    // This is just a placeholder comment - the actual CUDA linking is handled per-executable

    // Link essential LLD libraries for tests (static linking)
    exe.addIncludePath(.{ .cwd_relative = lld_include_dir });
    if (builtin.target.os.tag == .linux) {
        const lld_wrapper_step = buildLldWrapper(b, llvm_include_dir, lld_include_dir, lld_lib_dir);
        // Link with our comprehensive static library (depends on lld_wrapper_step)
        exe.addObjectFile(b.path("liblldwrapper.a"));
        exe.step.dependOn(lld_wrapper_step); // Main exe depends on LLD wrapper
    } else {

        // Auto-link all LLD libraries
        exe.addLibraryPath(.{ .cwd_relative = lld_lib_dir });
        try linkAllStaticLibraries(b, exe, lld_lib_dir, "lld", null);
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
    // Create CUDA stub copy step (but don't run it immediately)
    const cuda_stub_copy_step = createCudaStubCopyStep(b);

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

    // Make the executable depend on CUDA stub copying step
    exe.step.dependOn(cuda_stub_copy_step);

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

    // Get CUDA directories from environment (for cross-compilation)
    const cuda_include_dir = std.process.getEnvVarOwned(b.allocator, "CUDA_INCLUDE_DIR") catch null;
    const cuda_lib_dir = std.process.getEnvVarOwned(b.allocator, "CUDA_LIB_DIR") catch null;
    const cuda_stub_dir = std.process.getEnvVarOwned(b.allocator, "CUDA_STUB_DIR") catch null;
    defer if (cuda_include_dir) |dir| b.allocator.free(dir);
    defer if (cuda_lib_dir) |dir| b.allocator.free(dir);
    defer if (cuda_stub_dir) |dir| b.allocator.free(dir);

    exe.addIncludePath(.{ .cwd_relative = llvm_include_dir });
    exe.addLibraryPath(.{ .cwd_relative = llvm_lib_dir });
    // exe.addIncludePath(.{ .cwd_relative = lld_include_dir });
    exe.addLibraryPath(.{ .cwd_relative = "./" });

    try createLinking(b, exe, llvm_include_dir, llvm_lib_dir, lld_include_dir, lld_lib_dir, true);

    // Add our custom GPU to NVVM wrapper to main dcc build
    const mlir_include_dir_main = std.process.getEnvVarOwned(b.allocator, "MLIR_INCLUDE_DIR") catch null;
    const mlir_lib_dir_main = std.process.getEnvVarOwned(b.allocator, "MLIR_LIB_DIR") catch null;

    if (mlir_include_dir_main != null and mlir_lib_dir_main != null) {
        defer b.allocator.free(mlir_include_dir_main.?);
        defer b.allocator.free(mlir_lib_dir_main.?);

        // Add our GPU to NVVM wrapper for main dcc
        exe.addCSourceFile(.{
            .file = b.path("src/gpu_to_nvvm_wrapper.cpp"),
            .flags = &.{"-std=c++17"},
        });

        // Add include path for our wrapper header
        exe.addIncludePath(.{ .cwd_relative = "src" });
    }

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
    try createLinking(b, emit_ptx_exe, llvm_include_dir, llvm_lib_dir, lld_include_dir, lld_lib_dir, false);

    // Add our custom GPU to NVVM wrapper for emit_ptx
    const mlir_include_dir = std.process.getEnvVarOwned(b.allocator, "MLIR_INCLUDE_DIR") catch null;
    const mlir_lib_dir = std.process.getEnvVarOwned(b.allocator, "MLIR_LIB_DIR") catch null;

    if (mlir_include_dir != null and mlir_lib_dir != null) {
        defer b.allocator.free(mlir_include_dir.?);
        defer b.allocator.free(mlir_lib_dir.?);

        // Add our GPU to NVVM wrapper
        emit_ptx_exe.addCSourceFile(.{
            .file = b.path("src/gpu_to_nvvm_wrapper.cpp"),
            .flags = &.{"-std=c++17"},
        });

        // Add include path for our wrapper header
        emit_ptx_exe.addIncludePath(.{ .cwd_relative = "src" });
    }

    // Install emit_ptx executable
    b.installArtifact(emit_ptx_exe);

    // Create CUDA LLVM IR test executable
    const cuda_llvm_ir_test_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda_llvm_ir_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    const cuda_llvm_ir_test_exe = b.addExecutable(.{
        .name = "cuda_llvm_ir_test",
        .root_module = cuda_llvm_ir_test_mod,
    });

    // Link cuda_llvm_ir_test with the same libraries as main dcc
    try createLinking(b, cuda_llvm_ir_test_exe, llvm_include_dir, llvm_lib_dir, lld_include_dir, lld_lib_dir, false);

    // Install cuda_llvm_ir_test executable
    b.installArtifact(cuda_llvm_ir_test_exe);

    // Create run step for emit_ptx
    const run_emit_ptx_cmd = b.addRunArtifact(emit_ptx_exe);
    run_emit_ptx_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_emit_ptx_cmd.addArgs(args);
    }

    const run_emit_ptx_step = b.step("emit-ptx", "Run the emit_ptx tool");
    run_emit_ptx_step.dependOn(&run_emit_ptx_cmd.step);

    // Create run step for CUDA LLVM IR test
    const run_cuda_llvm_ir_test_cmd = b.addRunArtifact(cuda_llvm_ir_test_exe);
    run_cuda_llvm_ir_test_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cuda_llvm_ir_test_cmd.addArgs(args);
    }

    const run_cuda_llvm_ir_test_step = b.step("cuda-llvm-ir-test", "Run the CUDA LLVM IR generator test");
    run_cuda_llvm_ir_test_step.dependOn(&run_cuda_llvm_ir_test_cmd.step);

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

    try createLinking(b, exe_unit_tests, llvm_include_dir, llvm_lib_dir, lld_include_dir, lld_lib_dir, false);

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

    // Cross-compilation tests that require the dcc binary
    const cross_compilation_tests = b.addTest(.{
        .name = "cross-compilation-tests",
        .root_source_file = b.path("src/cross_compilation_tests.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add cross-compilation tests step
    const run_cross_compilation_tests = b.addRunArtifact(cross_compilation_tests);
    run_cross_compilation_tests.step.dependOn(b.getInstallStep()); // Ensure dcc binary is built first

    const cross_compilation_test_step = b.step("test-cross", "Run cross-compilation tests");
    cross_compilation_test_step.dependOn(&run_cross_compilation_tests.step);

    // Make the main test step depend on both unit and integration tests
    test_step.dependOn(&run_integration_tests.step);
    test_step.dependOn(&run_cross_compilation_tests.step);

    // Add specific MLIR codegen test step
    const mlir_codegen_tests = b.addTest(.{
        .name = "mlir-codegen-tests",
        .root_source_file = b.path("src/mlir_codegen.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Configure MLIR test with all necessary dependencies
    try createLinking(b, mlir_codegen_tests, llvm_include_dir, llvm_lib_dir, lld_include_dir, lld_lib_dir, false);

    const run_mlir_codegen_tests = b.addRunArtifact(mlir_codegen_tests);

    const mlir_test_step = b.step("test-mlir", "Run MLIR codegen tests");
    mlir_test_step.dependOn(&run_mlir_codegen_tests.step);

    // Also add MLIR tests to the main test step
    test_step.dependOn(&run_mlir_codegen_tests.step);
}
