const std = @import("std");
const cuda_llvm_ir_gen = @import("cuda_llvm_ir_gen.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Check for help flag
    if (args.len > 1 and (std.mem.eql(u8, args[1], "--help") or std.mem.eql(u8, args[1], "-h"))) {
        printUsage();
        return;
    }

    // Determine target triple
    var target_triple: []const u8 = "x86_64-unknown-linux-gnu"; // Default
    if (args.len > 1 and std.mem.startsWith(u8, args[1], "--target=")) {
        target_triple = args[1][9..]; // Skip "--target="
    }

    // Determine verbosity
    var verbose = false;
    // Determine if we should test complete executable generation
    var test_executable = false;
    for (args) |arg| {
        if (std.mem.eql(u8, arg, "--verbose") or std.mem.eql(u8, arg, "-v")) {
            verbose = true;
        } else if (std.mem.eql(u8, arg, "--test-executable") or std.mem.eql(u8, arg, "-e")) {
            test_executable = true;
        }
    }

    std.debug.print("🚀 CUDA LLVM IR Generator Test\n", .{});
    std.debug.print("Target: {s}\n", .{target_triple});
    std.debug.print("Verbose: {}\n", .{verbose});
    std.debug.print("Test Executable: {}\n", .{test_executable});
    std.debug.print("=" ** 50 ++ "\n", .{});

    // Initialize the generator
    var gen = cuda_llvm_ir_gen.CudaLLVMIRGen.init(allocator, target_triple, verbose) catch |err| {
        std.debug.print("❌ Failed to initialize generator: {}\n", .{err});
        return;
    };
    defer gen.deinit();

    if (verbose) {
        std.debug.print("✅ Generator initialized successfully\n", .{});
    }

    // Generate sample PTX (simulating what emit_ptx.zig would generate)
    const sample_ptx = generateSamplePTX();

    if (test_executable) {
        // Test complete executable generation (IR + compilation + linking)
        std.debug.print("🚀 Testing complete executable generation...\n", .{});
        const output_executable = "cuda_test_output";
        
        if (gen.generateExecutable(output_executable, sample_ptx, "demo")) {
            std.debug.print("✅ Executable generated successfully: {s}\n", .{output_executable});
            std.debug.print("   Note: The executable requires CUDA runtime libraries to run\n", .{});
            std.debug.print("   You can try running it with: ./{s}\n\n", .{output_executable});
        } else |err| {
            std.debug.print("⚠️  Executable generation failed (likely missing CUDA libraries): {}\n", .{err});
            std.debug.print("   This is expected if CUDA libraries are not installed\n", .{});
            std.debug.print("   Falling back to LLVM IR generation...\n\n", .{});
        }
    }

    // Generate CUDA wrapper functions (for IR output)
    std.debug.print("🔧 Generating CUDA wrapper functions...\n", .{});
    _ = try gen.generateCudaInitFunction();
    _ = try gen.generateCudaCreateContextFunction();
    _ = try gen.generateCudaLoadModuleFunction();
    _ = try gen.generateCudaCleanupFunction();

    // Generate sample PTX data
    std.debug.print("🔧 Generating sample PTX data...\n", .{});
    _ = try gen.generatePTXDataConstant("gpu_vector_add_ptx", sample_ptx);

    std.debug.print("✅ All CUDA functions and PTX constant generated\n", .{});

    // Output the generated LLVM IR
    std.debug.print("\n" ++ "=" ** 80 ++ "\n", .{});
    std.debug.print("🎯 Generated LLVM IR for CUDA Runtime + Embedded PTX:\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});

    if (verbose) {
        gen.printLLVMIR();
    } else {
        std.debug.print("(Use --verbose to see full LLVM IR)\n", .{});
    }

    std.debug.print("\n" ++ "=" ** 80 ++ "\n", .{});

    // Save to file
    const output_file = "cuda_output.ll";
    try gen.writeToFile(output_file);
    std.debug.print("✅ LLVM IR saved to: {s}\n", .{output_file});

    // Generate usage examples
    std.debug.print("\n💡 Usage Examples:\n", .{});
    std.debug.print("   📄 View generated IR: cat {s}\n", .{output_file});
    std.debug.print("   🔍 Verify IR syntax: llvm-as {s} -o /dev/null\n", .{output_file});
    std.debug.print("   🛠️  Compile to object: clang -c {s} -o cuda_program.o\n", .{output_file});
    std.debug.print("   🚀 Link with CUDA: clang cuda_program.o -lcuda -o cuda_program\n", .{});
    std.debug.print("   ⚡ Run (with CUDA): ./cuda_program\n", .{});

    std.debug.print("\n🎉 CUDA LLVM IR generation completed successfully!\n", .{});
}

fn generateSamplePTX() []const u8 {
    return
        \\// Generated PTX for function: gpu_vector_add
        \\.version 7.0
        \\.target sm_50
        \\.address_size 64
        \\
        \\.visible .entry gpu_vector_add(
        \\    .param .u64 gpu_vector_add_param_0,
        \\    .param .u64 gpu_vector_add_param_1,
        \\    .param .u64 gpu_vector_add_param_2,
        \\    .param .u32 gpu_vector_add_param_3
        \\) {
        \\    .reg .u64 %rd<10>;
        \\    .reg .u32 %r<10>;
        \\    .reg .f32 %f<10>;
        \\    
        \\    ld.param.u64 %rd1, [gpu_vector_add_param_0];
        \\    ld.param.u64 %rd2, [gpu_vector_add_param_1];
        \\    ld.param.u64 %rd3, [gpu_vector_add_param_2];
        \\    ld.param.u32 %r1, [gpu_vector_add_param_3];
        \\    
        \\    mov.u32 %r2, %tid.x;
        \\    mov.u32 %r3, %ntid.x;
        \\    mov.u32 %r4, %ctaid.x;
        \\    mad.lo.u32 %r5, %r4, %r3, %r2;
        \\    
        \\    setp.ge.u32 %p1, %r5, %r1;
        \\    @%p1 bra EXIT;
        \\    
        \\    cvta.to.global.u64 %rd4, %rd1;
        \\    cvta.to.global.u64 %rd5, %rd2;
        \\    cvta.to.global.u64 %rd6, %rd3;
        \\    
        \\    mul.wide.u32 %rd7, %r5, 4;
        \\    add.u64 %rd8, %rd4, %rd7;
        \\    add.u64 %rd9, %rd5, %rd7;
        \\    add.u64 %rd10, %rd6, %rd7;
        \\    
        \\    ld.global.f32 %f1, [%rd8];
        \\    ld.global.f32 %f2, [%rd9];
        \\    add.f32 %f3, %f1, %f2;
        \\    st.global.f32 [%rd10], %f3;
        \\    
        \\EXIT:
        \\    ret;
        \\}
    ;
}

fn printUsage() void {
    std.debug.print("CUDA LLVM IR Generator Test\n\n", .{});
    std.debug.print("Usage: cuda_llvm_ir_test [OPTIONS]\n\n", .{});
    std.debug.print("Options:\n", .{});
    std.debug.print("  --target=<triple>    Target triple (default: x86_64-unknown-linux-gnu)\n", .{});
    std.debug.print("  -v, --verbose        Enable verbose output\n", .{});
    std.debug.print("  -e, --test-executable Test complete executable generation (IR + compile + link)\n", .{});
    std.debug.print("  -h, --help          Show this help message\n", .{});
    std.debug.print("\nExamples:\n", .{});
    std.debug.print("  cuda_llvm_ir_test\n", .{});
    std.debug.print("  cuda_llvm_ir_test --verbose\n", .{});
    std.debug.print("  cuda_llvm_ir_test --test-executable\n", .{});
    std.debug.print("  cuda_llvm_ir_test --target=x86_64-pc-linux-gnu --verbose\n", .{});
    std.debug.print("  cuda_llvm_ir_test --target=aarch64-linux-gnu --test-executable\n", .{});
    std.debug.print("\nOutput:\n", .{});
    std.debug.print("  - Prints generated LLVM IR to stdout\n", .{});
    std.debug.print("  - Saves LLVM IR to cuda_output.ll\n", .{});
    std.debug.print("  - Shows compilation examples\n", .{});
} 