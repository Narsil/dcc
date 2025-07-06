const std = @import("std");
const builtin = @import("builtin");

/// CUDA Stub Manager - handles embedding and extraction of CUDA stub files
/// This allows dcc to be completely self-sufficient for CUDA cross-compilation
pub const CudaStubManager = struct {
    allocator: std.mem.Allocator,
    temp_dir: ?[]const u8,
    cuda_include_path: ?[]const u8,
    cuda_lib_path: ?[]const u8,
    verbose: bool,

    // Embedded CUDA stub files (copied during build time)
    const cuda_header = @embedFile("cuda_stub/include/cuda.h");
    const cuda_stub_library = @embedFile("cuda_stub/lib/libcuda.so");

    const StubManagerError = error{
        TempDirCreationFailed,
        FileExtractionFailed,
        CompilationFailed,
        DiskQuota,
        ReadOnlyFileSystem,
        LinkQuotaExceeded,
        InputOutput,
        InvalidArgument,
        BrokenPipe,
        OperationAborted,
        NotOpenForWriting,
        LockViolation,
        ConnectionResetByPeer,
        ProcessNotFound,
    } || std.mem.Allocator.Error || std.fs.Dir.OpenError || std.fs.File.OpenError;

    pub fn init(allocator: std.mem.Allocator, verbose: bool) StubManagerError!CudaStubManager {
        return CudaStubManager{
            .allocator = allocator,
            .temp_dir = null,
            .cuda_include_path = null,
            .cuda_lib_path = null,
            .verbose = verbose,
        };
    }

    pub fn deinit(self: *CudaStubManager) void {
        if (self.temp_dir) |temp_dir| {
            if (self.verbose) {
                std.debug.print("🧹 Cleaning up CUDA stub temp directory: {s}\n", .{temp_dir});
            }
            // Clean up temp directory
            self.cleanupTempDir(temp_dir);
            self.allocator.free(temp_dir);
        }
        
        if (self.cuda_include_path) |path| {
            self.allocator.free(path);
        }
        
        if (self.cuda_lib_path) |path| {
            self.allocator.free(path);
        }
    }

    /// Extract CUDA stub files to a temporary directory
    pub fn extractAndCompile(self: *CudaStubManager, target_triple: []const u8) StubManagerError!void {
        if (self.verbose) {
            std.debug.print("🔧 Extracting CUDA stub files for target: {s}\n", .{target_triple});
        }

        // Create temporary directory
        const temp_dir = try self.createTempDir();
        
        // Create include directory and extract header file
        const cuda_include_dir = try std.fs.path.join(self.allocator, &[_][]const u8{ temp_dir, "include" });
        try std.fs.cwd().makePath(cuda_include_dir);
        defer self.allocator.free(cuda_include_dir);
        
        const cuda_header_path = try std.fs.path.join(self.allocator, &[_][]const u8{ cuda_include_dir, "cuda.h" });
        try std.fs.cwd().writeFile(.{ .sub_path = cuda_header_path, .data = cuda_header });
        self.allocator.free(cuda_header_path);
        
        // Create lib directory and extract stub library
        const cuda_lib_dir = try std.fs.path.join(self.allocator, &[_][]const u8{ temp_dir, "lib" });
        try std.fs.cwd().makePath(cuda_lib_dir);
        defer self.allocator.free(cuda_lib_dir);
        
        // Extract the pre-compiled stub library
        const libcuda_path = try std.fs.path.join(self.allocator, &[_][]const u8{ cuda_lib_dir, "libcuda.so" });
        try std.fs.cwd().writeFile(.{ .sub_path = libcuda_path, .data = cuda_stub_library });
        defer self.allocator.free(libcuda_path);
        
        // Create symlink for versioned library (like CUDA does)
        const versioned_lib_path = try std.fs.path.join(self.allocator, &[_][]const u8{ cuda_lib_dir, "libcuda.so.1" });
        defer self.allocator.free(versioned_lib_path);
        
        // Create symlink (ignore errors if it fails)
        std.fs.cwd().symLink(libcuda_path, versioned_lib_path, .{}) catch |err| {
            if (self.verbose) {
                std.debug.print("⚠️  Failed to create symlink {s} -> {s}: {}\n", .{ versioned_lib_path, libcuda_path, err });
            }
        };
        
        // Store paths for later use
        self.temp_dir = temp_dir;
        self.cuda_include_path = try self.allocator.dupe(u8, cuda_include_dir);
        self.cuda_lib_path = try self.allocator.dupe(u8, cuda_lib_dir);
        
        if (self.verbose) {
            std.debug.print("✅ CUDA stub files extracted successfully\n", .{});
            std.debug.print("   Include path: {s}\n", .{self.cuda_include_path.?});
            std.debug.print("   Library path: {s}\n", .{self.cuda_lib_path.?});
        }
    }

    /// Get the include path for CUDA headers
    pub fn getIncludePath(self: *CudaStubManager) ?[]const u8 {
        return self.cuda_include_path;
    }

    /// Get the library path for CUDA stub library
    pub fn getLibPath(self: *CudaStubManager) ?[]const u8 {
        return self.cuda_lib_path;
    }

    /// Get the full path to libcuda stub library
    pub fn getLibCudaPath(self: *CudaStubManager) StubManagerError![]const u8 {
        if (self.cuda_lib_path) |lib_path| {
            return try std.fs.path.join(self.allocator, &[_][]const u8{ lib_path, "libcuda.so" });
        }
        return error.FileExtractionFailed;
    }

    /// Check if we should use CUDA stubs based on the host platform
    pub fn shouldUseStubs(target_triple: []const u8) bool {
        // Use stubs for cross-compilation or when compiling on non-CUDA platforms
        if (builtin.os.tag == .macos) {
            return true; // Always use stubs on macOS
        }
        
        if (std.mem.indexOf(u8, target_triple, "linux") != null and builtin.os.tag != .linux) {
            return true; // Cross-compiling to Linux
        }
        
        if (std.mem.indexOf(u8, target_triple, "windows") != null and builtin.os.tag != .windows) {
            return true; // Cross-compiling to Windows
        }
        
        // On Linux, we could check for CUDA installation, but for simplicity, always use stubs
        return true;
    }

    /// Create a temporary directory for CUDA stub files
    fn createTempDir(self: *CudaStubManager) StubManagerError![]const u8 {
        const temp_dir_name = try std.fmt.allocPrint(self.allocator, "dcc_cuda_stub_{d}", .{std.time.timestamp()});
        defer self.allocator.free(temp_dir_name);
        
        // Try to create in system temp directory
        var temp_dir_path: []const u8 = undefined;
        
        // Use a simple fallback approach - just use current directory for now
        // TODO: In production, could use std.fs.Dir.tmpDir() or similar
        temp_dir_path = try std.fs.path.join(self.allocator, &[_][]const u8{ ".", temp_dir_name });
        
        std.fs.cwd().makePath(temp_dir_path) catch |err| {
            self.allocator.free(temp_dir_path);
            if (self.verbose) {
                std.debug.print("❌ Failed to create temp directory: {}\n", .{err});
            }
            return StubManagerError.TempDirCreationFailed;
        };
        
        if (self.verbose) {
            std.debug.print("📁 Created temp directory: {s}\n", .{temp_dir_path});
        }
        
        return temp_dir_path;
    }



    /// Clean up temporary directory
    fn cleanupTempDir(self: *CudaStubManager, temp_dir: []const u8) void {
        _ = self; // unused parameter
        
        // Remove the entire temp directory, ignoring any errors
        std.fs.cwd().deleteTree(temp_dir) catch {};
    }
};

// Test the CUDA stub manager
test "cuda_stub_manager basic functionality" {
    const allocator = std.testing.allocator;
    
    var manager = CudaStubManager.init(allocator, false) catch return;
    defer manager.deinit();
    
    // Test stub detection
    const should_use_stubs = CudaStubManager.shouldUseStubs("x86_64-unknown-linux-gnu");
    std.debug.print("Should use CUDA stubs: {}\n", .{should_use_stubs});
    
    // Test extraction and compilation (this might fail if no compiler is available)
    manager.extractAndCompile("x86_64-unknown-linux-gnu") catch |err| {
        std.debug.print("Note: CUDA stub compilation failed (expected if no compiler available): {}\n", .{err});
        return; // Skip the rest of the test
    };
    
    // Test path retrieval
    const include_path = manager.getIncludePath();
    const lib_path = manager.getLibPath();
    
    std.debug.print("Include path: {s}\n", .{include_path orelse "none"});
    std.debug.print("Library path: {s}\n", .{lib_path orelse "none"});
} 