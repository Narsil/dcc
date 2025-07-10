const std = @import("std");
const parser = @import("../parser.zig");
const core = @import("core.zig");
const CodeGenError = core.CodeGenError;
const CodeGen = core.CodeGen;
const LLVM = core.LLVM;

// Darwin ARM64 system call numbers
const SYS_EXIT_DARWIN = 1; // exit() system call on Darwin

// LLD (LLVM Linker) C wrapper API
pub extern fn lld_main(args: [*]const [*:0]const u8, argc: c_int) c_int;

// Executable generation functions
pub fn generateExecutable(self: *CodeGen, output_path: []const u8, target: std.Target) CodeGenError!void {
    const llvm = try getLLVMTarget(self.allocator, target);
    defer LLVM.LLVMDisposeTargetMachine(llvm.machine);

    if (target.os.tag == .linux) {
        try generateStartFunction(self);
    }

    var error_message: [*c]u8 = undefined;

    // Generate object file
    const obj_path = try std.fmt.allocPrint(self.allocator, "{s}.o", .{output_path});
    defer self.allocator.free(obj_path);

    const obj_path_z = try self.allocator.dupeZ(u8, obj_path);
    defer self.allocator.free(obj_path_z);

    if (LLVM.LLVMTargetMachineEmitToFile(llvm.machine, self.module, obj_path_z.ptr, LLVM.LLVMObjectFile, &error_message) != 0) {
        defer LLVM.LLVMDisposeMessage(error_message);
        std.debug.print("Error generating object file: {s}\n", .{error_message});
        return error.CodeGenError;
    }

    if (self.verbose) {
        std.debug.print("Generated object file: {s}\n", .{obj_path});
    }

    // Link object file into executable
    try linkExecutable(self, obj_path, output_path, target);

    // Clean up object file
    std.fs.cwd().deleteFile(obj_path) catch {};
}

pub fn linkExecutable(self: *CodeGen, obj_path: []const u8, output_path: []const u8, target: std.Target) CodeGenError!void {
    // Use lld directly without external process calls
    var args = std.ArrayList([*:0]const u8).init(self.allocator);
    defer args.deinit();

    // Defer cleanup of allocated strings
    defer for (args.items) |arg| {
        self.allocator.free(std.mem.span(arg));
    };

    switch (target.os.tag) {
        .macos => {
            // Mach-O executable arguments
            try args.append(try self.allocator.dupeZ(u8, "ld64.lld"));
            try args.append(try self.allocator.dupeZ(u8, "-arch"));
            switch (target.cpu.arch) {
                .aarch64 => {
                    try args.append(try self.allocator.dupeZ(u8, "arm64"));
                },
                else => std.debug.panic("Unhandled arch on macos", .{}),
            }
            try args.append(try self.allocator.dupeZ(u8, "-platform_version"));
            try args.append(try self.allocator.dupeZ(u8, "macos"));
            try args.append(try self.allocator.dupeZ(u8, "10.15"));
            try args.append(try self.allocator.dupeZ(u8, "10.15"));
            try args.append(try self.allocator.dupeZ(u8, "-o"));
            try args.append(try self.allocator.dupeZ(u8, output_path));
        },
        .windows => {
            // COFF/PE executable arguments (note: no /dll flag for executables)
            try args.append(try self.allocator.dupeZ(u8, "lld-link"));
            try args.append(try self.allocator.dupeZ(u8, "/subsystem:console"));
            try args.append(try std.fmt.allocPrintZ(self.allocator, "/out:{s}", .{output_path}));
        },
        else => {
            // ELF executable arguments
            try args.append(try self.allocator.dupeZ(u8, "ld.lld"));
            try args.append(try self.allocator.dupeZ(u8, "--entry=_start"));
            // Only set dynamic linker if we're linking against dynamic libraries (i.e., when accelerator is present)
            if (self.accelerator != null) {
                try args.append(try self.allocator.dupeZ(u8, "--dynamic-linker=/lib64/ld-linux-x86-64.so.2"));
            }
            try args.append(try self.allocator.dupeZ(u8, "-o"));
            try args.append(try self.allocator.dupeZ(u8, output_path));
        },
    }

    // Add object file
    try args.append(try self.allocator.dupeZ(u8, obj_path));

    // No libc needed - using direct syscalls for IO

    // Add CUDA library linking for Linux targets if GPU code is present
    if (target.os.tag == .linux and self.accelerator != null) {
        // Check if we have CUDA stub libraries available
        if (self.accelerator) |*accel| {
            const stub_lib_path = accel.stub.getLibCudaPath() catch |err| {
                if (self.verbose) {
                    std.debug.print("⚠️  Warning: Failed to get CUDA stub library path: {}\n", .{err});
                }
                // Fall back to system libraries
                try args.append(try self.allocator.dupeZ(u8, "-lcuda"));

                if (self.verbose) {
                    std.debug.print("🔗 Added system CUDA library linking (fallback)\n", .{});
                }
                return;
            };
            defer self.allocator.free(stub_lib_path);

            // Add the stub library directory for linking
            const stub_lib_dir = std.fs.path.dirname(stub_lib_path) orelse ".";
            const lib_dir_flag = try std.fmt.allocPrintZ(self.allocator, "-L{s}", .{stub_lib_dir});
            try args.append(lib_dir_flag);

            // Set RPATH to standard Linux CUDA library paths for cross-compilation
            // For cross-compilation, use the most common Linux CUDA library paths
            const linux_cuda_paths = [_][]const u8{
                "/run/opengl-driver/lib", // NixOS
                "/usr/local/cuda/lib64", // Standard CUDA installation
                "/usr/lib/x86_64-linux-gnu", // Ubuntu/Debian
                "/usr/lib64", // RHEL/CentOS
            };

            // Add multiple RPATH entries for better compatibility
            for (linux_cuda_paths) |cuda_path| {
                const rpath_flag = try std.fmt.allocPrintZ(self.allocator, "--rpath={s}", .{cuda_path});
                try args.append(rpath_flag);
            }

            if (self.verbose) {
                std.debug.print("🔗 Using standard Linux CUDA library paths in RPATH for cross-compilation\n", .{});
            }

            try args.append(try self.allocator.dupeZ(u8, "-lcuda"));

            if (self.verbose) {
                std.debug.print("🔗 Added CUDA stub library for linking: {s}\n", .{stub_lib_path});
            }
        } else {
            // No accelerator, use system libraries
            try args.append(try self.allocator.dupeZ(u8, "-lcuda"));

            if (self.verbose) {
                std.debug.print("🔗 Added system CUDA library linking (no accelerator)\n", .{});
            }
        }
    }

    // Call lld_main with all arguments
    if (self.verbose) {
        std.debug.print("LLD Arguments: ", .{});
        for (args.items, 0..) |arg, i| {
            std.debug.print("{s}", .{std.mem.span(arg)});
            if (i < args.items.len - 1) {
                std.debug.print(" ", .{});
            }
        }
        std.debug.print("\n", .{});
    }
    const result = lld_main(args.items.ptr, @intCast(args.items.len));

    if (result != 0) {
        std.debug.print("lld executable linking failed with code: {}\n", .{result});
        return error.LinkingFailed;
    }

    if (self.verbose) {
        std.debug.print("Generated executable: {s}\n", .{output_path});
    }
}

pub fn generateSharedLibrary(self: *CodeGen, output_path: []const u8, target: std.Target) CodeGenError!void {
    const llvm = try getLLVMTarget(self.allocator, target);
    defer LLVM.LLVMDisposeTargetMachine(llvm.machine);

    // Generate object file
    const obj_path = try std.fmt.allocPrint(self.allocator, "{s}.o", .{output_path});
    defer self.allocator.free(obj_path);

    const obj_path_z = try self.allocator.dupeZ(u8, obj_path);
    defer self.allocator.free(obj_path_z);

    var error_message: [*c]u8 = undefined;
    if (LLVM.LLVMTargetMachineEmitToFile(llvm.machine, self.module, obj_path_z.ptr, LLVM.LLVMObjectFile, &error_message) != 0) {
        defer LLVM.LLVMDisposeMessage(error_message);
        std.debug.print("Error generating object file: {s}\n", .{error_message});
        return error.CodeGenError;
    }

    if (self.verbose) {
        std.debug.print("Generated object file: {s}\n", .{obj_path});
    }

    const lib_extension = switch (target.os.tag) {
        .macos => ".dylib",
        .windows => ".dll",
        else => ".so",
    };
    const lib_file = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ output_path, lib_extension });
    defer self.allocator.free(lib_file);

    // Link object file into shared library
    try linkSharedLibrary(self, obj_path, lib_file, target);

    // Clean up object file
    std.fs.cwd().deleteFile(obj_path) catch {};
}

pub fn linkSharedLibrary(self: *CodeGen, obj_path: []const u8, output_path: []const u8, target: std.Target) CodeGenError!void {
    // Use lld directly without external process calls
    var args = std.ArrayList([*:0]const u8).init(self.allocator);
    defer args.deinit();

    // Defer cleanup of allocated strings
    defer for (args.items) |arg| {
        self.allocator.free(std.mem.span(arg));
    };

    switch (target.os.tag) {
        .macos => {
            // Mach-O shared library (dylib) arguments
            try args.append(try self.allocator.dupeZ(u8, "ld64.lld"));
            try args.append(try self.allocator.dupeZ(u8, "-arch"));
            switch (target.cpu.arch) {
                .aarch64 => {
                    try args.append(try self.allocator.dupeZ(u8, "arm64"));
                },
                else => std.debug.panic("Unhandled arch on macos", .{}),
            }

            try args.append(try self.allocator.dupeZ(u8, "-platform_version"));
            try args.append(try self.allocator.dupeZ(u8, "macos"));
            try args.append(try self.allocator.dupeZ(u8, "10.15"));
            try args.append(try self.allocator.dupeZ(u8, "10.15"));
            try args.append(try self.allocator.dupeZ(u8, "-dylib"));
            try args.append(try self.allocator.dupeZ(u8, "-o"));
            try args.append(try self.allocator.dupeZ(u8, output_path));
        },
        .windows => {
            try args.append(try self.allocator.dupeZ(u8, "/dll"));
            try args.append(try std.fmt.allocPrintZ(self.allocator, "/out:{s}", .{output_path}));
        },
        else => {
            try args.append(try self.allocator.dupeZ(u8, "ld.lld"));
            try args.append(try self.allocator.dupeZ(u8, "--shared"));
            try args.append(try self.allocator.dupeZ(u8, "-o"));
            try args.append(try self.allocator.dupeZ(u8, output_path));
        },
    }

    // Add object file
    try args.append(try self.allocator.dupeZ(u8, obj_path));

    // Call lld_main with all arguments
    const result = lld_main(args.items.ptr, @intCast(args.items.len));

    if (result != 0) {
        std.debug.print("lld shared library linking failed with code: {}\n", .{result});
        return error.LinkingFailed;
    }

    if (self.verbose) {
        std.debug.print("Generated shared library: {s}\n", .{output_path});
    }
}

// Executable format creation functions
pub fn createELFExecutable(self: *CodeGen, obj_data: []const u8, output_path: []const u8) !void {
    _ = self;
    var elf_file = try std.fs.cwd().createFile(output_path, .{ .mode = 0o755 });
    defer elf_file.close();

    var writer = elf_file.writer();

    // ELF Header
    const ehdr = std.elf.Elf64_Ehdr{
        .e_ident = .{ 0x7f, 'E', 'L', 'F', std.elf.ELFCLASS64, std.elf.ELFDATA2LSB, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        .e_type = std.elf.ET.EXEC,
        .e_machine = std.elf.EM.X86_64,
        .e_version = 1,
        .e_entry = 0x401000, // Entry point
        .e_phoff = @sizeOf(std.elf.Elf64_Ehdr),
        .e_shoff = 0, // Will be set later
        .e_flags = 0,
        .e_ehsize = @sizeOf(std.elf.Elf64_Ehdr),
        .e_phentsize = @sizeOf(std.elf.Elf64_Phdr),
        .e_phnum = 1,
        .e_shentsize = @sizeOf(std.elf.Elf64_Shdr),
        .e_shnum = 0, // Will be set later
        .e_shstrndx = 0, // Will be set later
    };
    try writer.writeAll(std.mem.asBytes(&ehdr));

    // Program Header
    const phdr = std.elf.Elf64_Phdr{
        .p_type = std.elf.PT_LOAD,
        .p_flags = std.elf.PF_R | std.elf.PF_X,
        .p_offset = 0x1000,
        .p_vaddr = 0x401000,
        .p_paddr = 0x401000,
        .p_filesz = obj_data.len,
        .p_memsz = obj_data.len,
        .p_align = 0x1000,
    };
    try writer.writeAll(std.mem.asBytes(&phdr));

    // Pad to the text section
    const padding = 0x1000 - (@sizeOf(std.elf.Elf64_Ehdr) + @sizeOf(std.elf.Elf64_Phdr));
    for (0..padding) |_| {
        try writer.writeByte(0);
    }

    // Write the object data
    try writer.writeAll(obj_data);
}

pub fn createMachOExecutable(self: *CodeGen, object_data: []const u8, output_path: []const u8) CodeGenError!void {
    // Pure LLVM approach: Create Mach-O executable without any external process calls
    // Fix symbol naming issue by using proper entry point

    // Parse the object file to extract the code section
    var object_code: []const u8 = undefined;
    var text_size: u32 = 0;

    // Create a memory buffer from object data
    const memory_buffer = LLVM.LLVMCreateMemoryBufferWithMemoryRange(object_data.ptr, object_data.len, "object_buffer", 0 // don't require null termination
    );
    defer if (memory_buffer != null) {
        // Skip disposal to avoid potential crashes with LLVM memory management
        // LLVM.LLVMDisposeMemoryBuffer(memory_buffer);
    };

    if (memory_buffer == null) {
        std.debug.print("Failed to create memory buffer\n", .{});
        return CodeGenError.LinkingFailed;
    }

    // Create an object file from the memory buffer
    const object_file = LLVM.LLVMCreateObjectFile(memory_buffer.?);
    defer if (object_file != null) {
        LLVM.LLVMDisposeObjectFile(object_file.?);
    };

    if (object_file == null) {
        std.debug.print("Failed to create object file\n", .{});
        return CodeGenError.LinkingFailed;
    }

    // Get sections iterator
    const sections_iterator = LLVM.LLVMGetSections(object_file.?);
    defer LLVM.LLVMDisposeSectionIterator(sections_iterator);

    // Find the __text section
    while (LLVM.LLVMIsSectionIteratorAtEnd(object_file.?, sections_iterator) == 0) {
        const section_name_ptr = LLVM.LLVMGetSectionName(sections_iterator);
        const section_name = std.mem.span(section_name_ptr);

        if (std.mem.eql(u8, section_name, "__text")) {
            const section_size = LLVM.LLVMGetSectionSize(sections_iterator);
            const section_contents = LLVM.LLVMGetSectionContents(sections_iterator);

            text_size = @intCast(section_size);
            object_code = section_contents[0..text_size];
            break;
        }

        LLVM.LLVMMoveToNextSection(sections_iterator);
    }

    if (text_size == 0) {
        std.debug.print("No __text section found in object file\n", .{});
        return CodeGenError.LinkingFailed;
    }

    // Create pure LLVM Mach-O executable
    var executable_data = std.ArrayList(u8).init(self.allocator);
    defer executable_data.deinit();

    // Mach-O header for ARM64 - using _start as entry point with proper Darwin syscalls
    const header = std.mem.zeroes([32]u8);
    var header_data = header;

    std.mem.writeInt(u32, header_data[0..4], 0xfeedfacf, .little); // MH_MAGIC_64
    std.mem.writeInt(u32, header_data[4..8], 0x0100000c, .little); // CPU_TYPE_ARM64
    std.mem.writeInt(u32, header_data[8..12], 0, .little); // CPU_SUBTYPE_ARM_ALL
    std.mem.writeInt(u32, header_data[12..16], 2, .little); // MH_EXECUTE
    std.mem.writeInt(u32, header_data[16..20], 3, .little); // ncmds (3 load commands)
    std.mem.writeInt(u32, header_data[20..24], 176, .little); // sizeofcmds
    std.mem.writeInt(u32, header_data[24..28], 0x00200001, .little); // flags (NOUNDEFS)
    std.mem.writeInt(u32, header_data[28..32], 0, .little); // reserved

    try executable_data.appendSlice(&header_data);

    // LC_SEGMENT_64 for __PAGEZERO
    const pagezero_cmd = std.mem.zeroes([72]u8);
    var pagezero_data = pagezero_cmd;
    std.mem.writeInt(u32, pagezero_data[0..4], 0x19, .little); // LC_SEGMENT_64
    std.mem.writeInt(u32, pagezero_data[4..8], 72, .little); // cmdsize
    @memcpy(pagezero_data[8..24], "__PAGEZERO\x00\x00\x00\x00\x00\x00");
    std.mem.writeInt(u64, pagezero_data[24..32], 0, .little); // vmaddr
    std.mem.writeInt(u64, pagezero_data[32..40], 0x100000000, .little); // vmsize
    std.mem.writeInt(u64, pagezero_data[40..48], 0, .little); // fileoff
    std.mem.writeInt(u64, pagezero_data[48..56], 0, .little); // filesize
    std.mem.writeInt(u32, pagezero_data[56..60], 0, .little); // maxprot
    std.mem.writeInt(u32, pagezero_data[60..64], 0, .little); // initprot
    std.mem.writeInt(u32, pagezero_data[64..68], 0, .little); // nsects
    std.mem.writeInt(u32, pagezero_data[68..72], 0, .little); // flags

    try executable_data.appendSlice(&pagezero_data);

    // LC_SEGMENT_64 for __TEXT with __text section
    const text_file_offset: u32 = 32 + 176;
    const text_vaddr: u64 = 0x100000000;

    const text_cmd = std.mem.zeroes([152]u8);
    var text_data = text_cmd;
    std.mem.writeInt(u32, text_data[0..4], 0x19, .little); // LC_SEGMENT_64
    std.mem.writeInt(u32, text_data[4..8], 152, .little); // cmdsize
    @memcpy(text_data[8..24], "__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
    std.mem.writeInt(u64, text_data[24..32], text_vaddr, .little); // vmaddr
    std.mem.writeInt(u64, text_data[32..40], 0x1000, .little); // vmsize
    std.mem.writeInt(u64, text_data[40..48], text_file_offset, .little); // fileoff
    std.mem.writeInt(u64, text_data[48..56], text_size, .little); // filesize
    std.mem.writeInt(u32, text_data[56..60], 5, .little); // maxprot (READ | EXECUTE)
    std.mem.writeInt(u32, text_data[60..64], 5, .little); // initprot (READ | EXECUTE)
    std.mem.writeInt(u32, text_data[64..68], 1, .little); // nsects
    std.mem.writeInt(u32, text_data[68..72], 0, .little); // flags

    // __text section header
    @memcpy(text_data[72..88], "__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
    @memcpy(text_data[88..104], "__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
    std.mem.writeInt(u64, text_data[104..112], text_vaddr, .little); // addr
    std.mem.writeInt(u64, text_data[112..120], text_size, .little); // size
    std.mem.writeInt(u32, text_data[120..124], text_file_offset, .little); // offset
    std.mem.writeInt(u32, text_data[124..128], 2, .little); // align
    std.mem.writeInt(u32, text_data[128..132], 0, .little); // reloff
    std.mem.writeInt(u32, text_data[132..136], 0, .little); // nreloc
    std.mem.writeInt(u32, text_data[136..140], 0x80000400, .little); // flags
    std.mem.writeInt(u32, text_data[140..144], 0, .little); // reserved1
    std.mem.writeInt(u32, text_data[144..148], 0, .little); // reserved2
    std.mem.writeInt(u32, text_data[148..152], 0, .little); // reserved3

    try executable_data.appendSlice(&text_data);

    // LC_MAIN load command (use _start function as entry point)
    const main_cmd = std.mem.zeroes([24]u8);
    var main_data = main_cmd;
    std.mem.writeInt(u32, main_data[0..4], 0x80000028, .little); // LC_MAIN
    std.mem.writeInt(u32, main_data[4..8], 24, .little); // cmdsize
    // _start function is at offset 0x54 based on nm output
    std.mem.writeInt(u64, main_data[8..16], 0x54, .little); // entryoff to _start function
    std.mem.writeInt(u64, main_data[16..24], 0, .little); // stacksize

    try executable_data.appendSlice(&main_data);

    // Pad to text section file offset
    while (executable_data.items.len < text_file_offset) {
        try executable_data.append(0);
    }

    // Write the actual code section
    try executable_data.writer().writeAll(object_code);

    // Write to file
    var file = std.fs.cwd().createFile(output_path, .{ .mode = 0o755 }) catch |err| {
        std.debug.print("Failed to create output file: {}\n", .{err});
        return CodeGenError.LinkingFailed;
    };
    defer file.close();

    file.writeAll(executable_data.items) catch |err| {
        std.debug.print("Failed to write executable: {}\n", .{err});
        return CodeGenError.LinkingFailed;
    };

    std.debug.print("Created pure LLVM Mach-O executable: {s} ({} bytes) ✅\n", .{ output_path, executable_data.items.len });
}

pub fn createPEExecutable(self: *CodeGen, obj_data: []const u8, output_path: []const u8) CodeGenError!void {
    // Create a minimal PE executable
    var pe_data = std.ArrayList(u8).init(self.allocator);
    defer pe_data.deinit();

    // DOS Header
    try pe_data.appendSlice("MZ");
    try pe_data.appendNTimes(0, 58); // DOS header padding
    try pe_data.writer().writeInt(u32, 0x80, .little); // PE header offset

    // Write to file
    const file = std.fs.cwd().createFile(output_path, .{}) catch |err| {
        std.debug.print("Error creating PE executable: {}\n", .{err});
        return error.CodeGenError;
    };
    defer file.close();

    file.writeAll(pe_data.items) catch |err| {
        std.debug.print("Error writing PE data: {}\n", .{err});
        return error.CodeGenError;
    };
    file.writeAll(obj_data) catch |err| {
        std.debug.print("Error writing object data: {}\n", .{err});
        return error.CodeGenError;
    };
}

pub fn createELFSharedLibrary(self: *CodeGen, obj_data: []const u8, output_path: []const u8) anyerror!void {
    // Create a minimal ELF shared library
    try createELFExecutable(self, obj_data, output_path); // Simplified for now
}

pub fn createMachODylib(self: *CodeGen, obj_data: []const u8, output_path: []const u8) CodeGenError!void {
    // Create a minimal Mach-O dylib
    try createMachOExecutable(self, obj_data, output_path); // Simplified for now
}

pub fn createPEDLL(self: *CodeGen, obj_data: []const u8, output_path: []const u8) CodeGenError!void {
    // Create a minimal PE DLL
    try createPEExecutable(self, obj_data, output_path); // Simplified for now
}

pub fn generateSimpleEntryPoint(module: LLVM.LLVMModuleRef, context: LLVM.LLVMContextRef, builder: LLVM.LLVMBuilderRef, main_func: LLVM.LLVMValueRef) !LLVM.LLVMValueRef {
    // Create an entry point function that calls main() and then exits properly
    const i32_type = LLVM.LLVMInt32TypeInContext(context);
    const i64_type = LLVM.LLVMInt64TypeInContext(context);
    const void_type = LLVM.LLVMVoidTypeInContext(context);
    const entry_type = LLVM.LLVMFunctionType(i32_type, null, 0, 0);
    const entry_func = LLVM.LLVMAddFunction(module, "_start", entry_type);

    // Declare the external exit function from libSystem
    const exit_type = LLVM.LLVMFunctionType(void_type, &[_]LLVM.LLVMTypeRef{i32_type}, 1, 0);
    const exit_func = LLVM.LLVMAddFunction(module, "exit", exit_type);

    const entry_block = LLVM.LLVMAppendBasicBlockInContext(context, entry_func, "entry");
    LLVM.LLVMPositionBuilderAtEnd(builder, entry_block);

    // Get the main function type (should be i64 () )
    const main_type = LLVM.LLVMFunctionType(i64_type, null, 0, 0);

    // Call main function
    const main_call = LLVM.LLVMBuildCall2(builder, main_type, main_func, null, 0, "main_result");

    // Convert the exit code from main (i64) to i32
    const exit_code = LLVM.LLVMBuildTrunc(builder, main_call, i32_type, "exit_code");

    // Call exit() from libSystem
    _ = LLVM.LLVMBuildCall2(builder, exit_type, exit_func, &[_]LLVM.LLVMValueRef{exit_code}, 1, "");

    // This should never be reached, but add unreachable just in case
    _ = LLVM.LLVMBuildUnreachable(builder);

    return entry_func;
}

pub fn generateEntryPoint(module: LLVM.LLVMModuleRef, context: LLVM.LLVMContextRef, builder: LLVM.LLVMBuilderRef, main_func: LLVM.LLVMValueRef) !LLVM.LLVMValueRef {
    // Create entry point function that calls main() and then exits
    const i32_type = LLVM.LLVMInt32TypeInContext(context);
    const i64_type = LLVM.LLVMInt64TypeInContext(context);
    const entry_type = LLVM.LLVMFunctionType(i32_type, null, 0, 0);
    const entry_func = LLVM.LLVMAddFunction(module, "_start", entry_type);

    const entry_block = LLVM.LLVMAppendBasicBlockInContext(context, entry_func, "entry");
    LLVM.LLVMPositionBuilderAtEnd(builder, entry_block);

    // Get the main function type (should be i64 () )
    const main_type = LLVM.LLVMFunctionType(i64_type, null, 0, 0);

    // Call main function
    const main_call = LLVM.LLVMBuildCall2(builder, main_type, main_func, null, 0, "main_result");

    // Convert the exit code from main (i64) to i32 for the exit syscall
    const exit_code = LLVM.LLVMBuildTrunc(builder, main_call, i32_type, "exit_code");

    // Create Darwin ARM64 system call for exit
    // Darwin uses svc #0x80 instead of svc #0
    const inline_asm_type = LLVM.LLVMFunctionType(LLVM.LLVMVoidTypeInContext(context), &[_]LLVM.LLVMTypeRef{i32_type}, 1, 0);
    const inline_asm = LLVM.LLVMGetInlineAsm(inline_asm_type, "mov x16, #1\nmov x0, $0\nsvc #0x80", // Darwin ARM64 exit syscall
        "r", // input constraint: general register
        1, // has side effects
        0, // align stack
        0, // ATT dialect
        0 // can throw
    );

    // Call the inline assembly with exit code
    _ = LLVM.LLVMBuildCall2(builder, inline_asm_type, inline_asm, &[_]LLVM.LLVMValueRef{exit_code}, 1, "");

    // Add unreachable instruction (should never be reached)
    _ = LLVM.LLVMBuildUnreachable(builder);

    return entry_func;
}

pub fn getLLVMTarget(allocator: std.mem.Allocator, target: std.Target) !struct { target: LLVM.LLVMTargetRef, machine: LLVM.LLVMTargetMachineRef } {
    var target_llvm: LLVM.LLVMTargetRef = null;
    var error_message: [*c]u8 = undefined;
    const triple = try targetToTriple(allocator, target);
    defer allocator.free(triple);
    if (LLVM.LLVMGetTargetFromTriple(triple.ptr, &target_llvm, &error_message) != 0) {
        defer LLVM.LLVMDisposeMessage(error_message);
        std.debug.print("Error getting target: {s}\n", .{error_message});
        std.debug.print("Tried target triple: {s}\n", .{triple});
        return error.TargetError;
    }
    // Create target machine optimized for shared libraries
    const machine = LLVM.LLVMCreateTargetMachine(target_llvm.?, triple.ptr, "generic", "", LLVM.LLVMCodeGenLevelDefault, LLVM.LLVMRelocPIC, // Position Independent Code for shared libraries
        LLVM.LLVMCodeModelDefault);
    return .{ .target = target_llvm, .machine = machine };
}

/// Helper function to convert std.Target to target triple string
pub fn targetToTriple(allocator: std.mem.Allocator, target: std.Target) ![]u8 {
    const arch_str = switch (target.cpu.arch) {
        .x86_64 => "x86_64",
        .aarch64 => "aarch64",
        .arm => "arm",
        .riscv64 => "riscv64",
        else => @tagName(target.cpu.arch),
    };

    const os_str = switch (target.os.tag) {
        .macos => "apple-darwin",
        .linux => "unknown-linux-gnu",
        .windows => "pc-windows-msvc",
        else => @tagName(target.os.tag),
    };

    return try std.fmt.allocPrint(allocator, "{s}-{s}", .{ arch_str, os_str });
}

pub fn generateStartFunction(self: *CodeGen) CodeGenError!void {
    // Create _start function for Linux ELF executables
    // This function calls main() and returns the exit code

    const i32_type = LLVM.LLVMInt32TypeInContext(self.context);
    const start_function_type = LLVM.LLVMFunctionType(i32_type, null, 0, 0);
    const start_function = LLVM.LLVMAddFunction(self.module, "_start", start_function_type);

    // Set proper stack alignment attributes for x86-64 System V ABI compliance
    setStackAlignmentAttributes(self, start_function);

    // Create basic block
    const entry_block = LLVM.LLVMAppendBasicBlockInContext(self.context, start_function, "entry");
    LLVM.LLVMPositionBuilderAtEnd(self.builder, entry_block);

    // Add dummy alloca for stack alignment in _start (accounts for call instruction push)
    // Allocate 8 bytes of padding to help ensure 16-byte alignment before calling main
    const int8_type = LLVM.LLVMInt8TypeInContext(self.context);
    const dummy_array_type = LLVM.LLVMArrayType(int8_type, 8); // 8 bytes
    const dummy_name_start = "dummy_padding_start";
    const dummy_alloca_start = LLVM.LLVMBuildAlloca(self.builder, dummy_array_type, dummy_name_start);
    LLVM.LLVMSetAlignment(dummy_alloca_start, 16);

    // Get the main function
    const main_function_ = LLVM.LLVMGetNamedFunction(self.module, "main");
    if (main_function_ == null) {
        std.debug.print("Error: main function not found when generating _start\n", .{});
        return error.CodeGenError;
    }
    const main_function = main_function_.?;

    // Check if main returns void
    var exit_code: LLVM.LLVMValueRef = undefined;
    
    if (self.main_returns_void) {
        // Call void main function
        const void_type = LLVM.LLVMVoidTypeInContext(self.context);
        const main_function_type = LLVM.LLVMFunctionType(void_type, null, 0, 0);
        _ = LLVM.LLVMBuildCall2(self.builder, main_function_type, main_function, null, 0, "");
        
        // Use exit code 0 for void main
        exit_code = LLVM.LLVMConstInt(i32_type, 0, 0);
    } else {
        // Call main that returns i64
        const int64_type = LLVM.LLVMInt64TypeInContext(self.context);
        const main_function_type = LLVM.LLVMFunctionType(int64_type, null, 0, 0);
        const main_result = LLVM.LLVMBuildCall2(self.builder, main_function_type, main_function, null, 0, "main_result");
        
        // Convert i64 to i32 for exit code
        exit_code = LLVM.LLVMBuildTrunc(self.builder, main_result, i32_type, "exit_code");
    }

    // Store the exit code in a global variable for inspection
    const exit_code_global = LLVM.LLVMAddGlobal(self.module, i32_type, "program_exit_code");
    LLVM.LLVMSetInitializer(exit_code_global, LLVM.LLVMConstInt(i32_type, 0, 0));
    _ = LLVM.LLVMBuildStore(self.builder, exit_code, exit_code_global);

    // --- Exit the process via Linux x86_64 syscall ---
    const void_type = LLVM.LLVMVoidTypeInContext(self.context);
    const int64_type = LLVM.LLVMInt64TypeInContext(self.context);
    const param_types = [_]LLVM.LLVMTypeRef{int64_type};
    const syscall_asm_ty = LLVM.LLVMFunctionType(void_type, @constCast(&param_types[0]), 1, 0);
    const asm_str = "mov $0, %rdi\n mov $$231, %rax\nsyscall"; // rax=231 (SYS_exit_group), rdi=status

    const syscall_inline = LLVM.LLVMGetInlineAsm(syscall_asm_ty, asm_str, asm_str.len, "r", 1, // single general-purpose register input
        1, // has side effects
        0, // align stack
        0, // ATT dialect
        0 // can throw
    );

    const exit_code_64 = LLVM.LLVMBuildSExt(self.builder, exit_code, int64_type, "exit_code64");
    _ = LLVM.LLVMBuildCall2(self.builder, syscall_asm_ty, syscall_inline, @constCast(&[_]LLVM.LLVMValueRef{exit_code_64}), 1, "");

    // Mark unreachable as the syscall terminates the program
    _ = LLVM.LLVMBuildUnreachable(self.builder);

    if (self.verbose) {
        self.printIR();
    }
}

// Helper function to set stack alignment attributes on functions
pub fn setStackAlignmentAttributes(self: *CodeGen, function: LLVM.LLVMValueRef) void {
    // Set stack realignment for x86-64 System V ABI compliance
    const stackrealign_attr_name = "stackrealign";
    const stackrealign_kind = LLVM.LLVMGetEnumAttributeKindForName(stackrealign_attr_name.ptr, stackrealign_attr_name.len);
    const stackrealign_attr = LLVM.LLVMCreateEnumAttribute(self.context, stackrealign_kind, 0);
    LLVM.LLVMAddAttributeAtIndex(function, 0xffffffff, stackrealign_attr);

    // Set unwind table for proper exception handling and debugging
    const uwtable_attr_name = "uwtable";
    const uwtable_kind = LLVM.LLVMGetEnumAttributeKindForName(uwtable_attr_name.ptr, uwtable_attr_name.len);
    const uwtable_attr = LLVM.LLVMCreateEnumAttribute(self.context, uwtable_kind, 2); // sync uwtable
    LLVM.LLVMAddAttributeAtIndex(function, 0xffffffff, uwtable_attr);

    // Set proper calling convention for System V ABI
    LLVM.LLVMSetFunctionCallConv(function, 0); // C calling convention (System V ABI)

    // Ensure the backend maintains at least 16-byte stack alignment
    const alignstack_attr_name = "alignstack"; // LLVM attribute: alignstack(<alignment>)
    const alignstack_kind = LLVM.LLVMGetEnumAttributeKindForName(alignstack_attr_name.ptr, alignstack_attr_name.len);
    if (alignstack_kind != 0) { // only add if LLVM recognises the attribute
        const alignstack_attr = LLVM.LLVMCreateEnumAttribute(self.context, alignstack_kind, 16);
        LLVM.LLVMAddAttributeAtIndex(function, 0xffffffff, alignstack_attr);
    }
}
