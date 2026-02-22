add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

-- Use vcpkg manifest mode (vcpkg.json)
-- Dependencies are automatically managed by vcpkg manifest
set_policy("package.requires_lock", true)

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
    includes("xmake/test_cuda.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-cuda")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    add_files("src/llaisys/models/*.cc")
    set_installdir(".")

    -- CUDA device linking: required so nvcc performs the final device-code link
    -- when static libs containing .cu objects are linked into this shared library
    if has_config("nv-gpu") then
        add_rules("cuda")
        add_cugencodes("native")

        before_link(function (target)
            import("lib.detect.find_tool")
            local nvcc = find_tool("nvcc")
            if not nvcc then
                raise("nvcc not found!")
            end

            -- Collect all .cu object files from CUDA static lib dependencies
            -- Windows: .cu.obj, Linux: .cu.o
            local cu_objects = {}
            local is_win = target:is_plat("windows")
            local obj_pattern = is_win and "%.cu%.obj$" or "%.cu%.o$"

            -- On Windows, nvcc needs -ccbin to locate cl.exe
            local ccbin_dir = nil
            if is_win then
                local cl = find_tool("cl")
                if cl then
                    ccbin_dir = path.directory(cl.program)
                end
            end
            for _, dep in ipairs(target:orderdeps()) do
                for _, obj in ipairs(dep:objectfiles()) do
                    if obj:match(obj_pattern) then
                        table.insert(cu_objects, obj)
                    end
                end
            end

            if #cu_objects > 0 then
                local dlink_ext = is_win and ".obj" or ".o"
                local dlink_obj = path.join(target:objectdir(), "cuda_dlink" .. dlink_ext)
                local argv = {"-dlink"}
                if not is_win then
                    table.insert(argv, "-shared")
                end
                table.insert(argv, "-gencode")
                table.insert(argv, "arch=compute_86,code=sm_86")
                -- Match MSVC dynamic CRT in device-link step
                if is_win then
                    table.insert(argv, "-Xcompiler")
                    table.insert(argv, "/MD")
                    if ccbin_dir then
                        table.insert(argv, "-ccbin")
                        table.insert(argv, ccbin_dir)
                    end
                end
                table.insert(argv, "-o")
                table.insert(argv, dlink_obj)
                for _, obj in ipairs(cu_objects) do
                    table.insert(argv, obj)
                end
                os.vrunv(nvcc.program, argv)
                -- Insert device-link object at the front so the linker includes it
                local objs = target:objectfiles()
                table.insert(objs, 1, dlink_obj)
                target:set("objectfiles", objs)
            end
        end)
    end

    -- BLAS linking
    if has_config("mkl") then
        -- Intel MKL linking
        local mkl_default = is_plat("windows")
            and "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
            or  "/opt/intel/oneapi/mkl/latest"
        local mkl_root = os.getenv("MKLROOT") or mkl_default

        if not is_plat("windows") then
            local mkl_lib = path.join(mkl_root, "lib", "intel64")
            add_ldflags("-fopenmp")
            add_syslinks("gomp")

            add_linkdirs(mkl_lib)
            add_rpathdirs(mkl_lib)

            -- MKL link line: lp64 + gnu_thread + core (for GCC with OpenMP threading)
            -- Use --no-as-needed to force NEEDED entries for MKL shared libs
            add_shflags("-Wl,--no-as-needed", "-lmkl_intel_lp64", "-lmkl_gnu_thread", "-lmkl_core", "-Wl,--as-needed", {force = true})

            add_syslinks("pthread", "m", "dl")
        else
            add_ldflags("/openmp")
            add_linkdirs(path.join(mkl_root, "lib"))
            -- Intel OpenMP runtime (libiomp5md) is required by mkl_intel_thread
            local compiler_default = "C:/Program Files (x86)/Intel/oneAPI/compiler/latest"
            local compiler_root = os.getenv("CMPLR_ROOT") or compiler_default
            add_linkdirs(path.join(compiler_root, "lib"))
            add_links("mkl_intel_lp64", "mkl_intel_thread", "mkl_core", "libiomp5md")
        end
    else
        -- OpenBLAS linking (vcpkg manifest mode)
        local triplet = is_plat("windows") and "x64-windows" or "x64-linux"
        local vcpkg_lib = path.join(os.projectdir(), "vcpkg_installed", triplet, "lib")

        if not is_plat("windows") then
            add_ldflags("-fopenmp")
            add_syslinks("gomp")

            add_linkdirs(vcpkg_lib)

            add_ldflags("-Wl,--whole-archive")
            add_links("openblas")
            add_ldflags("-Wl,--no-whole-archive")

            add_syslinks("pthread", "gfortran")
        else
            add_ldflags("/openmp")
            add_linkdirs(vcpkg_lib)
            add_links("openblas")
        end
    end

    if is_mode("debug") then
        add_defines("DEBUG")
        -- CUDA line info for debug profiling
        if has_config("nv-gpu") then
            add_cuflags("-G", "-lineinfo", "-g", {force = true})
        end
    end

    -- CUDA runtime linking
    if has_config("nv-gpu") then
        if not is_plat("windows") then
            add_links("cudart")
            -- Force NEEDED entry for cublas/cublasLt: symbols are called from static
            -- .cu objects so --as-needed (the default) would drop the dependency.
            add_shflags("-Wl,--no-as-needed", "-lcublas", "-lcublasLt", "-Wl,--as-needed", {force = true})
        else
            -- On Windows, use standard link directives with CUDA toolkit path
            local cuda_path = os.getenv("CUDA_PATH") or "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"
            add_linkdirs(path.join(cuda_path, "lib/x64"))
            add_links("cudart", "cublas", "cublasLt")
        end
    end
    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        local targetfile = target:targetfile()
        os.cp(targetfile, "python/llaisys/libllaisys/")
    end)
target_end()