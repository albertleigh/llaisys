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

            -- Collect all .cu.o files from CUDA static lib dependencies
            local cu_objects = {}
            for _, dep in ipairs(target:orderdeps()) do
                for _, obj in ipairs(dep:objectfiles()) do
                    if obj:match("%.cu%.o$") then
                        table.insert(cu_objects, obj)
                    end
                end
            end

            if #cu_objects > 0 then
                local dlink_obj = path.join(target:objectdir(), "cuda_dlink.o")
                local argv = {"-dlink", "-shared",
                              "-gencode", "arch=compute_86,code=sm_86",
                              "-o", dlink_obj}
                for _, obj in ipairs(cu_objects) do
                    table.insert(argv, obj)
                end
                os.vrunv(nvcc.program, argv)
                -- Insert device-link object at the front so g++ includes it
                local objs = target:objectfiles()
                table.insert(objs, 1, dlink_obj)
                target:set("objectfiles", objs)
            end
        end)
    end

    -- BLAS linking
    if has_config("mkl") then
        -- Intel MKL linking
        local mkl_root = os.getenv("MKLROOT") or "/opt/intel/oneapi/mkl/latest"
        local mkl_lib = path.join(mkl_root, "lib", "intel64")

        if not is_plat("windows") then
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
            add_links("mkl_intel_lp64", "mkl_intel_thread", "mkl_core")
        end
    else
        -- OpenBLAS linking (fallback)
        local vcpkg_root = os.getenv("VCPKG_ROOT") or (is_plat("windows") and "C:/opt/vcpkg" or "~/opt/vcpkg")

        if not is_plat("windows") then
            add_ldflags("-fopenmp")
            add_syslinks("gomp")

            add_linkdirs(path.join(vcpkg_root, "installed/x64-linux/lib"))

            add_ldflags("-Wl,--whole-archive")
            add_links("openblas")
            add_ldflags("-Wl,--no-whole-archive")

            add_syslinks("pthread", "gfortran")
        else
            add_ldflags("/openmp")
            add_linkdirs(path.join(vcpkg_root, "installed/x64-windows/lib"))
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
        add_links("cudart")
        add_links("cublas")
    end
    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        local targetfile = target:targetfile()
        os.cp(targetfile, "python/llaisys/libllaisys/")
    end)
target_end()