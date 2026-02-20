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
        set_symbols("debug")
        set_optimize("none")
        add_defines("DEBUG")
    end
    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()