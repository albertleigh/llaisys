add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

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
    set_installdir(".")

    if is_mode("debug") then
        set_symbols("debug")
        set_optimize("none")
        add_defines("DEBUG")
    end
    
    after_install(function (target)
        local venv = os.getenv("VIRTUAL_ENV")
        if not venv then
            -- copy shared library to python package
            print("Copying llaisys to python/llaisys/libllaisys/ ..")
            if is_plat("windows") then
                os.cp("bin/*.dll", "python/llaisys/libllaisys/")
            end
            if is_plat("linux") then
                os.cp("lib/*.so", "python/llaisys/libllaisys/")
            end
            return
        end
        
        -- Get Python version
        local python_version = os.iorun("python -c \"import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')\"")
        python_version = python_version:trim()
        
        print("Copying llaisys to virtual environment: " .. venv)
        print("Python version detected: " .. python_version)
        
        if is_plat("windows") then
            local dest = path.join(venv, "Lib", "site-packages/llaisys/libllaisys")
            print("Destination: " .. dest)
            os.cp("bin/*.dll", dest)
        elseif is_plat("linux") then
            local dest = path.join(venv, "lib", python_version, "site-packages/llaisys/libllaisys")
            print("Destination: " .. dest)
            os.cp("lib/*.so", dest)
        end
    end)
target_end()