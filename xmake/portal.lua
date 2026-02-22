-- xmake/portal.lua
-- Builds the React + Vite portal UI (ui/portal)
-- Output goes to ui/portal/dist/

target("portal-ui")
    set_kind("phony")
    set_default(false)  -- opt-in: `xmake build portal-ui`

    on_build(function (target)
        import("core.base.option")
        local portal_dir = path.join(os.projectdir(), "ui", "portal")
        local node_modules = path.join(portal_dir, "node_modules")

        -- On Windows, npm is a .cmd script, not an .exe
        local npm = is_host("windows") and "npm.cmd" or "npm"

        -- Install npm dependencies if needed
        if not os.isdir(node_modules) then
            print("portal-ui: installing npm dependencies ...")
            os.vrunv(npm, {"install"}, {curdir = portal_dir})
        end

        -- Run the Vite production build
        print("portal-ui: building React app ...")
        os.vrunv(npm, {"run", "build"}, {curdir = portal_dir})

        cprint("${bright green}portal-ui: build complete → ui/portal/dist/${clear}")
    end)

    on_clean(function (target)
        local out = path.join(os.projectdir(), "ui", "portal", "dist")
        if os.isdir(out) then
            print("portal-ui: cleaning ui/portal/dist/ ...")
            os.rmdir(out)
        end
    end)

    on_install(function (target) end)
target_end()
