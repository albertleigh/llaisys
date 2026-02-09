//
// Created by ali on 2/8/26.
//

#include <fstream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#pragma once
namespace llaisys::debugger {

#ifdef _WIN32
// Windows implementation
bool isDebuggerAttached() {
    return IsDebuggerPresent() != 0;
}

void waitForDebugger() {
    char *wait_debug = nullptr;
    size_t len = 0;
    _dupenv_s(&wait_debug, &len, "LLAISYS_WAIT_DEBUGGER");

    if (wait_debug && std::string(wait_debug) == "1") {
        std::printf("PID %lu waiting for debugger to attach...\n", GetCurrentProcessId());
        while (!isDebuggerAttached()) {
            Sleep(1000);
        }
        std::printf("Debugger attached! Continuing execution...\n");
    }

    if (wait_debug) {
        free(wait_debug);
    }
}

#else
// Linux/Unix implementation
bool isDebuggerAttached() {
    std::ifstream statusFile("/proc/self/status");
    std::string line;
    while (std::getline(statusFile, line)) {
        if (line.compare(0, 9, "TracerPid") == 0) {
            // TracerPid: 0 means no debugger, non-zero means debugger attached
            size_t pos = line.find_first_of("0123456789");
            if (pos != std::string::npos) {
                int pid = std::stoi(line.substr(pos));
                return pid != 0;
            }
        }
    }
    return false;
}

void waitForDebugger() {
    const char *wait_debug = std::getenv("LLAISYS_WAIT_DEBUGGER");
    if (wait_debug && std::string(wait_debug) == "1") {
        std::printf("PID %d waiting for debugger to attach...\n", getpid());
        while (!isDebuggerAttached()) {
            sleep(1);
        }
        std::printf("Debugger attached! Continuing execution...\n");
    }
}
#endif

} // namespace llaisys::debugger