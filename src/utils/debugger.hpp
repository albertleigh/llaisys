//
// Created by ali on 2/8/26.
//

#include <fstream>
#include <string>
#include <unistd.h>

#pragma once
namespace llaisys::debugger {
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

} // namespace llaisys::debugger