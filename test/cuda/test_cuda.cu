//
// GTest for CUDA argmax kernel — debuggable with cuda-gdb.
//
// Build & run:
//   xmake f -m debug --nv-gpu=y && xmake build test-cuda
//   ./build/linux/x86_64/debug/test-cuda
//
// Debug:
//   cuda-gdb ./build/linux/x86_64/debug/test-cuda
//   (cuda-gdb) break argmax_single_block
//   (cuda-gdb) run
//
#include <gtest/gtest.h>

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
