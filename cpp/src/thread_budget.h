// SPDX-License-Identifier: GPL-3.0-or-later
// Process-wide native thread budget (RENDER_SCHEDULER_PLAN S3): 0 means
// "unlimited" (hardware concurrency); the Python renderer publishes each
// pooled section's per-worker share before the section runs, so the kernels
// stop stacking min(hw, 8) threads on top of the outer Python pool.
#pragma once

#include <algorithm>
#include <atomic>
#include <thread>

namespace dngscan_fast {

inline std::atomic<unsigned> g_thread_budget{0};

inline unsigned budgeted_workers(unsigned cap) {
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  const unsigned budget = g_thread_budget.load(std::memory_order_relaxed);
  unsigned allowed = budget == 0 ? hw : std::min(hw, budget);
  return std::max(1u, std::min(allowed, cap));
}

}  // namespace dngscan_fast
