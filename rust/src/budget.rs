// SPDX-License-Identifier: GPL-3.0-or-later
//! Process-wide native thread budget (RENDER_SCHEDULER_PLAN S3): 0 means
//! "unlimited" (hardware concurrency); the Python renderer publishes each
//! pooled section's per-worker share before the section runs, so the kernels
//! stop stacking min(hw, 8) threads on top of the outer Python pool.
use std::sync::atomic::{AtomicU32, Ordering};

pub static THREAD_BUDGET: AtomicU32 = AtomicU32::new(0);

/// Pixel count from which the per-pixel kernels fan out (the fixed realtime
/// preview is large enough for the pow-heavy cores to benefit; small buffers
/// avoid thread start-up).
pub const PARALLEL_THRESHOLD: usize = 128 * 1024;

pub fn budgeted_workers(cap: u32) -> u32 {
    let hw = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
        .max(1);
    let budget = THREAD_BUDGET.load(Ordering::Relaxed);
    let allowed = if budget == 0 { hw } else { hw.min(budget) };
    allowed.min(cap).max(1)
}

pub fn workers_for(pixel_count: usize) -> u32 {
    if pixel_count >= PARALLEL_THRESHOLD {
        budgeted_workers(8)
    } else {
        1
    }
}

/// Block size (in pixels) of the C++ kernels' partition: ceil(n / workers).
pub fn block_pixels(pixel_count: usize, workers: u32) -> usize {
    let w = workers.max(1) as usize;
    ((pixel_count + w - 1) / w).max(1)
}
