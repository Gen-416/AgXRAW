// SPDX-License-Identifier: GPL-3.0-or-later
//! Process-wide native thread budget (RENDER_SCHEDULER_PLAN S3): 0 means
//! "unlimited" (hardware concurrency); the Python renderer publishes each
//! pooled section's per-worker share before the section runs, so the kernels
//! stop stacking min(hw, 8) threads on top of the outer Python pool.
use std::sync::atomic::{AtomicU32, Ordering};

pub static THREAD_BUDGET: AtomicU32 = AtomicU32::new(0);

/// Explicitly join OS threads before releasing a batch's CPU budget.
/// Scope's implicit join only waits for thread functions; TLS destructors
/// may still run and overlap the next batch when handles are dropped.
pub fn join_workers(handles: Vec<std::thread::ScopedJoinHandle<'_, ()>>) {
    for handle in handles {
        handle.join().expect("native kernel worker panicked");
    }
}

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

#[cfg(test)]
mod tests {
    use super::join_workers;
    use std::cell::RefCell;
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::time::Duration;

    struct ExitGate(Sender<()>, Receiver<()>);
    impl Drop for ExitGate {
        fn drop(&mut self) {
            let _ = self.0.send(());
            let _ = self.1.recv_timeout(Duration::from_secs(5));
        }
    }
    thread_local! {
        static EXIT_GATE: RefCell<Option<ExitGate>> = const { RefCell::new(None) };
    }

    #[test]
    fn budget_is_not_released_during_thread_exit() {
        let (exiting_tx, exiting_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let caller = std::thread::spawn(move || {
            std::thread::scope(|scope| {
                let worker = scope.spawn(move || {
                    EXIT_GATE.with(|gate| {
                        *gate.borrow_mut() = Some(ExitGate(exiting_tx, release_rx));
                    });
                });
                join_workers(vec![worker]);
            });
            done_tx.send(()).unwrap();
        });
        exiting_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let early = done_rx.recv_timeout(Duration::from_millis(100));
        // Release before asserting so a regression cannot strand the worker.
        release_tx.send(()).unwrap();
        caller.join().unwrap();
        assert!(matches!(early, Err(mpsc::RecvTimeoutError::Timeout)));
    }
}
