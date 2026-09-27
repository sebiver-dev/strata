//! Keeps background work (generating, meshing, far tiles) inside a fixed slice
//! of each frame, so loading never shows up as stutter.

use web_time::Instant;

/// The time left for background work in the current frame.
pub struct Deadline {
    start: Instant,
    limit_ms: f64,
}

impl Deadline {
    pub fn new(limit_ms: f64) -> Self {
        Self {
            start: Instant::now(),
            limit_ms,
        }
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// Whether a job expected to take `cost` fits before `share` of the frame's
    /// work budget is used up. `must` lets the first job of a stage run anyway,
    /// so work always progresses even when one job exceeds the whole budget.
    pub fn fits(&self, cost: &Cost, share: f64, must: bool) -> bool {
        let used = self.elapsed_ms();
        let limit = self.limit_ms * share;
        (must && used < limit) || used + cost.estimate_ms() <= limit
    }
}

/// A running estimate of how long one job of a kind takes.
#[derive(Default, Clone, Copy)]
pub struct Cost {
    avg_ms: f64,
    samples: u32,
}

impl Cost {
    pub fn record(&mut self, ms: f64) {
        self.avg_ms = if self.samples == 0 {
            ms
        } else {
            self.avg_ms * 0.9 + ms * 0.1
        };
        self.samples = self.samples.saturating_add(1);
    }

    /// The average plus a margin, since jobs vary (a chunk full of trees costs
    /// more than one of open air).
    pub fn estimate_ms(&self) -> f64 {
        self.avg_ms * 1.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_job_always_runs_but_later_ones_must_fit() {
        let d = Deadline::new(4.0);
        let mut big = Cost::default();
        big.record(10.0);
        assert!(d.fits(&big, 1.0, true));
        assert!(!d.fits(&big, 1.0, false));
        let mut small = Cost::default();
        small.record(0.5);
        assert!(d.fits(&small, 1.0, false));
    }
}
