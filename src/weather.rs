//! The wind that moves the world: grass, leaves, banners, clouds and mist all
//! read the same wind so a gust rolls through everything at once. Shaders get
//! it through `Globals::wind`; other code (sound, particles) calls `wind`.

use glam::Vec2;

/// The wind at one moment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wind {
    /// Unit direction in the XZ plane the wind blows towards.
    pub dir: Vec2,
    /// Mean strength, 0 (still) to 1 (a strong breeze).
    pub strength: f32,
    /// How far in metres the gust pattern has travelled downwind. Shaders
    /// sample gusts at `position - dir * phase`, so gusts sweep across the
    /// valley as waves.
    pub phase: f32,
}

/// Speed in metres per second at which gusts sweep downwind.
const GUST_SPEED: f32 = 7.0;

/// The wind at `time` seconds. It veers slowly around a prevailing breeze
/// from the south-west and swells and calms over tens of seconds.
pub fn wind(time: f32) -> Wind {
    let slow = |f: f32, p: f32| (time * f + p).sin();
    let veer = 0.25 * slow(0.021, 0.0) + 0.1 * slow(0.057, 1.3);
    let base = 0.64f32;
    let a = base + veer;
    let swell = 0.5 + 0.5 * (0.6 * slow(0.043, 0.7) + 0.4 * slow(0.11, 2.1));
    Wind {
        dir: Vec2::new(a.cos(), a.sin()),
        strength: (0.35 + 0.5 * swell).clamp(0.0, 1.0),
        phase: time * GUST_SPEED,
    }
}

impl Wind {
    /// Packed for the shader: xy direction, z strength, w gust phase.
    pub fn uniform(&self) -> [f32; 4] {
        [self.dir.x, self.dir.y, self.strength, self.phase]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wind_is_a_gentle_varying_breeze() {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for i in 0..2000 {
            let w = wind(i as f32 * 0.5);
            assert!((w.dir.length() - 1.0).abs() < 1e-4);
            lo = lo.min(w.strength);
            hi = hi.max(w.strength);
        }
        assert!(lo >= 0.3 && hi <= 0.9 && hi - lo > 0.3, "{lo} {hi}");
        // It changes smoothly from frame to frame.
        let (a, b) = (wind(100.0), wind(100.016));
        assert!((a.strength - b.strength).abs() < 0.01);
        assert!(a.dir.dot(b.dir) > 0.9999);
    }
}
