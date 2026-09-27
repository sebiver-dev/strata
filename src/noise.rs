//! Deterministic hash-based value noise. Everything in the world is derived
//! from these functions and a seed, so the same seed always gives the same world.

#[inline]
pub fn hash2(seed: u32, x: i32, y: i32) -> u32 {
    let mut h = seed ^ (x as u32).wrapping_mul(0x27d4_eb2d) ^ (y as u32).wrapping_mul(0x1656_67b1);
    h = (h ^ (h >> 15)).wrapping_mul(0x85eb_ca6b);
    h = (h ^ (h >> 13)).wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

#[inline]
pub fn hash3(seed: u32, x: i32, y: i32, z: i32) -> u32 {
    hash2(seed ^ (z as u32).wrapping_mul(0x9e37_79b9), x, y)
}

/// Uniform float in [0, 1).
#[inline]
pub fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / (1u32 << 24) as f32
}

#[inline]
fn smooth(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// 2D value noise in [0, 1).
pub fn value2(seed: u32, x: f32, y: f32) -> f32 {
    let (xf, yf) = (x.floor(), y.floor());
    let (xi, yi) = (xf as i32, yf as i32);
    let (u, v) = (smooth(x - xf), smooth(y - yf));
    let a = unit(hash2(seed, xi, yi));
    let b = unit(hash2(seed, xi + 1, yi));
    let c = unit(hash2(seed, xi, yi + 1));
    let d = unit(hash2(seed, xi + 1, yi + 1));
    lerp(lerp(a, b, u), lerp(c, d, u), v)
}

/// 3D value noise in [0, 1).
pub fn value3(seed: u32, x: f32, y: f32, z: f32) -> f32 {
    let (xf, yf, zf) = (x.floor(), y.floor(), z.floor());
    let (xi, yi, zi) = (xf as i32, yf as i32, zf as i32);
    let (u, v, w) = (smooth(x - xf), smooth(y - yf), smooth(z - zf));
    let c = |dx, dy, dz| unit(hash3(seed, xi + dx, yi + dy, zi + dz));
    let x00 = lerp(c(0, 0, 0), c(1, 0, 0), u);
    let x10 = lerp(c(0, 1, 0), c(1, 1, 0), u);
    let x01 = lerp(c(0, 0, 1), c(1, 0, 1), u);
    let x11 = lerp(c(0, 1, 1), c(1, 1, 1), u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)
}

/// Fractal sum of 2D value noise, normalised to [0, 1).
pub fn fbm2(seed: u32, x: f32, y: f32, octaves: u32) -> f32 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 0.5, 1.0, 0.0);
    for i in 0..octaves {
        sum += amp * value2(seed.wrapping_add(i * 1013), x * freq, y * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.03;
    }
    sum / norm
}

/// Ridged fractal noise: sharp crests, useful for mountain ranges. In [0, 1].
pub fn ridged2(seed: u32, x: f32, y: f32, octaves: u32) -> f32 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 0.5, 1.0, 0.0);
    for i in 0..octaves {
        let n = 1.0 - (value2(seed.wrapping_add(i * 7919), x * freq, y * freq) * 2.0 - 1.0).abs();
        sum += amp * n * n;
        norm += amp;
        amp *= 0.5;
        freq *= 2.1;
    }
    sum / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        for i in 0..1000 {
            let x = i as f32 * 0.37 - 100.0;
            let y = i as f32 * 0.11 + 5.0;
            let a = fbm2(7, x, y, 5);
            assert_eq!(a, fbm2(7, x, y, 5));
            assert!((0.0..1.0).contains(&a));
            let r = ridged2(7, x, y, 4);
            assert!((0.0..=1.0).contains(&r));
            let v = value3(3, x, y, x - y);
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn continuous() {
        let a = value2(1, 10.4999, 3.2);
        let b = value2(1, 10.5001, 3.2);
        assert!((a - b).abs() < 0.01);
    }
}
