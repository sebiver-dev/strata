//! Grass and wildflowers as authored procedural models. Every air voxel of
//! tall grass holds one clump: curved, tapering blades fanning out from a
//! common root, with now and then a lupin (a rosette of palmate leaves under a
//! tall spike of pea-flowers) or a small cluster of daisies among them.
//! Verges of the roads and river banks grow lusher and flower more.
//!
//! Plant vertices keep the height along the plant in the occlusion bits
//! (0 at the root, 3 at the top), which the shader uses to bend them in the
//! wind, darken them towards the ground and sink them into it far away.

use crate::block::*;
use crate::mesh::{smooth_data, MeshData, Vertex};
use crate::noise::{fbm2, hash3, unit};
use crate::terrain::Terrain;
use glam::{IVec3, Vec2, Vec3};
use std::f32::consts::TAU;

/// What the ground around one grass voxel is like, 0..1 each.
#[derive(Clone, Copy, Debug, Default)]
pub struct Spot {
    /// Open meadow: thick, knee-high grass and daisies.
    pub meadow: f32,
    /// Tall-grass field: chest- to head-high grass drying to gold.
    pub field: f32,
    /// Along a road or a river bank: lush grass, lupins and daisies.
    pub verge: f32,
    /// Hard against a rock, a trunk or a wall: grass only, kept short, so
    /// nothing grows up through or stands on top of them.
    pub sheltered: bool,
}

impl Spot {
    pub fn at(terrain: &Terrain, w: IVec3) -> Self {
        let (xm, zm) = ((w.x as f32 + 0.5) * VOXEL_SIZE, (w.z as f32 + 0.5) * VOXEL_SIZE);
        Self {
            meadow: crate::terrain::meadow(terrain.seed, xm, zm),
            field: crate::terrain::tall_meadow(terrain.seed, xm, zm),
            verge: verge(terrain, xm, zm),
            sheltered: false,
        }
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How much a spot (metres) is the verge of a road or a river bank, 0..1.
pub fn verge(terrain: &Terrain, xm: f32, zm: f32) -> f32 {
    let road = 1.0 - smoothstep(2.6, 9.0, terrain.road_distance(xm, zm));
    // The water's edge lies 10 to 17 m from the river's centre line.
    let bank = 1.0 - smoothstep(16.0, 26.0, (xm - terrain.river_x(zm)).abs());
    road.max(bank)
}

/// How much a spot (metres) is the meadow in front of the arrival view, 0..1:
/// grass grows thickest and lushest there, as in the reference's foreground.
pub fn vista_meadow(xm: f32, zm: f32) -> f32 {
    let (ax, az) = crate::vista::ARRIVAL;
    let yaw = crate::vista::SPAWN_YAW;
    let centre = Vec2::new(ax, az) + Vec2::new(yaw.cos(), yaw.sin()) * 10.0;
    1.0 - smoothstep(14.0, 26.0, Vec2::new(xm, zm).distance(centre))
}

/// A stream of random numbers for one plant.
struct Rng(u32);

impl Rng {
    fn new(seed: u32, w: IVec3) -> Self {
        Self(hash3(seed, w.x, w.y, w.z))
    }
    /// Uniform in [0, 1).
    fn f(&mut self) -> f32 {
        self.0 = crate::noise::hash2(self.0, 0x51ed, 0x2705);
        unit(self.0)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

/// Vertex data for a thin plant part lit as if it faced the sky, so a
/// meadow shades as one soft surface whichever way its blades turn.
fn sky_data(mat: Block, ao: u32) -> u32 {
    2 | ((mat as u32) << 3) | (ao.min(3) << 11)
}

fn push(out: &mut MeshData, pos: Vec3, data: u32) -> u32 {
    out.vertices.push(Vertex {
        pos: pos.to_array(),
        data,
    });
    out.vertices.len() as u32 - 1
}

/// A triangle seen from both sides.
fn both(out: &mut MeshData, a: u32, b: u32, c: u32) {
    out.indices.extend_from_slice(&[a, b, c, a, c, b]);
}

/// A triangle wound counter-clockwise as seen from the side `towards` points to.
fn facing(out: &mut MeshData, [a, b, c]: [u32; 3], towards: Vec3) {
    let p = |i: u32| Vec3::from(out.vertices[i as usize].pos);
    let n = (p(b) - p(a)).cross(p(c) - p(a));
    if n.dot(towards) >= 0.0 {
        out.indices.extend_from_slice(&[a, b, c]);
    } else {
        out.indices.extend_from_slice(&[a, c, b]);
    }
}

/// A horizontal unit vector at angle `a`.
fn dir(a: f32) -> Vec3 {
    Vec3::new(a.cos(), 0.0, a.sin())
}

// ---------------------------------------------------------------- grass

/// One grass blade: a tapering ribbon that rises from `base` leaning towards
/// `toward` and arcs over more and more towards its tip. `lean` is its angle
/// from upright at the root and `arc` how much further it bends by the tip
/// (radians). Tip occlusion `top` sets how far the wind moves it.
#[allow(clippy::too_many_arguments)]
fn blade(out: &mut MeshData, base: Vec3, toward: Vec3, len: f32, lean: f32, arc: f32, half: f32, mat: Block, top: u32) {
    // Short blades are a single tapering triangle; only the longest, which
    // droop over furthest, get a third joint.
    let segs = if len < 0.45 {
        1
    } else if len < 1.0 {
        2
    } else {
        3
    };
    // The flat of the blade turns a little off square to the way it leans.
    let side = Vec3::new(-toward.z, 0.0, toward.x) * half;
    let ao = |t: f32| (t * top as f32 + 0.35).floor() as u32;
    let mut p = base;
    let mut prev = [
        push(out, base - side, sky_data(mat, 0)),
        push(out, base + side, sky_data(mat, 0)),
    ];
    for i in 0..segs {
        let t0 = i as f32 / segs as f32;
        let t1 = (i + 1) as f32 / segs as f32;
        // Steeper at the root, bending over towards the tip.
        let a = lean + arc * (t0 + t1) * 0.5 * (t0 + t1) * 0.5 * 1.3;
        p += (toward * a.sin() + Vec3::Y * a.cos()) * len / segs as f32;
        if i + 1 == segs {
            let tip = push(out, p, sky_data(mat, top));
            both(out, prev[0], prev[1], tip);
        } else {
            // Widest a little above the root, then tapering to a point.
            let w = (1.0 - t1).powf(0.8) * (1.0 + 0.25 * (1.0 - t1));
            let next = [
                push(out, p - side * w, sky_data(mat, ao(t1))),
                push(out, p + side * w, sky_data(mat, ao(t1))),
            ];
            both(out, prev[0], prev[1], next[1]);
            both(out, prev[0], next[1], next[0]);
            prev = next;
        }
    }
}

/// A clump of grass blades fanning out from one root in the air voxel `w`,
/// standing on `floor`.
fn grass(out: &mut MeshData, rng: &mut Rng, w: IVec3, floor: Vec3, spot: &Spot, crowded: bool) {
    // Heights vary in soft patches: ankle- to knee-high on open ground and
    // along verges, thigh- to waist-high in the tall fields.
    let patch = fbm2(97, w.x as f32 / 9.0, w.z as f32 / 9.0, 2);
    let open = 0.25 + 0.4 * patch * patch;
    let lush = 0.38 + 0.25 * patch;
    let tall = 0.7 + 0.35 * patch;
    let mut height = open + (lush - open).max(0.0) * spot.verge;
    height += (tall - height) * spot.field;

    let vista = vista_meadow(floor.x, floor.z);
    // The meadow in front of the arrival view stays knee-high, thick rather than tall.
    height += (lush - height) * vista;
    if spot.sheltered {
        height = height.min(0.3);
    }

    let thick = spot.meadow.max(spot.verge).max(spot.field);
    let mut n = 4 + (thick * 1.5 + rng.f() * 0.9).floor() as u32 + (spot.field > 0.5) as u32;
    n += (vista * 2.0).round() as u32;
    if crowded {
        n = n.saturating_sub(2).max(2);
    }
    // Dry, golden blades stand in the tall fields and on open ground, fresh
    // yellow-green ones along the verges and water.
    let dry = 0.14 + 0.3 * spot.field * (1.0 - spot.verge) + 0.1 * (1.0 - thick);
    let fresh = 0.2 + 0.4 * spot.verge;

    let root = floor + Vec3::new(rng.range(0.1, 0.4), 0.0, rng.range(0.1, 0.4));
    let turn = rng.f() * TAU;
    for k in 0..n {
        // The first blade stands tall in the middle; the rest fan out around it.
        let centre = k == 0;
        let az = turn + k as f32 * TAU / n as f32 + rng.range(-0.5, 0.5);
        let toward = dir(az);
        let base = root + toward * rng.range(0.01, 0.1);
        let len = height
            * if centre {
                rng.range(0.95, 1.15)
            } else {
                rng.range(0.5, 1.0)
            };
        let lean = if centre {
            rng.range(0.03, 0.15)
        } else {
            rng.range(0.12, 0.45)
        };
        // Long blades arc over further, so tall grass droops at the tips.
        let arc = rng.range(0.2, 0.8) + 0.6 * smoothstep(0.4, 1.4, len);
        // Broad, painterly blades rather than hair-thin ones.
        let half = (0.034 + 0.024 * len.min(1.4)) * rng.range(0.8, 1.25);
        let pick = rng.f();
        let mat = if pick < dry {
            DRY_GRASS
        } else if pick < dry + fresh {
            FRESH_GRASS
        } else {
            TALL_GRASS
        };
        // Short blades move less at the tip than long ones.
        let top = if len < 0.35 { 2 } else { 3 };
        blade(out, base, toward, len, lean, arc, half, mat, top);
    }
}

// ---------------------------------------------------------------- lupins

/// A lupin: a rosette of palmate leaves, a stem, and a tapering spike of
/// pea-flowers spiralling up it, smaller and tighter towards the tip.
fn lupin(out: &mut MeshData, rng: &mut Rng, at: Vec3, height: f32, mat: Block) {
    // Leaves: two or three on short stalks up the lower stem, each a star of
    // narrow leaflets.
    let spike_len = height * rng.range(0.38, 0.5);
    let leaves = 2 + (rng.f() < 0.25) as u32;
    let leaf_top = height - spike_len - 0.18;
    let turn = rng.f() * TAU;
    for k in 0..leaves {
        let a = turn + k as f32 * TAU / leaves as f32 + rng.range(-0.3, 0.3);
        let c = at
            + dir(a) * rng.range(0.06, 0.11)
            + Vec3::Y * (0.12 + 0.15 * height * k as f32 + rng.range(0.0, 0.06)).min(leaf_top);
        let len = rng.range(0.07, 0.1);
        palmate(out, rng, c, len);
    }

    let lean = dir(rng.f() * TAU) * rng.range(0.0, 0.1) * height;
    let bottom = at + lean * (1.0 - spike_len / height) + Vec3::Y * (height - spike_len);
    let top = at + lean + Vec3::Y * height;
    let axis = (top - bottom).normalize();

    // A slim three-sided stem up to the first flowers.
    let r = 0.009;
    let ring = |c: Vec3, r: f32, ph: f32| [0.0, 1.0, 2.0].map(|k| c + dir(ph + k * TAU / 3.0) * r);
    let lo = ring(at, r, turn);
    let hi = ring(bottom, r * 0.8, turn);
    for k in 0..3 {
        let j = (k + 1) % 3;
        let out_n = (lo[k] + lo[j] - at * 2.0).normalize();
        let ids = [
            push(out, lo[k], smooth_data(STEM, 0, out_n)),
            push(out, lo[j], smooth_data(STEM, 0, out_n)),
            push(out, hi[j], smooth_data(STEM, 2, out_n)),
            push(out, hi[k], smooth_data(STEM, 2, out_n)),
        ];
        facing(out, [ids[0], ids[1], ids[2]], out_n);
        facing(out, [ids[0], ids[2], ids[3]], out_n);
    }

    // The spike: a solid bullet-shaped body studded with florets set on
    // the golden angle, so it reads as a dense cone even from afar.
    let base_r = 0.065 + 0.02 * rng.f();
    let side = if axis.y > 0.99 {
        Vec3::X
    } else {
        Vec3::Y.cross(axis).normalize()
    };
    let fwd = axis.cross(side);
    let around = |a: f32| side * a.cos() + fwd * a.sin();
    let body = |t: f32| {
        if t < 0.5 {
            base_r * (0.55 - 0.3 * t)
        } else {
            base_r * 0.4 * (2.0 - 2.0 * t)
        }
    };
    let ring = |out: &mut MeshData, t: f32, ao: u32| -> Vec<u32> {
        let c = bottom + (top - bottom) * t;
        (0..6)
            .map(|k| {
                let d = around((k as f32 + t * 3.0) * TAU / 6.0);
                push(out, c + d * body(t), smooth_data(mat, ao, (d + axis * 0.3).normalize()))
            })
            .collect()
    };
    let r0 = ring(out, 0.0, 2);
    let r1 = ring(out, 0.5, 3);
    let apex = push(out, top + axis * 0.015, smooth_data(mat, 3, axis));
    for k in 0..6 {
        let j = (k + 1) % 6;
        let d = around((k as f32 + 0.5) * TAU / 6.0);
        facing(out, [r0[k], r0[j], r1[j]], d);
        facing(out, [r0[k], r1[j], r1[k]], d);
        facing(out, [r1[k], r1[j], apex], d + axis * 0.3);
    }

    // Florets stand in whorls of four or five, each turned half a step from
    // the one below, so the spike steps in soft tiers up to a point.
    let per = 4;
    let whorls = (4.0 + spike_len * 7.0).min(7.0) as u32;
    let phase = rng.f() * TAU;
    for i in 0..whorls {
        let t = (i as f32 + 0.3) / whorls as f32;
        let c = bottom + (top - bottom) * t;
        // The spike tapers to a point; its top florets are small unopened buds.
        let reach = body(t);
        let ao = if t < 0.5 { 2 } else { 3 };
        for k in 0..per {
            let a = phase + (k as f32 + 0.5 * (i % 2) as f32) * TAU / per as f32 + rng.range(-0.2, 0.2);
            let size = (0.06 + 0.014 * rng.f()) * (1.0 - 0.6 * t);
            floret(out, c, around(a), axis, reach, size, mat, ao);
        }
    }
}

/// One pea-flower on a lupin spike: a rounded, blunt bead set on the body
/// at `c` and facing out along `d`, a little taller than wide where its
/// banner petal rises. Its normals spread like a ball's, so it shades round.
#[allow(clippy::too_many_arguments)]
fn floret(out: &mut MeshData, c: Vec3, d: Vec3, axis: Vec3, reach: f32, size: f32, mat: Block, ao: u32) {
    let across = axis.cross(d);
    let heart = c + d * reach;
    let centre = heart - d * size * 0.35;
    let normal = |p: Vec3| (p - centre).normalize();
    let keel = heart + d * size * 0.45 - axis * size * 0.05;
    let apex = push(out, keel, smooth_data(mat, ao, normal(keel)));
    let rim = [0.0f32, 1.0, 2.0, 3.0, 4.0].map(|k| {
        let a = std::f32::consts::FRAC_PI_2 + k * TAU / 5.0;
        let up = if a.sin() > 0.0 { 0.65 } else { 0.5 };
        heart + (axis * a.sin() * up + across * a.cos() * 0.55) * size + d * size * 0.1
    });
    let ids = rim.map(|p| push(out, p, smooth_data(mat, ao, normal(p))));
    for k in 0..5 {
        let j = (k + 1) % 5;
        facing(out, [apex, ids[k], ids[j]], (keel + rim[k] + rim[j]) / 3.0 - centre);
    }
}

/// A palmate leaf: seven narrow leaflets radiating from `c`, drooping at the tips.
fn palmate(out: &mut MeshData, rng: &mut Rng, c: Vec3, len: f32) {
    let turn = rng.f() * TAU;
    let hub = push(out, c, sky_data(STEM, 1));
    for k in 0..7 {
        let a = turn + k as f32 * TAU / 7.0;
        let d = dir(a);
        let s = Vec3::new(-d.z, 0.0, d.x);
        let l = len * rng.range(0.8, 1.1);
        let mid = c + d * l * 0.55 + Vec3::Y * l * 0.08;
        let ids = [
            hub,
            push(out, mid - s * l * 0.13, sky_data(STEM, 1)),
            push(out, c + d * l - Vec3::Y * l * 0.25, sky_data(STEM, 1)),
            push(out, mid + s * l * 0.13, sky_data(STEM, 1)),
        ];
        facing(out, [ids[0], ids[1], ids[2]], Vec3::Y);
        facing(out, [ids[0], ids[2], ids[3]], Vec3::Y);
    }
}

// ---------------------------------------------------------------- daisies

/// A daisy: a slim stem holding a ring of white strap petals, cupped a
/// little, round a domed yellow heart that faces the sky.
fn daisy(out: &mut MeshData, rng: &mut Rng, at: Vec3, height: f32) {
    let tilt = dir(rng.f() * TAU) * rng.range(0.0, 0.35);
    let up = (Vec3::Y + tilt).normalize();
    let head = at + (Vec3::Y + tilt * 0.4) * height;
    // A single tapering stem blade, seen from both sides.
    let s = dir(rng.f() * TAU) * 0.006;
    let (a, b) = (
        push(out, at - s, sky_data(STEM, 0)),
        push(out, at + s, sky_data(STEM, 0)),
    );
    let t = push(out, head - up * 0.01, sky_data(STEM, 2));
    both(out, a, b, t);

    let u = up.cross(Vec3::Z).normalize();
    let v = up.cross(u);
    let heart_r = rng.range(0.016, 0.021);
    let petal = rng.range(0.05, 0.07);
    let petals = 10 + (rng.f() * 3.0) as u32;
    let turn = rng.f() * TAU;
    for k in 0..petals {
        let a = turn + k as f32 * TAU / petals as f32 + rng.range(-0.08, 0.08);
        let d = u * a.cos() + v * a.sin();
        let s = u * -a.sin() + v * a.cos();
        let w = TAU * heart_r / petals as f32 * 0.55 + 0.004;
        let len = petal * rng.range(0.85, 1.1);
        let root = head + d * heart_r * 0.8;
        // Cupped: the petal rises towards its rounded tip.
        let end = head + d * (heart_r + len) + up * len * rng.range(0.1, 0.3);
        let n = (up - d * 0.35).normalize();
        let ids = [
            push(out, root - s * w * 0.6, smooth_data(DAISY, 2, n)),
            push(out, root + s * w * 0.6, smooth_data(DAISY, 2, n)),
            push(out, end + s * w, smooth_data(DAISY, 2, n)),
            push(out, end - s * w, smooth_data(DAISY, 2, n)),
        ];
        facing(out, [ids[0], ids[1], ids[2]], up);
        facing(out, [ids[0], ids[2], ids[3]], up);
    }
    // The heart: a low dome standing proud of the petals.
    let rim: Vec<u32> = (0..6)
        .map(|k| {
            let a = turn + k as f32 * TAU / 6.0;
            let d = u * a.cos() + v * a.sin();
            push(
                out,
                head + d * heart_r + up * 0.004,
                smooth_data(DAISY_HEART, 2, (up + d).normalize()),
            )
        })
        .collect();
    let crown = push(out, head + up * heart_r * 0.9, smooth_data(DAISY_HEART, 2, up));
    for k in 0..6 {
        facing(out, [rim[k], rim[(k + 1) % 6], crown], up);
    }
}

// ---------------------------------------------------------------- placement

/// Everything growing in the tall-grass voxel `w`: one clump of grass and
/// perhaps a lupin or a few daisies.
pub fn tuft(out: &mut MeshData, seed: u32, w: IVec3, spot: &Spot) {
    // Rooted a little below the voxel floor, since the smooth ground can dip there.
    let floor = w.as_vec3() * VOXEL_SIZE - Vec3::Y * 0.12;
    let (xm, zm) = (w.x as f32 * VOXEL_SIZE, w.z as f32 * VOXEL_SIZE);
    let mut rng = Rng::new(seed ^ 0x9a55, w);

    // Lupins grow in patches, most of all along verges; daisies in drifts
    // through meadows and along the paths.
    let lupins = fbm2(seed.wrapping_add(50), xm / 14.0, zm / 14.0, 2) + 0.05 * spot.verge;
    let drifts = fbm2(seed.wrapping_add(51), xm / 6.0, zm / 6.0, 2);
    let daisies = (0.01 + 0.05 * spot.meadow + 0.12 * spot.verge) * smoothstep(0.42, 0.62, drifts) * (1.0 - spot.field);
    let roll = rng.f();
    let spot_at = |rng: &mut Rng| floor + Vec3::new(rng.range(0.08, 0.42), 0.02, rng.range(0.08, 0.42));
    let mut crowded = false;
    // Drifts: sparse at their edges, crowded in the middle.
    if spot.sheltered {
        crowded = true;
    } else if lupins > 0.68 && roll < 0.03 + 1.0 * (lupins - 0.68) {
        let at = spot_at(&mut rng);
        // Mostly violet, with pink and white spires mixed in; some drifts
        // run pinker or paler than others.
        let drift = fbm2(seed.wrapping_add(52), xm / 9.0, zm / 9.0, 2);
        let pick = rng.f() * 0.75 + drift * 0.35;
        let mat = if pick < 0.25 {
            LUPIN_DEEP
        } else if pick < 0.62 {
            LUPIN
        } else if pick < 0.82 {
            LUPIN_PINK
        } else {
            LUPIN_WHITE
        };
        let height = rng.range(0.5, 0.9);
        lupin(out, &mut rng, at, height, mat);
        crowded = true;
    } else if roll < daisies * 2.0 {
        let heads = 1 + (rng.f() * 1.6) as u32;
        let base = spot_at(&mut rng);
        for _ in 0..heads {
            let at = base + dir(rng.f() * TAU) * rng.range(0.0, 0.12);
            let height = rng.range(0.2, 0.42);
            daisy(out, &mut rng, at, height);
        }
        crowded = heads > 1;
    }
    grass(out, &mut rng, w, floor, spot, crowded);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat_of(v: &Vertex) -> Block {
        ((v.data >> 3) & 255) as Block
    }

    #[test]
    fn lupin_florets_stand_above_the_leaves() {
        for seed in 0..20 {
            let mut out = MeshData::default();
            let mut rng = Rng(seed);
            lupin(&mut out, &mut rng, Vec3::ZERO, 0.9, LUPIN);
            // The leaves are lit as if facing the sky (face 2); the stem is not.
            let leaf_top = out
                .vertices
                .iter()
                .filter(|v| mat_of(v) == STEM && v.data & 7 == 2)
                .map(|v| v.pos[1])
                .fold(f32::MIN, f32::max);
            let flowers: Vec<f32> = out
                .vertices
                .iter()
                .filter(|v| mat_of(v) == LUPIN)
                .map(|v| v.pos[1])
                .collect();
            assert!(flowers.len() >= 12 * 6, "{}", flowers.len());
            let lowest = flowers.iter().copied().fold(f32::MAX, f32::min);
            assert!(lowest > leaf_top + 0.1, "{lowest} vs {leaf_top}");
            assert!(flowers.iter().all(|&y| y < 0.9 + 0.05));
        }
    }

    #[test]
    fn florets_face_away_from_the_spike_and_daisies_face_up() {
        for seed in 0..10 {
            let mut out = MeshData::default();
            let mut rng = Rng(seed);
            lupin(&mut out, &mut rng, Vec3::ZERO, 1.0, LUPIN);
            daisy(&mut out, &mut rng, Vec3::new(2.0, 0.0, 0.0), 0.3);
            // The spike's axis runs from the root through its highest point.
            let top = out
                .vertices
                .iter()
                .filter(|v| mat_of(v) == LUPIN)
                .map(|v| Vec3::from(v.pos))
                .fold(Vec3::ZERO, |a, p| if p.y > a.y { p } else { a });
            let axis = top.normalize();
            for tri in out.indices.chunks(3) {
                let vs = [0, 1, 2].map(|k| out.vertices[tri[k] as usize]);
                let [a, b, c] = vs.map(|v| Vec3::from(v.pos));
                let n = (b - a).cross(c - a);
                let mid = (a + b + c) / 3.0;
                match mat_of(&vs[0]) {
                    LUPIN => {
                        let away = mid - axis * mid.dot(axis);
                        assert!(n.dot(away) > 0.0 || n.dot(axis) > 0.0, "{a} {b} {c}");
                    }
                    DAISY | DAISY_HEART => assert!(n.y > 0.0, "{a} {b} {c}"),
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn grass_stays_low_polygon() {
        let mut out = MeshData::default();
        let spot = Spot {
            meadow: 1.0,
            field: 1.0,
            verge: 1.0,
            sheltered: false,
        };
        for x in 0..32 {
            tuft(&mut out, 3, IVec3::new(x, 10, 0), &spot);
        }
        let per_voxel = out.indices.len() / 3 / 32;
        // The worst case: a verge in a lupin drift.
        assert!((10..=100).contains(&per_voxel), "{per_voxel}");
        assert!(out.vertices.iter().all(|v| v.pos[1] > 4.8 && v.pos[1] < 5.0 + 2.2));
    }
}
