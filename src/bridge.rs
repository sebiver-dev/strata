//! The stone bridge where the road crosses the river below the hamlet: two
//! broad round arches springing just above the water, a pier with pointed
//! cutwaters between them, a ring of wedge-shaped voussoirs round each arch,
//! rough coursed spandrel walls, a string course, a solid parapet capped with
//! coping stones, lanterns on stone pillars and a timber banner post with a
//! lantern at each end.
//!
//! It is an authored model laid stone by stone (see `model`), backed by a
//! plain core so no gap between stones ever shows daylight. The voxels under
//! it only give collision (`BUILT`, never drawn), carry the lanterns, and
//! clear the air over the deck and under the arches; the river keeps its
//! water.

use crate::block::*;
use crate::mesh::{smooth_data, MeshData, Vertex};
use crate::model;
use crate::noise::{hash2, unit};
use crate::terrain::{water_level, Terrain};
use glam::Vec3;

/// Outer half-width of the bridge, parapets included.
pub const BRIDGE_HALF_W: f32 = 2.75;
/// Half-width of the walkway between the parapets.
const DECK_HALF: f32 = 2.25;
/// Radius of each arch (a span of 7 m).
const ARCH_R: f32 = 3.5;
/// Depth of the voussoir ring round each arch.
const RING: f32 = 0.6;
/// Half the pier's width along the bridge.
const PIER_HALF: f32 = 1.2;
/// How far above the water the arches spring.
const SPRING_ABOVE: f32 = 0.4;
/// Half the length of the deck's rounded hump; beyond it the ramps run straight.
const HUMP: f32 = 10.0;
/// Rise per metre of the ramps (the hump's slope where it meets them).
const RAMP: f32 = 0.22;
/// Height of the parapet over the deck, coping aside.
const PARAPET_H: f32 = 1.0;
/// Top of the spandrel walls under the string course, below the deck.
const WALL_TOP: f32 = 0.38;
/// Wedge stones in each arch ring.
const VOUSSOIRS: usize = 15;
/// Depth of the face stones laid over the core.
const FACE_DEPTH: f32 = 0.34;
/// Gap left between neighbouring stones.
const JOINT: f32 = 0.022;
/// Lanterns on pillars stand over the pier and a little past each arch.
const PILLAR_OFFSETS: [f32; 3] = [-9.5, 0.0, 9.5];
/// Width of a banner; the shader's trim assumes edges on multiples of it.
pub const BANNER_W: f32 = 0.8;
/// Length of a banner.
const BANNER_H: f32 = 1.7;

/// A lantern on a stone pillar in the parapet.
#[derive(Clone, Debug)]
struct Pillar {
    x: f32,
    z: f32,
    /// Floor of the lantern voxel, the pillar's top.
    top: f32,
}

/// A timber post at the end of the bridge with a banner and a hanging lantern.
#[derive(Clone, Debug)]
struct BannerPost {
    x: f32,
    z: f32,
    ground: f32,
    /// Floor of the voxel the lantern hangs in.
    lamp: f32,
    /// Direction along X from the post to the lantern (the banner hangs the other way).
    side: f32,
}

/// A stone bridge of two round arches, crossing the river along X.
#[derive(Clone, Debug)]
pub struct Bridge {
    /// Centre line of the bridge.
    pub z: f32,
    /// X of the pier between the arches.
    pub centre: f32,
    /// Where the ramps meet the banks.
    pub x0: f32,
    pub x1: f32,
    /// Height of the walking surface over the pier.
    pub deck: f32,
    /// Water level under the bridge and where the arches spring.
    pub water: f32,
    pub spring: f32,
    /// Lowest ground under the bridge (the river bed).
    pub bed: f32,
    pub seed: u32,
    pillars: Vec<Pillar>,
    posts: Vec<BannerPost>,
}

/// Snaps a coordinate so a span of `2 * half` metres covers whole voxels.
fn snap(x: f32, half: f32) -> f32 {
    crate::cottage::snap(x, half)
}

/// Whether `p` lies in the voxel whose centre is nearest `target`.
fn in_voxel(p: Vec3, target: Vec3) -> bool {
    (p / VOXEL_SIZE).floor() == (target / VOXEL_SIZE).floor()
}

/// Where the bridge crosses the river (Z, metres): 72 m ahead of the spawn
/// point, where the road meets the river, with the falls upstream seen
/// through its arches. Swap for the vista's own constant when it lands.
pub const BRIDGE_Z: f32 = crate::vista::BRIDGE_Z;

/// The narrowest reach of water within a few metres of `BRIDGE_Z`,
/// preferring the middle and never near a fall.
pub fn site(t: &Terrain, _spawn: Vec3) -> f32 {
    let want = BRIDGE_Z;
    let mut best = (f32::MAX, want);
    for k in -2..=2 {
        let z = want + k as f32;
        let rx = t.river_x(z);
        let w = water_level(rx, z);
        let level = |dz: f32| water_level(rx, z + dz);
        if level(-6.0) != w || level(6.0) != w {
            continue;
        }
        let mut wet = 0.0;
        for i in -60..=60 {
            if t.height_at(rx + i as f32 * 0.5, z).0 < w {
                wet += 0.5;
            }
        }
        let score = wet + (k as f32).abs() * 0.3;
        if score < best.0 {
            best = (score, z);
        }
    }
    best.1
}

impl Bridge {
    pub fn plan(t: &Terrain, z: f32, seed: u32) -> Self {
        // Odd voxel counts across both put the parapets and pillars on voxel centres.
        let z = snap(z, BRIDGE_HALF_W);
        let centre = snap(t.river_x(z), 0.25);
        let water = water_level(centre, z);
        let spring = water + SPRING_ABOVE;
        let a = RAMP / (2.0 * HUMP);
        let crown_x = PIER_HALF + ARCH_R;
        // The deck clears the ring over each arch's crown with room for fill and paving.
        let deck = spring + ARCH_R + RING + 0.45 + a * crown_x * crown_x;
        let mut b = Bridge {
            z,
            centre,
            x0: centre,
            x1: centre,
            deck,
            water,
            spring,
            bed: water,
            seed,
            pillars: Vec::new(),
            posts: Vec::new(),
        };
        let ground = |x: f32| {
            [-BRIDGE_HALF_W, 0.0, BRIDGE_HALF_W]
                .iter()
                .map(|dz| t.height_at(x, z + dz).0)
                .fold(f32::MIN, f32::max)
        };
        // Ramps run down until they meet the bank.
        let abut = PIER_HALF + 2.0 * ARCH_R;
        for dir in [-1.0f32, 1.0] {
            let mut d = ((abut + 1.5) / VOXEL_SIZE).ceil() * VOXEL_SIZE;
            while d < 45.0 {
                let x = centre + dir * d;
                if b.top(x) <= ground(x) + 0.3 {
                    break;
                }
                d += 0.5;
            }
            if dir < 0.0 {
                b.x0 = centre - d;
            } else {
                b.x1 = centre + d;
            }
        }
        let mut x = b.x0;
        while x <= b.x1 {
            b.bed = b.bed.min(t.height_at(x, z).0);
            x += 1.0;
        }
        for off in PILLAR_OFFSETS {
            let x = centre + off;
            if x <= b.x0 + 1.5 || x >= b.x1 - 1.5 {
                continue;
            }
            let top = ((b.top(x) + PARAPET_H + 0.3) / VOXEL_SIZE).ceil() * VOXEL_SIZE;
            for s in [-1.0f32, 1.0] {
                b.pillars.push(Pillar {
                    x,
                    z: z + s * (BRIDGE_HALF_W - VOXEL_SIZE * 0.5),
                    top,
                });
            }
        }
        // Banner posts stand on the side facing the hamlet, just off each end.
        for (x, side) in [(b.x0 - 0.5, -1.0f32), (b.x1 + 0.5, 1.0)] {
            let pz = z + BRIDGE_HALF_W + 1.25;
            let g = t.height_at(x, pz).0;
            let lamp = ((g + 2.3) / VOXEL_SIZE).round() * VOXEL_SIZE;
            b.posts.push(BannerPost {
                x,
                z: pz,
                ground: g,
                lamp,
                side,
            });
        }
        b
    }

    /// Height of the walking surface: a gentle hump over the arches easing
    /// into straight ramps down to the banks.
    pub fn top(&self, x: f32) -> f32 {
        let a = RAMP / (2.0 * HUMP);
        let d = (x - self.centre).abs();
        if d <= HUMP {
            self.deck - a * d * d
        } else {
            self.deck - a * HUMP * HUMP - (d - HUMP) * RAMP
        }
    }

    /// Slope of the walking surface.
    fn slope(&self, x: f32) -> f32 {
        let d = x - self.centre;
        -(d.clamp(-HUMP, HUMP) * RAMP / HUMP)
    }

    fn arches(&self) -> [f32; 2] {
        let o = PIER_HALF + ARCH_R;
        [self.centre - o, self.centre + o]
    }

    /// Whether a point is inside an arch's opening (under the intrados).
    fn in_opening(&self, x: f32, y: f32) -> bool {
        self.arches().iter().any(|&ac| {
            let dx = x - ac;
            if y < self.spring {
                dx.abs() < ARCH_R
            } else {
                let dy = y - self.spring;
                dx * dx + dy * dy < ARCH_R * ARCH_R
            }
        })
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        (
            Vec3::new(self.x0 - 2.0, self.bed - 1.0, self.z - BRIDGE_HALF_W),
            Vec3::new(self.x1 + 2.0, self.deck + 4.0, self.z + BRIDGE_HALF_W + 1.5),
        )
    }

    /// Positions of every lantern voxel (centres), on pillars and on the banner posts.
    pub fn lanterns(&self) -> Vec<Vec3> {
        let h = VOXEL_SIZE * 0.5;
        let mut out: Vec<Vec3> = self.pillars.iter().map(|p| Vec3::new(p.x, p.top + h, p.z)).collect();
        out.extend(
            self.posts
                .iter()
                .map(|p| Vec3::new(p.x + p.side * 1.0, p.lamp + h, p.z)),
        );
        out
    }

    /// Collision and light voxels, and the air the bridge clears around itself.
    pub fn block(&self, p: Vec3, ground: f32) -> Option<Block> {
        let h = VOXEL_SIZE * 0.5;
        for post in &self.posts {
            if in_voxel(p, Vec3::new(post.x + post.side, post.lamp + h, post.z)) {
                return Some(LANTERN);
            }
            let foot = Vec3::new(post.x, p.y, post.z);
            if in_voxel(p, foot) && p.y >= ground - 0.5 && p.y < post.lamp + 1.0 {
                return Some(BUILT);
            }
        }
        let dz = (p.z - self.z).abs();
        if p.x < self.x0 || p.x > self.x1 || dz > BRIDGE_HALF_W {
            return None;
        }
        let top = self.top(p.x);
        if p.y < top {
            if p.y < ground - 1.0 {
                return None;
            }
            if self.in_opening(p.x, p.y) {
                return (p.y > ground && p.y > self.water).then_some(AIR);
            }
            return Some(BUILT);
        }
        if dz > DECK_HALF {
            for pl in &self.pillars {
                let col = Vec3::new(pl.x, p.y, pl.z);
                if in_voxel(p, col) {
                    if p.y >= pl.top + VOXEL_SIZE {
                        break;
                    }
                    if p.y >= pl.top {
                        return Some(LANTERN);
                    }
                    if p.y >= pl.top - VOXEL_SIZE {
                        // The lantern stands on a post voxel, hidden inside the pillar.
                        return Some(POST);
                    }
                    return Some(BUILT);
                }
            }
            if p.y - top < PARAPET_H {
                return Some(BUILT);
            }
        }
        (p.y - top < 3.0).then_some(AIR)
    }

    fn rnd(&self, i: i32, j: i32) -> f32 {
        unit(hash2(self.seed, i, j))
    }

    /// Tops of the masonry courses from below the river bed to the deck;
    /// the same on every face so the courses line up round the corners.
    fn courses(&self) -> Vec<f32> {
        let mut out = vec![self.bed - 0.6];
        let mut k = 0;
        while *out.last().unwrap() < self.deck {
            let h = 0.36 + 0.14 * self.rnd(k, 900);
            out.push(out.last().unwrap() + h);
            k += 1;
        }
        out
    }

    /// The bridge as an authored model, in world space.
    pub fn model(&self, t: &Terrain, out: &mut MeshData) {
        let courses = self.courses();
        self.core(t, out);
        self.spandrels(t, out, &courses);
        self.inner_faces(t, out, &courses);
        self.rings(out);
        self.cutwaters(out, &courses);
        self.parapets(out);
        self.deck_surface(out);
        self.pillar_models(out);
        for post in &self.posts {
            self.banner_post(post, out);
        }
    }

    /// Pieces along the bridge no longer than `step`, broken where the
    /// openings begin and end.
    fn pieces(&self, step: f32) -> Vec<(f32, f32)> {
        let mut cuts = vec![self.x0, self.x1];
        for ac in self.arches() {
            cuts.push(ac - ARCH_R);
            cuts.push(ac + ARCH_R);
        }
        cuts.sort_by(|a, b| a.total_cmp(b));
        let mut out = Vec::new();
        for w in cuts.windows(2) {
            let (a, b) = (w[0].max(self.x0), w[1].min(self.x1));
            if b - a < 1e-3 {
                continue;
            }
            let n = ((b - a) / step).ceil().max(1.0) as usize;
            for k in 0..n {
                out.push((
                    a + (b - a) * k as f32 / n as f32,
                    a + (b - a) * (k + 1) as f32 / n as f32,
                ));
            }
        }
        out
    }

    /// Height of an arch's intrados over `x`, if `x` lies within a span.
    fn intrados(&self, x: f32) -> Option<f32> {
        self.arches().iter().find_map(|&ac| {
            let dx = x - ac;
            (dx.abs() <= ARCH_R).then(|| self.spring + (ARCH_R * ARCH_R - dx * dx).max(0.0).sqrt())
        })
    }

    /// The fill behind the face stones: plain, dark slices from the ground
    /// (or the arch) up to just under the deck, so joints show shadow, not sky.
    fn core(&self, t: &Terrain, out: &mut MeshData) {
        let w = BRIDGE_HALF_W - FACE_DEPTH * 0.6;
        for (xa, xb) in self.pieces(0.5) {
            let xm = (xa + xb) * 0.5;
            let bottom = match self.intrados(xm) {
                Some(_) => self
                    .intrados(xa)
                    .unwrap_or(self.spring)
                    .max(self.intrados(xb).unwrap_or(self.spring)),
                None => {
                    let g = [-w, 0.0, w]
                        .iter()
                        .flat_map(|dz| [t.height_at(xa, self.z + dz).0, t.height_at(xb, self.z + dz).0])
                        .fold(f32::MAX, f32::min);
                    g - 0.6
                }
            };
            let (ta, tb) = (self.top(xa) - 0.03, self.top(xb) - 0.03);
            if ta.min(tb) <= bottom {
                continue;
            }
            hexa(
                out,
                [
                    Vec3::new(xa, bottom, self.z - w),
                    Vec3::new(xb, bottom, self.z - w),
                    Vec3::new(xb, bottom, self.z + w),
                    Vec3::new(xa, bottom, self.z + w),
                    Vec3::new(xa, ta, self.z - w),
                    Vec3::new(xb, tb, self.z - w),
                    Vec3::new(xb, tb, self.z + w),
                    Vec3::new(xa, ta, self.z + w),
                ],
                BRIDGE_STONE,
                0,
            );
        }
    }

    /// Lays courses of rough stones on one face. `at(s, y, d)` maps a
    /// distance along the face, a height and a depth out of the face (0 on
    /// the face plane) to world space; `runs(y0, y1)` gives the stretches a
    /// course may fill, `cap(s)` the highest a stone may reach and `ground(s)`
    /// the ground, below which stones are left out.
    #[allow(clippy::too_many_arguments)]
    fn lay(
        &self,
        out: &mut MeshData,
        key: i32,
        courses: &[f32],
        depth: f32,
        at: impl Fn(f32, f32, f32) -> Vec3,
        runs: impl Fn(f32, f32) -> Vec<(f32, f32)>,
        cap: impl Fn(f32) -> f32,
        ground: impl Fn(f32) -> f32,
    ) {
        for (row, c) in courses.windows(2).enumerate() {
            let (y0, y1) = (c[0], c[1]);
            let row = row as i32;
            for (ri, (s0, s1)) in runs(y0, y1).into_iter().enumerate() {
                let mut s = s0;
                let mut k = 0;
                // Bond: each course starts with a different stone length.
                let mut first = true;
                while s < s1 - 0.05 {
                    let kk = k;
                    let r = |j: i32| self.rnd(key * 7919 + row * 131 + ri as i32 * 17 + kk, j);
                    let mut len = 0.5 + 0.65 * r(1);
                    if first {
                        len *= 0.4 + 0.6 * r(5);
                        first = false;
                    }
                    let (sa, mut sb) = (s, (s + len).min(s1));
                    // Never leave a sliver at the end of a run.
                    if s1 - sb < 0.3 {
                        sb = s1;
                    }
                    s = sb;
                    k += 1;
                    let (ya, yb) = (y1.min(cap(sa)), y1.min(cap(sb)));
                    if ya.max(yb) - y0 < 0.12 || sb - sa < 0.12 {
                        continue;
                    }
                    if ya.max(yb) < ground(sa).min(ground(sb)) - 0.2 {
                        continue;
                    }
                    let jut = 0.035 * r(2);
                    let (ya, yb) = (ya.max(y0 + 0.08), yb.max(y0 + 0.08));
                    let (a0, a1) = (sa + JOINT, sb - JOINT);
                    let (b0, ta, tb) = (y0 + JOINT, ya - JOINT, yb - JOINT);
                    // A slight random tilt of the face so neighbours catch the light differently.
                    let tilt = (r(3) - 0.5) * 0.03;
                    hexa(
                        out,
                        [
                            at(a0, b0, -depth),
                            at(a1, b0, -depth),
                            at(a1, b0, jut - tilt),
                            at(a0, b0, jut + tilt),
                            at(a0, ta, -depth),
                            at(a1, tb, -depth),
                            at(a1, tb, jut - tilt * 0.5),
                            at(a0, ta, jut + tilt * 0.5),
                        ],
                        BRIDGE_STONE,
                        3,
                    );
                }
            }
        }
    }

    /// Stretches of a course along the bridge between the arch rings.
    fn spandrel_runs(&self, y0: f32, y1: f32) -> Vec<(f32, f32)> {
        let re = ARCH_R + RING;
        let mut blocked: Vec<(f32, f32)> = Vec::new();
        for ac in self.arches() {
            let half = if y1 <= self.spring {
                ARCH_R
            } else {
                let dy = y1 - self.spring;
                if dy >= re {
                    continue;
                }
                (re * re - dy * dy).sqrt()
            };
            blocked.push((ac - half, ac + half));
        }
        let _ = y0;
        let mut runs = Vec::new();
        let mut s = self.x0;
        for (a, b) in blocked {
            if a > s {
                runs.push((s - self.x0, a - self.x0));
            }
            s = s.max(b);
        }
        if self.x1 > s {
            runs.push((s - self.x0, self.x1 - self.x0));
        }
        runs
    }

    /// The two long faces: abutments, pier and spandrels in rough courses.
    fn spandrels(&self, t: &Terrain, out: &mut MeshData, courses: &[f32]) {
        for (k, side) in [-1.0f32, 1.0].into_iter().enumerate() {
            let zf = self.z + side * BRIDGE_HALF_W;
            let x0 = self.x0;
            self.lay(
                out,
                k as i32,
                courses,
                FACE_DEPTH,
                |s, y, d| Vec3::new(x0 + s, y, zf + side * d),
                |y0, y1| self.spandrel_runs(y0, y1),
                |s| self.top(x0 + s) - WALL_TOP,
                |s| t.height_at(x0 + s, zf).0,
            );
        }
    }

    /// The faces of the pier and abutments that look into the arches, below the springing.
    fn inner_faces(&self, t: &Terrain, out: &mut MeshData, courses: &[f32]) {
        let low: Vec<f32> = courses.iter().copied().take_while(|&y| y < self.spring + 0.4).collect();
        let z0 = self.z - BRIDGE_HALF_W;
        let mut key = 10;
        for ac in self.arches() {
            for dir in [-1.0f32, 1.0] {
                // The face at ac + dir * R looks back towards the arch centre.
                let xf = ac + dir * ARCH_R;
                let spring = self.spring;
                self.lay(
                    out,
                    key,
                    &low,
                    FACE_DEPTH,
                    |s, y, d| Vec3::new(xf - dir * d, y, z0 + s),
                    |_, _| vec![(0.0, 2.0 * BRIDGE_HALF_W)],
                    |_| spring,
                    |s| t.height_at(xf, z0 + s).0,
                );
                key += 1;
            }
        }
    }

    /// Wedge-shaped voussoirs round each arch, running right through the
    /// barrel in three staggered lengths, with a proud keystone.
    fn rings(&self, out: &mut MeshData) {
        let w = BRIDGE_HALF_W + 0.06;
        for (ai, ac) in self.arches().into_iter().enumerate() {
            for i in 0..VOUSSOIRS {
                let gap = 0.012;
                let a0 = std::f32::consts::PI * i as f32 / VOUSSOIRS as f32 + gap;
                let a1 = std::f32::consts::PI * (i + 1) as f32 / VOUSSOIRS as f32 - gap;
                let key = i == VOUSSOIRS / 2;
                let r = |j: i32| self.rnd(ai as i32 * 100 + i as i32, 300 + j);
                let re = ARCH_R + RING + if key { 0.18 } else { (r(1) - 0.5) * 0.1 };
                let proud = if key { 0.08 } else { 0.03 * r(2) };
                let pt =
                    |rad: f32, a: f32, z: f32| Vec3::new(ac + rad * a.cos(), self.spring + rad * a.sin(), self.z + z);
                // Staggered joints through the barrel.
                let m = if i % 2 == 0 { 0.7 } else { -0.7 };
                let cuts = [-w - proud, -1.1 + m * 0.5, 1.1 + m * 0.5, w + proud];
                for c in cuts.windows(2) {
                    let (z0, z1) = (c[0] + 0.01, c[1] - 0.01);
                    let (r0, r1) = (ARCH_R + 0.01, re);
                    hexa(
                        out,
                        [
                            pt(r0, a0, z0),
                            pt(r0, a1, z0),
                            pt(r0, a1, z1),
                            pt(r0, a0, z1),
                            pt(r1, a0, z0),
                            pt(r1, a1, z0),
                            pt(r1, a1, z1),
                            pt(r1, a0, z1),
                        ],
                        BRIDGE_STONE,
                        3,
                    );
                }
            }
        }
    }

    /// Pointed cutwaters on both sides of the pier, in courses, each capped
    /// with a sloping stone roof that runs back into the spandrel.
    fn cutwaters(&self, out: &mut MeshData, courses: &[f32]) {
        let c = self.centre;
        let top = self.spring + 0.9;
        for side in [-1.0f32, 1.0] {
            let zf = self.z + side * (BRIDGE_HALF_W - 0.1);
            let reach = 1.5;
            let at = |y: f32, inset: f32| {
                [
                    Vec3::new(c - PIER_HALF + inset, y, zf),
                    Vec3::new(c + PIER_HALF - inset, y, zf),
                    Vec3::new(c, y, zf + side * (reach - inset * 1.3)),
                ]
            };
            let mut y_top = top;
            for (k, w) in courses.windows(2).enumerate() {
                let (y0, y1) = (w[0], w[1].min(top));
                if y0 >= top {
                    break;
                }
                y_top = y1;
                let inset = 0.02 * self.rnd(k as i32, 400);
                let (lo, hi) = (at(y0 + JOINT, inset), at(y1 - JOINT, inset));
                hexa(
                    out,
                    [lo[0], lo[1], lo[2], lo[2], hi[0], hi[1], hi[2], hi[2]],
                    BRIDGE_STONE,
                    3,
                );
            }
            // The cap slopes up from the nose to the face of the spandrel.
            let lo = at(y_top, -0.06);
            let rise = 1.3;
            let back = |x: f32| Vec3::new(x, y_top + rise, zf);
            hexa(
                out,
                [
                    lo[0],
                    lo[1],
                    lo[2],
                    lo[2],
                    back(c - PIER_HALF - 0.06),
                    back(c + PIER_HALF + 0.06),
                    back(c),
                    back(c),
                ],
                BRIDGE_STONE,
                3,
            );
        }
    }

    /// A point following the deck: `s` along the bridge, `y` over the deck
    /// line, at `z` across.
    fn on_deck(&self, s: f32, y: f32, z: f32) -> Vec3 {
        let x = self.x0 + s;
        Vec3::new(x, self.top(x) + y, self.z + z)
    }

    /// The string course, the parapet walls with their core and coping.
    fn parapets(&self, out: &mut MeshData) {
        let len = self.x1 - self.x0;
        let newel = 0.35;
        // Parapet stones stop at the end pillars.
        let span = vec![(newel, len - newel)];
        let pillar_gaps = |s0: f32, s1: f32| {
            let mut runs = Vec::new();
            let mut s = s0;
            for p in self.pillars.iter().step_by(2) {
                let ps = p.x - self.x0;
                if ps - 0.3 > s {
                    runs.push((s, ps - 0.3));
                }
                s = s.max(ps + 0.3);
            }
            if s1 > s {
                runs.push((s, s1));
            }
            runs
        };
        for (k, side) in [-1.0f32, 1.0].into_iter().enumerate() {
            let outer = side * BRIDGE_HALF_W;
            let inner = side * DECK_HALF;
            // Core of the parapet, a little inside both faces.
            for (xa, xb) in self.pieces(0.5) {
                let (sa, sb) = (xa - self.x0, xb - self.x0);
                let (zi, zo) = (inner + side * 0.05, outer - side * 0.05);
                let (ylo, yhi) = (-WALL_TOP, PARAPET_H - 0.02);
                hexa(
                    out,
                    [
                        self.on_deck(sa, ylo, zi),
                        self.on_deck(sb, ylo, zi),
                        self.on_deck(sb, ylo, zo),
                        self.on_deck(sa, ylo, zo),
                        self.on_deck(sa, yhi, zi),
                        self.on_deck(sb, yhi, zi),
                        self.on_deck(sb, yhi, zo),
                        self.on_deck(sa, yhi, zo),
                    ],
                    BRIDGE_STONE,
                    0,
                );
            }
            // String course: a projecting band of long stones under the parapet.
            self.lay(
                out,
                20 + k as i32,
                &[-WALL_TOP, -0.06],
                0.3,
                |s, y, d| self.on_deck(s, y, outer + side * (d + 0.05)),
                |_, _| span.clone(),
                |_| f32::MAX,
                |_| f32::MIN,
            );
            // Two courses on each face of the parapet.
            let rows = [-0.06, 0.44, PARAPET_H];
            self.lay(
                out,
                30 + k as i32,
                &rows,
                0.16,
                |s, y, d| self.on_deck(s, y, outer + side * d),
                |_, _| span.clone(),
                |_| f32::MAX,
                |_| f32::MIN,
            );
            self.lay(
                out,
                40 + k as i32,
                &rows,
                0.16,
                |s, y, d| self.on_deck(s, y, inner - side * d),
                |_, _| pillar_gaps(newel, len - newel),
                |_| f32::MAX,
                |_| f32::MIN,
            );
            // Coping: broad, slightly rounded stones overhanging both faces.
            for (ri, (s0, s1)) in pillar_gaps(newel, len - newel).into_iter().enumerate() {
                let mut s = s0;
                let mut i = 0;
                while s < s1 - 0.05 {
                    let r = |j: i32| self.rnd(500 + k as i32 * 50 + ri as i32, i * 10 + j);
                    let mut sb = (s + 0.65 + 0.35 * r(1)).min(s1);
                    if s1 - sb < 0.3 {
                        sb = s1;
                    }
                    let (a, b) = (s + JOINT, sb - JOINT);
                    let lift = 0.02 * r(2);
                    let (y0, y1) = (PARAPET_H - 0.02, PARAPET_H + 0.15 + lift);
                    let (zi, zo) = (inner - side * 0.07, outer + side * 0.07);
                    hexa(
                        out,
                        [
                            self.on_deck(a, y0, zi),
                            self.on_deck(b, y0, zi),
                            self.on_deck(b, y0, zo),
                            self.on_deck(a, y0, zo),
                            self.on_deck(a, y1, zi + side * 0.03),
                            self.on_deck(b, y1, zi + side * 0.03),
                            self.on_deck(b, y1, zo - side * 0.03),
                            self.on_deck(a, y1, zo - side * 0.03),
                        ],
                        BRIDGE_STONE,
                        3,
                    );
                    s = sb;
                    i += 1;
                }
            }
            // Newels closing the parapet at both ends, with pyramid caps.
            for s in [newel, len - newel] {
                let x = self.x0 + s;
                let zc = self.z + side * (BRIDGE_HALF_W - 0.25);
                let t = self.top(x);
                let h = t + PARAPET_H + 0.3;
                let hw = 0.36;
                model::block(
                    out,
                    Vec3::new(x - hw, t - 1.2, zc - hw),
                    Vec3::new(x + hw, h, zc + hw),
                    BRIDGE_STONE,
                );
                pyramid(out, Vec3::new(x, h, zc), hw + 0.06, 0.35, BRIDGE_STONE);
            }
        }
    }

    /// The walking surface: cobbles over the fill, following the hump.
    fn deck_surface(&self, out: &mut MeshData) {
        for (xa, xb) in self.pieces(0.5) {
            let n = |x: f32| Vec3::new(-self.slope(x), 1.0, 0.0).normalize();
            let (za, zb) = (self.z - DECK_HALF - 0.05, self.z + DECK_HALF + 0.05);
            let start = out.vertices.len() as u32;
            for (x, z) in [(xa, za), (xa, zb), (xb, zb), (xb, za)] {
                out.vertices.push(Vertex {
                    pos: [x, self.top(x), z],
                    data: smooth_data(PATH, 3, n(x)),
                });
            }
            out.indices
                .extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
        }
    }

    /// Stone pillars in the parapet with a lantern standing on each.
    fn pillar_models(&self, out: &mut MeshData) {
        for p in &self.pillars {
            let t = self.top(p.x);
            let hw = 0.3;
            model::block(
                out,
                Vec3::new(p.x - hw, t - WALL_TOP, p.z - hw - 0.04),
                Vec3::new(p.x + hw, p.top - 0.14, p.z + hw + 0.04),
                BRIDGE_STONE,
            );
            // A moulded base course and a cap slab.
            model::block(
                out,
                Vec3::new(p.x - hw - 0.05, t + PARAPET_H - 0.05, p.z - hw - 0.09),
                Vec3::new(p.x + hw + 0.05, t + PARAPET_H + 0.12, p.z + hw + 0.09),
                BRIDGE_STONE,
            );
            model::block(
                out,
                Vec3::new(p.x - hw - 0.07, p.top - 0.14, p.z - hw - 0.1),
                Vec3::new(p.x + hw + 0.07, p.top, p.z + hw + 0.1),
                BRIDGE_STONE,
            );
        }
    }

    /// A rough timber post with a crossbar: a lantern hangs from one arm, a
    /// banner from the other.
    fn banner_post(&self, post: &BannerPost, out: &mut MeshData) {
        let foot = Vec3::new(post.x, post.ground - 0.5, post.z);
        // The lantern's rod reaches 0.63 m above its voxel floor.
        let bar_y = post.lamp + 0.7;
        let top = bar_y + 0.35;
        model::beam(out, foot, Vec3::new(post.x, top, post.z), 0.11, WOOD);
        model::block(
            out,
            Vec3::new(post.x - 0.14, top, post.z - 0.14),
            Vec3::new(post.x + 0.14, top + 0.06, post.z + 0.14),
            OAK,
        );
        // Banner width is fixed on a world grid so the shader can draw its trim.
        let mut bx = ((post.x - post.side * 0.8) / BANNER_W - 0.5).round() * BANNER_W + BANNER_W * 0.5;
        if (bx - post.x).abs() < BANNER_W * 0.5 + 0.15 {
            bx -= post.side * BANNER_W;
        }
        let lamp_x = post.x + post.side * 1.0;
        let (xa, xb) = (bx.min(lamp_x) - 0.5, bx.max(lamp_x) + 0.12);
        let xa = xa.min(post.x - 0.25);
        let xb = xb.max(post.x + 0.25);
        model::block(
            out,
            Vec3::new(xa, bar_y - 0.07, post.z - 0.07),
            Vec3::new(xb, bar_y + 0.07, post.z + 0.07),
            OAK,
        );
        // Knee braces under both arms.
        for s in [-1.0f32, 1.0] {
            model::beam(
                out,
                Vec3::new(post.x, bar_y - 0.6, post.z),
                Vec3::new(post.x + s * 0.55, bar_y - 0.05, post.z),
                0.05,
                OAK,
            );
        }
        // Banner rod on two short cords, then the cloth.
        let rod_y = bar_y - 0.22;
        model::beam(
            out,
            Vec3::new(bx - BANNER_W * 0.5 - 0.06, rod_y, post.z),
            Vec3::new(bx + BANNER_W * 0.5 + 0.06, rod_y, post.z),
            0.022,
            OAK,
        );
        for s in [-1.0f32, 1.0] {
            let x = bx + s * BANNER_W * 0.42;
            model::beam(
                out,
                Vec3::new(x, rod_y, post.z),
                Vec3::new(x, bar_y - 0.07, post.z),
                0.008,
                IRON,
            );
        }
        banner(out, Vec3::new(bx, rod_y - 0.02, post.z), BANNER_W, BANNER_H);
    }

    /// A light stand-in for far away with the same silhouette: slices under
    /// the deck with the arches cut out, the parapets and the pier.
    pub fn far(&self, out: &mut MeshData) {
        let w = BRIDGE_HALF_W;
        // The body, its underside following each arch's curve.
        for (xa, xb) in self.pieces(0.5) {
            let xm = (xa + xb) * 0.5;
            let (ba, bb) = match self.intrados(xm) {
                Some(_) => (
                    self.intrados(xa).unwrap_or(self.spring),
                    self.intrados(xb).unwrap_or(self.spring),
                ),
                None => (self.bed, self.bed),
            };
            let (ta, tb) = (self.top(xa), self.top(xb));
            slab(out, (xa, xb), (ba, bb), (ta, tb), (self.z - w, self.z + w));
        }
        // Parapets and their lantern pillars.
        for (xa, xb) in self.pieces(1.0) {
            let (ta, tb) = (self.top(xa), self.top(xb));
            let h = PARAPET_H + 0.12;
            for s in [-1.0f32, 1.0] {
                let (z0, z1) = (self.z + s * DECK_HALF, self.z + s * (w + 0.05));
                slab(out, (xa, xb), (ta, tb), (ta + h, tb + h), (z0.min(z1), z0.max(z1)));
            }
        }
        for p in &self.pillars {
            let t = self.top(p.x) + PARAPET_H;
            slab(
                out,
                (p.x - 0.35, p.x + 0.35),
                (t, t),
                (p.top, p.top),
                (p.z - 0.4, p.z + 0.4),
            );
        }
    }
}

/// A block of bridge stone between two X positions, with its bottom and top
/// sloping linearly between the given heights at each end.
fn slab(out: &mut MeshData, x: (f32, f32), lo: (f32, f32), hi: (f32, f32), z: (f32, f32)) {
    hexa(
        out,
        [
            Vec3::new(x.0, lo.0, z.0),
            Vec3::new(x.1, lo.1, z.0),
            Vec3::new(x.1, lo.1, z.1),
            Vec3::new(x.0, lo.0, z.1),
            Vec3::new(x.0, hi.0, z.0),
            Vec3::new(x.1, hi.1, z.0),
            Vec3::new(x.1, hi.1, z.1),
            Vec3::new(x.0, hi.0, z.1),
        ],
        BRIDGE_STONE,
        3,
    );
}

/// A convex six-sided solid from eight corners: the bottom four, then the
/// top four above them in the same order. Corners may coincide (wedges and
/// prisms); faces that collapse are left out. Normals lean a little away
/// from the centre so edges read soft. `ao` darkens it in the ambient light
/// (0 for the dark core showing through joints).
pub fn hexa(out: &mut MeshData, c: [Vec3; 8], mat: Block, ao: u32) {
    let centre = c.iter().copied().sum::<Vec3>() / 8.0;
    const FACES: [[usize; 4]; 6] = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ];
    for f in FACES {
        let q = f.map(|i| c[i]);
        let n = (q[2] - q[0]).cross(q[3] - q[1]);
        if n.length_squared() < 1e-8 {
            continue;
        }
        let mut n = n.normalize();
        let fc = (q[0] + q[1] + q[2] + q[3]) * 0.25;
        let flip = n.dot(fc - centre) < 0.0;
        if flip {
            n = -n;
        }
        let start = out.vertices.len() as u32;
        for v in q {
            let lean = (v - fc).normalize_or_zero() * 0.22;
            out.vertices.push(Vertex {
                pos: v.to_array(),
                data: smooth_data(mat, ao, (n + lean).normalize()),
            });
        }
        let tris = if flip {
            [start, start + 2, start + 1, start, start + 3, start + 2]
        } else {
            [start, start + 1, start + 2, start, start + 2, start + 3]
        };
        out.indices.extend_from_slice(&tris);
    }
}

/// A four-sided pyramid cap on a square of half-width `hw` centred at `base`.
fn pyramid(out: &mut MeshData, base: Vec3, hw: f32, h: f32, mat: Block) {
    let apex = base + Vec3::Y * h;
    let b = |x: f32, z: f32| base + Vec3::new(x * hw, 0.0, z * hw);
    hexa(
        out,
        [
            b(-1.0, -1.0),
            b(1.0, -1.0),
            b(1.0, 1.0),
            b(-1.0, 1.0),
            apex,
            apex,
            apex,
            apex,
        ],
        mat,
        3,
    );
}

/// A hanging banner in the plane facing ±Z, its top edge centred at `top`.
/// The ambient-occlusion bits carry how far down the cloth each vertex is
/// (0 at the rod, 3 at the tip): the shader sways the cloth by it and draws
/// the trim with it. The bottom ends in a point.
fn banner(out: &mut MeshData, top: Vec3, w: f32, h: f32) {
    let rows = [0.0f32, 1.0 / 3.0, 2.0 / 3.0, 1.0];
    for normal in [Vec3::Z, Vec3::NEG_Z] {
        let start = out.vertices.len() as u32;
        for (k, v) in rows.iter().enumerate() {
            for u in [-0.5f32, 0.0, 0.5] {
                // The last row forms the point: the corners stop short.
                let y = if k == 3 && u != 0.0 { h - 0.3 } else { h * v };
                let pos = top + Vec3::new(u * w, -y, if normal.z > 0.0 { 0.002 } else { -0.002 });
                out.vertices.push(Vertex {
                    pos: pos.to_array(),
                    data: smooth_data(BANNER, k as u32, normal),
                });
            }
        }
        for k in 0..3u32 {
            for j in 0..2u32 {
                let a = start + k * 3 + j;
                let (b, c, d) = (a + 1, a + 4, a + 3);
                if normal.z > 0.0 {
                    out.indices.extend_from_slice(&[a, d, c, a, c, b]);
                } else {
                    out.indices.extend_from_slice(&[a, b, c, a, c, d]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> Terrain {
        Terrain::new(20260927)
    }

    fn bridge(t: &Terrain) -> Bridge {
        t.structures
            .iter()
            .find_map(|s| match s {
                crate::structures::Structure::Bridge(b) => Some(b.clone()),
                _ => None,
            })
            .expect("no bridge")
    }

    /// Height of the top of the solid voxels in a column (metres), scanning down from `from`.
    fn floor_at(t: &Terrain, x: f32, z: f32, from: f32) -> f32 {
        let g = t.height_at(x, z).0;
        let mut y = ((from / VOXEL_SIZE).floor() + 0.5) * VOXEL_SIZE;
        while y > g - 2.0 {
            let b = t.structures.block_at(Vec3::new(x, y, z), g);
            let solid = match b {
                Some(b) => is_solid(b),
                None => y < g,
            };
            if solid {
                return y + VOXEL_SIZE * 0.5;
            }
            y -= VOXEL_SIZE;
        }
        g
    }

    #[test]
    fn the_bridge_spans_the_river_with_two_arches_over_the_water() {
        let t = world();
        let b = bridge(&t);
        let ends = |x: f32| t.height_at(x, b.z).0;
        assert!(ends(b.x0) > b.water && ends(b.x1) > b.water, "{b:?}");
        for ac in b.arches() {
            // Open water under each arch: the structure leaves the river alone.
            let under = Vec3::new(ac + 0.25, b.water - 0.25, b.z + 0.25);
            assert_eq!(t.structures.block_at(under, b.water - 2.5), None);
            let above = Vec3::new(ac + 0.25, b.spring + 1.0, b.z + 0.25);
            assert_eq!(t.structures.block_at(above, b.water - 2.5), Some(AIR));
        }
        // The pier and the fill over the crowns are solid.
        let pier = Vec3::new(b.centre + 0.25, b.water + 0.25, b.z + 0.25);
        assert_eq!(t.structures.block_at(pier, b.water - 2.5), Some(BUILT));
        let deck = Vec3::new(b.centre + 0.25, b.deck - 0.25, b.z + 0.25);
        assert_eq!(t.structures.block_at(deck, b.water - 2.5), Some(BUILT));
    }

    #[test]
    fn the_deck_is_walkable_end_to_end_in_half_metre_steps() {
        let t = world();
        let b = bridge(&t);
        let mut prev: Option<f32> = None;
        let mut x = b.x0 - 3.0;
        while x <= b.x1 + 3.0 {
            let f = floor_at(&t, x, b.z + 0.25, b.deck + 3.0);
            if let Some(p) = prev {
                assert!((f - p).abs() <= VOXEL_SIZE + 1e-3, "step of {} at x {x}", f - p);
            }
            // Headroom over the walkway.
            if x > b.x0 && x < b.x1 {
                for dy in [0.25f32, 1.25, 1.75] {
                    let p = Vec3::new(x, f + dy, b.z + 0.25);
                    let blk = t.structures.block_at(p, t.height_at(x, b.z).0);
                    assert!(blk.is_none_or(|k| !is_solid(k)), "blocked at x {x} +{dy}: {blk:?}");
                }
                // The walking surface stays within a quarter metre of the modelled deck.
                if t.height_at(x, b.z).0 < b.top(x) - 0.5 {
                    assert!((f - b.top(x)).abs() <= 0.26, "x {x}: floor {f} deck {}", b.top(x));
                }
            }
            prev = Some(f);
            x += 0.5;
        }
    }

    #[test]
    fn the_parapets_block_but_the_deck_between_them_is_open() {
        let t = world();
        let b = bridge(&t);
        let x = b.centre + 4.25;
        let top = b.top(x);
        let g = b.water - 2.5;
        for side in [-1.0f32, 1.0] {
            let wall = Vec3::new(x, top + 0.5, b.z + side * (BRIDGE_HALF_W - 0.25));
            assert_eq!(t.structures.block_at(wall, g), Some(BUILT));
            let inside = Vec3::new(x, top + 0.5, b.z + side * (DECK_HALF - 0.25));
            assert_eq!(t.structures.block_at(inside, g), Some(AIR));
        }
    }

    #[test]
    fn lanterns_stand_on_posts_on_the_pillars() {
        let t = world();
        let b = bridge(&t);
        assert!(b.pillars.len() >= 4);
        for p in &b.pillars {
            let l = Vec3::new(p.x, p.top + 0.25, p.z);
            assert_eq!(t.structures.block_at(l, b.water), Some(LANTERN));
            assert_eq!(t.structures.block_at(l - Vec3::Y * 0.5, b.water), Some(POST));
        }
        assert_eq!(b.lanterns().len(), b.pillars.len() + 2);
    }

    #[test]
    fn the_bridge_model_stays_within_a_triangle_budget() {
        let t = world();
        let b = bridge(&t);
        let mut out = MeshData::default();
        b.model(&t, &mut out);
        let tris = out.indices.len() / 3;
        assert!(tris > 3000 && tris < 30_000, "{tris}");
        let mut far = MeshData::default();
        b.far(&mut far);
        assert!(far.indices.len() / 3 < 2500);
    }

    #[test]
    fn the_site_is_near_the_road_crossing_ahead_of_spawn() {
        let t = world();
        let s = t.spawn_point();
        let z = site(&t, s);
        assert!((z - (s.z - 72.0)).abs() <= 4.0);
    }
}
