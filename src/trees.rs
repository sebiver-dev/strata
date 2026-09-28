//! Trees as authored models. A broadleaf tree is a thick tapered trunk with
//! a root flare and a slight lean that forks into a few boughs; each bough
//! ends in a rounded clump of foliage, and the clumps overlap into one full,
//! cloud-like canopy that always wraps the bough ends. Conifers are a straight
//! tapering trunk under stacked, drooping tiers of needles.
//!
//! The same `Shape` drives the near model (`append`, meshed into the chunk
//! that holds the tree's foot), the coarse far model (`far`) and the hidden
//! collision voxels in the trunk (`trunk_cells`), so all three agree.

use crate::block::{BARK, FOLIAGE, NEEDLES};
use crate::mesh::{smooth_data, MeshData, Vertex};
use crate::noise::{hash3, unit, value3};
use glam::{Vec2, Vec3};
use std::f32::consts::{PI, TAU};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Broadleaf,
    Conifer,
}

#[derive(Clone, Copy, Debug)]
pub struct Tree {
    /// Where the trunk meets the ground, in metres.
    pub base: Vec3,
    /// Overall height in metres.
    pub height: f32,
    pub kind: Kind,
    pub seed: u32,
    /// The great oak at the start: one long bough reaches out along `reach`.
    pub hero: Option<Vec2>,
}

/// A tapered tube along a bent spine.
struct Limb {
    spine: Vec<Vec3>,
    radii: Vec<f32>,
}

/// One rounded mass of foliage: an ellipsoid pushed out into lobes.
struct Clump {
    centre: Vec3,
    radii: Vec3,
    seed: u32,
}

/// One drooping skirt of a conifer.
struct Tier {
    /// Height of the tier's rim and its peak above the ground.
    rim: f32,
    peak: f32,
    radius: f32,
}

struct Shape {
    trunk: Limb,
    /// Height of the root flare's widest point relative to the trunk radius.
    flare: f32,
    limbs: Vec<Limb>,
    clumps: Vec<Clump>,
    tiers: Vec<Tier>,
    /// Middle of the canopy, which foliage normals lean away from.
    canopy: Vec3,
}

impl Tree {
    fn rand(&self, k: u32) -> f32 {
        unit(hash3(self.seed, k as i32, 7, 3))
    }

    /// Radius of the trunk at the ground.
    pub fn trunk_radius(&self) -> f32 {
        match (self.kind, self.hero) {
            (_, Some(_)) => 0.95,
            (Kind::Broadleaf, _) => 0.24 + 0.032 * self.height,
            (Kind::Conifer, _) => 0.16 + 0.011 * self.height,
        }
    }

    /// How far the model reaches out from the trunk in metres.
    pub fn spread(&self) -> f32 {
        match (self.kind, self.hero) {
            (_, Some(_)) => self.height * 1.05,
            (Kind::Broadleaf, _) => self.height * 0.95,
            (Kind::Conifer, _) => self.height * 0.3 + 0.6,
        }
    }

    fn shape(&self) -> Shape {
        match self.kind {
            Kind::Broadleaf => self.broadleaf(),
            Kind::Conifer => self.conifer(),
        }
    }

    fn lean(&self) -> Vec3 {
        let a = self.rand(1) * TAU;
        let amount = self.height * (0.02 + 0.06 * self.rand(2));
        match self.hero {
            Some(d) => Vec3::new(d.x, 0.0, d.y) * self.height * 0.08,
            None => Vec3::new(a.cos(), 0.0, a.sin()) * amount,
        }
    }

    fn broadleaf(&self) -> Shape {
        let h = self.height;
        let r0 = self.trunk_radius();
        let lean = self.lean();
        // The great oak forks higher, so its crown frames the sky rather than
        // hanging low over the view.
        let fork = h * if self.hero.is_some() {
            0.36
        } else {
            // A clear trunk below the crown, so the tree's structure shows.
            0.34 + 0.1 * self.rand(3)
        };
        // The trunk: a gentle curve up to the fork, with a slight kink.
        let side = Vec3::new(-lean.z, 0.0, lean.x).normalize_or_zero();
        // Old broadleaf trunks bend one way and back again on their way up.
        let wig = (self.rand(4) - 0.5) * if self.hero.is_some() { 0.3 } else { 1.1 } * r0;
        let bend = (self.rand(7) - 0.5) * if self.hero.is_some() { 0.0 } else { 0.9 } * r0;
        let fwd = lean.normalize_or_zero();
        let n = 6;
        let mut spine = Vec::new();
        let mut radii = Vec::new();
        for i in 0..=n {
            let t = i as f32 / n as f32;
            spine.push(
                self.base
                    + Vec3::Y * (fork * t)
                    + lean * t.powf(1.5)
                    + side * (wig * (t * PI).sin())
                    + fwd * (bend * (t * TAU).sin()),
            );
            radii.push(r0 * (1.0 - 0.38 * t));
        }
        let top = *spine.last().unwrap();
        let rt = *radii.last().unwrap();
        let trunk_dir = (top - spine[n - 1]).normalize();

        let mut limbs = Vec::new();
        let mut clumps = Vec::new();
        let count = if self.hero.is_some() {
            8
        } else {
            3 + (self.rand(5) * 3.99) as usize
        };
        let phase = self.rand(6) * TAU;
        let lean_dir = Vec2::new(lean.x, lean.z).normalize_or_zero();
        // The great oak's crown is many smaller clumps with sky between them.
        let cr = h * if self.hero.is_some() { 0.17 } else { 0.25 };
        let mut ends = Vec::new();
        for k in 0..count {
            let hero_bough = self.hero.is_some() && k == 0;
            let mut a = phase + (k as f32 + 0.35 * (self.rand(10 + k as u32) - 0.5)) * TAU / count as f32;
            if let (true, Some(d)) = (hero_bough, self.hero) {
                a = d.y.atan2(d.x);
            }
            let out = Vec2::new(a.cos(), a.sin());
            // Boughs reach a little further on the side the tree leans to.
            let long = 1.0 + 0.25 * out.dot(lean_dir);
            // Bough lengths vary a lot, so the crown's outline is ragged
            // rather than a dome.
            let mut length = h * (0.3 + 0.26 * self.rand(20 + k as u32)) * long;
            let mut rise = 0.15 + 0.45 * self.rand(30 + k as u32);
            if self.hero.is_some() {
                // The great oak spreads wide and low, its clumps far apart.
                length *= 1.25;
                rise *= 0.7;
            }
            if hero_bough {
                length = h * 0.6;
                rise = 0.3;
            }
            let at = 0.72 + 0.28 * self.rand(40 + k as u32);
            let start = spine[((at * n as f32) as usize).min(n)];
            let horiz = Vec3::new(out.x, 0.0, out.y);
            let end = start + horiz * length * rise.cos() + Vec3::Y * (length * rise.sin() + length * 0.22);
            // Bends: out first, then up towards the light.
            let ctrl = start + (horiz * rise.cos() * 0.8 + Vec3::Y * rise.sin() * 0.6) * length * 0.55;
            let r_start = rt * (0.62 + 0.15 * self.rand(50 + k as u32));
            let (spine, radii) = bezier(start, ctrl, end, r_start, 0.07, 5);
            if self.hero.is_none() && length > h * 0.44 {
                // Long boughs carry a smaller clump partway out, so the
                // crown spreads in layers like an old oak.
                let p = spine[3];
                clumps.push(Clump {
                    centre: p + Vec3::Y * cr * 0.25,
                    radii: Vec3::new(1.05, 0.75, 1.05) * cr * 0.6,
                    seed: self.seed.wrapping_add(70 + k as u32),
                });
            }
            if self.hero.is_some() {
                // Clumps along every bough, each a separate rounded mass, so
                // the crown breaks into many lobes with sky between them.
                let along: &[f32] = if hero_bough { &[0.45, 0.72] } else { &[0.6] };
                for (i, &f) in along.iter().enumerate() {
                    let p = spine[(f * 5.0) as usize];
                    let size = if hero_bough { 0.85 } else { 0.7 };
                    clumps.push(Clump {
                        centre: p + Vec3::Y * cr * 0.35,
                        radii: Vec3::new(1.05, 0.72, 1.05) * cr * size,
                        seed: self.seed.wrapping_add(90 + k as u32 * 4 + i as u32),
                    });
                }
            }
            limbs.push(Limb { spine, radii });
            ends.push(end);
        }
        // A leader carries on up the middle into the crown's top clump.
        // The higher fork of ordinary trees leaves less room above it.
        let lead = if self.hero.is_some() { 0.3 } else { 0.2 };
        let lead_top = top + trunk_dir * h * 0.15 + Vec3::Y * h * lead;
        let ctrl = top + trunk_dir * h * 0.15;
        let (s, r) = bezier(top, ctrl, lead_top, rt * 0.7, 0.08, 4);
        limbs.push(Limb { spine: s, radii: r });

        // A clump on every bough end, one on top, and fillers between
        // neighbouring ends so the canopy reads as one full mass.
        for (k, &e) in ends.iter().enumerate() {
            let s = 0.65 + 0.5 * self.rand(60 + k as u32);
            clumps.push(Clump {
                centre: e - Vec3::Y * cr * 0.15,
                radii: Vec3::new(1.05, 0.92, 1.05) * cr * s,
                seed: self.seed.wrapping_add(k as u32),
            });
        }
        clumps.push(Clump {
            centre: lead_top,
            radii: Vec3::new(1.1, 0.95, 1.1) * cr * 1.1,
            seed: self.seed.wrapping_add(40),
        });
        let mid = ends.iter().copied().sum::<Vec3>() / ends.len() as f32;
        for k in 0..ends.len() {
            let (a, b) = (ends[k], ends[(k + 1) % ends.len()]);
            let gap = if self.hero.is_some() { 3.2 } else { 2.1 };
            if a.distance(b) > cr * gap {
                let m = (a + b) * 0.5;
                clumps.push(Clump {
                    centre: m + (m - mid) * 0.1 - Vec3::Y * cr * 0.05,
                    radii: Vec3::new(1.0, 0.85, 1.0) * cr * 0.7,
                    seed: self.seed.wrapping_add(20 + k as u32),
                });
            }
        }
        let canopy = clumps.iter().map(|c| c.centre).sum::<Vec3>() / clumps.len() as f32;
        // Smaller puffs of leaves bulge out of each clump's upper side, so the
        // crown's outline breaks into many rounded masses like a painted oak.
        let big = clumps.len();
        for i in 0..big {
            let (c, radii) = (clumps[i].centre, clumps[i].radii);
            let away = (c - canopy).normalize_or_zero();
            // The great oak keeps only the upward puff, so sky shows between its clumps.
            let puffs = if self.hero.is_some() { 1 } else { 2 };
            for k in (2 - puffs)..2u32 {
                let a = self.rand(200 + i as u32 * 4 + k) * TAU;
                let side = Vec3::new(a.cos(), 0.0, a.sin());
                // One puff bulges out sideways and a little down, the other up.
                let d = (away * 0.9 + side * 0.8 + Vec3::Y * (-0.25 + 0.9 * k as f32)).normalize();
                clumps.push(Clump {
                    centre: c + d * radii * 0.9,
                    radii: radii * (0.46 + 0.1 * self.rand(300 + i as u32 * 4 + k)),
                    seed: self.seed.wrapping_add(500 + i as u32 * 4 + k),
                });
            }
        }
        Shape {
            trunk: Limb { spine, radii },
            flare: if self.hero.is_some() { 1.4 } else { 1.0 },
            limbs,
            clumps,
            tiers: Vec::new(),
            canopy,
        }
    }

    fn conifer(&self) -> Shape {
        let h = self.height;
        let r0 = self.trunk_radius();
        let lean = self.lean() * 0.3;
        let n = 5;
        let mut spine = Vec::new();
        let mut radii = Vec::new();
        for i in 0..=n {
            let t = i as f32 / n as f32;
            spine.push(self.base + Vec3::Y * (h * 0.97 * t) + lean * t);
            radii.push(r0 * (1.0 - 0.9 * t) + 0.03);
        }
        let count = 6 + (h / 3.5) as usize;
        let mut tiers = Vec::new();
        for k in 0..count {
            let t = k as f32 / count as f32;
            let rim = h * (0.2 + 0.72 * t);
            let radius = (h * 0.27 * (1.0 - t).powf(0.85) + 0.35) * (0.9 + 0.2 * self.rand(70 + k as u32));
            tiers.push(Tier {
                rim,
                peak: rim + h * 0.13 + radius * 0.35,
                radius,
            });
        }
        Shape {
            trunk: Limb { spine, radii },
            flare: 0.6,
            limbs: Vec::new(),
            clumps: Vec::new(),
            tiers,
            canopy: self.base + Vec3::Y * h * 0.55 + lean * 0.5,
        }
    }

    /// The whole near model.
    pub fn mesh(&self, out: &mut MeshData) {
        let s = self.shape();
        let sides = if self.hero.is_some() { 16 } else { 11 };
        trunk(out, &s.trunk, s.flare, sides, self.seed);
        for l in &s.limbs {
            tube(out, l, 7, 3);
        }
        let big = s.clumps.iter().map(|c| c.radii.x).fold(0.0, f32::max) * 0.75;
        for (i, c) in s.clumps.iter().enumerate() {
            let fine = if self.hero.is_some() { 1 } else { 0 };
            let mesh = if c.radii.x > big { ico(2 + fine) } else { ico(1 + fine) };
            clump(out, &s.clumps, i, s.canopy, mesh, c.seed);
        }
        for t in &s.tiers {
            tier(out, self, t, 14, &s);
        }
    }

    /// A lighter model for the far field: fewer sides, coarser clumps, no boughs.
    pub fn far(&self, out: &mut MeshData, level: u8) {
        let s = self.shape();
        let top = if self.kind == Kind::Broadleaf {
            // Up into the canopy so no gap shows under the clumps.
            s.canopy.y
        } else {
            self.base.y + self.height * 0.9
        };
        let r0 = self.trunk_radius();
        let a = self.base - Vec3::Y * 0.5;
        let b = Vec3::new(s.trunk.spine.last().unwrap().x, top, s.trunk.spine.last().unwrap().z);
        tube(
            out,
            &Limb {
                spine: vec![a, b],
                radii: vec![r0 * 1.1, r0 * 0.5],
            },
            5,
            2,
        );
        if level == 0 {
            // Boughs show through the gaps between clumps, so the crown
            // does not read as a ball on a stick.
            for l in &s.limbs {
                let spine: Vec<Vec3> = l.spine.iter().step_by(2).copied().collect();
                let radii: Vec<f32> = l.radii.iter().step_by(2).copied().collect();
                if spine.len() > 1 {
                    tube(out, &Limb { spine, radii }, 4, 2);
                }
            }
        }
        for (i, c) in s.clumps.iter().enumerate() {
            let puff = c.radii.x < self.height * 0.2;
            // Further out keep every other puff, so crowns still break into
            // lobes rather than reading as balls.
            if level > 1 && puff || level == 1 && puff && i % 2 == 1 {
                continue;
            }
            let mesh = ico(if level == 0 && !puff { 1 } else { 0 });
            clump(out, &s.clumps, i, s.canopy, mesh, c.seed);
        }
        let step = if level == 0 { 1 } else { 2 };
        for t in s.tiers.iter().step_by(step) {
            tier(out, self, t, if level == 0 { 8 } else { 6 }, &s);
        }
    }

    /// Voxels (in voxel coordinates) that the trunk fills, for collision.
    pub fn trunk_cells(&self, mut put: impl FnMut(glam::IVec3)) {
        let s = self.shape();
        let v = crate::block::VOXEL_SIZE;
        let spine = &s.trunk.spine;
        let top = spine.last().unwrap().y;
        let mut y = self.base.y;
        while y < top {
            let t = ((y - self.base.y) / (top - self.base.y)).clamp(0.0, 1.0);
            let f = t * (spine.len() - 1) as f32;
            let i = (f as usize).min(spine.len() - 2);
            let c = spine[i].lerp(spine[i + 1], f - i as f32);
            let r = s.trunk.radii[i] * 0.85;
            let lo = ((Vec2::new(c.x, c.z) - r) / v).floor();
            let hi = ((Vec2::new(c.x, c.z) + r) / v).floor();
            for z in lo.y as i32..=hi.y as i32 {
                for x in lo.x as i32..=hi.x as i32 {
                    let d = Vec2::new((x as f32 + 0.5) * v - c.x, (z as f32 + 0.5) * v - c.z);
                    if d.length() < r.max(0.3) {
                        put(glam::IVec3::new(x, (y / v).floor() as i32, z));
                    }
                }
            }
            y += v;
        }
    }
}

/// Appends every tree whose foot stands in chunk `cpos` (whole trees, even
/// where they reach into neighbouring chunks).
pub fn append(terrain: &crate::terrain::Terrain, cpos: glam::IVec3, out: &mut MeshData) {
    let side = crate::chunk::CHUNK as f32 * crate::block::VOXEL_SIZE;
    let lo = Vec2::new(cpos.x as f32, cpos.z as f32) * side;
    for t in terrain.trees_in(lo, lo + side) {
        // The foot belongs to the chunk holding the ground just under it.
        if ((t.base.y - 0.25) / side).floor() as i32 == cpos.y {
            t.mesh(out);
        }
    }
}

/// A quadratic curve from `a` over `c` to `b`, sampled at `n + 1` points,
/// with radii tapering from `r0` to `r1`.
fn bezier(a: Vec3, c: Vec3, b: Vec3, r0: f32, r1: f32, n: usize) -> (Vec<Vec3>, Vec<f32>) {
    let mut s = Vec::new();
    let mut r = Vec::new();
    for i in 0..=n {
        let t = i as f32 / n as f32;
        s.push(a * (1.0 - t) * (1.0 - t) + c * 2.0 * t * (1.0 - t) + b * t * t);
        r.push(r0 + (r1 - r0) * t.powf(0.8));
    }
    (s, r)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Two unit vectors across a direction.
fn across(d: Vec3) -> (Vec3, Vec3) {
    let helper = if d.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
    let u = d.cross(helper).normalize();
    (u, d.cross(u))
}

/// A tapered tube along a limb's spine, closed with a point at the tip.
fn tube(out: &mut MeshData, l: &Limb, sides: u32, ao: u32) {
    let n = l.spine.len();
    let start = out.vertices.len() as u32;
    let mut u_prev = None;
    for i in 0..n {
        let d = if i + 1 < n {
            l.spine[i + 1] - l.spine[i]
        } else {
            l.spine[i] - l.spine[i - 1]
        }
        .normalize_or_zero();
        // Carry the frame along so the tube does not twist.
        let (u, v) = match u_prev {
            Some(up) => {
                let u: Vec3 = (up - d * d.dot(up)).normalize_or_zero();
                if u.length_squared() < 0.5 {
                    across(d)
                } else {
                    (u, d.cross(u))
                }
            }
            None => across(d),
        };
        u_prev = Some(u);
        for k in 0..sides {
            let a = k as f32 / sides as f32 * TAU;
            let radial = u * a.cos() + v * a.sin();
            out.vertices.push(Vertex {
                pos: (l.spine[i] + radial * l.radii[i]).to_array(),
                data: smooth_data(BARK, ao, radial),
            });
        }
    }
    let tip_dir = (l.spine[n - 1] - l.spine[n - 2]).normalize_or_zero();
    let tip = out.vertices.len() as u32;
    out.vertices.push(Vertex {
        pos: (l.spine[n - 1] + tip_dir * l.radii[n - 1]).to_array(),
        data: smooth_data(BARK, ao, tip_dir),
    });
    rings(out, start, n as u32, sides);
    for k in 0..sides {
        let a = start + (n as u32 - 1) * sides + k;
        let b = start + (n as u32 - 1) * sides + (k + 1) % sides;
        out.indices.extend_from_slice(&[b, a, tip]);
    }
}

/// Indices joining `count` rings of `sides` vertices each, wound outward
/// for rings laid out anticlockwise around a spine running from first to last.
fn rings(out: &mut MeshData, start: u32, count: u32, sides: u32) {
    for r in 0..count - 1 {
        for k in 0..sides {
            let k1 = (k + 1) % sides;
            let a = start + r * sides + k;
            let b = start + r * sides + k1;
            let c = start + (r + 1) * sides + k1;
            let d = start + (r + 1) * sides + k;
            out.indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
}

/// The trunk: a tube whose foot swells into buttress roots that spread over
/// the ground and dip into it.
fn trunk(out: &mut MeshData, l: &Limb, flare: f32, sides: u32, seed: u32) {
    let base = l.spine[0];
    let r0 = l.radii[0];
    let lobes = 4.0 + (unit(seed.rotate_left(3)) * 2.0).floor();
    let phase = unit(seed.rotate_left(9)) * TAU;
    // Extra rings near the foot for the flare, then the trunk's own rings.
    let mut ring_at: Vec<(Vec3, f32, f32)> = Vec::new();
    for &hgt in &[-0.6f32, -0.05, 0.12, 0.3, 0.55, 0.9] {
        let h = hgt * r0 * 1.6;
        ring_at.push((base + Vec3::Y * h, r0, (-h.max(0.0) / (r0 * 0.9)).exp()));
    }
    let foot_top = ring_at.last().unwrap().0.y;
    for (p, r) in l.spine.iter().zip(&l.radii) {
        if p.y > foot_top + 0.1 {
            ring_at.push((*p, *r, (-(p.y - base.y) / (r0 * 0.9)).exp()));
        }
    }
    let n = ring_at.len();
    let start = out.vertices.len() as u32;
    let bark_ao = |y: f32| {
        if y < base.y + 0.3 {
            1
        } else if y < base.y + 1.2 {
            2
        } else {
            3
        }
    };
    for (i, &(p, r, f)) in ring_at.iter().enumerate() {
        let next = ring_at[(i + 1).min(n - 1)].0;
        let prev = ring_at[i.saturating_sub(1)].0;
        let d = (next - prev).normalize_or_zero();
        let d = if d == Vec3::ZERO { Vec3::Y } else { d };
        let (u, v) = {
            // Keep a fixed frame around the vertical so rings line up.
            let u = (Vec3::X - d * d.x).normalize();
            (u, d.cross(u))
        };
        // Above the foot the buttresses carry on up as ridges that twist
        // around the trunk, like the fluted bole of an old oak.
        let up = p.y - base.y;
        let ridge = 0.13 * (1.0 - f) * (1.0 - smoothstep(4.0, 9.0, up));
        for k in 0..sides {
            let a = k as f32 / sides as f32 * TAU;
            let lobe = ((a * lobes + phase).cos().max(0.0)).powi(2);
            let twist = a * lobes + phase + up * 0.45;
            let swell = (1.0 + flare * f * (0.25 + 0.8 * lobe)) * (1.0 + ridge * twist.cos());
            let radial = u * a.cos() + v * a.sin();
            let tangent = v * a.cos() - u * a.sin();
            // Roots dip into the ground as they spread.
            let dip = if p.y <= base.y + 0.01 {
                0.0
            } else {
                -f * f * lobe * r0 * 0.25 * flare
            };
            let pos = p + radial * r * swell + Vec3::Y * dip;
            // Normals tilt up where the foot spreads out.
            let normal =
                (radial + tangent * (ridge * lobes * twist.sin()) + Vec3::Y * (f * flare * (0.5 + lobe) * 0.8))
                    .normalize();
            out.vertices.push(Vertex {
                pos: pos.to_array(),
                data: smooth_data(BARK, bark_ao(p.y), normal),
            });
        }
    }
    rings(out, start, n as u32, sides);
    // Close the top inside the canopy.
    let tip = out.vertices.len() as u32;
    let last = ring_at[n - 1].0;
    out.vertices.push(Vertex {
        pos: (last + Vec3::Y * ring_at[n - 1].1).to_array(),
        data: smooth_data(BARK, 3, Vec3::Y),
    });
    for k in 0..sides {
        let a = start + (n as u32 - 1) * sides + k;
        let b = start + (n as u32 - 1) * sides + (k + 1) % sides;
        out.indices.extend_from_slice(&[b, a, tip]);
    }
}

/// A unit icosphere, subdivided `level` times (0 to 3).
struct Ico {
    verts: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
}

fn ico(level: usize) -> &'static Ico {
    static CACHE: [OnceLock<Ico>; 4] = [const { OnceLock::new() }; 4];
    CACHE[level.min(3)].get_or_init(|| {
        let t = (1.0 + 5f32.sqrt()) / 2.0;
        let mut verts: Vec<Vec3> = [
            (-1.0, t, 0.0),
            (1.0, t, 0.0),
            (-1.0, -t, 0.0),
            (1.0, -t, 0.0),
            (0.0, -1.0, t),
            (0.0, 1.0, t),
            (0.0, -1.0, -t),
            (0.0, 1.0, -t),
            (t, 0.0, -1.0),
            (t, 0.0, 1.0),
            (-t, 0.0, -1.0),
            (-t, 0.0, 1.0),
        ]
        .iter()
        .map(|&(x, y, z)| Vec3::new(x, y, z).normalize())
        .collect();
        let mut tris: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..level.min(3) {
            let mut mids = std::collections::HashMap::new();
            let mut mid = |a: u32, b: u32, verts: &mut Vec<Vec3>| {
                *mids.entry((a.min(b), a.max(b))).or_insert_with(|| {
                    verts.push(((verts[a as usize] + verts[b as usize]) * 0.5).normalize());
                    verts.len() as u32 - 1
                })
            };
            let mut next = Vec::new();
            for [a, b, c] in tris {
                let ab = mid(a, b, &mut verts);
                let bc = mid(b, c, &mut verts);
                let ca = mid(c, a, &mut verts);
                next.extend_from_slice(&[[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
            }
            tris = next;
        }
        Ico { verts, tris }
    })
}

/// Whether `p` lies well inside any clump other than `skip`.
fn buried(clumps: &[Clump], skip: usize, p: Vec3) -> bool {
    clumps
        .iter()
        .enumerate()
        .any(|(i, c)| i != skip && ((p - c.centre) / c.radii).length_squared() < 0.72)
}

/// One foliage clump: the sphere is pushed out into a handful of rounded
/// lobes with a leafy ripple over them and a flatter underside, like a
/// painted cumulus of leaves. Normals lean away from the middle of the whole
/// canopy so light rolls softly over it, and occlusion darkens the
/// undersides. Triangles buried inside neighbouring clumps are dropped.
fn clump(out: &mut MeshData, all: &[Clump], index: usize, canopy: Vec3, mesh: &Ico, seed: u32) {
    let c = &all[index];
    let r = |k: u32| unit(hash3(seed, k as i32, 11, 5));
    let lobes: Vec<(Vec3, f32)> = (0..11)
        .map(|k| {
            let a = r(k) * TAU;
            // Mostly on top, but some bulge down so a crown seen from below is not flat.
            let y = -0.75 + 1.7 * r(k + 10);
            let s = (1.0 - y * y).max(0.0).sqrt();
            (
                Vec3::new(a.cos() * s, y.min(1.0), a.sin() * s).normalize(),
                0.14 + 0.12 * r(k + 20),
            )
        })
        .collect();
    let positions: Vec<Vec3> = mesh
        .verts
        .iter()
        .map(|&d| {
            let mut s = 0.9;
            for &(l, amp) in &lobes {
                let t = ((d.dot(l) - 0.6) / 0.4).max(0.0);
                s += amp * t * t * (3.0 - 2.0 * t);
            }
            let w = c.centre + d * c.radii;
            s += 0.14 * (value3(seed, w.x * 0.9, w.y * 0.9, w.z * 0.9) - 0.5);
            s += 0.03 * (value3(seed ^ 0x55, w.x * 2.3, w.y * 2.3, w.z * 2.3) - 0.5);
            let mut q = d * c.radii * s;
            if q.y < 0.0 {
                q.y *= 0.95;
            }
            c.centre + q
        })
        .collect();
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for t in &mesh.tris {
        let [a, b, cc] = t.map(|i| positions[i as usize]);
        let n = (b - a).cross(cc - a);
        for &i in t {
            normals[i as usize] += n;
        }
    }
    let start = out.vertices.len() as u32;
    let size = c.radii.max_element();
    for (i, &p) in positions.iter().enumerate() {
        let own = normals[i].normalize_or_zero();
        let global = (p - canopy).normalize_or_zero();
        // A lift towards the sky lets undersides catch skylight instead of
        // going black, as in painted foliage.
        // Each clump keeps most of its own rounding, so the crown reads as
        // separate masses rather than one smooth dome.
        let n = (own + global * 0.15 + Vec3::Y * 0.3).normalize_or_zero();
        // Occlusion: each clump's own underside and the parts facing into
        // the canopy are darker, so shade gathers between the clumps.
        let depth = ((p - canopy).length() / (size * 1.6)).min(1.0);
        let local = ((p.y - c.centre.y) / c.radii.y).clamp(-1.0, 1.0);
        let light = 0.5 + 0.5 * local;
        let ao = ((light * 0.75 + depth * 0.45) * 3.0).round().clamp(1.0, 3.0) as u32;
        out.vertices.push(Vertex {
            pos: p.to_array(),
            data: smooth_data(FOLIAGE, ao, n),
        });
    }
    for t in &mesh.tris {
        let [a, b, cc] = t.map(|i| positions[i as usize]);
        if buried(all, index, a) && buried(all, index, b) && buried(all, index, cc) {
            continue;
        }
        // The icosphere's triangles wind anticlockwise seen from outside.
        out.indices.extend(t.iter().map(|i| start + i));
    }
}

/// One conifer tier: a drooping skirt from its peak down to a star-shaped rim
/// of branch tips, closed underneath back to the trunk.
fn tier(out: &mut MeshData, tree: &Tree, t: &Tier, sides: u32, s: &Shape) {
    let axis_at = |y: f32| {
        let spine = &s.trunk.spine;
        let top = spine.last().unwrap();
        let f = ((y - tree.base.y) / (top.y - tree.base.y)).clamp(0.0, 1.0);
        let x = spine[0].lerp(*top, f);
        Vec3::new(x.x, y, x.z)
    };
    let base_y = tree.base.y;
    let peak = axis_at(base_y + t.peak);
    let phase = unit(hash3(tree.seed, t.rim as i32, 3, 1)) * TAU;
    let start = out.vertices.len() as u32;
    // Rings: shoulder (0.45 of the way down), rim, and the underside's inner ring.
    for (ring, (down, reach)) in [(0.45f32, 0.62f32), (1.0, 1.0)].iter().enumerate() {
        let y = base_y + t.peak - (t.peak - t.rim) * down;
        let c = axis_at(y);
        for k in 0..sides {
            let a = phase + k as f32 / sides as f32 * TAU;
            let tip = if k % 2 == 0 { 1.0 } else { 0.78 };
            let jag = 0.9 + 0.2 * unit(hash3(tree.seed, k as i32, t.rim as i32, 9));
            let radial = Vec3::new(a.cos(), 0.0, a.sin());
            let rr = t.radius * reach * if ring == 1 { tip * jag } else { 1.0 };
            let droop = if ring == 1 { t.radius * 0.18 * tip } else { 0.0 };
            let normal = (radial + Vec3::Y * (0.9 - 0.3 * ring as f32)).normalize();
            let ao = if ring == 1 { 2 } else { 3 };
            out.vertices.push(Vertex {
                pos: (c + radial * rr - Vec3::Y * droop).to_array(),
                data: smooth_data(NEEDLES, ao, normal),
            });
        }
    }
    let apex = out.vertices.len() as u32;
    out.vertices.push(Vertex {
        pos: peak.to_array(),
        data: smooth_data(NEEDLES, 3, Vec3::Y),
    });
    // The underside shares the rim positions but faces down.
    let under = out.vertices.len() as u32;
    for k in 0..sides {
        let p = Vec3::from(out.vertices[(start + sides + k) as usize].pos);
        let radial = (p - axis_at(p.y)).normalize_or_zero();
        out.vertices.push(Vertex {
            pos: p.to_array(),
            data: smooth_data(NEEDLES, 0, (radial * 0.4 - Vec3::Y).normalize()),
        });
    }
    let hub = out.vertices.len() as u32;
    out.vertices.push(Vertex {
        pos: axis_at(base_y + t.rim + (t.peak - t.rim) * 0.2).to_array(),
        data: smooth_data(NEEDLES, 0, -Vec3::Y),
    });
    for k in 0..sides {
        let k1 = (k + 1) % sides;
        // Apex to shoulder, anticlockwise seen from above and outside.
        out.indices.extend_from_slice(&[apex, start + k1, start + k]);
        let (a, b) = (start + k, start + k1);
        let (c, d) = (start + sides + k1, start + sides + k);
        out.indices.extend_from_slice(&[a, b, c, a, c, d]);
        out.indices.extend_from_slice(&[hub, under + k, under + k1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(kind: Kind, hero: Option<Vec2>) -> Tree {
        Tree {
            base: Vec3::new(10.0, 30.0, 10.0),
            height: 12.0,
            kind,
            seed: 12345,
            hero,
        }
    }

    fn normal_of(d: u32) -> Vec3 {
        let q = |s: u32| ((d >> s) & 511) as f32 / 511.0 * 2.0 - 1.0;
        let (u, v) = (q(13), q(22));
        let mut m = Vec3::new(u, 1.0 - u.abs() - v.abs(), v);
        if m.y < 0.0 {
            let sign = |a: f32| if a >= 0.0 { 1.0 } else { -1.0 };
            (m.x, m.z) = ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v));
        }
        m.normalize()
    }

    #[test]
    fn triangles_wind_with_their_normals() {
        for t in [
            tree(Kind::Broadleaf, None),
            tree(Kind::Conifer, None),
            tree(Kind::Broadleaf, Some(Vec2::X)),
        ] {
            let mut out = MeshData::default();
            t.mesh(&mut out);
            let (mut good, mut total) = (0, 0);
            for tri in out.indices.chunks(3) {
                let v = [0, 1, 2].map(|k| out.vertices[tri[k] as usize]);
                let [a, b, c] = v.map(|v| Vec3::from(v.pos));
                let face = (b - a).cross(c - a);
                if face.length() > 1e-6 {
                    let n: Vec3 = v.iter().map(|v| normal_of(v.data)).sum();
                    total += 1;
                    good += (face.dot(n) > 0.0) as usize;
                }
            }
            // Lobes can fold a few tiny triangles; nearly all must agree.
            let bad: std::collections::BTreeMap<u32, usize> = out
                .indices
                .chunks(3)
                .filter(|tri| {
                    let v = [0, 1, 2].map(|k| out.vertices[tri[k] as usize]);
                    let [a, b, c] = v.map(|v| Vec3::from(v.pos));
                    let n: Vec3 = v.iter().map(|v| normal_of(v.data)).sum();
                    (b - a).cross(c - a).dot(n) <= 0.0
                })
                .fold(Default::default(), |mut m, tri| {
                    *m.entry((out.vertices[tri[0] as usize].data >> 3) & 255).or_default() += 1;
                    m
                });
            assert!(
                good as f32 > total as f32 * 0.97,
                "{:?}: {good} of {total} {bad:?}",
                t.kind
            );
        }
    }

    #[test]
    fn canopy_wraps_the_bough_ends() {
        let t = tree(Kind::Broadleaf, None);
        let s = t.shape();
        for l in &s.limbs {
            let end = *l.spine.last().unwrap();
            assert!(
                s.clumps.iter().any(|c| ((end - c.centre) / c.radii).length() < 0.8),
                "bough end {end} sticks out"
            );
        }
        // Everything stays within the advertised spread.
        let mut out = MeshData::default();
        t.mesh(&mut out);
        for v in &out.vertices {
            let d = Vec2::new(v.pos[0] - t.base.x, v.pos[2] - t.base.z).length();
            assert!(d < t.spread() + 0.5, "{d}");
            assert!(v.pos[1] < t.base.y + t.height * 1.25);
        }
    }

    #[test]
    fn trees_are_modest_in_triangles() {
        for kind in [Kind::Broadleaf, Kind::Conifer] {
            let mut out = MeshData::default();
            tree(kind, None).mesh(&mut out);
            let tris = out.indices.len() / 3;
            assert!(tris < 4000, "{kind:?}: {tris}");
            let mut far = MeshData::default();
            tree(kind, None).far(&mut far, 1);
            assert!(far.indices.len() / 3 < tris / 3, "{kind:?} far");
        }
    }

    #[test]
    fn trunk_cells_stand_on_the_base() {
        let t = tree(Kind::Broadleaf, None);
        let mut cells = Vec::new();
        t.trunk_cells(|c| cells.push(c));
        assert!(!cells.is_empty());
        let lowest = cells.iter().map(|c| c.y).min().unwrap();
        assert_eq!(lowest, (t.base.y / crate::block::VOXEL_SIZE).floor() as i32);
    }
}
