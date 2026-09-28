//! The castle on the cliff above the falls: a fairy-tale cluster of pale
//! stone. A crenellated curtain wall links round towers of many heights, each
//! under a tall, steep slate spire; a keep with steep roofs and dormers stands
//! at the back, a great tower rises from it higher than everything else, and a
//! small chapel leans on the west wall. Blue banners hang from the walls and
//! towers, a pennant flies from the highest spire, and pointed windows glow
//! from dusk on.
//!
//! Like the cottages it is an authored model: every part is a shaped piece
//! (revolved shafts and spires, boxes, panels) in world metres. The voxels
//! underneath only carry collision (`BUILT`, never drawn), clear the courtyard
//! and hold the lanterns at the gate. The player can climb the approach steps,
//! walk through the gate into the courtyard and take the stair up onto the
//! east wall walk. Far away the same builder runs with fewer segments and no
//! small parts, so the far stand-in has the near model's silhouette.
//!
//! The castle is laid out in its own frame: local X runs across, local Z
//! towards the front, which faces the player's arrival point; the gate is at
//! the back. Heights are metres above the courtyard floor.

use crate::block::*;
use crate::mesh::{smooth_data, MeshData, Vertex};
use crate::model::{beam, panel, soft_box};
use crate::terrain::Terrain;
use crate::vista;
use glam::{Vec2, Vec3};
use std::f32::consts::{PI, TAU};

/// Height of the wall walk above the courtyard.
const WALK: f32 = 9.0;
/// Half the curtain wall's thickness.
const WALL_HALF: f32 = 1.0;
/// Thickness of the parapet along the wall's outer edge.
const PARAPET: f32 = 0.5;
/// Height of the parapet's merlons above the wall walk.
const MERLON_H: f32 = 1.35;
/// Merlons repeat this often along the wall.
const MERLON_PITCH: f32 = 1.5;
/// Corners of the curtain wall, going round; a tower stands on each.
const RING: [Vec2; 7] = [
    Vec2::new(-16.5, 8.5),
    Vec2::new(0.0, 12.0),
    Vec2::new(16.5, 8.5),
    Vec2::new(16.5, -11.5),
    Vec2::new(12.0, -15.0),
    Vec2::new(3.5, -15.0),
    Vec2::new(-16.5, -11.5),
];
/// The ring edge the gatehouse replaces.
const GATE_EDGE: usize = 4;
/// Centre of the gate passage across the castle, and its half width.
const GATE_X: f32 = 7.75;
const GATE_HALF: f32 = 1.5;
/// Where the gate's pointed arch springs, and its apex.
const GATE_SPRING: f32 = 3.0;
const GATE_APEX: f32 = 5.2;
/// The gatehouse block (local corners) and its height.
const GATEHOUSE_LO: Vec2 = Vec2::new(3.5, -17.0);
const GATEHOUSE_HI: Vec2 = Vec2::new(12.0, -13.0);
const GATEHOUSE_TOP: f32 = 12.0;
/// The stair up to the east wall walk: across range, where it starts, and
/// how far each half-metre step runs.
const STAIR_X: (f32, f32) = (14.0, 15.5);
const STAIR_Z0: f32 = -8.5;
const STEP: f32 = 0.5;
/// The approach outside the gate: half width, and run of each step.
const APPROACH_HALF: f32 = 2.5;
const APPROACH_RUN: f32 = 1.0;
/// The land the castle claims, as local corners (approach steps extra).
const CLAIM_LO: Vec2 = Vec2::new(-19.5, -17.5);
const CLAIM_HI: Vec2 = Vec2::new(19.5, 15.5);
/// Terrain inside the walls is cleared up to this height above the floor.
const COURT_AIR: f32 = 12.0;
/// The courtyard well: local centre and outer radius.
const WELL: Vec2 = Vec2::new(9.5, 3.0);
const WELL_R: f32 = 1.0;

/// A round tower as laid out: local centre, radius, where its shaft starts
/// (0 for the ground, or a height for a turret corbelled out of a wall), where
/// its shaft ends and how tall its spire is, whether it stands on the curtain
/// wall, how many dormers its spire has, and whether it flies the pennant.
#[derive(Clone, Copy, Debug)]
struct TowerSpec {
    at: Vec2,
    r: f32,
    foot: f32,
    top: f32,
    spire: f32,
    perimeter: bool,
    dormers: u8,
    pennant: bool,
}

const fn spec(x: f32, z: f32, r: f32, top: f32, spire: f32, perimeter: bool) -> TowerSpec {
    TowerSpec {
        at: Vec2::new(x, z),
        r,
        foot: 0.0,
        top,
        spire,
        perimeter,
        dormers: 0,
        pennant: false,
    }
}

const TOWERS: [TowerSpec; 13] = [
    // On the ring, in RING order.
    spec(-16.5, 8.5, 2.7, 15.0, 14.5, true),
    TowerSpec {
        dormers: 3,
        ..spec(0.0, 12.0, 3.1, 19.0, 17.0, true)
    },
    spec(16.5, 8.5, 2.7, 15.0, 14.5, true),
    spec(16.5, -11.5, 2.5, 13.0, 12.5, true),
    spec(12.0, -15.0, 2.4, 14.0, 10.5, true),
    spec(3.5, -15.0, 2.4, 14.0, 10.5, true),
    spec(-16.5, -11.5, 2.5, 13.0, 12.5, true),
    // The great tower, its stair turret, and the tower at the keep's west end.
    TowerSpec {
        dormers: 4,
        pennant: true,
        ..spec(-4.0, -1.5, 4.0, 34.0, 23.0, false)
    },
    spec(0.2, 1.2, 1.4, 37.0, 7.0, false),
    TowerSpec {
        dormers: 2,
        ..spec(-12.0, -5.8, 2.8, 24.0, 14.0, false)
    },
    // Turrets corbelled out of the keep's corners.
    TowerSpec {
        foot: 10.0,
        ..spec(2.7, -9.7, 1.0, 16.5, 4.5, false)
    },
    TowerSpec {
        foot: 10.0,
        ..spec(-11.0, -9.9, 1.0, 16.5, 4.5, false)
    },
    TowerSpec {
        foot: 11.0,
        ..spec(2.7, -2.2, 1.0, 17.0, 4.5, false)
    },
];

/// A hall under a steep gabled roof: local corners, eave height, whether the
/// ridge runs along local X, and the roof's rise per metre.
#[derive(Clone, Copy, Debug)]
struct Hall {
    lo: Vec2,
    hi: Vec2,
    eave: f32,
    along_x: bool,
    pitch: f32,
}

const KEEP: Hall = Hall {
    lo: Vec2::new(-11.0, -9.5),
    hi: Vec2::new(2.5, -2.0),
    eave: 16.0,
    along_x: true,
    pitch: 1.9,
};
const CHAPEL: Hall = Hall {
    lo: Vec2::new(-15.5, 0.0),
    hi: Vec2::new(-10.0, 5.5),
    eave: 9.0,
    along_x: false,
    pitch: 1.9,
};
const HALLS: [Hall; 2] = [KEEP, CHAPEL];

impl Hall {
    fn mid(&self) -> Vec2 {
        (self.lo + self.hi) * 0.5
    }

    /// Half the width under the roof, and half the length along the ridge.
    fn halves(&self) -> (f32, f32) {
        let e = (self.hi - self.lo) * 0.5;
        if self.along_x {
            (e.y, e.x)
        } else {
            (e.x, e.y)
        }
    }

    /// Local unit vectors along the ridge and across it.
    fn axes(&self) -> (Vec2, Vec2) {
        if self.along_x {
            (Vec2::X, Vec2::Y)
        } else {
            (Vec2::Y, Vec2::X)
        }
    }

    fn ridge(&self) -> f32 {
        self.eave + self.halves().0 * self.pitch
    }

    fn contains(&self, q: Vec2, margin: f32) -> bool {
        q.x > self.lo.x - margin && q.x < self.hi.x + margin && q.y > self.lo.y - margin && q.y < self.hi.y + margin
    }

    /// Height of the roof's outer surface over a local point.
    fn roof_at(&self, q: Vec2) -> f32 {
        let (half_w, _) = self.halves();
        let (_, across) = self.axes();
        let b = (q - self.mid()).dot(across).abs();
        self.eave + (half_w - b) * self.pitch
    }
}

/// A tower as planned: its layout, and the ground under it.
#[derive(Clone, Debug)]
struct Tower {
    s: TowerSpec,
    /// Where the footing starts, under the lowest ground (absolute metres).
    base: f32,
    /// Highest ground under the tower (absolute metres).
    ground_hi: f32,
}

impl Tower {
    /// Radius of the spire's eaves.
    fn eave_r(&self) -> f32 {
        self.s.r + if self.s.foot > 0.0 { 0.45 } else { 0.75 }
    }

    /// Radius of the spire at `h` metres above its eaves.
    fn spire_r(&self, h: f32) -> f32 {
        let u = (h / self.s.spire).clamp(0.0, 1.0);
        self.eave_r() * (1.0 - u).powf(1.3)
    }
}

#[derive(Clone, Debug)]
pub struct Castle {
    /// Centre in world metres (X, Z).
    pub c: Vec2,
    /// Unit vector (world X, Z) the front faces.
    pub fwd: Vec2,
    /// Height of the courtyard floor.
    pub floor: f32,
    /// Lowest ground inside the walls.
    pub ground_lo: f32,
    towers: Vec<Tower>,
    /// Bottom of each ring edge's footing (absolute metres).
    wall_base: Vec<f32>,
    /// Bottom of the gatehouse's footing.
    gate_base: f32,
    /// Top of each approach step outside the gate, going out, and the ground under it.
    approach: Vec<(f32, f32)>,
    /// Voxel centres of the lanterns.
    lanterns: Vec<Vec3>,
    lo: Vec3,
    hi: Vec3,
}

/// Whether `p` lies in the voxel whose centre is nearest `target`.
fn in_voxel(p: Vec3, target: Vec3) -> bool {
    (p / VOXEL_SIZE).floor() == (target / VOXEL_SIZE).floor()
}

/// The inward unit normal of ring edge `i`.
fn edge_normal(i: usize) -> Vec2 {
    let (a, b) = (RING[i], RING[(i + 1) % RING.len()]);
    let d = (b - a).normalize();
    let n = Vec2::new(-d.y, d.x);
    let centroid = RING.iter().copied().sum::<Vec2>() / RING.len() as f32;
    if (centroid - a).dot(n) > 0.0 {
        n
    } else {
        -n
    }
}

/// Distance inside the ring's centre line (negative outside).
fn inward(q: Vec2) -> f32 {
    (0..RING.len())
        .map(|i| (q - RING[i]).dot(edge_normal(i)))
        .fold(f32::MAX, f32::min)
}

/// Top of the wall stair's step under a local point, above the floor.
fn stair_top(q: Vec2) -> Option<f32> {
    if q.x < STAIR_X.0 || q.x >= STAIR_X.1 || q.y < STAIR_Z0 {
        return None;
    }
    let k = ((q.y - STAIR_Z0) / STEP).floor();
    (k < WALK / STEP).then_some((k + 1.0) * STEP)
}

/// Height of the pointed gate arch's soffit at `dx` from the passage centre.
fn arch_y(dx: f32) -> f32 {
    let w = GATE_HALF;
    // Each half is an arc centred on the far springing's side.
    let r = ((GATE_APEX - GATE_SPRING).powi(2) + w * w) / (2.0 * w);
    let o = r - w;
    let x = dx.abs() + o;
    GATE_SPRING + (r * r - x * x).max(0.0).sqrt()
}

impl Castle {
    /// Sites the castle and fits it to the ground there. `spawn` is where the
    /// player arrives; the castle's front faces it.
    pub fn plan(t: &Terrain, spawn: Vec3) -> Option<Self> {
        let (c, fwd) = site(spawn);
        let mut castle = Castle {
            c,
            fwd,
            floor: 0.0,
            ground_lo: 0.0,
            towers: Vec::new(),
            wall_base: Vec::new(),
            gate_base: 0.0,
            approach: Vec::new(),
            lanterns: Vec::new(),
            lo: Vec3::ZERO,
            hi: Vec3::ZERO,
        };
        let frame = castle.clone();
        let ground = |q: Vec2| {
            let w = frame.world(q);
            t.height_at(w.x, w.y).0
        };
        // The floor sits a little above the middle of the ground inside the walls.
        let mut hs = Vec::new();
        let mut z = RING[5].y;
        while z <= RING[1].y {
            let mut x = RING[0].x;
            while x <= RING[2].x {
                let q = Vec2::new(x, z);
                if inward(q) > WALL_HALF {
                    hs.push(ground(q));
                }
                x += 2.0;
            }
            z += 2.0;
        }
        hs.sort_by(f32::total_cmp);
        castle.ground_lo = hs.first().copied().unwrap_or(40.0);
        let median = hs.get(hs.len() / 2).copied().unwrap_or(40.0);
        let top = hs.last().copied().unwrap_or(median);
        let floor = ((median + top) * 0.5 + 0.5).min(median + 3.0);
        castle.floor = (floor / VOXEL_SIZE).round() * VOXEL_SIZE;

        // Ground under a disc and along a band, lowest and highest.
        let disc = |at: Vec2, r: f32| {
            let (mut lo, mut hi) = (ground(at), ground(at));
            for k in 0..12 {
                let a = k as f32 / 12.0 * TAU;
                let g = ground(at + Vec2::new(a.cos(), a.sin()) * (r + 0.5));
                lo = lo.min(g);
                hi = hi.max(g);
            }
            (lo, hi)
        };
        castle.towers = TOWERS
            .iter()
            .map(|s| {
                let (lo, hi) = disc(s.at, s.r);
                Tower {
                    s: *s,
                    base: lo.min(castle.floor) - 1.5,
                    ground_hi: hi,
                }
            })
            .collect();
        castle.wall_base = (0..RING.len())
            .map(|i| {
                let (a, b) = (RING[i], RING[(i + 1) % RING.len()]);
                let n = edge_normal(i);
                let len = a.distance(b);
                let mut lo = castle.floor;
                let mut s = 0.0;
                while s <= len {
                    let p = a + (b - a) * (s / len);
                    for off in [-WALL_HALF - 0.4, 0.0, WALL_HALF] {
                        lo = lo.min(ground(p + n * off));
                    }
                    s += 1.0;
                }
                lo - 1.5
            })
            .collect();
        let mut g_lo = castle.floor;
        for x in [GATEHOUSE_LO.x, GATE_X, GATEHOUSE_HI.x] {
            for z in [GATEHOUSE_LO.y - 0.5, GATEHOUSE_HI.y] {
                g_lo = g_lo.min(ground(Vec2::new(x, z)));
            }
        }
        castle.gate_base = g_lo - 1.5;
        // Steps run out from the gate, half a metre at a time, until they meet the ground.
        let mut s = castle.floor;
        for k in 0..40 {
            let z = GATEHOUSE_LO.y - (k as f32 + 0.5) * APPROACH_RUN;
            let g = [-APPROACH_HALF, 0.0, APPROACH_HALF]
                .iter()
                .map(|dx| ground(Vec2::new(GATE_X + dx, z)))
                .fold(f32::MAX, f32::min);
            let gm = ground(Vec2::new(GATE_X, z));
            s += (gm - s).clamp(-STEP, STEP);
            castle.approach.push((s, g));
            if (gm - s).abs() < 0.3 {
                break;
            }
        }
        castle.lanterns = castle.lantern_spots();
        castle.set_bounds();
        Some(castle)
    }

    fn set_bounds(&mut self) {
        let run = self.approach.len() as f32 * APPROACH_RUN;
        let corners = [
            Vec2::new(CLAIM_LO.x, CLAIM_LO.y - run),
            Vec2::new(CLAIM_HI.x, CLAIM_LO.y - run),
            Vec2::new(CLAIM_LO.x, CLAIM_HI.y),
            Vec2::new(CLAIM_HI.x, CLAIM_HI.y),
        ];
        let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for q in corners {
            let w = self.world(q);
            lo = lo.min(w);
            hi = hi.max(w);
        }
        let mut bottom = self.gate_base;
        for t in &self.towers {
            bottom = bottom.min(t.base);
        }
        for &b in &self.wall_base {
            bottom = bottom.min(b);
        }
        for &(s, g) in &self.approach {
            bottom = bottom.min(g.min(s) - 1.5);
        }
        self.lo = Vec3::new(lo.x, bottom - 1.0, lo.y);
        self.hi = Vec3::new(hi.x, self.tip() + 1.0, hi.y);
    }

    /// Height of the pennant pole's top, the highest point of the castle.
    pub fn tip(&self) -> f32 {
        self.towers
            .iter()
            .map(|t| self.floor + t.s.top + 0.15 + t.s.spire + if t.s.pennant { 4.6 } else { 1.3 })
            .fold(f32::MIN, f32::max)
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.lo, self.hi)
    }

    // ---- The castle's frame ----

    fn right(&self) -> Vec2 {
        Vec2::new(self.fwd.y, -self.fwd.x)
    }

    /// World X/Z of a local point.
    fn world(&self, q: Vec2) -> Vec2 {
        self.c + self.right() * q.x + self.fwd * q.y
    }

    /// Local point of a world X/Z.
    pub fn local(&self, x: f32, z: f32) -> Vec2 {
        let d = Vec2::new(x, z) - self.c;
        Vec2::new(d.dot(self.right()), d.dot(self.fwd))
    }

    /// World point of a local point at a height above the floor.
    fn pt(&self, q: Vec2, dy: f32) -> Vec3 {
        let w = self.world(q);
        Vec3::new(w.x, self.floor + dy, w.y)
    }

    /// World direction of a local direction.
    fn dir(&self, v: Vec2) -> Vec3 {
        let w = self.right() * v.x + self.fwd * v.y;
        Vec3::new(w.x, 0.0, w.y)
    }

    /// A box square to the castle between local corners and absolute heights.
    fn lbox(&self, out: &mut MeshData, lo: Vec2, hi: Vec2, y0: f32, y1: f32, mat: Block) {
        let c = self.world((lo + hi) * 0.5);
        let e = (hi - lo) * 0.5;
        soft_box(
            out,
            Vec3::new(c.x, (y0 + y1) * 0.5, c.y),
            [self.dir(Vec2::X), Vec3::Y, self.dir(Vec2::Y)],
            Vec3::new(e.x, (y1 - y0) * 0.5, e.y),
            mat,
        );
    }

    /// Voxel centres of the lanterns: either side of the gate, inside and
    /// out, and either side of the keep door.
    fn lantern_spots(&self) -> Vec<Vec3> {
        let snap = |v: Vec3| ((v / VOXEL_SIZE).floor() + 0.5) * VOXEL_SIZE;
        let mut out = Vec::new();
        for s in [-1.0f32, 1.0] {
            let x = GATE_X + s * (GATE_HALF + 0.9);
            out.push(snap(self.pt(Vec2::new(x, GATEHOUSE_LO.y - 0.45), 3.3)));
            out.push(snap(self.pt(Vec2::new(x, GATEHOUSE_HI.y + 0.45), 3.3)));
        }
        let door = self.keep_door();
        for s in [-1.0f32, 1.0] {
            out.push(snap(self.pt(door + Vec2::new(s * 1.3, 0.45), 2.6)));
        }
        out
    }

    /// Local centre of the keep door, on its front wall.
    fn keep_door(&self) -> Vec2 {
        Vec2::new(KEEP.hi.x - 1.2, KEEP.hi.y)
    }

    /// Which step of the approach a local point is over, if any.
    fn approach_step(&self, q: Vec2) -> Option<usize> {
        if (q.x - GATE_X).abs() >= APPROACH_HALF || q.y >= GATEHOUSE_LO.y {
            return None;
        }
        let k = ((GATEHOUSE_LO.y - q.y) / APPROACH_RUN).floor() as usize;
        (k < self.approach.len()).then_some(k)
    }

    // ---- Collision ----

    /// Collision and light voxels, and the air the castle clears. `ground` is
    /// the column's terrain height.
    pub fn block(&self, p: Vec3, ground: f32) -> Option<Block> {
        let q = self.local(p.x, p.z);
        let dy = p.y - self.floor;
        let solid_from = |bottom: f32| (p.y >= bottom.min(ground - 1.0)).then_some(BUILT);
        if dy > -1.0 && dy < 4.0 && self.lanterns.iter().any(|&v| in_voxel(p, v)) {
            return Some(LANTERN);
        }
        for t in &self.towers {
            let d = q.distance(t.s.at);
            if d > t.eave_r() + 0.5 {
                continue;
            }
            let bottom = if t.s.foot > 0.0 {
                self.floor + t.s.foot - 2.0
            } else {
                t.base
            };
            if d < t.s.r && p.y >= bottom && dy < t.s.top {
                return Some(BUILT);
            }
            if dy >= t.s.top && d < t.spire_r(dy - t.s.top).max(0.3) && dy < t.s.top + t.s.spire {
                return Some(BUILT);
            }
        }
        for h in &HALLS {
            if h.contains(q, 0.0) {
                if dy < 0.0 {
                    return solid_from(self.floor - 1.0);
                }
                return (dy < h.roof_at(q)).then_some(BUILT);
            }
        }
        if q.x > GATEHOUSE_LO.x && q.x < GATEHOUSE_HI.x && q.y > GATEHOUSE_LO.y && q.y < GATEHOUSE_HI.y {
            if dy < 0.0 {
                return solid_from(self.gate_base);
            }
            if (q.x - GATE_X).abs() < GATE_HALF && dy < GATE_SPRING + 1.5 {
                return Some(AIR);
            }
            let parapet = q.y < GATEHOUSE_LO.y + PARAPET;
            let top = GATEHOUSE_TOP + if parapet { MERLON_H } else { 0.0 };
            return (dy < top).then_some(BUILT);
        }
        if let Some(k) = self.approach_step(q) {
            let (s, _) = self.approach[k];
            if p.y < s {
                return solid_from(s - 2.0);
            }
            return (p.y < s + 4.0).then_some(AIR);
        }
        let e = inward(q);
        if e < -WALL_HALF {
            return None;
        }
        if e <= WALL_HALF {
            if dy < WALK {
                return solid_from(self.floor - 1.0);
            }
            if e < -WALL_HALF + PARAPET && dy < WALK + MERLON_H {
                return Some(BUILT);
            }
            return (dy < WALK + 3.0).then_some(AIR);
        }
        // The courtyard.
        if dy < 0.0 {
            return solid_from(self.floor - 1.0);
        }
        if let Some(top) = stair_top(q) {
            if dy < top {
                return Some(BUILT);
            }
        }
        if q.distance(WELL) < WELL_R + 0.1 && dy < 0.9 {
            return Some(BUILT);
        }
        (dy < COURT_AIR).then_some(AIR)
    }

    // ---- The model ----

    /// The castle as an authored model, in world space.
    pub fn model(&self, out: &mut MeshData) {
        self.build(out, true);
    }

    /// The far stand-in: the same shapes with fewer segments and no small parts.
    pub fn far(&self, out: &mut MeshData) {
        self.build(out, false);
    }

    fn build(&self, out: &mut MeshData, near: bool) {
        self.walls(out, near);
        self.gatehouse(out, near);
        for t in &self.towers {
            self.tower(out, t, near);
        }
        for h in &HALLS {
            self.hall(out, h, near);
        }
        self.keep_extras(out, near);
        self.chapel_extras(out, near);
        self.banners(out, near);
        if near {
            self.courtyard(out);
            self.approach_model(out);
            self.lantern_brackets(out);
        }
    }

    /// Whether a point just outside some surface (local, height above floor)
    /// is buried in another part, so a window there would not show.
    fn buried(&self, q: Vec2, dy: f32, own: Option<usize>) -> bool {
        for (i, t) in self.towers.iter().enumerate() {
            if Some(i) != own && q.distance(t.s.at) < t.s.r + 0.2 && dy < t.s.top + 1.0 && dy > t.s.foot - 1.0 {
                return true;
            }
        }
        if HALLS.iter().any(|h| h.contains(q, 0.2) && dy < h.roof_at(q) + 1.0) {
            return true;
        }
        let e = inward(q);
        if e > -WALL_HALF - 0.3 && e < WALL_HALF + 0.3 && dy < WALK + MERLON_H + 0.5 {
            return true;
        }
        let g = q.x > GATEHOUSE_LO.x - 0.3
            && q.x < GATEHOUSE_HI.x + 0.3
            && q.y > GATEHOUSE_LO.y - 0.3
            && q.y < GATEHOUSE_HI.y + 0.3;
        g && dy < GATEHOUSE_TOP + MERLON_H + 0.5
    }

    fn walls(&self, out: &mut MeshData, near: bool) {
        for (i, &a) in RING.iter().enumerate() {
            if i == GATE_EDGE {
                continue;
            }
            let j = (i + 1) % RING.len();
            let b = RING[j];
            let n = edge_normal(i);
            let d = (b - a).normalize();
            let len = a.distance(b);
            let mid = (a + b) * 0.5;
            let base = self.wall_base[i];
            let f = self.floor;
            let along = self.dir(d);
            let axes = [along, Vec3::Y, along.cross(Vec3::Y)];
            let body = |out: &mut MeshData, q: Vec2, half_len: f32, y0: f32, y1: f32, half_t: f32, mat: Block| {
                let w = self.world(q);
                soft_box(
                    out,
                    Vec3::new(w.x, (y0 + y1) * 0.5, w.y),
                    axes,
                    Vec3::new(half_len, (y1 - y0) * 0.5, half_t),
                    mat,
                );
            };
            // Coursed footing, the pale wall, a moulded string course, and the
            // parapet along the outer edge.
            body(out, mid, len * 0.5, base, f + 0.6, WALL_HALF + 0.2, MASONRY);
            body(out, mid, len * 0.5, f + 0.5, f + WALK, WALL_HALF, ASHLAR);
            body(
                out,
                mid - n * 0.05,
                len * 0.5,
                f + WALK - 0.45,
                f + WALK - 0.2,
                WALL_HALF + 0.12,
                ASHLAR,
            );
            let edge = mid - n * (WALL_HALF - PARAPET * 0.5);
            body(out, edge, len * 0.5, f + WALK, f + WALK + 0.5, PARAPET * 0.5, ASHLAR);
            // Merlons, clear of the towers at either end.
            let (ra, rb) = (self.towers[i].s.r + 0.5, self.towers[j].s.r + 0.5);
            let count = ((len - ra - rb) / MERLON_PITCH).floor().max(0.0) as i32;
            let start = ra + (len - ra - rb - (count - 1) as f32 * MERLON_PITCH) * 0.5;
            for k in 0..count {
                let s = start + k as f32 * MERLON_PITCH;
                let q = a + d * s - n * (WALL_HALF - PARAPET * 0.5);
                body(
                    out,
                    q,
                    0.42,
                    f + WALK + 0.45,
                    f + WALK + MERLON_H,
                    PARAPET * 0.5,
                    ASHLAR,
                );
            }
            // A low kerb along the inner edge of the walk.
            if near {
                let q = mid + n * (WALL_HALF - 0.1);
                body(out, q, len * 0.5, f + WALK - 0.1, f + WALK + 0.2, 0.12, ASHLAR);
            }
        }
    }

    fn gatehouse(&self, out: &mut MeshData, near: bool) {
        let f = self.floor;
        let (lo, hi) = (GATEHOUSE_LO, GATEHOUSE_HI);
        let (g0, g1) = (GATE_X - GATE_HALF, GATE_X + GATE_HALF);
        // Footing, piers either side of the passage, and the block over it.
        self.lbox(out, lo - 0.2, hi + 0.2, self.gate_base, f + 0.4, MASONRY);
        self.lbox(out, lo, Vec2::new(g0, hi.y), f + 0.3, f + GATEHOUSE_TOP, ASHLAR);
        self.lbox(out, Vec2::new(g1, lo.y), hi, f + 0.3, f + GATEHOUSE_TOP, ASHLAR);
        self.lbox(
            out,
            Vec2::new(g0, lo.y),
            Vec2::new(g1, hi.y),
            f + GATE_APEX,
            f + GATEHOUSE_TOP,
            ASHLAR,
        );
        // String course and parapet with merlons on the outer face.
        self.lbox(
            out,
            lo - Vec2::new(0.0, 0.12),
            hi + Vec2::new(0.0, 0.12),
            f + GATEHOUSE_TOP - 0.45,
            f + GATEHOUSE_TOP - 0.2,
            ASHLAR,
        );
        self.lbox(
            out,
            lo,
            Vec2::new(hi.x, lo.y + PARAPET),
            f + GATEHOUSE_TOP,
            f + GATEHOUSE_TOP + 0.5,
            ASHLAR,
        );
        let mut x = lo.x + 2.9;
        while x < hi.x - 2.8 {
            self.lbox(
                out,
                Vec2::new(x - 0.42, lo.y),
                Vec2::new(x + 0.42, lo.y + PARAPET),
                f + GATEHOUSE_TOP + 0.45,
                f + GATEHOUSE_TOP + MERLON_H,
                ASHLAR,
            );
            x += MERLON_PITCH;
        }
        // The pointed arch: spandrels on both faces and a vaulted soffit.
        let segs = if near { 8 } else { 4 };
        let arc = |k: usize, side: f32| {
            let t = k as f32 / segs as f32;
            let dx = GATE_HALF * (1.0 - t);
            (GATE_X + side * dx, arch_y(dx))
        };
        for side in [-1.0f32, 1.0] {
            for (z, push) in [(lo.y, -0.01), (hi.y, 0.01)] {
                let corner = self.pt(Vec2::new(GATE_X + side * GATE_HALF, z + push), GATE_APEX);
                let mut pts = vec![corner];
                for k in 0..=segs {
                    let (x, y) = arc(k, side);
                    pts.push(self.pt(Vec2::new(x, z + push), y));
                }
                panel(out, &pts, ASHLAR);
            }
            if near {
                for k in 0..segs {
                    let (x0, y0) = arc(k, side);
                    let (x1, y1) = arc(k + 1, side);
                    let p0 = self.pt(Vec2::new(x0, lo.y), y0);
                    let p1 = self.pt(Vec2::new(x1, lo.y), y1);
                    let p2 = self.pt(Vec2::new(x1, hi.y), y1);
                    let p3 = self.pt(Vec2::new(x0, hi.y), y0);
                    panel(out, &[p0, p1, p2, p3], ASHLAR);
                    // Voussoirs: a moulded ring of stones round the outer arch.
                    let o = self.dir(-Vec2::Y) * 0.08;
                    beam(out, p0 + o, p1 + o, 0.13, ASHLAR);
                }
            }
        }
        if !near {
            return;
        }
        // Passage floor, and the raised portcullis.
        self.lbox(out, Vec2::new(g0, lo.y), Vec2::new(g1, hi.y), f - 0.3, f, MASONRY);
        let zp = lo.y + 0.9;
        let mut x = -GATE_HALF + 0.3;
        while x < GATE_HALF - 0.2 {
            let top = arch_y(x.abs()) - 0.05;
            beam(
                out,
                self.pt(Vec2::new(GATE_X + x, zp), GATE_SPRING + 0.9),
                self.pt(Vec2::new(GATE_X + x, zp), top),
                0.04,
                IRON,
            );
            x += 0.45;
        }
        for y in [GATE_SPRING + 1.0, GATE_SPRING + 1.6] {
            let w = (arch_y(0.0) - y).clamp(0.0, GATE_HALF) * 0.95;
            let dx = if y > GATE_SPRING + 1.2 {
                w * 0.8
            } else {
                GATE_HALF - 0.05
            };
            beam(
                out,
                self.pt(Vec2::new(GATE_X - dx, zp), y),
                self.pt(Vec2::new(GATE_X + dx, zp), y),
                0.04,
                IRON,
            );
        }
        // Two plank doors swung back against the passage walls.
        for side in [-1.0f32, 1.0] {
            let x = GATE_X + side * (GATE_HALF - 0.08);
            let z0 = hi.y - 0.4;
            let z1 = z0 - GATE_HALF;
            self.lbox(
                out,
                Vec2::new(x - 0.07, z1),
                Vec2::new(x + 0.07, z0),
                f,
                f + GATE_SPRING + 0.8,
                BOARDS,
            );
            for y in [0.5, 1.6, 2.8] {
                self.lbox(
                    out,
                    Vec2::new(x - side * 0.1 - 0.03, z1 + 0.05),
                    Vec2::new(x - side * 0.1 + 0.03, z0 - 0.05),
                    f + y,
                    f + y + 0.12,
                    IRON,
                );
            }
        }
    }

    fn tower(&self, out: &mut MeshData, t: &Tower, near: bool) {
        let s = &t.s;
        let c = self.world(s.at);
        let big = s.r > 2.0;
        let segs = match (near, big) {
            (true, true) => 24,
            (true, false) => 16,
            (false, true) => 16,
            (false, false) => 10,
        };
        let f = self.floor;
        let (r, top) = (s.r, f + s.top);
        let ring_r = r + if s.foot > 0.0 { 0.15 } else { 0.3 };
        if s.foot > 0.0 {
            // A turret on a corbelled cone out of the wall.
            let y0 = f + s.foot;
            lathe(
                out,
                c,
                &[(y0 - 2.2, 0.0), (y0 - 0.4, r * 0.8), (y0, r)],
                segs,
                ASHLAR,
                true,
            );
            lathe(
                out,
                c,
                &[(y0, r), (top - 0.6, r), (top - 0.3, ring_r), (top, ring_r)],
                segs,
                ASHLAR,
                false,
            );
        } else {
            let plinth = if s.perimeter {
                (t.ground_hi + 1.0).max(t.base + 2.0).min(f + 2.0)
            } else {
                f + 0.6
            };
            lathe(
                out,
                c,
                &[(t.base, r + 0.45), (plinth - 0.4, r + 0.3), (plinth, r + 0.02)],
                segs,
                MASONRY,
                false,
            );
            lathe(
                out,
                c,
                &[(plinth, r), (top - 1.6, r), (top - 0.9, ring_r), (top, ring_r)],
                segs,
                ASHLAR,
                false,
            );
            if near {
                // A thin moulded band a storey below the top.
                let y = top - 5.5;
                if y > f + WALK + 2.0 {
                    lathe(
                        out,
                        c,
                        &[(y, r), (y + 0.1, r + 0.12), (y + 0.3, r + 0.12), (y + 0.4, r)],
                        segs,
                        ASHLAR,
                        false,
                    );
                }
            }
        }
        // Eaves, and the tall spire flaring out over them.
        let re = t.eave_r();
        let e0 = top + 0.15;
        lathe(out, c, &[(top, ring_r), (e0, re)], segs, OAK, false);
        let mut prof = vec![(e0, re)];
        for u in [0.03f32, 0.08, 0.15, 0.24, 0.35, 0.48, 0.62, 0.76, 0.88, 0.96, 1.0] {
            prof.push((e0 + u * s.spire, re * (1.0 - u).powf(1.3)));
        }
        lathe(out, c, &prof, segs, ROOF, true);
        let tip = e0 + s.spire;
        let tip3 = Vec3::new(c.x, tip, c.y);
        // A gilt ball and spike, or the pennant's pole.
        let ball = if big { 0.26 } else { 0.18 };
        lathe(
            out,
            c,
            &[
                (tip - 0.2, 0.0),
                (tip - 0.05, ball),
                (tip + 0.2, ball),
                (tip + 0.38, 0.0),
            ],
            if near { 10 } else { 6 },
            GILT,
            true,
        );
        if s.pennant {
            let pole_top = tip + 4.5;
            beam(out, tip3, Vec3::new(c.x, pole_top, c.y), 0.07, IRON);
            self.pennant(out, Vec3::new(c.x, pole_top - 0.15, c.y), near);
        } else {
            beam(out, tip3 + Vec3::Y * 0.3, tip3 + Vec3::Y * 1.3, 0.035, GILT);
        }
        // Spire dormers.
        let own = self.towers.iter().position(|o| o.s.at == s.at);
        for k in 0..s.dormers {
            let a = (k as f32 + 0.5) / s.dormers as f32 * TAU + PI * 0.5;
            let dl = Vec2::new(a.cos(), a.sin());
            let h0 = s.spire * 0.07;
            let rc = t.spire_r(h0 + 0.8);
            // Not where a neighbouring tower rises through the eaves.
            let q = s.at + dl * (rc + 0.6);
            let dy = s.top + h0;
            let blocked = self.towers.iter().enumerate().any(|(i, o)| {
                Some(i) != own && q.distance(o.s.at) < o.eave_r() + 1.5 && o.s.top + o.s.spire * 0.5 > dy
            });
            if blocked {
                continue;
            }
            let foot = self.pt(s.at + dl * (rc + 0.15), s.top + 0.15 + h0);
            let w = (r * 0.2).clamp(0.45, 0.7);
            dormer(out, foot, self.dir(dl), w, w * 2.4, rc * 0.9, near);
        }
        // Pointed windows in rows round the shaft, each row turned a little.
        let n = ((r * 1.3).round() as i32).clamp(2, 6);
        let (ww, wh, gap) = if big { (0.42, 1.6, 4.0) } else { (0.3, 1.1, 3.4) };
        let mut dy = if s.foot > 0.0 { s.foot + 1.0 } else { 4.0 };
        let mut row = 0;
        let outward = s.at.normalize_or_zero();
        while dy + wh + ww * 1.2 < s.top - 1.9 {
            for k in 0..n {
                let a = (k as f32 + 0.5 * (row % 2) as f32 + 0.25) / n as f32 * TAU + s.at.x * 0.37;
                let dl = Vec2::new(a.cos(), a.sin());
                let probe = s.at + dl * (r + 0.4);
                if self.buried(probe, dy + wh * 0.5, own) || self.buried(probe, dy + wh + ww, own) {
                    continue;
                }
                // Low down, ringside towers only look out, over ground they clear.
                if s.perimeter && (dl.dot(outward) < 0.3 || f + dy < t.ground_hi + 2.0) && dy < WALK + 1.5 {
                    continue;
                }
                let n3 = self.dir(dl);
                let foot = self.pt(s.at + dl * r, dy);
                window(out, foot, n3, ww, wh, near);
            }
            dy += gap;
            row += 1;
        }
    }

    /// A swallow-tailed pennant flying from a pole top, rippling as it goes.
    fn pennant(&self, out: &mut MeshData, top: Vec3, near: bool) {
        let fly = self.dir(Vec2::new(0.8, -0.6).normalize());
        let side = fly.cross(Vec3::Y);
        let n = if near { 5 } else { 3 };
        let (len, h) = (4.0, 1.1);
        let at = |i: usize, up: f32| {
            let s = i as f32 / n as f32;
            let half = h * 0.5 * (1.0 - 0.8 * s);
            top + fly * (s * len) + side * (0.3 * (s * PI * 1.6).sin() * s) - Vec3::Y * (h * 0.5 + 0.35 * s * s)
                + Vec3::Y * (up * half)
        };
        for i in 0..n {
            panel(
                out,
                &[at(i, -1.0), at(i + 1, -1.0), at(i + 1, 1.0), at(i, 1.0)],
                CASTLE_BANNER,
            );
        }
    }

    fn hall(&self, out: &mut MeshData, h: &Hall, near: bool) {
        let f = self.floor;
        let (half_w, half_l) = h.halves();
        let (al, ac) = h.axes();
        let mid = h.mid();
        self.lbox(out, h.lo - 0.1, h.hi + 0.1, f - 1.0, f + 0.7, MASONRY);
        self.lbox(out, h.lo, h.hi, f + 0.6, f + h.eave, ASHLAR);
        self.lbox(
            out,
            h.lo - 0.15,
            h.hi + 0.15,
            f + h.eave - 0.4,
            f + h.eave - 0.05,
            ASHLAR,
        );
        // Roof slabs over the eaves, a ridge, and the gable ends.
        let ridge = h.ridge();
        let a3 = self.dir(al);
        for s in [-1.0f32, 1.0] {
            let eave = self.pt(mid + ac * s * (half_w + 0.5), h.eave - 0.5 * h.pitch);
            let top = self.pt(mid, ridge);
            let slope = (top - eave).normalize();
            let mut nrm = a3.cross(slope);
            if nrm.y < 0.0 {
                nrm = -nrm;
            }
            soft_box(
                out,
                (eave + top) * 0.5 + nrm * 0.15,
                [a3, slope, a3.cross(slope)],
                Vec3::new(half_l + 0.35, eave.distance(top) * 0.5 + 0.1, 0.15),
                ROOF,
            );
        }
        beam(
            out,
            self.pt(mid - al * (half_l + 0.35), ridge + 0.25),
            self.pt(mid + al * (half_l + 0.35), ridge + 0.25),
            0.16,
            ROOF,
        );
        for e in [-1.0f32, 1.0] {
            let end = mid + al * e * half_l;
            panel(
                out,
                &[
                    self.pt(end - ac * half_w, h.eave - 0.05),
                    self.pt(end + ac * half_w, h.eave - 0.05),
                    self.pt(end, ridge - 0.1),
                ],
                ASHLAR,
            );
            if near {
                let peak = self.pt(end + al * e * 0.1, ridge + 0.3);
                beam(out, peak, peak + Vec3::Y * 1.0, 0.05, GILT);
            }
        }
        // Rows of tall pointed windows on every wall.
        let rows: &[f32] = if h.eave > 12.0 { &[2.2, 7.2, 12.2] } else { &[2.8] };
        let (ww, wh) = if h.eave > 12.0 { (0.5, 2.2) } else { (0.55, 3.8) };
        for s in [-1.0f32, 1.0] {
            for (dir, half_run, half_depth) in [(ac, half_l, half_w), (al, half_w, half_l)] {
                let n3 = self.dir(dir * s);
                let count = ((2.0 * half_run - 1.5) / 3.0).floor().max(0.0) as i32 + 1;
                let along = if dir == ac { al } else { ac };
                for k in 0..count {
                    let off = (k as f32 - (count - 1) as f32 * 0.5) * 3.0;
                    for &dy in rows {
                        let q = mid + dir * s * half_depth + along * off;
                        let probe = q + dir * s * 0.4;
                        if self.buried(probe, dy + wh * 0.5, None) || self.buried(probe, dy + wh, None) {
                            continue;
                        }
                        if (q - self.keep_door()).length() < 1.5 && dy < 3.0 {
                            continue;
                        }
                        window(out, self.pt(q, dy), n3, ww, wh, near);
                    }
                }
            }
        }
    }

    /// The keep's dormers, chimneys and door.
    fn keep_extras(&self, out: &mut MeshData, near: bool) {
        let h = &KEEP;
        let f = self.floor;
        let (half_w, _) = h.halves();
        let mid = h.mid();
        for (s, xs) in [(1.0f32, &[-8.5f32, 1.0][..]), (-1.0, &[-8.0, -3.5, 1.0][..])] {
            for &x in xs {
                let b = half_w - 1.3;
                let q = Vec2::new(x, mid.y + s * b);
                if self.towers.iter().any(|t| q.distance(t.s.at) < t.s.r + 1.2) {
                    continue;
                }
                let foot = self.pt(q, h.eave + 1.3 * h.pitch + 0.05);
                dormer(out, foot, self.dir(Vec2::Y * s), 0.6, 1.5, 2.2, near);
            }
        }
        for (x, z) in [(-9.2f32, -7.5f32), (1.4, -8.2)] {
            let q = Vec2::new(x, z);
            let y0 = f + h.roof_at(q) - 0.6;
            let y1 = f + h.ridge() + 1.8;
            self.lbox(
                out,
                q - Vec2::new(0.45, 0.35),
                q + Vec2::new(0.45, 0.35),
                y0,
                y1,
                ASHLAR,
            );
            self.lbox(
                out,
                q - Vec2::new(0.6, 0.5),
                q + Vec2::new(0.6, 0.5),
                y1,
                y1 + 0.2,
                ASHLAR,
            );
        }
        if near {
            // The door: an arched plank door in a moulded frame.
            let door = self.keep_door();
            let n3 = self.dir(Vec2::Y);
            let foot = self.pt(door, 0.0) + n3 * 0.04;
            let t3 = Vec3::new(-n3.z, 0.0, n3.x);
            let (w, hgt) = (0.75, 2.1);
            panel(out, &arch_pts(foot, t3, w, hgt), BOARDS);
            let pts = arch_pts(foot + n3 * 0.06, t3, w + 0.12, hgt);
            for k in 1..pts.len() {
                beam(out, pts[k - 1], pts[k], 0.1, ASHLAR);
            }
            for y in [0.5, 1.4] {
                let p = foot + n3 * 0.03 + Vec3::Y * y;
                beam(out, p - t3 * (w - 0.1), p + t3 * (w - 0.1), 0.035, IRON);
            }
            self.lbox(
                out,
                door - Vec2::new(w + 0.4, 0.0),
                door + Vec2::new(w + 0.4, 0.9),
                f - 0.2,
                f + 0.25,
                ASHLAR,
            );
        }
    }

    /// The chapel's rose window, door and slender flèche.
    fn chapel_extras(&self, out: &mut MeshData, near: bool) {
        let h = &CHAPEL;
        let mid = h.mid();
        // Rose window high in the front gable.
        let front = Vec2::new(mid.x, h.hi.y);
        let n3 = self.dir(Vec2::Y);
        let t3 = Vec3::new(-n3.z, 0.0, n3.x);
        let centre = self.pt(front, h.eave + 1.6) + n3 * 0.05;
        let k = if near { 16 } else { 8 };
        let ring = |r: f32| -> Vec<Vec3> {
            (0..k)
                .map(|i| {
                    let a = i as f32 / k as f32 * TAU;
                    centre + t3 * (a.cos() * r) + Vec3::Y * (a.sin() * r)
                })
                .collect()
        };
        panel(out, &ring(1.1), WINDOW);
        if near {
            let pts = ring(1.25);
            for i in 0..k {
                beam(out, pts[i] + n3 * 0.05, pts[(i + 1) % k] + n3 * 0.05, 0.1, ASHLAR);
            }
            for i in 0..4 {
                let a = i as f32 / 4.0 * PI;
                let d = t3 * a.cos() + Vec3::Y * a.sin();
                beam(
                    out,
                    centre + n3 * 0.03 - d * 1.1,
                    centre + n3 * 0.03 + d * 1.1,
                    0.05,
                    ASHLAR,
                );
            }
            // Door onto the courtyard.
            let side = Vec2::new(h.hi.x, mid.y);
            let n3 = self.dir(Vec2::X);
            let t3 = Vec3::new(-n3.z, 0.0, n3.x);
            let foot = self.pt(side, 0.0) + n3 * 0.04;
            panel(out, &arch_pts(foot, t3, 0.6, 1.9), BOARDS);
            let pts = arch_pts(foot + n3 * 0.06, t3, 0.72, 1.9);
            for k in 1..pts.len() {
                beam(out, pts[k - 1], pts[k], 0.09, ASHLAR);
            }
        }
        // Flèche on the ridge.
        let c = self.world(mid);
        let ridge = self.floor + h.ridge();
        let segs = if near { 12 } else { 8 };
        lathe(
            out,
            c,
            &[(ridge - 1.0, 0.75), (ridge + 1.8, 0.75), (ridge + 2.0, 0.95)],
            segs,
            ASHLAR,
            false,
        );
        let mut prof = vec![(ridge + 2.0, 0.95)];
        for u in [0.1f32, 0.3, 0.55, 0.8, 1.0] {
            prof.push((ridge + 2.0 + u * 6.0, 0.95 * (1.0 - u).powf(1.2)));
        }
        lathe(out, c, &prof, segs, ROOF, true);
        let tip = ridge + 8.0;
        beam(
            out,
            Vec3::new(c.x, tip - 0.1, c.y),
            Vec3::new(c.x, tip + 1.0, c.y),
            0.04,
            GILT,
        );
        if near {
            // Bell-cote openings, lit from within.
            for i in 0..4 {
                let a = i as f32 / 4.0 * TAU + PI * 0.25;
                let d = Vec3::new(a.cos(), 0.0, a.sin());
                let foot = Vec3::new(c.x, ridge + 0.2, c.y) + d * 0.72;
                window(out, foot, d, 0.22, 0.8, false);
            }
        }
    }

    fn banners(&self, out: &mut MeshData, near: bool) {
        // Pairs on the two front walls, hung from under the parapet.
        for i in [0usize, 1] {
            let (a, b) = (RING[i], RING[i + 1]);
            let n = -edge_normal(i);
            let d = (b - a).normalize();
            for s in [0.36f32, 0.64] {
                let q = a + (b - a) * s + n * (WALL_HALF + 0.12);
                let top = self.pt(q, WALK - 0.55);
                banner(out, top, self.dir(d), self.dir(n), 0.65, 4.2, near);
            }
        }
        // Tall ones on the front tower, the great tower and the gatehouse.
        let hang_round = |out: &mut MeshData, ti: usize, dy: f32, len: f32, dir: Vec2| {
            let t = &self.towers[ti];
            let dl = dir.normalize();
            let q = t.s.at + dl * (t.s.r + 0.12);
            let n3 = self.dir(dl);
            let t3 = Vec3::new(-n3.z, 0.0, n3.x);
            banner(out, self.pt(q, dy), t3, n3, 0.7, len, near);
        };
        hang_round(out, 1, 17.0, 6.5, Vec2::Y);
        hang_round(out, 7, 31.0, 7.5, Vec2::new(0.25, 1.0));
        hang_round(out, 9, 21.5, 5.5, Vec2::new(-0.2, 1.0));
        for x in [GATE_X - 3.2, GATE_X + 3.2] {
            let q = Vec2::new(x, GATEHOUSE_LO.y - 0.12);
            banner(
                out,
                self.pt(q, GATEHOUSE_TOP - 0.7),
                self.dir(Vec2::X),
                self.dir(-Vec2::Y),
                0.6,
                4.0,
                near,
            );
        }
    }

    /// Courtyard paving, the wall stair and the well.
    fn courtyard(&self, out: &mut MeshData) {
        let f = self.floor;
        let pts: Vec<Vec3> = RING.iter().map(|&q| self.pt(q, 0.0)).collect();
        panel(out, &pts, MASONRY);
        // The stair: solid steps against the east wall, rising to the walk.
        let steps = (WALK / STEP) as i32;
        for k in 0..steps {
            let z0 = STAIR_Z0 + k as f32 * STEP;
            self.lbox(
                out,
                Vec2::new(STAIR_X.0, z0),
                Vec2::new(STAIR_X.1 + 0.05, z0 + STEP),
                f - 0.1,
                f + (k + 1) as f32 * STEP,
                ASHLAR,
            );
        }
        // The well: a stone ring, two posts, a winding beam and a little roof.
        let c = self.world(WELL);
        lathe(
            out,
            c,
            &[
                (f - 0.2, WELL_R),
                (f + 0.85, WELL_R),
                (f + 0.95, WELL_R - 0.12),
                (f + 0.95, WELL_R - 0.35),
                (f + 0.3, WELL_R - 0.35),
                (f + 0.3, 0.0),
            ],
            16,
            MASONRY,
            false,
        );
        let x3 = self.dir(Vec2::X);
        let z3 = self.dir(Vec2::Y);
        let base = Vec3::new(c.x, f + 0.9, c.y);
        for s in [-1.0f32, 1.0] {
            beam(out, base + x3 * s * 0.8, base + x3 * s * 0.8 + Vec3::Y * 1.6, 0.07, OAK);
            let eave = base + x3 * s * 0.95 + Vec3::Y * 1.45;
            let ridge = base + Vec3::Y * 2.2;
            let slope = (ridge - eave).normalize();
            soft_box(
                out,
                (eave + ridge) * 0.5,
                [z3, slope, z3.cross(slope)],
                Vec3::new(0.7, eave.distance(ridge) * 0.5 + 0.05, 0.05),
                ROOF,
            );
        }
        beam(
            out,
            base - x3 * 0.85 + Vec3::Y * 1.2,
            base + x3 * 0.85 + Vec3::Y * 1.2,
            0.06,
            OAK,
        );
    }

    /// Steps down (or a cut up) from the gate to the ground outside.
    fn approach_model(&self, out: &mut MeshData) {
        for (k, &(s, g)) in self.approach.iter().enumerate() {
            let z1 = GATEHOUSE_LO.y - k as f32 * APPROACH_RUN;
            let z0 = z1 - APPROACH_RUN - 0.05;
            let lo = Vec2::new(GATE_X - APPROACH_HALF, z0);
            let hi = Vec2::new(GATE_X + APPROACH_HALF, z1);
            let bottom = (g - 0.6).min(s - 0.3);
            self.lbox(out, lo, hi, bottom, s, MASONRY);
        }
    }

    /// Iron brackets from the walls out over each lantern.
    fn lantern_brackets(&self, out: &mut MeshData) {
        for &v in &self.lanterns {
            let q = self.local(v.x, v.z);
            // The nearest wall face is along local Z at the gate, and the keep door.
            let face_z = if q.y < GATEHOUSE_LO.y {
                GATEHOUSE_LO.y
            } else if q.y < GATEHOUSE_HI.y + 1.0 {
                GATEHOUSE_HI.y
            } else {
                KEEP.hi.y
            };
            let y = v.y + 0.45;
            let wall = self.pt(Vec2::new(q.x, face_z), y - self.floor);
            let tip = Vec3::new(v.x, y, v.z);
            beam(out, wall, tip, 0.035, IRON);
            beam(out, wall - Vec3::Y * 0.4, (wall + tip) * 0.5, 0.025, IRON);
        }
    }
}

/// Where the castle stands and which way it faces: on the cliff top, its
/// front turned towards the player's arrival point, squared to the world axes
/// so its walls, steps and stair line up with the voxels that carry collision.
fn site(spawn: Vec3) -> (Vec2, Vec2) {
    let c = Vec2::new(vista::CASTLE_TOP.0, vista::CASTLE_TOP.2);
    let d = Vec2::new(spawn.x, spawn.z) - c;
    let fwd = if d.x.abs() > d.y.abs() {
        Vec2::new(d.x.signum(), 0.0)
    } else {
        Vec2::new(0.0, if d.y < 0.0 { -1.0 } else { 1.0 })
    };
    ((c / VOXEL_SIZE).round() * VOXEL_SIZE, fwd)
}

/// Revolves a profile of (height, radius) points, traced from the bottom
/// outwards and up, about a vertical axis at `c`. Smooth profiles share
/// normals between bands; others get a crisp crease at every point.
fn lathe(out: &mut MeshData, c: Vec2, prof: &[(f32, f32)], segs: u32, mat: Block, smooth: bool) {
    // Outward normal of each band in (radial, up).
    let band = |i: usize| {
        let (y0, r0) = prof[i];
        let (y1, r1) = prof[i + 1];
        Vec2::new(y1 - y0, -(r1 - r0)).normalize_or(Vec2::X)
    };
    let ring = |out: &mut MeshData, (y, r): (f32, f32), n: Vec2| {
        let start = out.vertices.len() as u32;
        for k in 0..segs {
            let a = k as f32 / segs as f32 * TAU;
            let (cs, sn) = (a.cos(), a.sin());
            out.vertices.push(Vertex {
                pos: [c.x + cs * r, y, c.y + sn * r],
                data: smooth_data(mat, 3, Vec3::new(cs * n.x, n.y, sn * n.x)),
            });
        }
        start
    };
    let join = |out: &mut MeshData, s0: u32, s1: u32| {
        for k in 0..segs {
            let k1 = (k + 1) % segs;
            let (a, b, c, d) = (s0 + k, s0 + k1, s1 + k1, s1 + k);
            out.indices.extend_from_slice(&[a, c, b, a, d, c]);
        }
    };
    let bands = prof.len() - 1;
    if smooth {
        let mut prev = None;
        for (i, &point) in prof.iter().enumerate() {
            let n = match (i.checked_sub(1), (i < bands).then_some(i)) {
                (Some(a), Some(b)) => (band(a) + band(b)).normalize_or(band(b)),
                (Some(a), None) => band(a),
                (None, Some(b)) => band(b),
                (None, None) => Vec2::X,
            };
            let s = ring(out, point, n);
            if let Some(p) = prev {
                join(out, p, s);
            }
            prev = Some(s);
        }
    } else {
        for i in 0..bands {
            let n = band(i);
            let s0 = ring(out, prof[i], n);
            let s1 = ring(out, prof[i + 1], n);
            join(out, s0, s1);
        }
    }
}

/// Outline of a pointed (equilateral) arch standing on `foot`: half width
/// `w`, straight sides `h` tall, going round from the bottom left. Each side
/// is an arc of radius 2w centred on the opposite springing.
fn arch_pts(foot: Vec3, t: Vec3, w: f32, h: f32) -> Vec<Vec3> {
    let at = |x: f32, y: f32| foot + t * x + Vec3::Y * y;
    let mut pts = vec![at(-w, 0.0)];
    for k in 0..=4 {
        let a = k as f32 / 4.0 * (PI / 3.0);
        pts.push(at(w - 2.0 * w * a.cos(), h + 2.0 * w * a.sin()));
    }
    for k in (0..4).rev() {
        let a = k as f32 / 4.0 * (PI / 3.0);
        pts.push(at(-w + 2.0 * w * a.cos(), h + 2.0 * w * a.sin()));
    }
    pts.push(at(w, 0.0));
    pts
}

/// A pointed window: warm panes in a stone surround with a sill. `foot` is
/// the bottom centre on the wall face, `n` the face's outward normal.
fn window(out: &mut MeshData, foot: Vec3, n: Vec3, w: f32, h: f32, near: bool) {
    let t = Vec3::new(-n.z, 0.0, n.x);
    let glass = foot + n * 0.06;
    let pts = arch_pts(glass, t, w, h);
    panel(out, &pts, WINDOW);
    if !near {
        return;
    }
    let frame = arch_pts(foot + n * 0.1, t, w + 0.07, h);
    for k in 1..frame.len() {
        beam(out, frame[k - 1], frame[k], 0.075, ASHLAR);
    }
    // Sill.
    soft_box(
        out,
        foot + n * 0.12 - Vec3::Y * 0.08,
        [t, Vec3::Y, t.cross(Vec3::Y)],
        Vec3::new(w + 0.18, 0.07, 0.14),
        ASHLAR,
    );
}

/// A little gabled dormer standing out of a roof: `foot` is the bottom centre
/// of its front face, `n` the way it looks, `depth` how far back it runs.
fn dormer(out: &mut MeshData, foot: Vec3, n: Vec3, w: f32, h: f32, depth: f32, near: bool) {
    let t = Vec3::new(-n.z, 0.0, n.x);
    // Cheeks and front, running back into the roof.
    soft_box(
        out,
        foot + Vec3::Y * (h * 0.5 - 0.3) - n * (depth * 0.5),
        [t, Vec3::Y, t.cross(Vec3::Y)],
        Vec3::new(w, h * 0.5 + 0.3, depth * 0.5),
        ASHLAR,
    );
    let apex = h + w * 1.1;
    panel(
        out,
        &[
            foot + t * w + Vec3::Y * h + n * 0.01,
            foot + Vec3::Y * apex + n * 0.01,
            foot - t * w + Vec3::Y * h + n * 0.01,
        ],
        ASHLAR,
    );
    for s in [-1.0f32, 1.0] {
        let eave = foot + t * s * (w + 0.2) + Vec3::Y * (h - 0.2) + n * 0.15;
        let ridge = foot + Vec3::Y * (apex + 0.1) + n * 0.15;
        let slope = (ridge - eave).normalize();
        let back = -n;
        let mut nrm = back.cross(slope);
        if nrm.y < 0.0 {
            nrm = -nrm;
        }
        soft_box(
            out,
            (eave + ridge) * 0.5 + back * (depth * 0.5) + nrm * 0.06,
            [back, slope, back.cross(slope)],
            Vec3::new(depth * 0.5 + 0.15, eave.distance(ridge) * 0.5, 0.06),
            ROOF,
        );
    }
    let (wx, y0, y1) = (w * 0.55, 0.25, h - 0.15);
    let g = foot + n * 0.04;
    panel(
        out,
        &[
            g - t * wx + Vec3::Y * y0,
            g + t * wx + Vec3::Y * y0,
            g + t * wx + Vec3::Y * y1,
            g + Vec3::Y * (y1 + wx * 0.8),
            g - t * wx + Vec3::Y * y1,
        ],
        WINDOW,
    );
    if near {
        beam(out, g + n * 0.03 + Vec3::Y * y0, g + n * 0.03 + Vec3::Y * y1, 0.03, OAK);
    }
}

/// A long cloth banner with a pointed foot, hung from a gilt rod. `top` is
/// the middle of its top edge, `t` runs across it, `n` out from the wall.
fn banner(out: &mut MeshData, top: Vec3, t: Vec3, n: Vec3, w: f32, len: f32, near: bool) {
    let down = -Vec3::Y;
    let pts = [
        top - t * w,
        top + t * w,
        top + t * w + down * len,
        top + down * (len + w * 1.1),
        top - t * w + down * len,
    ];
    panel(out, &pts, CASTLE_BANNER);
    beam(
        out,
        top - t * (w + 0.18) + Vec3::Y * 0.05,
        top + t * (w + 0.18) + Vec3::Y * 0.05,
        0.05,
        GILT,
    );
    // A gilt lozenge on the field.
    let e = top + down * (len * 0.42) + n * 0.02;
    let (a, b) = (w * 0.45, w * 0.7);
    panel(out, &[e - t * a, e + down * b, e + t * a, e - down * b], GILT);
    if near {
        // Gilt edging down the sides and round the point.
        let o = n * 0.02;
        for k in 1..pts.len() {
            if k == 1 {
                continue;
            }
            beam(out, pts[k - 1] + o, pts[k] + o, 0.03, GILT);
        }
        beam(out, pts[4] + o, pts[0] + o, 0.03, GILT);
        // Finial knobs on the rod ends.
        for s in [-1.0f32, 1.0] {
            let p = top + t * s * (w + 0.22) + Vec3::Y * 0.05;
            beam(out, p - Vec3::Y * 0.07, p + Vec3::Y * 0.07, 0.07, GILT);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn castle() -> (Terrain, Castle) {
        let t = Terrain::new(20260927);
        let spawn = t.spawn_point();
        let c = Castle::plan(&t, spawn).expect("castle");
        (t, c)
    }

    /// What the castle puts in the voxel holding a local point.
    fn at(t: &Terrain, c: &Castle, q: Vec2, dy: f32) -> Option<Block> {
        let p = c.pt(q, dy);
        let v = ((p / VOXEL_SIZE).floor() + 0.5) * VOXEL_SIZE;
        c.block(v, t.height_at(v.x, v.z).0)
    }

    fn solid(b: Option<Block>) -> bool {
        b.is_some_and(is_solid)
    }

    #[test]
    fn the_courtyard_is_open_and_floored() {
        let (t, c) = castle();
        for q in [
            Vec2::new(8.0, 5.0),
            Vec2::new(0.0, 6.0),
            Vec2::new(12.0, -4.0),
            GATE_X * Vec2::X,
        ] {
            assert!(solid(at(&t, &c, q, -0.25)), "no floor at {q}");
            for dy in [0.25, 0.75, 1.25, 1.75] {
                assert!(!solid(at(&t, &c, q, dy)), "blocked at {q} {dy}");
            }
        }
    }

    #[test]
    fn the_gate_is_open_and_the_walls_are_solid() {
        let (t, c) = castle();
        // Through the passage, from outside the gatehouse to the courtyard.
        let mut z = GATEHOUSE_LO.y + 0.25;
        while z < GATEHOUSE_HI.y + 1.0 {
            for dy in [0.25, 1.25, 2.25] {
                assert!(
                    !solid(at(&t, &c, Vec2::new(GATE_X, z), dy)),
                    "gate shut at z {z} dy {dy}"
                );
            }
            z += 0.5;
        }
        // The piers beside the passage, and every curtain wall, are solid.
        assert!(solid(at(&t, &c, Vec2::new(GATE_X + GATE_HALF + 0.5, -15.0), 1.0)));
        assert!(solid(at(&t, &c, Vec2::new(GATE_X - GATE_HALF - 0.5, -15.0), 1.0)));
        for i in 0..RING.len() {
            if i == GATE_EDGE {
                continue;
            }
            let m = (RING[i] + RING[(i + 1) % RING.len()]) * 0.5;
            for dy in [0.25, 3.0, 8.0] {
                assert!(solid(at(&t, &c, m, dy)), "wall {i} open at {dy}");
            }
            // The parapet stops a fall from the walk.
            let parapet = (0..6).any(|k| {
                let q = m - edge_normal(i) * (WALL_HALF - 0.1 * k as f32);
                solid(at(&t, &c, q, WALK + 0.5))
            });
            assert!(parapet, "no parapet on wall {i}");
        }
        // Towers and the keep are solid.
        assert!(solid(at(&t, &c, TOWERS[7].at, 20.0)));
        assert!(solid(at(&t, &c, KEEP.mid(), 5.0)));
    }

    #[test]
    fn steps_lead_from_the_ground_to_the_gate() {
        let (t, c) = castle();
        let mut prev = c.floor;
        for (k, &(s, _)) in c.approach.iter().enumerate() {
            assert!((s - prev).abs() <= STEP + 1e-4, "step {k} rises {}", s - prev);
            let q = Vec2::new(GATE_X, GATEHOUSE_LO.y - (k as f32 + 0.5) * APPROACH_RUN);
            let p = c.pt(q, s - c.floor);
            let g = t.height_at(p.x, p.z).0;
            assert!(solid(c.block(p - Vec3::Y * 0.25, g)), "no tread on step {k}");
            assert!(!solid(c.block(p + Vec3::Y * 0.75, g)), "blocked over step {k}");
            prev = s;
        }
        let (last, _) = *c.approach.last().unwrap();
        let q = Vec2::new(GATE_X, GATEHOUSE_LO.y - c.approach.len() as f32 * APPROACH_RUN);
        let w = c.world(q);
        assert!((t.height_at(w.x, w.y).0 - last).abs() < 1.0, "steps end off the ground");
    }

    #[test]
    fn the_stair_climbs_to_the_wall_walk() {
        let (t, c) = castle();
        let x = (STAIR_X.0 + STAIR_X.1) * 0.5;
        for k in 0..(WALK / STEP) as i32 {
            let z = STAIR_Z0 + (k as f32 + 0.5) * STEP;
            let top = (k + 1) as f32 * STEP;
            assert!(solid(at(&t, &c, Vec2::new(x, z), top - 0.25)));
            assert!(!solid(at(&t, &c, Vec2::new(x, z), top + 0.25)));
            assert!(!solid(at(&t, &c, Vec2::new(x, z), top + 1.75)));
        }
        // Off the top step, onto the walk along the east wall.
        let walk = Vec2::new(RING[2].x - 0.25, 2.0);
        assert!(solid(at(&t, &c, walk, WALK - 0.25)));
        assert!(!solid(at(&t, &c, walk, WALK + 0.25)));
        assert!(!solid(at(&t, &c, walk, WALK + 1.75)));
    }

    #[test]
    fn lanterns_light_the_gate() {
        let (t, c) = castle();
        let n = c
            .lanterns
            .iter()
            .filter(|&&v| c.block(v, t.height_at(v.x, v.z).0) == Some(LANTERN))
            .count();
        assert_eq!(n, c.lanterns.len());
        assert!(n >= 4);
    }

    #[test]
    fn spires_are_tall_and_below_the_sky() {
        let (_, c) = castle();
        assert!(c.tip() < crate::terrain::WORLD_CHUNKS_Y as f32 * 16.0 - 2.0);
        for s in TOWERS {
            assert!(s.spire >= 2.0 * 2.0 * s.r, "{s:?}");
        }
    }

    #[test]
    fn nothing_floats() {
        let (t, c) = castle();
        for tw in &c.towers {
            if tw.s.foot > 0.0 {
                continue;
            }
            let w = c.world(tw.s.at);
            assert!(tw.base < t.height_at(w.x, w.y).0, "{:?}", tw.s);
        }
        for i in 0..RING.len() {
            if i == GATE_EDGE {
                continue;
            }
            let m = c.world((RING[i] + RING[(i + 1) % RING.len()]) * 0.5);
            assert!(c.wall_base[i] < t.height_at(m.x, m.y).0);
        }
    }

    #[test]
    fn the_model_stays_in_budget_and_the_far_one_matches_it() {
        let (_, c) = castle();
        let mut near = MeshData::default();
        c.model(&mut near);
        let mut far = MeshData::default();
        c.far(&mut far);
        let (nt, ft) = (near.indices.len() / 3, far.indices.len() / 3);
        assert!(nt < 45_000, "near model has {nt} triangles");
        assert!(ft < 12_000 && ft * 2 < nt, "far model has {ft} triangles");
        let top = |m: &MeshData| m.vertices.iter().map(|v| v.pos[1]).fold(f32::MIN, f32::max);
        assert!((top(&near) - top(&far)).abs() < 0.5);
        assert!(top(&near) <= c.tip() + 0.5);
        let (lo, hi) = c.bounds();
        for v in near.vertices.iter().chain(&far.vertices) {
            let p = Vec3::from(v.pos);
            assert!(
                p.cmpge(lo - 0.5).all() && p.cmple(hi + 0.5).all(),
                "{p} outside {lo} {hi}"
            );
        }
    }

    #[test]
    fn revolved_shapes_face_outward() {
        let mut out = MeshData::default();
        let prof = [(0.0, 0.0), (0.0, 2.0), (3.0, 2.0), (3.2, 2.6), (8.0, 0.0)];
        lathe(&mut out, Vec2::new(5.0, 5.0), &prof, 16, ROOF, false);
        for tri in out.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|k| Vec3::from(out.vertices[tri[k] as usize].pos));
            let n = (b - a).cross(c - a);
            if n.length() < 1e-6 {
                continue;
            }
            let centre = (a + b + c) / 3.0;
            let radial = Vec3::new(centre.x - 5.0, 0.0, centre.z - 5.0);
            // Outward means away from the axis, or up on the top and down on the bottom.
            let out_dir = if radial.length() > 1e-3 { radial } else { Vec3::Y };
            assert!(n.dot(out_dir) > -1e-4 || n.y.abs() > n.length() * 0.9);
        }
    }
}
