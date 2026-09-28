//! Built places in the valley: a stone arch bridge over the river, a hamlet of
//! timber cottages (one on a deck over the water), rail fences along the road,
//! a stone watchtower on a hill and a castle on the heights up the valley.
//!
//! Like the terrain they are pure functions of the seed. Each structure says
//! which block, if any, it puts in a voxel, and the terrain generator stamps
//! them over the ground after trees and lanterns. The smooth mesher rounds them
//! like everything else, so they come out soft rather than boxy. Far away,
//! where there are no voxels, each is drawn as a few simple shapes.

use crate::block::*;
use crate::bridge::{self, Bridge, BRIDGE_HALF_W};
use crate::chunk::{chunk_of, local_index, CHUNK, CHUNK_VOLUME};
use crate::cottage::Cottage;
use crate::far::{face, revolve};
use crate::mesh::{smooth_data, MeshData, Vertex};
use crate::model;
use crate::noise::{hash2, unit};
use crate::terrain::Terrain;
use crate::vista;
use crate::watchtower::Watchtower;
use glam::{IVec3, Vec2, Vec3};
use std::collections::HashMap;

/// Fences stand this far from the road's centre line.
const FENCE_OFFSET_M: f32 = 3.6;
/// Distance between fence posts.
const FENCE_POST_SPACING: f32 = 2.4;

/// Snaps a coordinate so a span of `2 * half` metres covers whole voxels.
fn snap(x: f32, half: f32) -> f32 {
    let voxels = (2.0 * half / VOXEL_SIZE).round() as i32;
    let base = (x / VOXEL_SIZE).round() * VOXEL_SIZE;
    if voxels % 2 == 0 {
        base
    } else {
        base + VOXEL_SIZE * 0.5
    }
}

/// Lowest and highest ground over a rectangle, sampled every metre.
fn ground_range(t: &Terrain, lo: Vec2, hi: Vec2) -> (f32, f32) {
    let (mut a, mut b) = (f32::MAX, f32::MIN);
    let mut z = lo.y;
    while z <= hi.y {
        let mut x = lo.x;
        while x <= hi.x {
            let h = t.height_at(x, z).0;
            a = a.min(h);
            b = b.max(h);
            x += 1.0;
        }
        z += 1.0;
    }
    (a, b)
}

/// A round stone tower with a timber lookout room and a pointed roof, or with
/// just a spire when it is part of the castle.
#[derive(Clone, Debug)]
pub struct Tower {
    pub c: Vec2,
    pub r: f32,
    /// Where the plinth starts, under the ground.
    pub base: f32,
    /// Top of the stone shaft.
    pub top: f32,
    /// Height of the timber lookout room above the shaft (0 for none).
    pub room: f32,
    pub cone_r: f32,
    pub cone_h: f32,
    pub seed: u32,
}

impl Tower {
    fn bounds(&self) -> (Vec3, Vec3) {
        let e = self.cone_r.max(self.r + 0.6) + 0.5;
        (
            Vec3::new(self.c.x - e, self.base - 1.0, self.c.y - e),
            Vec3::new(self.c.x + e, self.top + self.room + self.cone_h + 0.5, self.c.y + e),
        )
    }

    fn block(&self, p: Vec3) -> Option<Block> {
        let d = Vec2::new(p.x, p.z).distance(self.c);
        let angle = (p.z - self.c.y).atan2(p.x - self.c.x);
        let shell = |r: f32| d > r - 0.6;
        if p.y >= self.base && p.y < self.top && d <= self.r {
            let dy = p.y - self.base;
            // Narrow lit slits spiralling up the shaft.
            let turn = (angle / std::f32::consts::TAU * 4.0 + dy / 9.0).rem_euclid(1.0);
            if shell(self.r) && dy > 4.0 && dy % 3.5 < 1.0 && (0.47..0.53).contains(&turn) {
                return Some(WINDOW);
            }
            return Some(MASONRY);
        }
        let room_top = self.top + self.room;
        if self.room > 0.0 && p.y >= self.top && p.y < room_top {
            let rr = self.r + 0.6;
            if d <= rr {
                let dy = p.y - self.top;
                if dy < VOXEL_SIZE || dy >= self.room - VOXEL_SIZE {
                    return Some(WOOD);
                }
                let bay = (angle / std::f32::consts::TAU * 8.0).rem_euclid(1.0);
                if shell(rr) && (0.2..0.8).contains(&bay) && (0.75..2.25).contains(&dy) {
                    return Some(WINDOW);
                }
                return Some(if shell(rr) && !(0.2..0.8).contains(&bay) {
                    WOOD
                } else {
                    PLANKS
                });
            }
        }
        if p.y >= room_top && p.y < room_top + self.cone_h {
            let t = (p.y - room_top) / self.cone_h;
            // Slightly concave, flaring at the eaves like a real spire.
            let rr = self.cone_r * (1.0 - t).powf(1.4);
            if d <= rr.max(0.3) {
                return Some(ROOF);
            }
        }
        None
    }

    fn far(&self, out: &mut MeshData) {
        let room_top = self.top + self.room;
        let shaft = [
            (self.base, 0.0),
            (self.base, self.r),
            (self.top, self.r),
            (self.top, self.r + if self.room > 0.0 { 0.6 } else { 0.0 }),
            (room_top, self.r + if self.room > 0.0 { 0.6 } else { 0.0 }),
            (room_top, 0.0),
        ];
        revolve(out, self.c, &shaft, 10, MASONRY, 0);
        let mut cone = vec![(room_top, 0.0)];
        for i in 0..=4 {
            let t = i as f32 / 4.0;
            cone.push((room_top + t * self.cone_h, self.cone_r * (1.0 - t).powf(1.4)));
        }
        revolve(out, self.c, &cone, 10, ROOF, 0);
    }
}

/// A castle on the heights: a walled bailey on a masonry terrace, a keep,
/// and round towers crowned with slate spires.
#[derive(Clone, Debug)]
pub struct Castle {
    pub c: Vec2,
    /// Height of the terrace the castle stands on.
    pub floor: f32,
    /// Lowest ground under the terrace.
    pub base: f32,
    pub towers: Vec<Tower>,
}

const BAILEY: Vec2 = Vec2::new(22.0, 15.0);
const KEEP_LO: Vec2 = Vec2::new(-12.0, -7.0);
const KEEP_HI: Vec2 = Vec2::new(4.0, 7.0);
const CURTAIN_H: f32 = 8.0;
const KEEP_H: f32 = 16.0;

impl Castle {
    fn plan(t: &Terrain, x: f32, z: f32, seed: u32) -> Self {
        let c = Vec2::new(snap(x, 0.5), snap(z, 0.5));
        let (lo, hi) = ground_range(t, c - BAILEY, c + BAILEY);
        let floor = (hi / VOXEL_SIZE).round() * VOXEL_SIZE + 1.0;
        let tower = |dx: f32, dz: f32, r: f32, h: f32, cone: f32, s: u32| Tower {
            c: c + Vec2::new(dx, dz),
            r,
            base: lo - 1.0,
            top: floor + h,
            room: 0.0,
            cone_r: r + 0.7,
            cone_h: cone,
            seed: seed.wrapping_add(s),
        };
        let (bx, bz) = (BAILEY.x, BAILEY.y);
        let towers = vec![
            tower(-bx, -bz, 3.5, 13.0, 8.0, 1),
            tower(bx, -bz, 3.5, 13.0, 8.0, 2),
            tower(-bx, bz, 3.5, 13.0, 8.0, 3),
            tower(bx, bz, 3.5, 13.0, 8.0, 4),
            // The great tower and two slender ones rising from the bailey.
            tower(10.0, -3.0, 4.5, 24.0, 11.0, 5),
            tower(-12.0, 7.0, 2.5, 24.0, 9.0, 6),
            tower(3.0, 8.0, 2.5, 21.0, 8.0, 7),
            tower(-4.0, -7.0, 2.2, 26.0, 9.0, 8),
        ];
        Castle {
            c,
            floor,
            base: lo - 1.0,
            towers,
        }
    }

    fn bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::new(self.c.x - BAILEY.x - 1.0, self.base, self.c.y - BAILEY.y - 1.0);
        let mut hi = Vec3::new(
            self.c.x + BAILEY.x + 1.0,
            self.floor + KEEP_H + 8.0,
            self.c.y + BAILEY.y + 1.0,
        );
        for t in &self.towers {
            let (a, b) = t.bounds();
            lo = lo.min(a);
            hi = hi.max(b);
        }
        (lo, hi)
    }

    fn block(&self, p: Vec3) -> Option<Block> {
        for t in &self.towers {
            if let Some(b) = t.block(p) {
                return Some(b);
            }
        }
        let q = Vec2::new(p.x, p.z) - self.c;
        let inside = q.x.abs() <= BAILEY.x && q.y.abs() <= BAILEY.y;
        if !inside {
            return None;
        }
        if p.y < self.floor {
            return (p.y >= self.base).then_some(MASONRY);
        }
        let dy = p.y - self.floor;
        // Curtain wall with crenellations.
        let wall = q.x.abs() > BAILEY.x - 1.5 || q.y.abs() > BAILEY.y - 1.5;
        if wall {
            if dy < CURTAIN_H {
                return Some(MASONRY);
            }
            let merlon = ((q.x + q.y) / 1.0).floor() as i32 % 2 == 0;
            let outer = q.x.abs() > BAILEY.x - 0.5 || q.y.abs() > BAILEY.y - 0.5;
            return (dy < CURTAIN_H + 1.0 && outer && merlon).then_some(MASONRY);
        }
        // The keep: a tall hall with lit windows and a steep slate roof.
        if q.x >= KEEP_LO.x && q.x <= KEEP_HI.x && q.y >= KEEP_LO.y && q.y <= KEEP_HI.y {
            let half = (KEEP_HI.y - KEEP_LO.y) * 0.5;
            let across = (q.y - (KEEP_LO.y + KEEP_HI.y) * 0.5).abs();
            if dy < KEEP_H {
                let shell = q.x < KEEP_LO.x + 0.5 || q.x > KEEP_HI.x - 0.5 || across > half - 0.5;
                let along = if across > half - 0.5 { q.x } else { q.y };
                if shell && dy > 3.0 && dy % 4.0 < 1.5 && along.rem_euclid(3.0) < 1.0 {
                    return Some(WINDOW);
                }
                return Some(MASONRY);
            }
            let rt = (half - across) * 1.4;
            if dy - KEEP_H < rt {
                return Some(if dy - KEEP_H > rt - 1.0 { ROOF } else { MASONRY });
            }
        }
        None
    }

    fn far(&self, out: &mut MeshData) {
        let (lo, hi) = (self.c - BAILEY, self.c + BAILEY);
        boxed(
            out,
            Vec3::new(lo.x, self.base, lo.y),
            Vec3::new(hi.x, self.floor + CURTAIN_H, hi.y),
            MASONRY,
        );
        let (klo, khi) = (self.c + KEEP_LO, self.c + KEEP_HI);
        let top = self.floor + KEEP_H;
        boxed(
            out,
            Vec3::new(klo.x, self.floor, klo.y),
            Vec3::new(khi.x, top, khi.y),
            MASONRY,
        );
        let mid = (klo.y + khi.y) * 0.5;
        let ridge = top + (khi.y - klo.y) * 0.5 * 1.4;
        for (a, b) in [(klo.y, mid), (khi.y, mid)] {
            polygon(
                out,
                &[
                    Vec3::new(klo.x, top, a),
                    Vec3::new(khi.x, top, a),
                    Vec3::new(khi.x, ridge, b),
                    Vec3::new(klo.x, ridge, b),
                ],
                ROOF,
            );
        }
        for t in &self.towers {
            t.far(out);
        }
        // Rows of lit windows on the keep, where the voxel keep has them.
        let mut dy = 4.0;
        while dy + 1.5 < KEEP_H {
            let y = self.floor + dy + 0.75;
            let v = Vec3::Y * 0.75;
            // Window columns sit where the keep's local coordinate is a multiple of 3 m.
            let mut x = self.c.x + (KEEP_LO.x / 3.0).ceil() * 3.0 + 0.5;
            while x < khi.x - 0.5 {
                for z in [klo.y - 0.03, khi.y + 0.03] {
                    let (c, u) = (Vec3::new(x, y, z), Vec3::X * 0.5);
                    polygon(out, &[c - u - v, c + u - v, c + u + v, c - u + v], WINDOW);
                }
                x += 3.0;
            }
            let mut z = self.c.y + (KEEP_LO.y / 3.0).ceil() * 3.0 + 0.5;
            while z < khi.y - 0.5 {
                for x in [klo.x - 0.03, khi.x + 0.03] {
                    let (c, u) = (Vec3::new(x, y, z), Vec3::Z * 0.5);
                    polygon(out, &[c - u - v, c + u - v, c + u + v, c - u + v], WINDOW);
                }
                z += 3.0;
            }
            dy += 4.0;
        }
    }
}

/// A run of post-and-rail fence beside the valley road.
#[derive(Clone, Debug)]
pub struct Fence {
    pub z0: f32,
    pub z1: f32,
    /// +1 east of the road, -1 west of it.
    pub side: f32,
    /// Centre line X for each half-metre row of Z from `z0`, and how far it
    /// shifts to the next row.
    line: Vec<(f32, f32)>,
    lo: Vec3,
    hi: Vec3,
}

impl Fence {
    fn plan(t: &Terrain, z0: f32, z1: f32, side: f32) -> Self {
        let z0 = (z0 / VOXEL_SIZE).floor() * VOXEL_SIZE;
        let rows = ((z1 - z0) / VOXEL_SIZE).ceil() as usize + 1;
        let x_at = |z: f32| t.road_x(z) + side * FENCE_OFFSET_M;
        let line: Vec<(f32, f32)> = (0..rows)
            .map(|k| {
                let z = z0 + (k as f32 + 0.5) * VOXEL_SIZE;
                (x_at(z), (x_at(z + VOXEL_SIZE) - x_at(z)).abs())
            })
            .collect();
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for (k, (x, _)) in line.iter().enumerate() {
            let z = z0 + (k as f32 + 0.5) * VOXEL_SIZE;
            let h = t.height_at(*x, z).0;
            lo = lo.min(Vec3::new(x - 1.0, h - 1.0, z - 0.5));
            hi = hi.max(Vec3::new(x + 1.0, h + 2.0, z + 0.5));
        }
        Fence {
            z0,
            z1,
            side,
            line,
            lo,
            hi,
        }
    }

    /// Invisible collision along the fence line; the model draws it.
    fn block(&self, p: Vec3, ground: f32) -> Option<Block> {
        if p.z < self.z0 || p.z > self.z1 {
            return None;
        }
        let &(fx, slope) = self.line.get(((p.z - self.z0) / VOXEL_SIZE) as usize)?;
        // Wide enough that neighbouring rows touch where the road bends.
        if (p.x - fx).abs() > 0.26 + slope * 0.6 {
            return None;
        }
        (0.0..1.5).contains(&(p.y - ground)).then_some(BUILT)
    }

    /// Split-rail fence: rough posts every couple of metres with two rails
    /// between them, following the road's curve and the slope of the ground.
    fn model(&self, t: &Terrain, out: &mut MeshData) {
        let seed = (self.z0 * 7.0) as i32 ^ (self.side as i32 * 131);
        let n = (((self.z1 - self.z0) / FENCE_POST_SPACING).round() as usize).max(1);
        let step = (self.z1 - self.z0) / n as f32;
        let foot = |k: usize| {
            let z = self.z0 + k as f32 * step;
            let row = (((z - self.z0) / VOXEL_SIZE) as usize).min(self.line.len() - 1);
            let x = self.line[row].0;
            Vec3::new(x, t.height_at(x, z).0, z)
        };
        let feet: Vec<Vec3> = (0..=n).map(foot).collect();
        for (k, &f) in feet.iter().enumerate() {
            let r = |j: i32| unit(hash2(seed as u32, k as i32, j)) - 0.5;
            let lean = Vec3::new(r(1) * 0.08, 0.0, r(2) * 0.08);
            let h = 1.35 + r(3) * 0.12;
            model::beam(out, f - Vec3::Y * 0.4, f + Vec3::Y * h + lean, 0.09, WOOD);
            // A little cap so the post top reads as cut timber.
            model::block(
                out,
                f + lean + Vec3::new(-0.1, h, -0.1),
                f + lean + Vec3::new(0.1, h + 0.04, 0.1),
                OAK,
            );
        }
        for (k, pair) in feet.windows(2).enumerate() {
            let (a, b) = (pair[0], pair[1]);
            for (j, y) in [0.55f32, 1.05].iter().enumerate() {
                let r = |i: i32| unit(hash2(seed as u32 ^ 77, k as i32, i + j as i32 * 5)) - 0.5;
                // Rails overlap the posts a little and sag or rise a hand's width.
                let d = (b - a).normalize_or_zero() * 0.12;
                let a1 = a - d + Vec3::new(0.0, y + r(0) * 0.06, 0.0);
                let b1 = b + d + Vec3::new(0.0, y + r(1) * 0.06, 0.0);
                model::beam(out, a1, b1, 0.055 + r(2) * 0.01, OAK);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum Structure {
    Bridge(Bridge),
    Cottage(Cottage),
    Watchtower(Watchtower),
    Castle(Castle),
    Fence(Fence),
}

/// Everything built in the world, with the boxes (metres) each one occupies.
#[derive(Default)]
pub struct Structures {
    list: Vec<(Vec3, Vec3, Structure)>,
    /// Authored models, cut up by the chunk each triangle's centre falls in.
    models: HashMap<IVec3, MeshData>,
    /// The outside of each cottage, by its index in `list`, drawn in the far field
    /// so a house keeps its timbers, windows and roof at any distance.
    far_models: HashMap<usize, MeshData>,
}

impl Structures {
    /// Lays out the hamlet around the spawn point and the landmarks up and down the valley.
    pub fn plan(t: &Terrain) -> Self {
        let mut s = Structures::default();
        let spawn = t.spawn_point();
        let seed = t.seed;
        // The player starts facing down the valley (towards -Z), so the hamlet lies ahead.
        let zb = bridge::site(t, spawn);
        let bridge = Bridge::plan(t, zb, seed ^ 60);
        let zb = bridge.z;
        let (bx0, bx1) = (bridge.x0, bridge.x1);
        s.add(Structure::Bridge(bridge));

        let river = |z: f32| t.river_x(z);
        let road = |z: f32| t.road_x(z);
        let mut cottages = Vec::new();
        // A cottage on the east bank with a deck on stilts over the water.
        let z = spawn.z - 40.0;
        cottages.push(Cottage::plan(t, river(z) + 17.5, z, false, 2, 1.0, 7.0, seed ^ 1));
        // Cottages along the east side of the road, doors facing it.
        for (k, dz) in [-55.0f32, -86.0, -104.0].iter().enumerate() {
            let z = spawn.z + dz;
            let storeys = if k == 0 { 2 } else { 1 };
            cottages.push(Cottage::plan(
                t,
                road(z) + 9.5,
                z,
                false,
                storeys,
                -1.0,
                0.0,
                seed ^ (2 + k as u32),
            ));
        }
        // Across the bridge, on the west bank.
        for (k, dz) in [-12.0f32, 14.0].iter().enumerate() {
            let z = zb + dz;
            cottages.push(Cottage::plan(
                t,
                bx0 - 5.0,
                z,
                true,
                1,
                -dz.signum(),
                0.0,
                seed ^ (8 + k as u32),
            ));
        }
        for c in cottages {
            s.add(Structure::Cottage(c));
        }
        // Fences line the road through the hamlet, broken where paths leave it.
        let (zlo, zhi) = (spawn.z - 120.0, spawn.z - 8.0);
        for side in [-1.0f32, 1.0] {
            let mut z0 = zlo;
            let mut z = zlo;
            while z <= zhi {
                let x = road(z) + side * FENCE_OFFSET_M;
                let gap = (z - zb).abs() < BRIDGE_HALF_W + 2.0 || cottage_door_near(&s, x, z);
                if gap || z >= zhi {
                    if z - z0 > 4.0 {
                        s.add(Structure::Fence(Fence::plan(t, z0, z - 1.0, side)));
                    }
                    z0 = z + 1.0;
                }
                z += 0.5;
            }
        }
        let _ = bx1;

        // The watchtower and the castle stand on the knoll and the bluff composed
        // for the view on arrival (see `vista`).
        let (tx, _, tz) = vista::TOWER_KNOLL;
        s.add(Structure::Watchtower(Watchtower::plan(
            t,
            Vec2::new(tx, tz),
            Vec2::new(spawn.x, spawn.z),
            seed ^ 20,
        )));
        let (cx, _, cz) = vista::CASTLE_TOP;
        s.add(Structure::Castle(Castle::plan(t, cx, cz, seed ^ 40)));
        s.build_models(t);
        s
    }

    /// Builds the authored models and cuts them up by chunk, each triangle
    /// going to the chunk its centre lies in.
    fn build_models(&mut self, t: &Terrain) {
        let mut all = MeshData::default();
        for (k, (_, _, st)) in self.list.iter().enumerate() {
            if let Structure::Cottage(c) = st {
                let mut far = MeshData::default();
                c.exterior(t, &mut far);
                self.far_models.insert(k, far);
            }
            match st {
                Structure::Cottage(c) => c.model(t, &mut all),
                Structure::Fence(f) => f.model(t, &mut all),
                Structure::Watchtower(w) => w.model(t, &mut all),
                Structure::Bridge(b) => b.model(t, &mut all),
                _ => {}
            }
        }
        let mut remap: HashMap<(IVec3, u32), u32> = HashMap::new();
        for tri in all.indices.chunks(3) {
            let centre = tri
                .iter()
                .map(|&i| Vec3::from(all.vertices[i as usize].pos))
                .sum::<Vec3>()
                / 3.0;
            let cpos = chunk_of((centre / VOXEL_SIZE).floor().as_ivec3());
            let dst = self.models.entry(cpos).or_default();
            for &i in tri {
                let j = *remap.entry((cpos, i)).or_insert_with(|| {
                    dst.vertices.push(all.vertices[i as usize]);
                    dst.vertices.len() as u32 - 1
                });
                dst.indices.push(j);
            }
        }
    }

    /// Adds the model pieces that fall in chunk `cpos`, in metres like the chunk mesh.
    pub fn append_models(&self, cpos: IVec3, out: &mut MeshData) {
        if let Some(m) = self.models.get(&cpos) {
            model::append(out, m);
        }
    }

    /// Whether any model piece falls in chunk `cpos`.
    pub fn has_models(&self, cpos: IVec3) -> bool {
        self.models.contains_key(&cpos)
    }

    fn add(&mut self, st: Structure) {
        let (lo, hi) = match &st {
            Structure::Bridge(b) => b.bounds(),
            Structure::Cottage(c) => c.bounds(),
            Structure::Watchtower(w) => w.bounds(),
            Structure::Castle(c) => c.bounds(),
            Structure::Fence(f) => (f.lo, f.hi),
        };
        self.list.push((lo, hi, st));
    }

    pub fn iter(&self) -> impl Iterator<Item = &Structure> {
        self.list.iter().map(|(_, _, s)| s)
    }

    /// Whether a tree trunk at this point would stand in or crowd a structure.
    pub fn blocks_tree(&self, x: f32, z: f32) -> bool {
        self.list.iter().any(|(lo, hi, st)| {
            !matches!(st, Structure::Fence(_)) && x > lo.x - 4.0 && x < hi.x + 4.0 && z > lo.z - 4.0 && z < hi.z + 4.0
        })
    }

    /// Whether any structure touches a box of metres; cheap test before stamping.
    pub fn touches(&self, lo: Vec3, hi: Vec3) -> bool {
        self.list
            .iter()
            .any(|(a, b, _)| a.x < hi.x && b.x > lo.x && a.y < hi.y && b.y > lo.y && a.z < hi.z && b.z > lo.z)
    }

    /// Stamps every structure that reaches into a chunk over its voxels.
    /// `origin` is the chunk's first voxel and `ground(x, z)` the ground height
    /// of its local column. Solid parts of one structure win over the air
    /// another clears, and cleared air never drains the river.
    pub fn stamp(&self, origin: IVec3, ground: impl Fn(i32, i32) -> f32, data: &mut [Block; CHUNK_VOLUME]) {
        let lo_m = origin.as_vec3() * VOXEL_SIZE;
        let hi_m = lo_m + CHUNK as f32 * VOXEL_SIZE;
        let mut placed: Option<Vec<bool>> = None;
        for (lo, hi, st) in &self.list {
            if lo.x >= hi_m.x || hi.x <= lo_m.x || lo.y >= hi_m.y || hi.y <= lo_m.y || lo.z >= hi_m.z || hi.z <= lo_m.z
            {
                continue;
            }
            let placed = placed.get_or_insert_with(|| vec![false; CHUNK_VOLUME]);
            // Local voxel range covered by the box.
            let range = |a: f32, b: f32, o: f32| {
                let first = (((a - o) / VOXEL_SIZE).floor() as i32).max(0);
                let last = (((b - o) / VOXEL_SIZE).ceil() as i32).min(CHUNK);
                first..last
            };
            let (xr, yr, zr) = (
                range(lo.x, hi.x, lo_m.x),
                range(lo.y, hi.y, lo_m.y),
                range(lo.z, hi.z, lo_m.z),
            );
            for z in zr {
                for x in xr.clone() {
                    let g = ground(x, z);
                    for y in yr.clone() {
                        let p = ((origin + IVec3::new(x, y, z)).as_vec3() + 0.5) * VOXEL_SIZE;
                        let b = match st {
                            Structure::Bridge(b) => b.block(p, g),
                            Structure::Cottage(c) => c.block(p, g),
                            Structure::Watchtower(w) => w.block(p, g),
                            Structure::Castle(c) => c.block(p),
                            Structure::Fence(f) => f.block(p, g),
                        };
                        let Some(b) = b else { continue };
                        let i = local_index(x, y, z);
                        if b != AIR {
                            data[i] = b;
                            placed[i] = true;
                        } else if !placed[i] && data[i] != WATER {
                            data[i] = AIR;
                        }
                    }
                }
            }
        }
    }

    /// Which block the structures put at one voxel centre (metres), given its
    /// column's ground height. For tests and tools; generation uses `stamp`.
    pub fn block_at(&self, p: Vec3, ground: f32) -> Option<Block> {
        let v = (p / VOXEL_SIZE).floor().as_ivec3();
        let origin = crate::chunk::chunk_of(v) * CHUNK;
        let mut data = Box::new([STONE; CHUNK_VOLUME]);
        let l = v - origin;
        data[local_index(l.x, l.y, l.z)] = crate::block::GRAVEL;
        self.stamp(origin, |_, _| ground, &mut data);
        match data[local_index(l.x, l.y, l.z)] {
            crate::block::GRAVEL => None,
            b => Some(b),
        }
    }

    /// Whether a column is part of a path leading off the road to a structure.
    pub fn path_at(&self, t: &Terrain, x: f32, z: f32) -> bool {
        self.list.iter().any(|(_, _, st)| match st {
            // The approach from the valley road onto the bridge.
            Structure::Bridge(b) => {
                let road = t.road_x(b.z);
                (z - b.z).abs() < 2.1 && ((x > b.x1 - 1.0 && x < road) || (x < b.x0 + 1.0 && x > b.x0 - 12.0))
            }
            _ => false,
        })
    }

    /// Adds the far-away shapes of every structure whose centre lies in the
    /// rectangle of metres [lo, hi).
    pub fn far(&self, out: &mut MeshData, lo: Vec2, hi: Vec2) {
        for (k, (a, b, st)) in self.list.iter().enumerate() {
            let c = Vec2::new(a.x + b.x, a.z + b.z) * 0.5;
            if matches!(st, Structure::Fence(_)) || c.x < lo.x || c.y < lo.y || c.x >= hi.x || c.y >= hi.y {
                continue;
            }
            match st {
                Structure::Bridge(b) => b.far(out),
                Structure::Cottage(c) => match self.far_models.get(&k) {
                    Some(m) => model::append(out, m),
                    None => c.far(out),
                },
                Structure::Watchtower(w) => w.far(out),
                Structure::Castle(c) => c.far(out),
                Structure::Fence(_) => {}
            }
        }
    }
}

/// Which way the player faces on arrival (`Player::new`), as a yaw in radians.
pub const SPAWN_YAW: f32 = crate::vista::SPAWN_YAW;

fn cottage_door_near(s: &Structures, x: f32, z: f32) -> bool {
    s.list.iter().any(|(_, _, st)| match st {
        Structure::Cottage(c) => {
            let door = c.world(0.0, c.front * c.half_wid);
            (door - Vec2::new(x, z)).length() < 5.0
        }
        _ => false,
    })
}

/// A flat, convex polygon given counter-clockwise from outside; drawn from both
/// sides so its winding never hides it.
fn polygon(out: &mut MeshData, pts: &[Vec3], mat: Block) {
    let n = (pts[1] - pts[0]).cross(pts[2] - pts[0]).normalize_or_zero();
    for normal in [n, -n] {
        let start = out.vertices.len() as u32;
        for q in pts {
            out.vertices.push(Vertex {
                pos: q.to_array(),
                data: smooth_data(mat, 3, normal),
            });
        }
        for k in 1..pts.len() as u32 - 1 {
            if normal == n {
                out.indices.extend_from_slice(&[start, start + k, start + k + 1]);
            } else {
                out.indices.extend_from_slice(&[start, start + k + 1, start + k]);
            }
        }
    }
}

/// All six faces of an axis-aligned box.
pub(crate) fn boxed(out: &mut MeshData, lo: Vec3, hi: Vec3, mat: Block) {
    for fi in 0..6 {
        face(&mut out.vertices, &mut out.indices, lo, hi, fi, mat);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> Terrain {
        Terrain::new(20260927)
    }

    #[test]
    fn the_hamlet_and_landmarks_are_planned() {
        let t = world();
        let s = &t.structures;
        let count = |f: fn(&Structure) -> bool| s.iter().filter(|x| f(x)).count();
        assert_eq!(count(|x| matches!(x, Structure::Bridge(_))), 1);
        assert!(count(|x| matches!(x, Structure::Cottage(_))) >= 5);
        assert!(count(|x| matches!(x, Structure::Fence(_))) >= 2);
        assert_eq!(count(|x| matches!(x, Structure::Watchtower(_))), 1);
        assert_eq!(count(|x| matches!(x, Structure::Castle(_))), 1);
    }

    #[test]
    fn far_cottages_keep_their_timbers_and_windows() {
        let t = world();
        let (k, c) = t
            .structures
            .list
            .iter()
            .enumerate()
            .find_map(|(k, (_, _, s))| match s {
                Structure::Cottage(c) => Some((k, c)),
                _ => None,
            })
            .unwrap();
        let m = &t.structures.far_models[&k];
        let mats = |mat: Block| m.vertices.iter().filter(|v| (v.data >> 3) & 0xff == mat as u32).count();
        assert!(mats(OAK) > 100 && mats(WINDOW) > 8 && mats(ROOF) > 100, "{c:?}");
        assert!(m.indices.len() / 3 < 20_000, "{} triangles", m.indices.len() / 3);
    }

    #[test]
    fn far_shapes_exist_for_landmarks() {
        let t = world();
        let mut out = MeshData::default();
        t.structures
            .far(&mut out, Vec2::ZERO, Vec2::splat(crate::terrain::WORLD_SIZE_M));
        assert!(out.indices.len() > 300);
    }
}
