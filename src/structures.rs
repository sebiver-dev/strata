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
use crate::castle::Castle;
use crate::chunk::{chunk_of, local_index, CHUNK, CHUNK_VOLUME};
use crate::cottage::Cottage;
use crate::far::face;
use crate::mesh::MeshData;
use crate::model;
use crate::noise::{hash2, unit};
use crate::path_fence::PathFence;
use crate::terrain::Terrain;
use crate::vista;
use crate::watchtower::Watchtower;
use glam::{IVec3, Vec2, Vec3};
use std::collections::HashMap;

/// Fences stand this far from the road's centre line.
const FENCE_OFFSET_M: f32 = 3.6;
/// Distance between fence posts.
const FENCE_POST_SPACING: f32 = 2.4;
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
    /// The rustic fence along the path from the spawn rise to the bridge.
    PathFence(PathFence),
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

        let road = |z: f32| t.road_x(z);
        let bridge_box = (s.list[0].0, s.list[0].1);
        let mut decked = false;
        for c in hamlet(t, zb, bx0, seed) {
            if c.deck > 0.0 && decked {
                continue;
            }
            if cottage_site_ok(t, &c, bridge_box, &s) {
                decked |= c.deck > 0.0;
                s.add(Structure::Cottage(c));
            }
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
        if let Some(c) = Castle::plan(t, spawn) {
            s.add(Structure::Castle(c));
        }
        // A rustic fence along the path down to the bridge, broken wherever
        // something else stands: buildings, doorways, the road and its fences.
        let taken: Vec<(Vec3, Vec3)> = s
            .list
            .iter()
            .filter(|(_, _, st)| !matches!(st, Structure::Fence(_)))
            .map(|(lo, hi, _)| (*lo, *hi))
            .collect();
        let blocked = |p: Vec2| {
            taken
                .iter()
                .any(|(lo, hi)| p.x > lo.x - 0.6 && p.x < hi.x + 0.6 && p.y > lo.z - 0.6 && p.y < hi.z + 0.6)
                || cottage_door_near(&s, p.x, p.y)
                || cottage_approach(&s, p)
                || t.road_distance(p.x, p.y) < FENCE_OFFSET_M + 1.2
        };
        let path_fences = PathFence::plan(t, zb, seed ^ 70, blocked);
        for f in path_fences {
            s.add(Structure::PathFence(f));
        }
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
                Structure::PathFence(f) => f.model(t, &mut all),
                Structure::Watchtower(w) => w.model(t, &mut all),
                Structure::Bridge(b) => b.model(t, &mut all),
                Structure::Castle(c) => c.model(&mut all),
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
            Structure::PathFence(f) => f.bounds(),
        };
        self.list.push((lo, hi, st));
    }

    pub fn iter(&self) -> impl Iterator<Item = &Structure> {
        self.list.iter().map(|(_, _, s)| s)
    }

    /// Whether a tree trunk at this point would stand in or crowd a structure.
    pub fn blocks_tree(&self, x: f32, z: f32) -> bool {
        self.list.iter().any(|(lo, hi, st)| {
            !matches!(st, Structure::Fence(_) | Structure::PathFence(_))
                && x > lo.x - 4.0
                && x < hi.x + 4.0
                && z > lo.z - 4.0
                && z < hi.z + 4.0
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
                            Structure::Castle(c) => c.block(p, g),
                            Structure::Fence(f) => f.block(p, g),
                            Structure::PathFence(f) => f.block(p, g),
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
            if matches!(st, Structure::Fence(_) | Structure::PathFence(_))
                || c.x < lo.x
                || c.y < lo.y
                || c.x >= hi.x
                || c.y >= hi.y
            {
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
                Structure::Fence(_) | Structure::PathFence(_) => {}
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

/// Whether a point lies on the way out from a cottage's front door: the
/// strip of ground straight out from the door for fifteen metres.
fn cottage_approach(s: &Structures, p: Vec2) -> bool {
    s.list.iter().any(|(_, _, st)| match st {
        Structure::Cottage(c) => {
            let door = c.front_door();
            let out = (c.world(0.0, c.front * (c.half_wid + 1.0)) - door).normalize_or_zero();
            let along = (p - door).dot(out);
            (0.0..15.0).contains(&along) && out.perp_dot(p - door).abs() < 2.2
        }
        _ => false,
    })
}

/// The hamlet's houses, clustered along both banks of the river around the
/// bridge at `zb` (whose west end is at `bx0`): a riverside house with a deck
/// on stilts over the water below the spawn rise, a row facing the path on its
/// right, houses on the east bank upstream of the bridge facing the river, and
/// a cluster on the west bank around the bridge's far end. Sites are measured
/// from the river, the path and the bridge so they follow the terrain.
fn hamlet(t: &Terrain, zb: f32, bx0: f32, seed: u32) -> Vec<Cottage> {
    use crate::cottage::Style;
    let river = |z: f32| t.river_x(z);
    let path = |z: f32| vista::path_x(z).unwrap_or(river(z) + 22.0);
    let cottage = Style::cottage();
    let house = Style::house();
    // (x, z, ridge along X, front side, deck, style)
    let mut sites: Vec<(f32, f32, bool, f32, f32, Style)> = Vec::new();
    let mut at = |x: f32, z: f32, along_x: bool, front: f32, deck: f32, style: Style| {
        sites.push((x, z, along_x, front, deck, style));
    };
    // On the near bank below the rise, between the path and the water: door
    // to the path, a deck on stilts out over the river. It stands where the
    // gap between the sight lines to the bridge and to the castle is widest.
    // The path is still being tuned, so try spots nearest the old one first;
    // the first that passes `cottage_site_ok` is built and the rest dropped.
    for dz in [
        21.0f32, 18.0, 24.0, 15.0, 27.0, 30.0, 12.0, 33.0, 36.0, 40.0, 44.0, 48.0, -14.0, -20.0, -26.0,
    ] {
        let z = zb + dz;
        for dx in [0.0f32, -2.0, 2.0, -4.0, -6.0, -8.0, -10.0, -12.0] {
            let x = (path(z) - 7.0 + dx).max(river(z) + 13.0);
            let deck = ((x - 3.0 - (river(z) + 8.0)) / VOXEL_SIZE).round() * VOXEL_SIZE;
            if deck >= 2.0 {
                at(x, z, false, 1.0, deck, house.size(4.5, 3.0).pitch(1.4));
            }
        }
    }
    // East bank upstream of the bridge, facing the river, and one by the road.
    let z = zb - 11.0;
    at(
        river(z) + 19.0,
        z,
        false,
        -1.0,
        0.0,
        house.jettied().size(4.5, 3.25).pitch(1.3),
    );
    let z = zb - 27.0;
    at(
        river(z) + 20.0,
        z,
        false,
        -1.0,
        0.0,
        cottage.size(4.5, 3.0).pitch(1.45).with_dormers(),
    );
    let z = zb - 44.0;
    at(river(z) + 18.5, z, false, -1.0, 0.0, house.size(5.0, 3.5).pitch(1.15));
    let z = zb - 30.0;
    at(river(z) + 29.0, z, true, -1.0, 0.0, cottage.pitch(1.25));
    let z = zb - 8.0;
    at(t.road_x(z) + 11.0, z, false, -1.0, 0.0, house.pitch(1.3));
    // West bank: flanking the bridge's far end, then along the river.
    let z = zb - 11.0;
    at(bx0 - 7.0, z, true, 1.0, 0.0, house.jettied().size(4.5, 3.25).pitch(1.3));
    let z = zb + 13.0;
    at(
        bx0 - 8.0,
        z,
        true,
        -1.0,
        0.0,
        cottage.size(4.5, 3.0).pitch(1.4).with_dormers(),
    );
    let z = zb + 1.0;
    at(bx0 - 21.0, z, false, 1.0, 0.0, house.size(4.0, 3.5).pitch(1.5));
    let z = zb + 24.0;
    at(river(z) - 18.0, z, false, 1.0, 0.0, cottage.size(4.0, 3.0).pitch(1.3));
    let z = zb + 36.0;
    at(
        river(z) - 19.0,
        z,
        false,
        1.0,
        0.0,
        house.jettied().pitch(1.35).with_dormers(),
    );
    let z = zb - 30.0;
    at(river(z) - 20.0, z, false, 1.0, 0.0, house.size(4.5, 3.25).pitch(1.2));
    let z = zb - 50.0;
    at(river(z) - 19.0, z, false, 1.0, 0.0, cottage.pitch(1.2).with_dormers());
    sites
        .into_iter()
        .enumerate()
        .map(|(k, (x, z, along_x, front, deck, style))| {
            Cottage::plan(t, x, z, along_x, front, deck, style, seed ^ (1 + k as u32))
        })
        .collect()
}

/// Whether a planned cottage keeps off everything it must not crowd: its walls
/// stay out of the sight lines from the rise (`vista::keeps_clear`, whose
/// verge along the path is for trees; walls keep a smaller one), and its walls,
/// steps and deck stay off the path, the bridge, the road and its fences, and
/// the other houses. A low deck may reach into a sight line.
fn cottage_site_ok(t: &Terrain, c: &Cottage, bridge: (Vec3, Vec3), s: &Structures) -> bool {
    let (lo, hi) = c.footprint();
    let w = c.upper_wid();
    let (p, q) = (c.world(-c.half_len, -w), c.world(c.half_len, w));
    let (wlo, whi) = (p.min(q), p.max(q));
    let mut ok = true;
    let mut z = lo.y;
    while z <= hi.y && ok {
        let mut x = lo.x;
        while x <= hi.x && ok {
            let walls = x >= wlo.x && x <= whi.x && z >= wlo.y && z <= whi.y;
            let path = vista::path_distance(x, z);
            let sight = vista::keeps_clear(x, z) && path >= 3.5;
            ok = !(walls && (sight || path < vista::PATH_HALF_WIDTH_M + 1.3))
                && path > vista::PATH_HALF_WIDTH_M + 0.5
                && (t.road_x(z) - x).abs() > crate::terrain::ROAD_HALF_WIDTH_M + 3.5
                && !(x > bridge.0.x - 1.0 && x < bridge.1.x + 1.0 && z > bridge.0.z - 1.0 && z < bridge.1.z + 1.0);
            x += 0.5;
        }
        z += 0.5;
    }
    // The path also keeps clear of the house's whole bounding box, which
    // `vista`'s path test checks against.
    let (blo, bhi) = c.bounds();
    let mut z = blo.z;
    while z <= bhi.z && ok {
        if let Some(px) = vista::path_x(z) {
            let (x0, x1) = (px - vista::PATH_HALF_WIDTH_M, px + vista::PATH_HALF_WIDTH_M);
            ok = !(blo.x < x1 && bhi.x > x0);
        }
        z += 0.5;
    }
    ok && s.list.iter().all(|(_, _, st)| match st {
        Structure::Cottage(o) => {
            let (a, b) = o.footprint();
            a.x > hi.x + 1.5 || b.x < lo.x - 1.5 || a.y > hi.y + 1.5 || b.y < lo.y - 1.5
        }
        _ => true,
    })
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

    fn cottages(t: &Terrain) -> Vec<&Cottage> {
        t.structures
            .iter()
            .filter_map(|s| match s {
                Structure::Cottage(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_hamlet_clusters_on_both_banks_off_the_path_and_the_bridge() {
        let t = world();
        let houses = cottages(&t);
        assert!(houses.len() >= 12, "{} houses", houses.len());
        let bridge = t
            .structures
            .iter()
            .find_map(|s| match s {
                Structure::Bridge(b) => Some(b.clone()),
                _ => None,
            })
            .unwrap();
        let (mut east, mut west, mut upstream) = (0, 0, 0);
        for c in &houses {
            let rx = t.river_x(c.c.y);
            if c.c.x > rx {
                east += 1;
            } else {
                west += 1;
            }
            upstream += (c.c.y < bridge.z - 10.0) as i32;
            // Clustered around the bridge.
            assert!((c.c.y - bridge.z).abs() < 70.0, "{:?} strays from the bridge", c.c);
            // Walls, steps and deck stay off the path and the bridge.
            let (lo, hi) = c.footprint();
            let mut z = lo.y;
            while z <= hi.y {
                let mut x = lo.x;
                while x <= hi.x {
                    assert!(
                        vista::path_distance(x, z) > vista::PATH_HALF_WIDTH_M,
                        "{:?} on the path at {x},{z}",
                        c.c
                    );
                    let on_bridge =
                        x > bridge.x0 - 0.5 && x < bridge.x1 + 0.5 && (z - bridge.z).abs() < BRIDGE_HALF_W + 0.5;
                    assert!(!on_bridge, "{:?} on the bridge at {x},{z}", c.c);
                    x += 0.5;
                }
                z += 0.5;
            }
            // Houses don't overlap each other.
            for o in &houses {
                if std::ptr::eq(*c, *o) {
                    continue;
                }
                let (a, b) = o.footprint();
                assert!(
                    a.x >= hi.x || b.x <= lo.x || a.y >= hi.y || b.y <= lo.y,
                    "{:?} overlaps {:?}",
                    c.c,
                    o.c
                );
            }
            // The floor stands a step or so above the ground at every corner,
            // neither buried nor floating, and the steps come down to the ground.
            for (a, b) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let q = c.world(a * c.half_len, b * c.half_wid);
                let g = t.height_at(q.x, q.y).0;
                assert!(
                    c.floor > g && c.floor - g < 3.5,
                    "{:?} floor {} ground {g}",
                    c.c,
                    c.floor
                );
            }
            let foot = c.floor - c.steps as f32 * VOXEL_SIZE;
            assert!(
                foot - c.step_ground(&t, c.steps) <= VOXEL_SIZE + 0.1,
                "{:?} steps stop {} above the ground",
                c.c,
                foot - c.step_ground(&t, c.steps)
            );
        }
        assert!(
            east >= 4 && west >= 4 && upstream >= 4,
            "{east} east, {west} west, {upstream} upstream"
        );
    }

    #[test]
    fn the_riverside_deck_stands_on_posts_over_the_water() {
        let t = world();
        let houses = cottages(&t);
        let c = houses.iter().find(|c| c.deck > 0.0).expect("a house with a deck");
        // It stands on the spawn side of the river near the bridge, out of the
        // sight lines from the rise (checked when it is sited).
        assert!(
            c.c.x > t.river_x(c.c.y) && (c.c.y - vista::BRIDGE_Z).abs() < 50.0,
            "{:?}",
            c.c
        );
        // Every post runs from below the ground or river bed up to the deck.
        let posts = c.deck_posts(&t);
        assert!(posts.len() >= 6);
        let mut wet = 0;
        for &(foot, head) in &posts {
            let g = t.height_at(foot.x, foot.z).0;
            assert!(foot.y < g && head.y > g + 0.5, "post {foot} to {head}, ground {g}");
            assert!((head.y - (c.floor - 0.2)).abs() < 0.01);
            wet += (g < crate::terrain::water_level(foot.x, foot.z)) as i32;
        }
        assert!(wet >= 2, "only {wet} posts stand in the water");
        // The deck is walkable: boards underfoot, headroom above, a way out
        // through the back door, and a rail along its outer edge.
        let s = &t.structures;
        let at = |a: f32, dy: f32, bb: f32| {
            let q = c.world(a, -c.front * bb);
            let p = Vec3::new(q.x, c.floor + dy, q.y);
            let p = ((p / VOXEL_SIZE).floor() + 0.5) * VOXEL_SIZE;
            s.block_at(p, t.height_at(p.x, p.z).0)
        };
        let mid = c.half_wid + c.deck * 0.5;
        for a in [-2.0f32, 0.0, 2.0] {
            assert_eq!(at(a, -0.25, mid), Some(BUILT), "boards at {a}");
            for dy in [0.25, 1.25, 1.75] {
                assert_ne!(at(a, dy, mid), Some(BUILT), "headroom at {a}, {dy}");
            }
        }
        assert_eq!(at(1.5, 1.0, c.half_wid - 0.25), Some(AIR), "back door");
        assert_eq!(at(0.0, 0.25, c.half_wid + c.deck - 0.25), Some(BUILT), "rail");
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
