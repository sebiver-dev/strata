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
use crate::chunk::{local_index, CHUNK, CHUNK_VOLUME};
use crate::far::{face, revolve};
use crate::mesh::{smooth_data, MeshData, Vertex};
use crate::noise::{hash2, unit};
use crate::terrain::{Terrain, WATER_LEVEL_M};
use glam::{IVec3, Vec2, Vec3};

/// Outer half-width of the bridge, parapets included.
const BRIDGE_HALF_W: f32 = 2.5;
/// Radius of each of the bridge's three arches.
const ARCH_R: f32 = 4.5;
/// Distance between arch centres.
const ARCH_SPACING: f32 = 11.0;
/// Height where the arches spring from their piers.
const ARCH_SPRING: f32 = WATER_LEVEL_M - 0.5;
/// Half the length of the bridge's level middle part.
const BRIDGE_FLAT: f32 = 14.0;
/// Rise per metre of the bridge's ramps.
const BRIDGE_RAMP: f32 = 0.3;
/// Fences stand this far from the road's centre line.
const FENCE_OFFSET_M: f32 = 3.6;

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

/// A stone bridge of three round arches, crossing the river along X.
#[derive(Clone, Debug)]
pub struct Bridge {
    pub z: f32,
    pub centre: f32,
    pub x0: f32,
    pub x1: f32,
    /// Height of the walking surface over the middle.
    pub deck: f32,
}

impl Bridge {
    fn plan(t: &Terrain, z: f32) -> Self {
        let z = snap(z, BRIDGE_HALF_W);
        let centre = snap(t.river_x(z), 0.25);
        let deck = ARCH_SPRING + ARCH_R + 1.5;
        let mut b = Bridge {
            z,
            centre,
            x0: centre,
            x1: centre,
            deck,
        };
        // Ramps run down until they meet the bank.
        for dir in [-1.0f32, 1.0] {
            let mut d = BRIDGE_FLAT;
            while d < 45.0 {
                let x = centre + dir * d;
                if b.top(x) <= t.height_at(x, z).0 + 0.3 {
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
        b
    }

    fn top(&self, x: f32) -> f32 {
        self.deck - ((x - self.centre).abs() - BRIDGE_FLAT).max(0.0) * BRIDGE_RAMP
    }

    fn bounds(&self) -> (Vec3, Vec3) {
        (
            Vec3::new(self.x0, WATER_LEVEL_M - 6.0, self.z - BRIDGE_HALF_W),
            Vec3::new(self.x1, self.deck + 4.0, self.z + BRIDGE_HALF_W),
        )
    }

    fn block(&self, p: Vec3, ground: f32) -> Option<Block> {
        let dz = (p.z - self.z).abs();
        let top = self.top(p.x);
        let edge = dz > BRIDGE_HALF_W - VOXEL_SIZE;
        if p.y < top {
            if p.y < ground - 1.0 {
                return None;
            }
            for k in -1..=1 {
                let dx = p.x - (self.centre + k as f32 * ARCH_SPACING);
                let dy = (p.y - ARCH_SPRING).max(0.0);
                if dx * dx + dy * dy < ARCH_R * ARCH_R {
                    return None;
                }
            }
            return Some(MASONRY);
        }
        let h = p.y - top;
        if edge {
            // Parapets, with taller piers carrying lanterns at the middle and the ends of the level part.
            let pier = [-BRIDGE_FLAT, 0.0, BRIDGE_FLAT]
                .iter()
                .any(|o| (p.x - (self.centre + o)).abs() < 0.5);
            if h < 1.0 || (pier && h < 1.5) {
                return Some(MASONRY);
            }
            if pier && h < 2.0 && (p.x - self.centre).abs() % BRIDGE_FLAT < 0.25 {
                return Some(LANTERN);
            }
        }
        (h < 3.0).then_some(AIR)
    }

    fn far(&self, out: &mut MeshData) {
        let w = BRIDGE_HALF_W;
        for (a, b) in [
            (self.x0, self.centre - BRIDGE_FLAT),
            (self.centre + BRIDGE_FLAT, self.x1),
        ] {
            let (lo, hi) = (a, b);
            let mid = (lo + hi) * 0.5;
            boxed(
                out,
                Vec3::new(lo, self.top(mid) - 3.0, self.z - w),
                Vec3::new(hi, self.top(mid) + 1.0, self.z + w),
                MASONRY,
            );
        }
        let (lo, hi) = (self.centre - BRIDGE_FLAT, self.centre + BRIDGE_FLAT);
        boxed(
            out,
            Vec3::new(lo, self.deck - 1.2, self.z - w),
            Vec3::new(hi, self.deck + 1.0, self.z + w),
            MASONRY,
        );
        for k in [-1.5f32, -0.5, 0.5, 1.5] {
            let x = self.centre + k * ARCH_SPACING;
            boxed(
                out,
                Vec3::new(x - 1.0, WATER_LEVEL_M - 3.0, self.z - w),
                Vec3::new(x + 1.0, self.deck, self.z + w),
                MASONRY,
            );
        }
    }
}

/// A timber-framed cottage with lime plaster, a steep roof and warm windows.
#[derive(Clone, Debug)]
pub struct Cottage {
    pub c: Vec2,
    /// Whether the ridge runs along X (otherwise along Z).
    pub along_x: bool,
    pub half_len: f32,
    pub half_wid: f32,
    /// Height of the ground floor.
    pub floor: f32,
    pub storeys: u8,
    /// Which long side has the door: +1 or -1 along the across axis.
    pub front: f32,
    /// Metres of plank deck on stilts behind the back wall; 0 for none.
    pub deck: f32,
    /// Lowest ground under the cottage, where the plinth starts.
    pub base: f32,
    pub seed: u32,
}

const PITCH: f32 = 1.15;
const EAVE: f32 = 0.5;
const ROOF_THICK: f32 = 0.8;

impl Cottage {
    #[allow(clippy::too_many_arguments)]
    fn plan(t: &Terrain, x: f32, z: f32, along_x: bool, storeys: u8, front: f32, deck: f32, seed: u32) -> Self {
        let (half_len, half_wid) = if storeys > 1 { (4.0, 3.25) } else { (4.0, 2.75) };
        let (hx, hz) = if along_x {
            (half_len, half_wid)
        } else {
            (half_wid, half_len)
        };
        let c = Vec2::new(snap(x, hx), snap(z, hz));
        let (lo, hi) = ground_range(t, c - Vec2::new(hx, hz), c + Vec2::new(hx, hz));
        // Floors sit a step above the ground, never below the flood line.
        let floor = ((hi.min(lo + 2.0) + 0.5).max(WATER_LEVEL_M + 1.5) / VOXEL_SIZE).round() * VOXEL_SIZE;
        Cottage {
            c,
            along_x,
            half_len,
            half_wid,
            floor,
            storeys,
            front,
            deck,
            base: lo.min(WATER_LEVEL_M - 3.0 + if deck > 0.0 { 0.0 } else { 10.0 }),
            seed,
        }
    }

    fn wall_height(&self) -> f32 {
        if self.storeys > 1 {
            5.5
        } else {
            3.0
        }
    }

    fn ridge(&self) -> f32 {
        self.floor + self.wall_height() + (self.half_wid + EAVE) * PITCH
    }

    /// Converts world X/Z to (along, across) around the centre.
    fn local(&self, x: f32, z: f32) -> (f32, f32) {
        let d = Vec2::new(x, z) - self.c;
        if self.along_x {
            (d.x, d.y)
        } else {
            (d.y, d.x)
        }
    }

    fn world(&self, a: f32, b: f32) -> Vec2 {
        if self.along_x {
            self.c + Vec2::new(a, b)
        } else {
            self.c + Vec2::new(b, a)
        }
    }

    /// Outline of the land the cottage claims, deck and yard included.
    fn bounds(&self) -> (Vec3, Vec3) {
        let reach_b = self.half_wid + 1.5 + self.deck;
        let reach_a = self.half_len + 1.5;
        let (ex, ez) = if self.along_x {
            (reach_a, reach_b)
        } else {
            (reach_b, reach_a)
        };
        (
            Vec3::new(self.c.x - ex, self.base - 1.5, self.c.y - ez),
            Vec3::new(self.c.x + ex, self.ridge() + 2.0, self.c.y + ez),
        )
    }

    fn wall(&self, a: f32, b: f32, dy: f32) -> Block {
        let (l, w, h) = (self.half_len, self.half_wid, self.wall_height());
        let long_face = b.abs() > w - VOXEL_SIZE;
        let end_face = a.abs() > l - VOXEL_SIZE;
        if !(long_face || end_face) {
            return PLASTER;
        }
        if long_face && end_face {
            return WOOD;
        }
        if long_face && b * self.front > 0.0 && a.abs() < 0.5 && (0.5..2.5).contains(&dy) {
            return PLANKS;
        }
        if dy < 0.5 || dy >= h - VOXEL_SIZE || (self.storeys > 1 && (2.5..3.0).contains(&dy)) {
            return WOOD;
        }
        let (t, half) = if long_face { (a, l) } else { (b, w) };
        let ti = ((t + half) / VOXEL_SIZE).floor() as i32;
        if ti % 5 == 0 {
            return WOOD;
        }
        let storey_dy = if dy >= 3.0 { dy - 3.0 } else { dy };
        let bay = ti / 5;
        let side = if long_face { b.signum() } else { 2.0 + a.signum() };
        let lit = unit(hash2(self.seed, bay * 4 + side as i32, (dy >= 3.0) as i32)) < 0.8;
        let near_door = long_face && b * self.front > 0.0 && dy < 3.0 && a.abs() < 1.5;
        if (ti % 5 == 2 || ti % 5 == 3) && (1.0..2.0).contains(&storey_dy) && lit && !near_door {
            return WINDOW;
        }
        PLASTER
    }

    fn block(&self, p: Vec3, ground: f32) -> Option<Block> {
        let (a, b) = self.local(p.x, p.z);
        let (l, w, h) = (self.half_len, self.half_wid, self.wall_height());
        let top = self.floor + h;
        let r = p.y - top;
        // Chimney through the back slope.
        let chimney = (a - l * 0.45).abs() < 0.5 && (b + self.front * w * 0.4).abs() < 0.5;
        if chimney && p.y >= self.floor && r < (w + EAVE) * PITCH * 0.75 + 1.0 {
            return Some(MASONRY);
        }
        if a.abs() <= l + EAVE && b.abs() <= w + EAVE {
            let rt = (w + EAVE - b.abs()) * PITCH;
            if r < rt && r >= rt - ROOF_THICK {
                return Some(ROOF);
            }
        }
        if a.abs() <= l && b.abs() <= w && p.y >= self.floor {
            let rt = (w + EAVE - b.abs()) * PITCH;
            if r < 0.0 {
                return Some(self.wall(a, b, p.y - self.floor));
            }
            if r < rt - ROOF_THICK {
                // Gables carry a timber tie beam and a king post.
                return Some(if r < VOXEL_SIZE || a.abs() > l - VOXEL_SIZE && b.abs() < VOXEL_SIZE {
                    WOOD
                } else {
                    PLASTER
                });
            }
        }
        if a.abs() <= l + 0.25 && b.abs() <= w + 0.25 && p.y < self.floor {
            return (p.y >= self.base - 1.0).then_some(MASONRY);
        }
        let bb = -b * self.front;
        // A lantern on a bracket beside the door.
        if b * self.front > w && b * self.front <= w + VOXEL_SIZE && (0.75..1.25).contains(&a) {
            let dy = p.y - self.floor;
            if (2.0..2.5).contains(&dy) {
                return Some(LANTERN);
            }
            if (2.5..3.0).contains(&dy) {
                return Some(WOOD);
            }
        }
        if self.deck > 0.0 && bb > w && bb <= w + self.deck && a.abs() <= l + 0.5 {
            let dy = p.y - self.floor;
            let outer = bb > w + self.deck - VOXEL_SIZE || a.abs() > l;
            let ai = ((a + l + 0.5) / VOXEL_SIZE).floor() as i32;
            let post = outer && (ai % 4 == 0 || a.abs() > l || bb > w + self.deck - VOXEL_SIZE && ai == 17);
            if (-VOXEL_SIZE..0.0).contains(&dy) {
                return Some(PLANKS);
            }
            if dy < -VOXEL_SIZE {
                let stilt = ai % 6 == 0 && (outer || (bb - w - self.deck * 0.5).abs() < 0.25);
                return (stilt && p.y >= ground - 1.0).then_some(WOOD);
            }
            if outer && (1.0..1.5).contains(&dy) {
                return Some(WOOD);
            }
            if outer && post && dy < 1.0 {
                return Some(WOOD);
            }
            let corner = a > l && bb > w + self.deck - VOXEL_SIZE;
            if corner && (1.5..2.0).contains(&dy) {
                return Some(LANTERN);
            }
            return (dy < 3.0).then_some(AIR);
        }
        // Clear a small yard of grass, trees and ground above the floor.
        let yard = a.abs() <= l + 1.5 && b.abs() <= w + 1.5;
        (yard && p.y >= self.floor && p.y < self.ridge() + 2.0).then_some(AIR)
    }

    fn far(&self, out: &mut MeshData) {
        let (l, w) = (self.half_len, self.half_wid);
        let top = self.floor + self.wall_height();
        let lo = self.world(-l, -w);
        let hi = self.world(l, w);
        let (lo, hi) = (lo.min(hi), lo.max(hi));
        boxed(
            out,
            Vec3::new(lo.x, self.base, lo.y),
            Vec3::new(hi.x, top, hi.y),
            PLASTER,
        );
        // Roof: two sloped planes and two gable triangles.
        let (l, w) = (l + EAVE, w + EAVE);
        let ridge = top + w * PITCH;
        let p = |a: f32, b: f32, y: f32| {
            let q = self.world(a, b);
            Vec3::new(q.x, y, q.y)
        };
        for s in [-1.0f32, 1.0] {
            let quad = [
                p(-l, s * w, top - 0.3),
                p(l, s * w, top - 0.3),
                p(l, 0.0, ridge),
                p(-l, 0.0, ridge),
            ];
            polygon(out, &quad, ROOF);
            polygon(
                out,
                &[
                    p(s * (l - EAVE), -w + EAVE, top),
                    p(s * (l - EAVE), w - EAVE, top),
                    p(s * (l - EAVE), 0.0, ridge - 0.5),
                ],
                PLASTER,
            );
        }
    }
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

    /// Post-and-rail: posts every two metres and a rail at hip height.
    fn block(&self, p: Vec3, ground: f32) -> Option<Block> {
        if p.z < self.z0 || p.z > self.z1 {
            return None;
        }
        let &(fx, slope) = self.line.get(((p.z - self.z0) / VOXEL_SIZE) as usize)?;
        // Wide enough that neighbouring rows touch where the road bends.
        if (p.x - fx).abs() > 0.26 + slope * 0.6 {
            return None;
        }
        let dy = p.y - ground;
        let post = ((p.z / VOXEL_SIZE).floor() as i32).rem_euclid(4) == 0 && (p.x - fx).abs() <= 0.26;
        ((0.0..1.0).contains(&dy) && post || (1.0..1.5).contains(&dy)).then_some(WOOD)
    }
}

#[derive(Clone, Debug)]
pub enum Structure {
    Bridge(Bridge),
    Cottage(Cottage),
    Tower(Tower),
    Castle(Castle),
    Fence(Fence),
}

/// Everything built in the world, with the boxes (metres) each one occupies.
#[derive(Default)]
pub struct Structures {
    list: Vec<(Vec3, Vec3, Structure)>,
}

impl Structures {
    /// Lays out the hamlet around the spawn point and the landmarks up and down the valley.
    pub fn plan(t: &Terrain) -> Self {
        let mut s = Structures::default();
        let spawn = t.spawn_point();
        let seed = t.seed;
        // The player starts facing down the valley (towards -Z), so the hamlet lies ahead.
        let zb = spawn.z - 72.0;
        let bridge = Bridge::plan(t, zb);
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

        // The watchtower takes the highest knoll in view a few hundred metres out, the
        // castle a broad height further off, so both frame the view on arrival.
        let tower = best_site(
            t,
            spawn,
            (SPAWN_YAW - 0.6, SPAWN_YAW + 0.5),
            (130.0, 320.0),
            3.0,
            72.0,
            2.5,
            0.02,
        );
        if let Some(c) = tower {
            let c = Vec2::new(snap(c.x, 0.25), snap(c.y, 0.25));
            let (lo, hi) = ground_range(t, c - 3.0, c + 3.0);
            s.add(Structure::Tower(Tower {
                c,
                r: 2.6,
                base: lo - 1.0,
                top: hi + 11.0,
                room: 3.0,
                cone_r: 4.0,
                cone_h: 5.0,
                seed: seed ^ 20,
            }));
        }
        if let Some(c) = best_site(
            t,
            spawn,
            (SPAWN_YAW - 0.65, SPAWN_YAW + 0.35),
            (380.0, 700.0),
            20.0,
            82.0,
            16.0,
            0.03,
        ) {
            s.add(Structure::Castle(Castle::plan(t, c.x, c.y, seed ^ 40)));
        }
        s
    }

    fn add(&mut self, st: Structure) {
        let (lo, hi) = match &st {
            Structure::Bridge(b) => b.bounds(),
            Structure::Cottage(c) => c.bounds(),
            Structure::Tower(t) => t.bounds(),
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
                            Structure::Tower(t) => t.block(p),
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
                (z - b.z).abs() < 1.6 && ((x > b.x1 - 1.0 && x < road) || (x < b.x0 + 1.0 && x > b.x0 - 12.0))
            }
            _ => false,
        })
    }

    /// Adds the far-away shapes of every structure whose centre lies in the
    /// rectangle of metres [lo, hi).
    pub fn far(&self, out: &mut MeshData, lo: Vec2, hi: Vec2) {
        for (a, b, st) in &self.list {
            let c = Vec2::new(a.x + b.x, a.z + b.z) * 0.5;
            if matches!(st, Structure::Fence(_)) || c.x < lo.x || c.y < lo.y || c.x >= hi.x || c.y >= hi.y {
                continue;
            }
            match st {
                Structure::Bridge(b) => b.far(out),
                Structure::Cottage(c) => c.far(out),
                Structure::Tower(t) => t.far(out),
                Structure::Castle(c) => c.far(out),
                Structure::Fence(_) => {}
            }
        }
    }
}

/// Which way the player faces on arrival (`Player::new`), as a yaw in radians.
const SPAWN_YAW: f32 = -1.2;

/// The highest, most level site (centre in metres) within a wedge seen from
/// `from`: yaws and distances as ranges, a square of half-size `half` whose
/// ground varies by less than `rough` metres, below `max_h`, away from the
/// river and road. `far_penalty` trades height for nearness.
#[allow(clippy::too_many_arguments)]
fn best_site(
    t: &Terrain,
    from: Vec3,
    yaws: (f32, f32),
    dists: (f32, f32),
    half: f32,
    max_h: f32,
    rough: f32,
    far_penalty: f32,
) -> Option<Vec2> {
    let mut best: Option<(f32, Vec2)> = None;
    for j in 0..24 {
        for i in 0..24 {
            let d = dists.0 + (dists.1 - dists.0) * j as f32 / 23.0;
            let yaw = yaws.0 + (yaws.1 - yaws.0) * i as f32 / 23.0;
            let p = Vec2::new(from.x + d * yaw.cos(), from.z + d * yaw.sin());
            if (p.x - t.river_x(p.y)).abs() < half + 30.0 || t.road_distance(p.x, p.y) < half + 8.0 {
                continue;
            }
            let (lo, hi) = ground_range(t, p - half, p + half);
            let h = t.height_at(p.x, p.y).0;
            let score = h - (hi - lo) * 1.5 - d * far_penalty;
            if h < max_h && hi - lo < rough && best.is_none_or(|b| score > b.0) {
                best = Some((score, p));
            }
        }
    }
    best.map(|b| b.1)
}

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
fn boxed(out: &mut MeshData, lo: Vec3, hi: Vec3, mat: Block) {
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
        assert_eq!(count(|x| matches!(x, Structure::Tower(_))), 1);
        assert_eq!(count(|x| matches!(x, Structure::Castle(_))), 1);
    }

    #[test]
    fn the_bridge_spans_the_river_above_the_water() {
        let t = world();
        let Some(Structure::Bridge(b)) = t.structures.iter().find(|s| matches!(s, Structure::Bridge(_))) else {
            panic!("no bridge");
        };
        assert!(b.x0 < b.centre - 15.0 && b.x1 > b.centre + 15.0, "{b:?}");
        // The middle of the deck is stone and there is open water under the central arch.
        let deck = Vec3::new(b.centre + 0.25, b.deck - 0.25, b.z + 0.25);
        assert_eq!(t.structures.block_at(deck, 21.0), Some(MASONRY));
        let under = Vec3::new(b.centre + 0.25, WATER_LEVEL_M + 0.25, b.z + 0.25);
        assert_eq!(t.structures.block_at(under, 21.0), None);
    }

    #[test]
    fn cottages_have_lit_windows_and_a_roof() {
        let t = world();
        let c = t
            .structures
            .iter()
            .find_map(|s| match s {
                Structure::Cottage(c) => Some(c.clone()),
                _ => None,
            })
            .unwrap();
        let mut windows = 0;
        let mut roof = 0;
        let (lo, hi) = c.bounds();
        let mut p = lo + 0.25;
        while p.y < hi.y {
            p.z = lo.z + 0.25;
            while p.z < hi.z {
                p.x = lo.x + 0.25;
                while p.x < hi.x {
                    match c.block(p, c.base) {
                        Some(WINDOW) => windows += 1,
                        Some(ROOF) => roof += 1,
                        _ => {}
                    }
                    p.x += VOXEL_SIZE;
                }
                p.z += VOXEL_SIZE;
            }
            p.y += VOXEL_SIZE;
        }
        assert!(windows >= 8, "{windows} window voxels");
        assert!(roof > 100, "{roof} roof voxels");
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
