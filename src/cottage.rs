//! Timber-framed cottages you can walk into. Each is an authored model: a
//! coursed stone plinth, an oak frame with braces, plaster panels, framed
//! windows with shutters, plank doors hung open, rafters under rows of slate,
//! a chimney, board floors, stairs to the upper floor, furniture and hanging
//! lanterns. The voxels underneath only give it collision (`BUILT`, which is
//! never drawn) and carry the lanterns that light it.

use crate::block::*;
use crate::mesh::MeshData;
use crate::model::{beam, block, board, panel};
use crate::noise::{hash2, unit};
use crate::terrain::{Terrain, WATER_LEVEL_M};
use glam::{Vec2, Vec3};

/// Roofs rise at 45 degrees.
pub const PITCH: f32 = 1.0;
/// How far the roof overhangs the walls.
pub const EAVE: f32 = 0.5;
/// Wall thickness, equal to one voxel so collision matches the model.
const WALL: f32 = 0.5;
/// Height from the ground floor to the underside of the upper floor.
const STOREY: f32 = 3.0;
/// Thickness of the upper floor.
const FLOOR_T: f32 = 0.5;
/// Door openings: half width and height.
const DOOR_HALF_W: f32 = 0.5;
const DOOR_H: f32 = 2.5;
/// Half thickness of the main framing timbers.
const TIMBER: f32 = 0.11;
/// The roof's outer surface sits this far above the wall plate line.
const ROOF_LIFT: f32 = 0.35;
/// Collision thickness under the roof surface.
const ROOF_SOLID: f32 = 0.8;
/// Stairs: steps of one voxel rise and one voxel run.
const STEPS: i32 = 7;

#[derive(Clone, Debug)]
pub struct Cottage {
    pub c: Vec2,
    /// Whether the ridge runs along X (otherwise along Z).
    pub along_x: bool,
    pub half_len: f32,
    pub half_wid: f32,
    /// Height of the ground floor's walking surface.
    pub floor: f32,
    pub storeys: u8,
    /// Which long side has the front door: +1 or -1 along the across axis.
    pub front: f32,
    /// Metres of plank deck on stilts behind the back wall; 0 for none.
    pub deck: f32,
    /// Lowest ground under the cottage and its deck, where footings start.
    pub base: f32,
    pub seed: u32,
}

/// Snaps a coordinate so a span of `2 * half` metres covers whole voxels.
pub fn snap(x: f32, half: f32) -> f32 {
    let voxels = (2.0 * half / VOXEL_SIZE).round() as i32;
    let base = (x / VOXEL_SIZE).round() * VOXEL_SIZE;
    if voxels % 2 == 0 {
        base
    } else {
        base + VOXEL_SIZE * 0.5
    }
}

/// Whether `p` lies in the voxel whose centre is nearest `target`.
fn in_voxel(p: Vec3, target: Vec3) -> bool {
    let v = (target / VOXEL_SIZE).floor();
    (p / VOXEL_SIZE).floor() == v
}

impl Cottage {
    #[allow(clippy::too_many_arguments)]
    pub fn plan(t: &Terrain, x: f32, z: f32, along_x: bool, storeys: u8, front: f32, deck: f32, seed: u32) -> Self {
        let (half_len, half_wid) = if storeys > 1 { (4.0, 3.25) } else { (4.0, 2.75) };
        let (hx, hz) = if along_x {
            (half_len, half_wid)
        } else {
            (half_wid, half_len)
        };
        let c = Vec2::new(snap(x, hx), snap(z, hz));
        let lo_hi = |e: Vec2| {
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            let mut z = c.y - e.y;
            while z <= c.y + e.y {
                let mut x = c.x - e.x;
                while x <= c.x + e.x {
                    let h = t.height_at(x, z).0;
                    lo = lo.min(h);
                    hi = hi.max(h);
                    x += 1.0;
                }
                z += 1.0;
            }
            (lo, hi)
        };
        let (lo, hi) = lo_hi(Vec2::new(hx, hz));
        // Floors sit a step above the ground, never below the flood line.
        let floor = ((hi.min(lo + 2.0) + 0.5).max(WATER_LEVEL_M + 1.5) / VOXEL_SIZE).round() * VOXEL_SIZE;
        let reach = Vec2::new(hx, hz)
            + if along_x {
                Vec2::new(0.5, deck)
            } else {
                Vec2::new(deck, 0.5)
            };
        let (deck_lo, _) = lo_hi(reach);
        Cottage {
            c,
            along_x,
            half_len,
            half_wid,
            floor,
            storeys,
            front,
            deck,
            base: lo.min(deck_lo),
            seed,
        }
    }

    pub fn wall_height(&self) -> f32 {
        if self.storeys > 1 {
            5.5
        } else {
            3.0
        }
    }

    fn top(&self) -> f32 {
        self.floor + self.wall_height()
    }

    /// Height of the roof's outer surface at `b` across.
    fn roof_y(&self, b: f32) -> f32 {
        self.top() + (self.half_wid - b.abs()) * PITCH + ROOF_LIFT
    }

    pub fn ridge(&self) -> f32 {
        self.roof_y(0.0)
    }

    /// Converts world X/Z to (along, across) around the centre.
    pub fn local(&self, x: f32, z: f32) -> (f32, f32) {
        let d = Vec2::new(x, z) - self.c;
        if self.along_x {
            (d.x, d.y)
        } else {
            (d.y, d.x)
        }
    }

    pub fn world(&self, a: f32, b: f32) -> Vec2 {
        if self.along_x {
            self.c + Vec2::new(a, b)
        } else {
            self.c + Vec2::new(b, a)
        }
    }

    /// A point in world space from local (along, height, across).
    fn p(&self, a: f32, y: f32, b: f32) -> Vec3 {
        let q = self.world(a, b);
        Vec3::new(q.x, y, q.y)
    }

    fn along(&self) -> Vec3 {
        if self.along_x {
            Vec3::X
        } else {
            Vec3::Z
        }
    }

    fn across(&self) -> Vec3 {
        if self.along_x {
            Vec3::Z
        } else {
            Vec3::X
        }
    }

    /// Outline of the land the cottage claims, deck and yard included.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let reach_b = self.half_wid + 1.5 + self.deck;
        let reach_a = self.half_len + 1.5;
        let (ex, ez) = if self.along_x {
            (reach_a, reach_b)
        } else {
            (reach_b, reach_a)
        };
        (
            Vec3::new(self.c.x - ex, self.base - 1.5, self.c.y - ez),
            Vec3::new(self.c.x + ex, self.ridge() + 2.5, self.c.y + ez),
        )
    }

    /// Door openings: the front door in the middle of the front wall and, with
    /// a deck, a back door onto it.
    fn doors(&self) -> Vec<(f32, f32)> {
        let mut d = vec![(0.0, self.front)];
        if self.deck > 0.0 {
            d.push((1.5, -self.front));
        }
        d
    }

    fn in_door(&self, a: f32, b: f32, dy: f32) -> bool {
        dy < DOOR_H
            && self
                .doors()
                .iter()
                .any(|&(da, side)| b * side > 0.0 && (a - da).abs() < DOOR_HALF_W)
    }

    /// Top of the entry step under `(a, b)` outside the front door, if any.
    /// Steps drop half a metre each going out from the wall.
    fn step_top(&self, a: f32, b: f32) -> Option<f32> {
        let out = b * self.front - self.half_wid;
        if out < 0.0 || a.abs() >= DOOR_HALF_W + 0.25 {
            return None;
        }
        let k = (out / VOXEL_SIZE).floor();
        Some(self.floor - (k + 1.0) * VOXEL_SIZE).filter(|top| *top > self.base - 0.25)
    }

    /// Where the stairs start along the house, and the across range (towards
    /// the back wall) of the stair band.
    fn stair_band(&self, b: f32) -> bool {
        let bb = -b * self.front;
        let inner = self.half_wid - WALL;
        self.storeys > 1 && bb > inner - 1.0 && bb <= inner
    }

    fn stair_start(&self) -> f32 {
        -self.half_len + WALL
    }

    /// Voxel centres of the lanterns that light the cottage.
    fn lanterns(&self) -> Vec<Vec3> {
        let snap_v = |v: Vec3| ((v / VOXEL_SIZE).floor() + 0.5) * VOXEL_SIZE;
        let f = self.floor;
        let mut out = vec![snap_v(self.p(0.25, f + 2.75, 0.25))];
        if self.storeys > 1 {
            out.push(snap_v(self.p(0.25, f + 5.75, 0.25)));
        }
        // Beside the front door.
        out.push(snap_v(self.p(1.0, f + 2.25, self.front * (self.half_wid + 0.25))));
        if self.deck > 0.0 {
            let (l, w) = (self.half_len, self.half_wid);
            out.push(snap_v(self.p(l + 0.25, f + 1.75, -self.front * (w + self.deck - 0.25))));
        }
        out
    }

    /// Collision and light voxels, and the air the cottage clears around itself.
    pub fn block(&self, p: Vec3, ground: f32) -> Option<Block> {
        let (a, b) = self.local(p.x, p.z);
        let (l, w, h) = (self.half_len, self.half_wid, self.wall_height());
        let dy = p.y - self.floor;
        if self.lanterns().iter().any(|&v| in_voxel(p, v)) {
            return Some(LANTERN);
        }
        let inside = a.abs() < l && b.abs() < w;
        if inside {
            if dy < 0.0 {
                return (p.y >= self.base - 1.0).then_some(BUILT);
            }
            let shell = a.abs() > l - WALL || b.abs() > w - WALL;
            if dy < h {
                if shell {
                    return Some(if self.in_door(a, b, dy) { AIR } else { BUILT });
                }
                if self.storeys > 1 {
                    let a0 = self.stair_start();
                    let on_stairs = self.stair_band(b) && a >= a0 && a < a0 + STEPS as f32 * VOXEL_SIZE;
                    if on_stairs {
                        let k = ((a - a0) / VOXEL_SIZE).floor();
                        if dy < (k + 1.0) * VOXEL_SIZE {
                            return Some(BUILT);
                        }
                    }
                    if (STOREY..STOREY + FLOOR_T).contains(&dy) {
                        let opening = self.stair_band(b) && a >= a0 + VOXEL_SIZE && a < a0 + STEPS as f32 * VOXEL_SIZE;
                        return Some(if opening { AIR } else { BUILT });
                    }
                    // A rail along the stairwell upstairs.
                    let bb = -b * self.front;
                    let rail_line = bb > w - WALL - 1.5 && bb <= w - WALL - 1.0;
                    if rail_line
                        && a >= a0
                        && a < a0 + STEPS as f32 * VOXEL_SIZE
                        && (STOREY + FLOOR_T..STOREY + FLOOR_T + 1.0).contains(&dy)
                    {
                        return Some(BUILT);
                    }
                }
                return Some(AIR);
            }
        }
        if let Some(top) = self.step_top(a, b) {
            if p.y < top && p.y + VOXEL_SIZE > ground {
                return Some(BUILT);
            }
        }
        // Roof and gables.
        if a.abs() <= l + EAVE && b.abs() <= w + EAVE && dy >= h - EAVE * PITCH - 0.5 {
            let ry = self.roof_y(b);
            if p.y < ry && p.y >= ry - ROOF_SOLID {
                return Some(BUILT);
            }
            if p.y < ry && a.abs() < l && b.abs() < w && dy >= h {
                return Some(if a.abs() > l - WALL { BUILT } else { AIR });
            }
        }
        // The deck: boards, and a rail along its open sides.
        let bb = -b * self.front;
        if self.deck > 0.0 && bb >= w && bb < w + self.deck && a.abs() < l + 0.5 {
            if (-VOXEL_SIZE..0.0).contains(&dy) {
                return Some(BUILT);
            }
            let outer = bb >= w + self.deck - VOXEL_SIZE || a.abs() >= l;
            if outer && (0.0..1.0).contains(&dy) {
                return Some(BUILT);
            }
            if (0.0..3.0).contains(&dy) {
                return Some(AIR);
            }
            let _ = ground;
            return None;
        }
        // Clear a small yard of grass, trees and ground above the floor.
        let yard = a.abs() <= l + 1.5 && b.abs() <= w + 1.5;
        (yard && dy >= 0.0 && p.y < self.ridge() + 2.0).then_some(AIR)
    }

    /// The cottage as an authored model, in world space.
    pub fn model(&self, t: &Terrain, out: &mut MeshData) {
        self.plinth(t, out);
        self.floors(out);
        for side in [-1.0f32, 1.0] {
            self.long_wall(out, side);
            self.end_wall(out, side);
        }
        self.roof(out);
        self.chimney(out);
        self.stairs(out);
        self.furniture(out);
        self.fittings(out);
        if self.deck > 0.0 {
            self.deck_model(t, out);
        }
    }

    /// Just the outside: footings, walls, roof, chimney and deck, for the far field.
    pub fn exterior(&self, t: &Terrain, out: &mut MeshData) {
        self.plinth(t, out);
        for side in [-1.0f32, 1.0] {
            self.long_wall(out, side);
            self.end_wall(out, side);
        }
        self.roof(out);
        self.chimney(out);
        if self.deck > 0.0 {
            self.deck_model(t, out);
        }
    }

    /// A light stand-in for far away: body, timber bands, roof and lit windows.
    pub fn far(&self, out: &mut MeshData) {
        let (l, w) = (self.half_len, self.half_wid);
        let top = self.top();
        block(out, self.p(-l, self.base, -w), self.p(l, top, w), PLASTER);
        for (y0, y1) in [(self.floor, self.floor + 0.25), (top - 0.25, top)] {
            block(
                out,
                self.p(-l - 0.05, y0, -w - 0.05),
                self.p(l + 0.05, y1, w + 0.05),
                OAK,
            );
        }
        let (le, we) = (l + EAVE, w + EAVE);
        let ridge = self.ridge();
        let eave = self.roof_y(we);
        for s in [-1.0f32, 1.0] {
            panel(
                out,
                &[
                    self.p(-le, eave, s * we),
                    self.p(le, eave, s * we),
                    self.p(le, ridge, 0.0),
                    self.p(-le, ridge, 0.0),
                ],
                ROOF,
            );
            panel(
                out,
                &[
                    self.p(s * l, top, -w),
                    self.p(s * l, top, w),
                    self.p(s * l, ridge - ROOF_LIFT, 0.0),
                ],
                PLASTER,
            );
        }
        // One lit window per wall bay and storey, matching the near model roughly.
        let storeys: &[f32] = if self.storeys > 1 { &[1.3, 4.3] } else { &[1.3] };
        for &dy in storeys {
            for side in [-1.0f32, 1.0] {
                let studs = Self::studs(l);
                for pair in studs.windows(2) {
                    let t = (pair[0] + pair[1]) * 0.5;
                    if dy < 2.0 && (t - 0.0).abs() < 0.8 && side == self.front {
                        continue;
                    }
                    let q = |dt: f32, y: f32| self.p(t + dt, self.floor + y, side * (w + 0.03));
                    panel(
                        out,
                        &[
                            q(-0.45, dy - 0.5),
                            q(0.45, dy - 0.5),
                            q(0.45, dy + 0.5),
                            q(-0.45, dy + 0.5),
                        ],
                        WINDOW,
                    );
                }
            }
        }
    }

    fn rnd(&self, i: i32, j: i32) -> f32 {
        unit(hash2(self.seed, i, j))
    }

    /// Coursed stone footings from the ground up to the floor.
    fn plinth(&self, t: &Terrain, out: &mut MeshData) {
        let (l, w) = (self.half_len + 0.08, self.half_wid + 0.08);
        let course = 0.42;
        // Walk the perimeter, one side at a time: (start a, start b, direction, length).
        let sides = [
            (Vec2::new(-l, -w), Vec2::X, 2.0 * l),
            (Vec2::new(l, -w), Vec2::Y, 2.0 * w),
            (Vec2::new(l, w), Vec2::NEG_X, 2.0 * l),
            (Vec2::new(-l, w), Vec2::NEG_Y, 2.0 * w),
        ];
        for (si, (start, dir, len)) in sides.into_iter().enumerate() {
            let normal = Vec2::new(dir.y, -dir.x);
            let mut row = 0;
            let mut y1 = self.floor - 0.02;
            while row < 12 {
                let y0 = y1 - course;
                let mut s = if row % 2 == 1 { -0.35 } else { 0.0 };
                let mut k = 0;
                let mut any = false;
                while s < len {
                    let stone = 0.55 + 0.45 * self.rnd(si as i32 * 1000 + row * 50 + k, 1);
                    let (s0, s1) = (s.max(0.0), (s + stone).min(len));
                    let mid = start + dir * ((s0 + s1) * 0.5);
                    let q = self.world(mid.x, mid.y);
                    let g = t.height_at(q.x, q.y).0;
                    if y1 > g - 0.3 && s1 - s0 > 0.1 {
                        any = true;
                        let jut = 0.02 + 0.05 * self.rnd(si as i32 * 1000 + row * 50 + k, 2);
                        let centre = start + dir * ((s0 + s1) * 0.5) - normal * (0.2 - jut);
                        let along = self.along() * dir.x + self.across() * dir.y;
                        let outward = self.along() * normal.x + self.across() * normal.y;
                        board(
                            out,
                            self.p(centre.x, (y0 + y1) * 0.5, centre.y),
                            along,
                            Vec3::Y,
                            Vec3::new((s1 - s0) * 0.5 - 0.025, course * 0.5 - 0.025, 0.22),
                            MASONRY,
                        );
                        let _ = outward;
                    }
                    s += stone;
                    k += 1;
                }
                if !any {
                    break;
                }
                y1 = y0;
                row += 1;
            }
        }
    }

    fn floors(&self, out: &mut MeshData) {
        let (l, w) = (self.half_len - WALL + 0.02, self.half_wid - WALL + 0.02);
        let f = self.floor;
        // Ground floor boards over the plinth.
        block(out, self.p(-l, f - 0.08, -w), self.p(l, f, w), BOARDS);
        if self.storeys > 1 {
            let y = f + STOREY;
            let a0 = self.stair_start();
            let open_hi = a0 + STEPS as f32 * VOXEL_SIZE;
            let inner = self.half_wid - WALL;
            // Joists across the house, visible from below.
            let mut a = -l + 0.3;
            while a < l {
                beam(out, self.p(a, y + 0.12, -w), self.p(a, y + 0.12, w), 0.1, OAK);
                a += 0.8;
            }
            // Boards, leaving the stairwell open: the part beside the stairwell,
            // then the rest of the floor past its end.
            let (b_well0, b_well1) = (-self.front * inner, -self.front * (inner - 1.0));
            let (bl, bh) = (b_well0.min(b_well1), b_well0.max(b_well1));
            let top = y + FLOOR_T;
            let slab = |a0: f32, a1: f32, b0: f32, b1: f32, out: &mut MeshData| {
                block(out, self.p(a0, top - 0.1, b0), self.p(a1, top, b1), BOARDS);
            };
            slab(-l, a0 + VOXEL_SIZE, -w, w, out);
            slab(open_hi, l, -w, w, out);
            if self.front > 0.0 {
                slab(a0 + VOXEL_SIZE, open_hi, -w, bl, out);
            } else {
                slab(a0 + VOXEL_SIZE, open_hi, bh, w, out);
            }
            // Rail round the stairwell.
            let rb = -self.front * (inner - 1.0 - 0.25);
            let mut a = a0 + 0.1;
            while a <= open_hi {
                beam(out, self.p(a, top, rb), self.p(a, top + 0.95, rb), 0.04, OAK);
                a += 0.6;
            }
            beam(
                out,
                self.p(a0, top + 0.95, rb),
                self.p(open_hi, top + 0.95, rb),
                0.05,
                OAK,
            );
        }
    }

    /// Positions of the studs along a wall of half-length `half`.
    fn studs(half: f32) -> Vec<f32> {
        let n = (2.0 * half / 1.6).ceil().max(2.0) as i32;
        (0..=n).map(|i| -half + 2.0 * half * i as f32 / n as f32).collect()
    }

    /// One long wall (side = +1 or -1 across): frame, braces, plaster panels,
    /// windows and doors.
    fn long_wall(&self, out: &mut MeshData, side: f32) {
        let (l, w) = (self.half_len, self.half_wid);
        let b_out = side * w;
        let b_mid = side * (w - WALL * 0.5);
        self.wall(out, l - 0.12, side, |a, y| self.p(a, y, b_mid), b_out, true);
    }

    fn end_wall(&self, out: &mut MeshData, side: f32) {
        let (l, w) = (self.half_len, self.half_wid);
        let a_mid = side * (l - WALL * 0.5);
        // Local coordinates along the end wall are `b`.
        self.wall(out, w - 0.12, side, |b, y| self.p(a_mid, y, b), side * l, false);
        // Gable above the wall plate: plaster both sides, king post and collar.
        let top = self.top();
        let ridge = self.ridge() - ROOF_LIFT - 0.05;
        for depth in [side * (l - 0.02), side * (l - WALL + 0.02)] {
            panel(
                out,
                &[
                    self.p(depth, top, -w + 0.02),
                    self.p(depth, top, w - 0.02),
                    self.p(depth, ridge, 0.0),
                ],
                PLASTER,
            );
        }
        let a_out = side * (l + 0.02);
        beam(out, self.p(a_out, top, 0.0), self.p(a_out, ridge, 0.0), TIMBER, OAK);
        let collar = top + (ridge - top) * 0.45;
        let cw = w - (collar - top) / PITCH;
        beam(out, self.p(a_out, collar, -cw), self.p(a_out, collar, cw), TIMBER, OAK);
        if self.storeys > 1 {
            // A small lit window in the gable.
            let y = top + 0.4;
            let q = |b: f32, y: f32| self.p(side * (l + 0.03), y, b);
            panel(
                out,
                &[q(-0.35, y), q(0.35, y), q(0.35, y + 0.8), q(-0.35, y + 0.8)],
                WINDOW,
            );
            for (b0, y0, b1, y1) in [
                (-0.4, y, 0.4, y),
                (-0.4, y + 0.85, 0.4, y + 0.85),
                (-0.4, y, -0.4, y + 0.85),
                (0.4, y, 0.4, y + 0.85),
            ] {
                beam(out, q(b0, y0), q(b1, y1), 0.05, OAK);
            }
        }
    }

    /// A timber-framed wall. `at(t, y)` places a point on the wall's centre
    /// plane at `t` along it; `outer` is the local coordinate of the outside
    /// face and `long` whether this is a long wall (so doors apply).
    fn wall(&self, out: &mut MeshData, half: f32, side: f32, at: impl Fn(f32, f32) -> Vec3, outer: f32, long: bool) {
        let f = self.floor;
        let h = self.wall_height();
        let depth = WALL * 0.5 + 0.03;
        // Outward unit vector and along-wall unit vector in world space.
        let (t_dir, n_dir) = if long {
            (self.along(), self.across() * side)
        } else {
            (self.across(), self.along() * side)
        };
        let _ = outer;
        let studs = Self::studs(half);
        // Horizontal members: sill, (mid rail), wall plate.
        let mut rails = vec![f + TIMBER, f + h - TIMBER];
        if self.storeys > 1 {
            rails.push(f + STOREY + FLOOR_T * 0.5);
        }
        for &y in &rails {
            board(
                out,
                at(0.0, y),
                t_dir,
                n_dir,
                Vec3::new(half + 0.12, depth, TIMBER),
                OAK,
            );
        }
        for &t in &studs {
            // Leave door openings clear of studs.
            let blocked = long
                && self
                    .doors()
                    .iter()
                    .any(|&(da, s)| s == side && (t - da).abs() < DOOR_HALF_W + 0.05);
            if !blocked {
                board(
                    out,
                    at(t, f + h * 0.5),
                    Vec3::Y,
                    n_dir,
                    Vec3::new(h * 0.5, depth, TIMBER),
                    OAK,
                );
            }
        }
        // Storeys: (bottom, top) of each panel band between rails.
        let mut bands = vec![(f + 2.0 * TIMBER, f + h - 2.0 * TIMBER)];
        if self.storeys > 1 {
            let mid = f + STOREY + FLOOR_T * 0.5;
            bands = vec![(f + 2.0 * TIMBER, mid - TIMBER), (mid + TIMBER, f + h - 2.0 * TIMBER)];
        }
        for (bi, pair) in studs.windows(2).enumerate() {
            let (t0, t1) = (pair[0] + TIMBER, pair[1] - TIMBER);
            let tm = (t0 + t1) * 0.5;
            for (si, &(y0, y1)) in bands.iter().enumerate() {
                let door = long
                    && si == 0
                    && self
                        .doors()
                        .iter()
                        .any(|&(da, s)| s == side && da > t0 - 0.6 && da < t1 + 0.6);
                let window = !door
                    && t1 - t0 > 1.0
                    && self.rnd(bi as i32 + (side * 7.0) as i32 + if long { 0 } else { 40 }, si as i32) < 0.8;
                let fill = |ya: f32, yb: f32, ta: f32, tb: f32, out: &mut MeshData| {
                    if yb - ya > 0.02 && tb - ta > 0.02 {
                        board(
                            out,
                            at((ta + tb) * 0.5, (ya + yb) * 0.5),
                            t_dir,
                            n_dir,
                            Vec3::new((tb - ta) * 0.5, WALL * 0.5, (yb - ya) * 0.5),
                            PLASTER,
                        );
                    }
                };
                if door {
                    let da = self
                        .doors()
                        .into_iter()
                        .find(|&(da, s)| s == side && da > t0 - 0.6 && da < t1 + 0.6)
                        .unwrap()
                        .0;
                    let (d0, d1) = (da - DOOR_HALF_W, da + DOOR_HALF_W);
                    fill(f + DOOR_H + 0.12, y1, t0, t1, out);
                    fill(y0, f + DOOR_H + 0.12, t0, d0 - 0.08, out);
                    fill(y0, f + DOOR_H + 0.12, d1 + 0.08, t1, out);
                    self.door(out, &at, t_dir, n_dir, da, side);
                    continue;
                }
                if window {
                    let (wy0, wy1) = (y0 + 0.75, (y0 + 1.85).min(y1 - 0.3));
                    let hw = ((t1 - t0) * 0.5 - 0.25).min(0.5);
                    let (w0, w1) = (tm - hw, tm + hw);
                    fill(y0, wy0, t0, t1, out);
                    fill(wy1, y1, t0, t1, out);
                    fill(wy0, wy1, t0, w0, out);
                    fill(wy0, wy1, w1, t1, out);
                    self.window(out, &at, t_dir, n_dir, w0, w1, wy0, wy1, bi as i32 * 3 + si as i32);
                } else {
                    fill(y0, y1, t0, t1, out);
                    // A brace across plain bays at the ends of the wall.
                    if bi == 0 || bi == studs.len() - 2 {
                        let (ta, tb) = if bi == 0 { (t0, t1) } else { (t1, t0) };
                        let off = n_dir * (WALL * 0.5 + 0.02);
                        beam(out, at(ta, y0) + off, at(tb, y1) + off, 0.07, OAK);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn window(
        &self,
        out: &mut MeshData,
        at: &impl Fn(f32, f32) -> Vec3,
        t_dir: Vec3,
        n_dir: Vec3,
        w0: f32,
        w1: f32,
        y0: f32,
        y1: f32,
        k: i32,
    ) {
        // Glass set back in the middle of the wall; frame, mullion and transom.
        let pane = |t: f32, y: f32| at(t, y);
        panel(out, &[pane(w0, y0), pane(w1, y0), pane(w1, y1), pane(w0, y1)], WINDOW);
        let depth = WALL * 0.5 + 0.01;
        for (ta, ya, tb, yb) in [(w0, y0, w1, y0), (w0, y1, w1, y1), (w0, y0, w0, y1), (w1, y0, w1, y1)] {
            let c = (at(ta, ya) + at(tb, yb)) * 0.5;
            let along = if ya == yb { t_dir } else { Vec3::Y };
            let len = if ya == yb {
                (tb - ta) * 0.5 + 0.06
            } else {
                (yb - ya) * 0.5
            };
            board(out, c, along, n_dir, Vec3::new(len, depth, 0.06), OAK);
        }
        let tm = (w0 + w1) * 0.5;
        let ym = (y0 + y1) * 0.5;
        board(
            out,
            at(tm, ym),
            Vec3::Y,
            n_dir,
            Vec3::new((y1 - y0) * 0.5, 0.04, 0.025),
            OAK,
        );
        board(
            out,
            at(tm, ym),
            t_dir,
            n_dir,
            Vec3::new((w1 - w0) * 0.5, 0.04, 0.025),
            OAK,
        );
        // Sill board jutting out, and shutters swung open against the wall.
        board(
            out,
            at(tm, y0 - 0.05) + n_dir * 0.1,
            t_dir,
            Vec3::Y,
            Vec3::new((w1 - w0) * 0.5 + 0.12, 0.035, WALL * 0.5 + 0.08),
            BOARDS,
        );
        if self.rnd(k, 9) < 0.7 {
            let sw = (w1 - w0) * 0.5;
            for (edge, dir) in [(w0, -1.0f32), (w1, 1.0)] {
                let c = at(edge + dir * (sw * 0.5 + 0.08), (y0 + y1) * 0.5) + n_dir * (WALL * 0.5 + 0.06);
                board(
                    out,
                    c,
                    Vec3::Y,
                    n_dir,
                    Vec3::new((y1 - y0) * 0.5, 0.025, sw * 0.5),
                    BOARDS,
                );
            }
        }
    }

    /// A plank door hung open inwards, with its frame, lintel and a step.
    fn door(&self, out: &mut MeshData, at: &impl Fn(f32, f32) -> Vec3, t_dir: Vec3, n_dir: Vec3, da: f32, side: f32) {
        let f = self.floor;
        let (d0, d1) = (da - DOOR_HALF_W, da + DOOR_HALF_W);
        let depth = WALL * 0.5 + 0.03;
        for t in [d0 - 0.06, d1 + 0.06] {
            board(
                out,
                at(t, f + DOOR_H * 0.5),
                Vec3::Y,
                n_dir,
                Vec3::new(DOOR_H * 0.5, depth, 0.07),
                OAK,
            );
        }
        board(
            out,
            at(da, f + DOOR_H + 0.08),
            t_dir,
            n_dir,
            Vec3::new(DOOR_HALF_W + 0.2, depth, 0.1),
            OAK,
        );
        // The leaf, hinged at d0 and swung about 100 degrees into the room.
        let hinge = at(d0, f + 0.02) - n_dir * (WALL * 0.5);
        let open = (-n_dir * 0.98 + t_dir * -0.17).normalize();
        let leaf_c = hinge + open * (DOOR_HALF_W - 0.02) + Vec3::Y * (DOOR_H - 0.1) * 0.5;
        board(
            out,
            leaf_c,
            open,
            open.cross(Vec3::Y),
            Vec3::new(DOOR_HALF_W - 0.03, 0.035, (DOOR_H - 0.1) * 0.5),
            BOARDS,
        );
        // Ledges across the back of the leaf.
        for y in [0.35, DOOR_H - 0.5] {
            let c = hinge + open * (DOOR_HALF_W - 0.02) + Vec3::Y * y + open.cross(Vec3::Y) * 0.05;
            board(out, c, open, Vec3::Y, Vec3::new(DOOR_HALF_W - 0.08, 0.06, 0.02), BOARDS);
        }
        // Worn stone steps down to the ground, but not onto a deck.
        if side == self.front {
            let mut k = 0.0;
            while f - (k + 1.0) * VOXEL_SIZE > self.base - 0.25 {
                let top = f - (k + 1.0) * VOXEL_SIZE;
                let o = WALL * 0.5 + k * VOXEL_SIZE;
                block(
                    out,
                    at(d0 - 0.2, self.base - 0.5) + n_dir * o,
                    at(d1 + 0.2, top) + n_dir * (o + VOXEL_SIZE),
                    MASONRY,
                );
                k += 1.0;
            }
        }
    }

    fn roof(&self, out: &mut MeshData) {
        let (l, w) = (self.half_len + EAVE, self.half_wid + EAVE);
        let ridge = self.ridge();
        let slope_len = (w * w + (w * PITCH).powi(2)).sqrt();
        for side in [-1.0f32, 1.0] {
            // Down-slope direction and outward normal of this roof plane.
            let down = (self.across() * side - Vec3::Y * PITCH).normalize();
            let normal = (self.across() * side + Vec3::Y / PITCH).normalize();
            // Rafters under the tiles.
            let mut a = -self.half_len + 0.15;
            while a <= self.half_len {
                let top = self.p(a, ridge - 0.3, 0.0);
                beam(out, top, top + down * (slope_len - 0.05), 0.07, OAK);
                a += 0.8;
            }
            // Boards lining the underside, seen from inside.
            let under = -normal * 0.2;
            panel(
                out,
                &[
                    self.p(-l, ridge, 0.0) + under,
                    self.p(l, ridge, 0.0) + under,
                    self.p(l, ridge, 0.0) + under + down * slope_len,
                    self.p(-l, ridge, 0.0) + under + down * slope_len,
                ],
                BOARDS,
            );
            // Rows of slate, each lapping over the one below and tipped a little
            // steeper so its lower edge kicks out.
            let row = 0.34;
            let n_rows = (slope_len / row).ceil() as i32;
            let tilt = (down - normal * 0.12).normalize();
            let tnorm = tilt.cross(self.along()).normalize() * if side > 0.0 { -1.0 } else { 1.0 };
            let tnorm = if tnorm.y < 0.0 { -tnorm } else { tnorm };
            for k in 0..n_rows {
                let s = (k as f32 + 0.5) * row;
                let c = self.p(0.0, ridge, 0.0) + down * s + normal * 0.03;
                let jitter = (self.rnd(k, side as i32 + 5) - 0.5) * 0.04;
                board(
                    out,
                    c + tnorm * jitter.abs(),
                    self.along(),
                    tnorm,
                    Vec3::new(l + 0.05, 0.03, row * 0.62),
                    ROOF,
                );
            }
            // Bargeboards along the rakes at both gable ends.
            for end in [-1.0f32, 1.0] {
                let a = end * (l + 0.02);
                let top = self.p(a, ridge + 0.02, 0.0);
                beam(
                    out,
                    top - normal * 0.08,
                    top + down * slope_len - normal * 0.08,
                    0.06,
                    OAK,
                );
            }
        }
        // Ridge cap.
        beam(
            out,
            self.p(-l - 0.05, ridge + 0.04, 0.0),
            self.p(l + 0.05, ridge + 0.04, 0.0),
            0.13,
            ROOF,
        );
    }

    fn chimney(&self, out: &mut MeshData) {
        let (l, w) = (self.half_len, self.half_wid);
        let a = l - WALL - 0.55;
        let b = -self.front * (w - 1.3);
        let f = self.floor;
        let top = self.roof_y(b) + 1.2;
        // Hearth and chimney breast inside, the stack through the roof.
        block(
            out,
            self.p(a - 0.55, f, b - 0.75),
            self.p(a + 0.55, f + 0.12, b + 0.75),
            MASONRY,
        );
        let mut y = f + 0.12;
        let mut k = 0;
        while y < top {
            let course = 0.4;
            let (hw, hd) = if y < f + 1.2 { (0.5, 0.7) } else { (0.4, 0.45) };
            let nudge = (self.rnd(k, 31) - 0.5) * 0.03;
            block(
                out,
                self.p(a - hw + nudge, y, b - hd),
                self.p(a + hw + nudge, (y + course - 0.02).min(top), b + hd),
                MASONRY,
            );
            y += course;
            k += 1;
        }
        block(
            out,
            self.p(a - 0.48, top, b - 0.5),
            self.p(a + 0.48, top + 0.12, b + 0.5),
            MASONRY,
        );
    }

    fn stairs(&self, out: &mut MeshData) {
        if self.storeys < 2 {
            return;
        }
        let f = self.floor;
        let a0 = self.stair_start();
        let inner = self.half_wid - WALL;
        let (b0, b1) = (-self.front * inner, -self.front * (inner - 1.0));
        for k in 0..STEPS {
            let a = a0 + k as f32 * VOXEL_SIZE;
            let y = f + (k + 1) as f32 * VOXEL_SIZE;
            block(
                out,
                self.p(a, y - 0.07, b0),
                self.p(a + VOXEL_SIZE + 0.04, y, b1),
                BOARDS,
            );
            block(
                out,
                self.p(a + 0.02, y - VOXEL_SIZE, b0),
                self.p(a + 0.05, y - 0.07, b1),
                BOARDS,
            );
        }
        // Stringer along the open side and a handrail.
        let bs = b1 + (b1 - b0).signum() * 0.05;
        let (lo, hi) = (
            self.p(a0, f, bs),
            self.p(a0 + STEPS as f32 * VOXEL_SIZE, f + STEPS as f32 * VOXEL_SIZE, bs),
        );
        beam(out, lo + Vec3::Y * 0.2, hi + Vec3::Y * 0.2, 0.09, OAK);
        beam(out, lo + Vec3::Y * 1.1, hi + Vec3::Y * 1.1, 0.04, OAK);
        beam(out, lo, lo + Vec3::Y * 1.15, 0.05, OAK);
    }

    fn furniture(&self, out: &mut MeshData) {
        let f = self.floor;
        let (l, w) = (self.half_len - WALL, self.half_wid - WALL);
        let fr = self.front;
        // Table with benches, near the front windows.
        let (ta, tb) = (1.4, fr * (w - 1.3));
        let top = f + 0.78;
        block(
            out,
            self.p(ta - 0.7, top - 0.05, tb - 0.4),
            self.p(ta + 0.7, top, tb + 0.4),
            BOARDS,
        );
        for (da, db) in [(-0.6, -0.3), (0.6, -0.3), (-0.6, 0.3), (0.6, 0.3)] {
            beam(
                out,
                self.p(ta + da, f, tb + db),
                self.p(ta + da, top - 0.05, tb + db),
                0.04,
                OAK,
            );
        }
        for s in [-1.0f32, 1.0] {
            let bb = tb + s * 0.65;
            block(
                out,
                self.p(ta - 0.65, f + 0.42, bb - 0.15),
                self.p(ta + 0.65, f + 0.46, bb + 0.15),
                BOARDS,
            );
            for da in [-0.55, 0.55] {
                beam(out, self.p(ta + da, f, bb), self.p(ta + da, f + 0.42, bb), 0.035, OAK);
            }
        }
        // A chest against the end wall and a shelf with jars above it.
        let (ca, cb) = (-l + 0.35, fr * (w - 0.6));
        block(
            out,
            self.p(ca - 0.3, f, cb - 0.5),
            self.p(ca + 0.3, f + 0.55, cb + 0.5),
            BOARDS,
        );
        block(
            out,
            self.p(ca - 0.32, f + 0.5, cb - 0.52),
            self.p(ca + 0.32, f + 0.6, cb + 0.52),
            OAK,
        );
        block(
            out,
            self.p(-l, f + 1.6, cb - 0.6),
            self.p(-l + 0.28, f + 1.64, cb + 0.6),
            BOARDS,
        );
        for k in 0..3 {
            let jb = cb - 0.4 + k as f32 * 0.4;
            block(
                out,
                self.p(-l + 0.05, f + 1.64, jb - 0.08),
                self.p(-l + 0.21, f + 1.84 + 0.05 * k as f32, jb + 0.08),
                MASONRY,
            );
        }
        // Beds: upstairs in a two-storey house, by the end wall otherwise.
        let bed_floor = if self.storeys > 1 { f + STOREY + FLOOR_T } else { f };
        let (ba, bb) = (l - 1.05, -fr * (w - 0.55) * if self.storeys > 1 { -1.0 } else { 1.0 });
        let bed_a = if self.storeys > 1 { ba } else { ba - 1.3 };
        block(
            out,
            self.p(bed_a - 1.0, bed_floor + 0.25, bb - 0.5),
            self.p(bed_a + 1.0, bed_floor + 0.38, bb + 0.5),
            OAK,
        );
        block(
            out,
            self.p(bed_a - 0.95, bed_floor + 0.38, bb - 0.46),
            self.p(bed_a + 0.95, bed_floor + 0.52, bb + 0.46),
            CLOTH,
        );
        block(
            out,
            self.p(bed_a + 0.6, bed_floor + 0.52, bb - 0.35),
            self.p(bed_a + 0.9, bed_floor + 0.62, bb + 0.35),
            PLASTER,
        );
        block(
            out,
            self.p(bed_a + 0.95, bed_floor, bb - 0.5),
            self.p(bed_a + 1.05, bed_floor + 0.95, bb + 0.5),
            OAK,
        );
        for (da, db) in [(-0.95, -0.45), (-0.95, 0.45), (0.95, -0.45), (0.95, 0.45)] {
            beam(
                out,
                self.p(bed_a + da, bed_floor, bb + db),
                self.p(bed_a + da, bed_floor + 0.3, bb + db),
                0.05,
                OAK,
            );
        }
    }

    /// Beams the lanterns hang from and the bracket beside the door.
    fn fittings(&self, out: &mut MeshData) {
        let (l, w) = (self.half_len, self.half_wid);
        let f = self.floor;
        // Tie beams across the house over each lantern.
        let tie = |y: f32, out: &mut MeshData| {
            beam(out, self.p(0.25, y, -w + WALL), self.p(0.25, y, w - WALL), 0.1, OAK);
        };
        if self.storeys > 1 {
            tie(f + 6.3, out);
        } else {
            tie(f + 3.28, out);
        }
        // Door bracket: out from the wall above the lantern.
        let door_l = self.lanterns()[if self.storeys > 1 { 2 } else { 1 }];
        let wall = self.p(1.0, door_l.y + 0.45, self.front * w);
        let (la, _) = self.local(door_l.x, door_l.z);
        let tip = self.p(la, door_l.y + 0.45, self.front * (w + 0.35));
        beam(out, wall, tip, 0.04, OAK);
        let _ = l;
    }

    fn deck_model(&self, t: &Terrain, out: &mut MeshData) {
        let (l, w, d) = (self.half_len + 0.5, self.half_wid, self.deck);
        let f = self.floor;
        let back = -self.front;
        // Individual boards with gaps, running along the house.
        let mut b = w + 0.02;
        let mut k = 0;
        while b < w + d - 0.05 {
            let bc = back * (b + 0.1);
            let sag = (self.rnd(k, 51) - 0.5) * 0.01;
            block(
                out,
                self.p(-l, f - 0.07 + sag, bc - 0.095),
                self.p(l, f + sag, bc + 0.095),
                BOARDS,
            );
            b += 0.22;
            k += 1;
        }
        // Joists and posts down into the river bed.
        let mut a = -l + 0.1;
        while a <= l {
            beam(
                out,
                self.p(a, f - 0.18, back * w),
                self.p(a, f - 0.18, back * (w + d)),
                0.1,
                OAK,
            );
            for bb in [w + d * 0.5, w + d - 0.15] {
                let q = self.world(a, back * bb);
                let g = t.height_at(q.x, q.y).0;
                beam(
                    out,
                    self.p(a, g - 0.5, back * bb),
                    self.p(a, f - 0.2, back * bb),
                    0.11,
                    WOOD,
                );
            }
            a += 1.9;
        }
        // Split-rail railing on the open sides with a lantern arm at the corner.
        let rail_pts = [
            (-l + 0.1, w + 0.1),
            (-l + 0.1, w + d - 0.15),
            (l - 0.1, w + d - 0.15),
            (l - 0.1, w + 0.1),
        ];
        for pair in rail_pts.windows(2) {
            let (p0, p1) = (pair[0], pair[1]);
            let len = ((p1.0 - p0.0).powi(2) + (p1.1 - p0.1).powi(2)).sqrt();
            let n = (len / 1.6).ceil() as i32;
            for i in 0..=n {
                let s = i as f32 / n as f32;
                let (pa, pb) = (p0.0 + (p1.0 - p0.0) * s, p0.1 + (p1.1 - p0.1) * s);
                beam(
                    out,
                    self.p(pa, f - 0.1, back * pb),
                    self.p(pa, f + 1.1, back * pb),
                    0.06,
                    WOOD,
                );
            }
            for y in [0.5, 1.0] {
                beam(
                    out,
                    self.p(p0.0, f + y, back * p0.1),
                    self.p(p1.0, f + y, back * p1.1),
                    0.045,
                    WOOD,
                );
            }
        }
        let corner = self.p(l - 0.1, f, back * (w + d - 0.15));
        beam(out, corner, corner + Vec3::Y * 2.3, 0.07, WOOD);
        let lantern = self.lanterns().last().copied().unwrap();
        beam(
            out,
            corner + Vec3::Y * 2.15,
            Vec3::new(lantern.x, f + 2.15, lantern.z),
            0.035,
            WOOD,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cottage(storeys: u8, deck: f32) -> Cottage {
        let t = Terrain::new(20260927);
        Cottage::plan(&t, 900.0, 1100.0, false, storeys, 1.0, deck, 7)
    }

    #[test]
    fn the_front_door_is_open_and_the_walls_are_solid() {
        let c = cottage(1, 0.0);
        let f = c.floor;
        let door = c.p(0.0, f + 1.0, c.half_wid - 0.25);
        assert_eq!(c.block(door, f - 1.0), Some(AIR));
        let wall = c.p(-2.0, f + 1.0, c.half_wid - 0.25);
        assert_eq!(c.block(wall, f - 1.0), Some(BUILT));
        let room = c.p(-1.0, f + 1.0, 0.25);
        assert_eq!(c.block(room, f - 1.0), Some(AIR));
        assert_eq!(c.block(c.p(-1.0, f - 0.25, 0.25), f - 1.0), Some(BUILT));
    }

    #[test]
    fn steps_lead_from_the_ground_to_the_front_door() {
        let mut c = cottage(1, 0.0);
        // Raise the floor well above the lowest ground so a flight is needed.
        c.floor = c.base + 1.5;
        let mut height = c.floor;
        let mut k = 0.0;
        while let Some(top) = c.step_top(0.0, c.front * (c.half_wid + k * VOXEL_SIZE + 0.25)) {
            assert!(height - top <= VOXEL_SIZE + 1e-4, "rise {} at step {k}", height - top);
            height = top;
            k += 1.0;
        }
        assert!(
            height <= c.base + VOXEL_SIZE,
            "flight stops at {height}, ground {}",
            c.base
        );
    }

    #[test]
    fn stairs_climb_to_the_upper_floor() {
        let c = cottage(2, 0.0);
        let f = c.floor;
        let a0 = c.stair_start();
        let b = -c.front * (c.half_wid - WALL - 0.25);
        for k in 0..STEPS {
            let a = a0 + (k as f32 + 0.5) * VOXEL_SIZE;
            let tread = c.p(a, f + (k as f32 + 0.5) * VOXEL_SIZE, b);
            let above = c.p(a, f + (k as f32 + 1.5) * VOXEL_SIZE, b);
            assert_eq!(c.block(tread, f - 1.0), Some(BUILT), "step {k}");
            assert_ne!(c.block(above, f - 1.0), Some(BUILT), "headroom over step {k}");
        }
        // The upper floor is solid away from the stairwell.
        assert_eq!(c.block(c.p(2.0, f + STOREY + 0.25, 0.25), f - 1.0), Some(BUILT));
    }

    #[test]
    fn a_cottage_model_is_a_few_thousand_triangles() {
        let t = Terrain::new(20260927);
        let c = cottage(2, 7.0);
        let mut out = MeshData::default();
        c.model(&t, &mut out);
        let tris = out.indices.len() / 3;
        assert!(tris > 1500 && tris < 30_000, "{tris}");
    }
}
