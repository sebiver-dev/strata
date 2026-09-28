//! Timber-framed cottages you can walk into. Each is an authored model: a
//! coursed stone plinth, an oak frame with braces, plaster panels, framed
//! windows with shutters, plank doors hung open, rafters under rows of slate,
//! a chimney, board floors, stairs to the upper floor, furniture and hanging
//! lanterns. Houses vary in size, roof pitch, number of storeys, jettied upper
//! floors and dormers. The voxels underneath only give it collision (`BUILT`,
//! which is never drawn) and carry the lanterns that light it.

use crate::block::*;
use crate::mesh::MeshData;
use crate::model::{beam, block, board, panel};
use crate::noise::{hash2, unit};
use crate::terrain::{Terrain, WATER_LEVEL_M};
use glam::{Vec2, Vec3};

/// Default roof pitch (rise over run): 45 degrees.
pub const PITCH: f32 = 1.0;
/// How far the roof overhangs the walls.
pub const EAVE: f32 = 0.5;
/// How far a jettied upper storey oversails the wall below, on both long
/// sides; one voxel, so collision matches the model.
pub const JETTY: f32 = 0.5;
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
/// Most entry steps a front door gets.
const MAX_ENTRY_STEPS: i32 = 12;

/// What a roof is covered with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Roof {
    /// Blue-grey slate.
    Slate,
    /// Warm red-brown clay tile.
    Tile,
    /// Weathered brown shingle.
    Shingle,
}

impl Roof {
    /// The material and the value of the vertex AO bits that pick it (see
    /// `ROOF_TILE`).
    fn material(self) -> (Block, u32) {
        match self {
            Roof::Slate => (ROOF, 3),
            Roof::Tile => (ROOF_TILE, 3),
            Roof::Shingle => (ROOF_TILE, 2),
        }
    }
}

/// The shape of a cottage: storeys, footprint, roof and trimmings.
#[derive(Clone, Copy, Debug)]
pub struct Style {
    pub storeys: u8,
    /// Half the length along the ridge and half the depth across it (metres,
    /// multiples of a quarter metre so the walls cover whole voxels).
    pub half_len: f32,
    pub half_wid: f32,
    /// Roof rise over run.
    pub pitch: f32,
    /// Whether the upper storey oversails the ground floor (two storeys only).
    pub jetty: bool,
    /// Gabled dormer windows on the front slope of the roof.
    pub dormers: bool,
    pub roof: Roof,
}

impl Style {
    /// A single-storey cottage.
    pub const fn cottage() -> Self {
        Style {
            storeys: 1,
            half_len: 4.0,
            half_wid: 2.75,
            pitch: PITCH,
            jetty: false,
            dormers: false,
            roof: Roof::Slate,
        }
    }

    /// A two-storey house.
    pub const fn house() -> Self {
        Style {
            storeys: 2,
            half_len: 4.0,
            half_wid: 3.25,
            pitch: PITCH,
            jetty: false,
            dormers: false,
            roof: Roof::Slate,
        }
    }

    pub const fn size(self, half_len: f32, half_wid: f32) -> Self {
        Style {
            half_len,
            half_wid,
            ..self
        }
    }

    pub const fn pitch(self, pitch: f32) -> Self {
        Style { pitch, ..self }
    }

    pub const fn jettied(self) -> Self {
        Style { jetty: true, ..self }
    }

    pub const fn with_dormers(self) -> Self {
        Style { dormers: true, ..self }
    }

    pub const fn roofed(self, roof: Roof) -> Self {
        Style { roof, ..self }
    }
}

#[derive(Clone, Debug)]
pub struct Cottage {
    pub c: Vec2,
    /// Whether the ridge runs along X (otherwise along Z).
    pub along_x: bool,
    pub half_len: f32,
    /// Half depth of the ground floor across the ridge.
    pub half_wid: f32,
    /// Height of the ground floor's walking surface.
    pub floor: f32,
    pub storeys: u8,
    /// Roof rise over run.
    pub pitch: f32,
    /// How far the upper storey oversails the long walls below; 0 for none.
    pub jetty: f32,
    pub dormers: bool,
    pub roof: Roof,
    /// Which long side has the front door: +1 or -1 along the across axis.
    pub front: f32,
    /// Metres of plank deck on stilts behind the back wall; 0 for none.
    pub deck: f32,
    /// Lowest ground under the cottage and its deck, where footings start.
    pub base: f32,
    /// Stone steps down from the front door to the ground.
    pub steps: i32,
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

/// What fills one bay of a wall between two studs in one storey band.
enum Fill {
    Plain,
    /// A door centred at this position along the wall.
    Door(f32),
    /// A window: its extent along the wall and its sill and head heights.
    Window(f32, f32, f32, f32),
}

/// One bay of a wall: its index along the wall, whether it is an end bay, its
/// extent between the studs and between the rails, and what fills it.
struct Bay {
    k: usize,
    end: bool,
    t0: f32,
    t1: f32,
    y0: f32,
    y1: f32,
    fill: Fill,
}

/// One storey band of the walls: bottom and top heights, whether a sill rail
/// runs along its bottom, and the half depth of the house at that height.
#[derive(Clone, Copy)]
struct Span {
    y0: f32,
    y1: f32,
    sill: bool,
    wid: f32,
}

impl Cottage {
    /// Lays out a cottage near (x, z): snapped to the voxel grid, its floor a
    /// step above the ground it stands on and its entry steps down to the ground.
    #[allow(clippy::too_many_arguments)]
    pub fn plan(t: &Terrain, x: f32, z: f32, along_x: bool, front: f32, deck: f32, style: Style, seed: u32) -> Self {
        let Style {
            storeys,
            half_len,
            half_wid,
            pitch,
            jetty,
            dormers,
            roof,
        } = style;
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
        let mut cottage = Cottage {
            c,
            along_x,
            half_len,
            half_wid,
            floor,
            storeys,
            pitch,
            jetty: if jetty && storeys > 1 { JETTY } else { 0.0 },
            dormers,
            roof,
            front,
            deck,
            base: lo.min(deck_lo),
            steps: 0,
            seed,
        };
        cottage.steps = cottage.count_steps(t);
        cottage
    }

    /// How many entry steps the front door needs: one for each half metre the
    /// floor stands above the ground in front of it, walking outwards.
    pub fn count_steps(&self, t: &Terrain) -> i32 {
        let mut k = 0;
        while k < MAX_ENTRY_STEPS {
            let top = self.floor - (k + 1) as f32 * VOXEL_SIZE;
            let ground = self.step_ground(t, k);
            if top <= ground + 0.05 {
                break;
            }
            k += 1;
        }
        k
    }

    /// Highest ground under entry step `k` (counted outwards from the wall).
    pub fn step_ground(&self, t: &Terrain, k: i32) -> f32 {
        let out = self.half_wid + (k as f32 + 0.5) * VOXEL_SIZE;
        [-DOOR_HALF_W, 0.0, DOOR_HALF_W]
            .iter()
            .map(|&a| {
                let q = self.world(a, self.front * out);
                t.height_at(q.x, q.y).0
            })
            .fold(f32::MIN, f32::max)
    }

    pub fn wall_height(&self) -> f32 {
        if self.storeys > 1 {
            5.5
        } else {
            3.0
        }
    }

    /// Half depth of the upper storey, oversailing the ground floor when jettied.
    pub fn upper_wid(&self) -> f32 {
        self.half_wid + self.jetty
    }

    fn top(&self) -> f32 {
        self.floor + self.wall_height()
    }

    /// Height of the roof's outer surface at `b` across.
    fn roof_y(&self, b: f32) -> f32 {
        self.top() + (self.upper_wid() - b.abs()) * self.pitch + ROOF_LIFT
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
        let reach_b = self.upper_wid() + 1.5 + self.deck;
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

    /// The footprint of the walls, deck and entry steps as world (x, z)
    /// corners, for keeping houses off paths and bridges.
    pub fn footprint(&self) -> (Vec2, Vec2) {
        let w = self.upper_wid();
        let steps = self.steps as f32 * VOXEL_SIZE;
        let (b0, b1) = if self.front > 0.0 {
            (-w - self.deck, w.max(self.half_wid + steps))
        } else {
            (-w.max(self.half_wid + steps), w + self.deck)
        };
        let l = self.half_len + if self.deck > 0.0 { 0.5 } else { 0.0 };
        let (p, q) = (self.world(-l, b0), self.world(l, b1));
        (p.min(q), p.max(q))
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

    /// Where the front door opens, in world (x, z).
    pub fn front_door(&self) -> Vec2 {
        self.world(0.0, self.front * self.half_wid)
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
        let k = (out / VOXEL_SIZE).floor() as i32;
        (k < self.steps).then(|| self.floor - (k + 1) as f32 * VOXEL_SIZE)
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
            // One on an arm over each outer corner of the deck.
            let (l, w) = (self.half_len, self.half_wid);
            for s in [1.0f32, -1.0] {
                out.push(snap_v(self.p(
                    s * (l + 0.25),
                    f + 1.75,
                    -self.front * (w + self.deck - 0.25),
                )));
            }
        }
        out
    }

    /// Collision and light voxels, and the air the cottage clears around itself.
    pub fn block(&self, p: Vec3, ground: f32) -> Option<Block> {
        let (a, b) = self.local(p.x, p.z);
        let (l, w, h) = (self.half_len, self.half_wid, self.wall_height());
        let wu = self.upper_wid();
        let dy = p.y - self.floor;
        if self.lanterns().iter().any(|&v| in_voxel(p, v)) {
            return Some(LANTERN);
        }
        // A jettied upper storey is wider than the ground floor.
        let wb = if self.jetty > 0.0 && dy >= STOREY { wu } else { w };
        let inside = a.abs() < l && b.abs() < wb;
        if inside {
            if dy < 0.0 {
                return (p.y >= self.base - 1.0).then_some(BUILT);
            }
            let shell = a.abs() > l - WALL || b.abs() > wb - WALL;
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
        if a.abs() <= l + EAVE && b.abs() <= wu + EAVE && dy >= h - EAVE * self.pitch - 0.5 {
            let ry = self.roof_y(b);
            if p.y < ry && p.y >= ry - ROOF_SOLID {
                return Some(BUILT);
            }
            if p.y < ry && a.abs() < l && b.abs() < wu && dy >= h {
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
            return None;
        }
        // Clear a small yard of grass, trees and ground above the floor.
        let yard = a.abs() <= l + 1.5 && b.abs() <= w + 1.5;
        (yard && dy >= 0.0 && p.y < self.ridge() + 2.0).then_some(AIR)
    }

    /// The cottage as an authored model, in world space.
    pub fn model(&self, t: &Terrain, out: &mut MeshData) {
        let from = out.vertices.len();
        self.plinth(t, out);
        self.floors(out);
        for side in [-1.0f32, 1.0] {
            self.long_wall(out, side);
            self.end_wall(out, side);
            self.gable(out, side);
        }
        self.jetty_model(out);
        self.roof(out, false);
        self.dormer_models(out);
        self.chimney(out, false);
        self.stairs(out);
        self.furniture(out);
        self.fittings(out);
        if self.deck > 0.0 {
            self.deck_model(t, out, false);
        }
        self.finish(out, from);
    }

    /// Just the outside, pared down for the far field: footings, walls with
    /// their timbers and lit windows as flat strips, gables, the roof in a few
    /// rows of slate, dormers, chimney and deck.
    pub fn exterior(&self, t: &Terrain, out: &mut MeshData) {
        let from = out.vertices.len();
        let (l, f) = (self.half_len, self.floor);
        let w = self.half_wid;
        block(
            out,
            self.p(-l - 0.08, self.base - 0.3, -w - 0.08),
            self.p(l + 0.08, f - 0.02, w + 0.08),
            MASONRY,
        );
        for (si, span) in self.spans().into_iter().enumerate() {
            block(
                out,
                self.p(-l, span.y0, -span.wid),
                self.p(l, span.y1, span.wid),
                PLASTER,
            );
            for side in [-1.0f32, 1.0] {
                // Long wall: `t` runs along the house, the face just proud of the plaster.
                let long = |t: f32, y: f32| self.p(t, y, side * (span.wid + 0.02));
                self.flat_wall(out, l, side, true, span, si, &long);
                let end = |t: f32, y: f32| self.p(side * (l + 0.02), y, t);
                self.flat_wall(out, span.wid, side, false, span, si, &end);
            }
        }
        for side in [-1.0f32, 1.0] {
            self.gable(out, side);
        }
        // Doors read as dark openings at a distance.
        for (da, s) in self.doors() {
            let q = |t: f32, y: f32| self.p(t, y, s * (w + 0.03));
            let (d0, d1) = (da - DOOR_HALF_W, da + DOOR_HALF_W);
            panel(out, &[q(d0, f), q(d1, f), q(d1, f + DOOR_H), q(d0, f + DOOR_H)], OAK);
        }
        if self.steps > 0 {
            let k = self.steps as f32;
            block(
                out,
                self.p(-DOOR_HALF_W - 0.2, f - k * VOXEL_SIZE - 0.6, self.front * w),
                self.p(DOOR_HALF_W + 0.2, f - VOXEL_SIZE, self.front * (w + k * VOXEL_SIZE)),
                MASONRY,
            );
        }
        self.jetty_model(out);
        self.roof(out, true);
        self.dormer_models(out);
        self.chimney(out, true);
        if self.deck > 0.0 {
            self.deck_model(t, out, true);
        }
        self.finish(out, from);
    }

    /// A light stand-in for far away: body, timber bands, roof and lit windows.
    pub fn far(&self, out: &mut MeshData) {
        let from = out.vertices.len();
        let (l, w) = (self.half_len, self.half_wid);
        let wu = self.upper_wid();
        let top = self.top();
        block(out, self.p(-l, self.base, -w), self.p(l, top, w), PLASTER);
        if self.jetty > 0.0 {
            block(out, self.p(-l, self.floor + STOREY, -wu), self.p(l, top, wu), PLASTER);
        }
        for (y0, y1, bw) in [(self.floor, self.floor + 0.25, w), (top - 0.25, top, wu)] {
            block(
                out,
                self.p(-l - 0.05, y0, -bw - 0.05),
                self.p(l + 0.05, y1, bw + 0.05),
                OAK,
            );
        }
        let (le, we) = (l + EAVE, wu + EAVE);
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
                    self.p(s * l, top, -wu),
                    self.p(s * l, top, wu),
                    self.p(s * l, ridge - ROOF_LIFT, 0.0),
                ],
                PLASTER,
            );
        }
        // One lit window per wall bay and storey, matching the near model roughly.
        let storeys: &[f32] = if self.storeys > 1 { &[1.3, 4.3] } else { &[1.3] };
        for &dy in storeys {
            let bw = if dy > STOREY { wu } else { w };
            for side in [-1.0f32, 1.0] {
                let studs = Self::studs(l);
                for pair in studs.windows(2) {
                    let t = (pair[0] + pair[1]) * 0.5;
                    if dy < 2.0 && (t - 0.0).abs() < 0.8 && side == self.front {
                        continue;
                    }
                    let q = |dt: f32, y: f32| self.p(t + dt, self.floor + y, side * (bw + 0.03));
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
        self.finish(out, from);
    }

    /// Gives the vertices added since `from` their roof covering, and each
    /// window its own lamp: bright, ordinary, dim or dark. Both ride in the
    /// vertex AO bits, which authored models otherwise leave at 3; the shader
    /// reads them back (`shade_terrain`). Every window is a four-cornered
    /// panel drawn from both sides, so each takes eight vertices in a row.
    fn finish(&self, out: &mut MeshData, from: usize) {
        let (roof, tone) = self.roof.material();
        let set =
            |data: u32, mat: Block, ao: u32| (data & !(0xff << 3) & !(3 << 11)) | ((mat as u32) << 3) | (ao << 11);
        let mat_of = |data: u32| ((data >> 3) & 0xff) as Block;
        let mut i = from;
        while i < out.vertices.len() {
            let mat = mat_of(out.vertices[i].data);
            if mat == ROOF {
                out.vertices[i].data = set(out.vertices[i].data, roof, tone);
                i += 1;
            } else if mat == WINDOW && i + 8 <= out.vertices.len() {
                let c = out.vertices[i..i + 4].iter().map(|v| Vec3::from(v.pos)).sum::<Vec3>() / 4.0;
                let lamp = self.window_lamp(c);
                for v in &mut out.vertices[i..i + 8] {
                    debug_assert_eq!(mat_of(v.data), WINDOW);
                    v.data = set(v.data, WINDOW, lamp);
                }
                i += 8;
            } else {
                i += 1;
            }
        }
    }

    /// How the window centred at `c` is lit, as the AO bits the shader reads:
    /// 0 dark, 1 bright, 2 dim, 3 ordinary. Keyed on the wall, the storey and
    /// the place along the wall, so the near model and the far-field exterior
    /// agree.
    pub fn window_lamp(&self, c: Vec3) -> u32 {
        let (a, b) = self.local(c.x, c.z);
        let storey = ((c.y - self.floor) / STOREY).floor() as i32;
        let wid = if storey >= 1 { self.upper_wid() } else { self.half_wid };
        let end = a.abs() - self.half_len > b.abs() - wid;
        let (wall, along) = if end {
            (2 + (a > 0.0) as i32, b)
        } else {
            ((b > 0.0) as i32, a)
        };
        let key = wall * 1000 + storey * 100 + (along * 2.0).round() as i32;
        let u = unit(hash2(self.seed ^ 0x9e37, key, 17));
        if u < 0.22 {
            0
        } else if u < 0.48 {
            2
        } else if u < 0.8 {
            3
        } else {
            1
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
                        board(
                            out,
                            self.p(centre.x, (y0 + y1) * 0.5, centre.y),
                            along,
                            Vec3::Y,
                            Vec3::new((s1 - s0) * 0.5 - 0.025, course * 0.5 - 0.025, 0.22),
                            MASONRY,
                        );
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
            let wu = self.upper_wid() - WALL + 0.02;
            // Joists across the house, visible from below; under a jetty their
            // ends run out to carry the oversailing wall.
            let reach = if self.jetty > 0.0 { self.upper_wid() + 0.12 } else { w };
            let mut a = -l + 0.3;
            while a < l {
                beam(out, self.p(a, y + 0.12, -reach), self.p(a, y + 0.12, reach), 0.1, OAK);
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
            slab(-l, a0 + VOXEL_SIZE, -wu, wu, out);
            slab(open_hi, l, -wu, wu, out);
            // Beside the stairwell: the strip between it and the back wall, and
            // the room on its other side.
            if self.front > 0.0 {
                slab(a0 + VOXEL_SIZE, open_hi, -wu, bl, out);
                slab(a0 + VOXEL_SIZE, open_hi, bh, wu, out);
            } else {
                slab(a0 + VOXEL_SIZE, open_hi, bh, wu, out);
                slab(a0 + VOXEL_SIZE, open_hi, -wu, bl, out);
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

    /// The storey bands of the walls. A jettied house has two separate frames,
    /// the upper one wider; otherwise the posts run the full height.
    fn spans(&self) -> Vec<Span> {
        let f = self.floor;
        let (w, h) = (self.half_wid, self.wall_height());
        if self.storeys < 2 {
            vec![Span {
                y0: f,
                y1: f + h,
                sill: true,
                wid: w,
            }]
        } else if self.jetty > 0.0 {
            vec![
                Span {
                    y0: f,
                    y1: f + STOREY,
                    sill: true,
                    wid: w,
                },
                Span {
                    y0: f + STOREY,
                    y1: f + h,
                    sill: true,
                    wid: self.upper_wid(),
                },
            ]
        } else {
            let mid = f + STOREY + FLOOR_T * 0.5 + TIMBER;
            vec![
                Span {
                    y0: f,
                    y1: mid,
                    sill: true,
                    wid: w,
                },
                Span {
                    y0: mid,
                    y1: f + h,
                    sill: false,
                    wid: w,
                },
            ]
        }
    }

    /// The bays of one wall in one storey band, and what fills each.
    fn bays(&self, half: f32, side: f32, long: bool, span: Span, si: usize) -> Vec<Bay> {
        let y0 = span.y0 + if span.sill { 2.0 * TIMBER } else { 0.0 };
        let y1 = span.y1 - 2.0 * TIMBER;
        let studs = Self::studs(half);
        let n = studs.len() - 1;
        let doors = if long && si == 0 { self.doors() } else { Vec::new() };
        studs
            .windows(2)
            .enumerate()
            .map(|(k, pair)| {
                let (t0, t1) = (pair[0] + TIMBER, pair[1] - TIMBER);
                let door = doors
                    .iter()
                    .find(|&&(da, s)| s == side && da > t0 - 0.6 && da < t1 + 0.6);
                let fill = if let Some(&(da, _)) = door {
                    Fill::Door(da)
                } else if t1 - t0 > 1.0
                    && self.rnd(k as i32 + (side * 7.0) as i32 + if long { 0 } else { 40 }, si as i32) < 0.8
                {
                    let (wy0, wy1) = (y0 + 0.75, (y0 + 1.85).min(y1 - 0.3));
                    let tm = (t0 + t1) * 0.5;
                    let hw = ((t1 - t0) * 0.5 - 0.25).min(0.5);
                    Fill::Window(tm - hw, tm + hw, wy0, wy1)
                } else {
                    Fill::Plain
                };
                Bay {
                    k,
                    end: k == 0 || k == n - 1,
                    t0,
                    t1,
                    y0,
                    y1,
                    fill,
                }
            })
            .collect()
    }

    /// One long wall (side = +1 or -1 across): frame, braces, plaster panels,
    /// windows and doors, storey by storey.
    fn long_wall(&self, out: &mut MeshData, side: f32) {
        let l = self.half_len;
        for (si, span) in self.spans().into_iter().enumerate() {
            let b_mid = side * (span.wid - WALL * 0.5);
            self.wall(out, l - 0.12, side, |a, y| self.p(a, y, b_mid), true, span, si);
        }
    }

    fn end_wall(&self, out: &mut MeshData, side: f32) {
        let l = self.half_len;
        let a_mid = side * (l - WALL * 0.5);
        // Local coordinates along the end wall are `b`.
        for (si, span) in self.spans().into_iter().enumerate() {
            self.wall(out, span.wid - 0.12, side, |b, y| self.p(a_mid, y, b), false, span, si);
        }
    }

    /// The gable above the wall plate at one end: plaster both sides, king
    /// post and collar, and on a tall house a small lit window.
    fn gable(&self, out: &mut MeshData, side: f32) {
        let (l, w) = (self.half_len, self.upper_wid());
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
        let cw = w - (collar - top) / self.pitch;
        beam(out, self.p(a_out, collar, -cw), self.p(a_out, collar, cw), TIMBER, OAK);
        if self.storeys > 1 || self.pitch > 1.2 {
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

    /// A timber-framed wall in one storey band. `at(t, y)` places a point on
    /// the wall's centre plane at `t` along it; `long` says whether this is a
    /// long wall (so doors apply) and `si` which band it is.
    #[allow(clippy::too_many_arguments)]
    fn wall(
        &self,
        out: &mut MeshData,
        half: f32,
        side: f32,
        at: impl Fn(f32, f32) -> Vec3,
        long: bool,
        span: Span,
        si: usize,
    ) {
        let depth = WALL * 0.5 + 0.03;
        // Outward unit vector and along-wall unit vector in world space.
        let (t_dir, n_dir) = if long {
            (self.along(), self.across() * side)
        } else {
            (self.across(), self.along() * side)
        };
        let studs = Self::studs(half);
        // Horizontal members: sill (when this band has one) and wall plate.
        let mut rails = vec![span.y1 - TIMBER];
        if span.sill {
            rails.push(span.y0 + TIMBER);
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
        let doors = if long && si == 0 { self.doors() } else { Vec::new() };
        let (sy0, sy1) = (span.y0, span.y1);
        for &t in &studs {
            // Leave door openings clear of studs.
            let blocked = doors
                .iter()
                .any(|&(da, s)| s == side && (t - da).abs() < DOOR_HALF_W + 0.05);
            if !blocked {
                board(
                    out,
                    at(t, (sy0 + sy1) * 0.5),
                    Vec3::Y,
                    n_dir,
                    Vec3::new((sy1 - sy0) * 0.5, depth, TIMBER),
                    OAK,
                );
            }
        }
        let f = self.floor;
        for bay in self.bays(half, side, long, span, si) {
            let Bay {
                k, end, t0, t1, y0, y1, ..
            } = bay;
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
            match bay.fill {
                Fill::Door(da) => {
                    let (d0, d1) = (da - DOOR_HALF_W, da + DOOR_HALF_W);
                    fill(f + DOOR_H + 0.12, y1, t0, t1, out);
                    fill(y0, f + DOOR_H + 0.12, t0, d0 - 0.08, out);
                    fill(y0, f + DOOR_H + 0.12, d1 + 0.08, t1, out);
                    self.door(out, &at, t_dir, n_dir, da, side);
                }
                Fill::Window(w0, w1, wy0, wy1) => {
                    fill(y0, wy0, t0, t1, out);
                    fill(wy1, y1, t0, t1, out);
                    fill(wy0, wy1, t0, w0, out);
                    fill(wy0, wy1, w1, t1, out);
                    self.window(out, &at, t_dir, n_dir, w0, w1, wy0, wy1, k as i32 * 3 + si as i32);
                }
                Fill::Plain => {
                    fill(y0, y1, t0, t1, out);
                    // A brace across plain bays at the ends of the wall.
                    if end {
                        let (ta, tb) = if k == 0 { (t0, t1) } else { (t1, t0) };
                        let off = n_dir * (WALL * 0.5 + 0.02);
                        beam(out, at(ta, y0) + off, at(tb, y1) + off, 0.07, OAK);
                    }
                }
            }
        }
    }

    /// The far-field version of a wall: timbers, windows and braces as flat
    /// strips on the face given by `at(t, y)`.
    #[allow(clippy::too_many_arguments)]
    fn flat_wall(
        &self,
        out: &mut MeshData,
        half: f32,
        side: f32,
        long: bool,
        span: Span,
        si: usize,
        at: &impl Fn(f32, f32) -> Vec3,
    ) {
        let strip = |t0: f32, y0: f32, t1: f32, y1: f32, mat: Block, out: &mut MeshData| {
            panel(out, &[at(t0, y0), at(t1, y0), at(t1, y1), at(t0, y1)], mat);
        };
        let n_dir = if long {
            self.across() * side
        } else {
            self.along() * side
        };
        let doors = if long && si == 0 { self.doors() } else { Vec::new() };
        let (sy0, sy1) = (span.y0, span.y1);
        strip(-half, sy1 - 2.0 * TIMBER, half, sy1, OAK, out);
        if span.sill {
            strip(-half, sy0, half, sy0 + 2.0 * TIMBER, OAK, out);
        }
        for t in Self::studs(half - 0.12) {
            if !doors
                .iter()
                .any(|&(da, s)| s == side && (t - da).abs() < DOOR_HALF_W + 0.05)
            {
                strip(t - TIMBER, sy0, t + TIMBER, sy1, OAK, out);
            }
        }
        let lift = n_dir * 0.01;
        for bay in self.bays(half - 0.12, side, long, span, si) {
            match bay.fill {
                Fill::Window(w0, w1, wy0, wy1) => {
                    strip(w0 - 0.08, wy0 - 0.08, w1 + 0.08, wy1 + 0.08, OAK, out);
                    panel(
                        out,
                        &[
                            at(w0, wy0) + lift,
                            at(w1, wy0) + lift,
                            at(w1, wy1) + lift,
                            at(w0, wy1) + lift,
                        ],
                        WINDOW,
                    );
                }
                Fill::Plain if bay.end => {
                    let (ta, tb) = if bay.k == 0 { (bay.t0, bay.t1) } else { (bay.t1, bay.t0) };
                    // A strip along the diagonal, as wide as a brace.
                    let d = Vec2::new(tb - ta, bay.y1 - bay.y0).normalize() * 0.07;
                    let (pt, py) = (-d.y, d.x);
                    panel(
                        out,
                        &[
                            at(ta + pt, bay.y0 + py) + lift,
                            at(tb + pt, bay.y1 + py) + lift,
                            at(tb - pt, bay.y1 - py) + lift,
                            at(ta - pt, bay.y0 - py) + lift,
                        ],
                        OAK,
                    );
                }
                _ => {}
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
            let foot = f - self.steps as f32 * VOXEL_SIZE - 0.6;
            for k in 0..self.steps {
                let top = f - (k + 1) as f32 * VOXEL_SIZE;
                let o = WALL * 0.5 + k as f32 * VOXEL_SIZE;
                block(
                    out,
                    at(d0 - 0.2, foot) + n_dir * o,
                    at(d1 + 0.2, top) + n_dir * (o + VOXEL_SIZE),
                    MASONRY,
                );
            }
        }
    }

    /// Under a jettied storey: a soffit under the oversail and a moulded
    /// bressumer beam along its foot on both long sides.
    fn jetty_model(&self, out: &mut MeshData) {
        if self.jetty <= 0.0 {
            return;
        }
        let (l, w, wu) = (self.half_len, self.half_wid, self.upper_wid());
        let y = self.floor + STOREY;
        for s in [-1.0f32, 1.0] {
            block(out, self.p(-l, y, s * w), self.p(l, y + 0.05, s * (wu - 0.02)), BOARDS);
            beam(
                out,
                self.p(-l - 0.1, y + 0.1, s * (wu + 0.02)),
                self.p(l + 0.1, y + 0.1, s * (wu + 0.02)),
                0.13,
                OAK,
            );
            // Curved brackets under the corners, carrying the oversail.
            for a in [-l + 0.15, l - 0.15] {
                beam(
                    out,
                    self.p(a, y - 0.9, s * (w + 0.05)),
                    self.p(a, y, s * (wu - 0.1)),
                    0.07,
                    OAK,
                );
            }
        }
    }

    fn roof(&self, out: &mut MeshData, lod: bool) {
        let (l, w) = (self.half_len + EAVE, self.upper_wid() + EAVE);
        let pitch = self.pitch;
        let ridge = self.ridge();
        let slope_len = (w * w + (w * pitch).powi(2)).sqrt();
        for side in [-1.0f32, 1.0] {
            // Down-slope direction and outward normal of this roof plane.
            let down = (self.across() * side - Vec3::Y * pitch).normalize();
            let normal = (self.across() * side + Vec3::Y / pitch).normalize();
            if !lod {
                // Rafters under the tiles.
                let mut a = -self.half_len + 0.15;
                while a <= self.half_len {
                    let top = self.p(a, ridge - 0.3, 0.0);
                    beam(out, top, top + down * (slope_len - 0.05), 0.07, OAK);
                    a += 0.8;
                }
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
            // steeper so its lower edge kicks out. Far away, fewer and deeper rows.
            let row = if lod { 0.68 } else { 0.34 };
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

    /// Along-positions of the dormers on the front slope.
    fn dormer_spots(&self) -> Vec<f32> {
        if !self.dormers {
            return Vec::new();
        }
        if self.half_len >= 4.5 {
            vec![-self.half_len * 0.5, self.half_len * 0.5]
        } else {
            vec![-1.5]
        }
    }

    /// Gabled dormers on the front slope: a plaster front with a lit window,
    /// cheeks down to the roof and a little slate roof of their own.
    fn dormer_models(&self, out: &mut MeshData) {
        let s = self.front;
        let wu = self.upper_wid();
        let (hw, rise) = (0.8, 0.55);
        for ac in self.dormer_spots() {
            // The front stands over the wall line, a little up the slope.
            let bf = wu - 0.7;
            let yb = self.roof_y(bf) - 0.15;
            let yt = yb + 1.3;
            let yr = yt + rise;
            // Where the main roof rises to the dormer's eaves and ridge.
            let back = |y: f32| wu - (y - self.top() - ROOF_LIFT) / self.pitch;
            let (bt, br) = (back(yt), back(yr));
            let q = |a: f32, y: f32, b: f32| self.p(ac + a, y, s * b);
            panel(
                out,
                &[
                    q(-hw, yb, bf),
                    q(hw, yb, bf),
                    q(hw, yt, bf),
                    q(0.0, yr, bf),
                    q(-hw, yt, bf),
                ],
                PLASTER,
            );
            for e in [-hw, hw] {
                panel(out, &[q(e, yb, bf), q(e, yt, bf), q(e, yt, bt)], PLASTER);
            }
            // Window with its frame, just proud of the front.
            let (wy0, wy1) = (yb + 0.3, yb + 1.1);
            let fb = bf + 0.03;
            panel(
                out,
                &[q(-0.45, wy0, fb), q(0.45, wy0, fb), q(0.45, wy1, fb), q(-0.45, wy1, fb)],
                WINDOW,
            );
            for (a0, y0, a1, y1) in [
                (-0.5, wy0, 0.5, wy0),
                (-0.5, wy1, 0.5, wy1),
                (-0.5, wy0, -0.5, wy1),
                (0.5, wy0, 0.5, wy1),
                (0.0, wy0, 0.0, wy1),
            ] {
                beam(out, q(a0, y0, fb), q(a1, y1, fb), 0.045, OAK);
            }
            for (a0, a1) in [(-hw, hw), (-hw, -hw), (hw, hw)] {
                beam(out, q(a0, yt, fb), q(a1, if a0 == a1 { yb } else { yt }, fb), 0.06, OAK);
            }
            // Its roof: two slopes from its ridge out past the front.
            let over = bf + 0.25;
            for e in [-hw - 0.2, hw + 0.2] {
                let lo = yt - 0.2 * rise / hw;
                panel(
                    out,
                    &[
                        q(0.0, yr + 0.08, over),
                        q(0.0, yr + 0.08, br),
                        q(e, lo, bt),
                        q(e, lo, over),
                    ],
                    ROOF,
                );
                beam(out, q(0.0, yr + 0.1, over), q(e, lo, over), 0.05, OAK);
            }
        }
    }

    fn chimney(&self, out: &mut MeshData, lod: bool) {
        let (l, w) = (self.half_len, self.half_wid);
        let a = l - WALL - 0.55;
        let b = -self.front * (w - 1.3);
        let f = self.floor;
        let top = self.roof_y(b) + 1.2;
        if lod {
            let y0 = self.roof_y(b) - 1.0;
            block(
                out,
                self.p(a - 0.4, y0, b - 0.45),
                self.p(a + 0.4, top, b + 0.45),
                MASONRY,
            );
        } else {
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
        let w = self.half_wid;
        let f = self.floor;
        // Tie beams across the house over each lantern.
        let tie = |y: f32, wid: f32, out: &mut MeshData| {
            beam(out, self.p(0.25, y, -wid + WALL), self.p(0.25, y, wid - WALL), 0.1, OAK);
        };
        if self.storeys > 1 {
            tie(f + 6.3, self.upper_wid(), out);
        } else {
            tie(f + 3.28, w, out);
        }
        // Door bracket: out from the wall above the lantern.
        let door_l = self.lanterns()[if self.storeys > 1 { 2 } else { 1 }];
        let wall = self.p(1.0, door_l.y + 0.45, self.front * w);
        let (la, _) = self.local(door_l.x, door_l.z);
        let tip = self.p(la, door_l.y + 0.45, self.front * (w + 0.35));
        beam(out, wall, tip, 0.04, OAK);
    }

    /// The stilts under the deck: (foot, head) of each post, the foot half a
    /// metre into the ground or river bed.
    pub fn deck_posts(&self, t: &Terrain) -> Vec<(Vec3, Vec3)> {
        let (l, w, d) = (self.half_len + 0.5, self.half_wid, self.deck);
        let back = -self.front;
        let mut posts = Vec::new();
        if d <= 0.0 {
            return posts;
        }
        let n = ((2.0 * l - 0.2) / 1.9).ceil().max(1.0) as i32;
        for i in 0..=n {
            let a = -l + 0.1 + (2.0 * l - 0.2) * i as f32 / n as f32;
            for bb in [w + d * 0.5, w + d - 0.15] {
                let q = self.world(a, back * bb);
                let g = t.height_at(q.x, q.y).0;
                posts.push((self.p(a, g - 0.5, back * bb), self.p(a, self.floor - 0.2, back * bb)));
            }
        }
        posts
    }

    fn deck_model(&self, t: &Terrain, out: &mut MeshData, lod: bool) {
        let (l, w, d) = (self.half_len + 0.5, self.half_wid, self.deck);
        let f = self.floor;
        let back = -self.front;
        if lod {
            block(out, self.p(-l, f - 0.1, back * w), self.p(l, f, back * (w + d)), BOARDS);
        } else {
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
        }
        // Joists and posts down into the river bed, braced across.
        let posts = self.deck_posts(t);
        for pair in posts.chunks(2) {
            let (a, _) = self.local(pair[0].1.x, pair[0].1.z);
            beam(
                out,
                self.p(a, f - 0.18, back * w),
                self.p(a, f - 0.18, back * (w + d)),
                0.1,
                OAK,
            );
            for &(foot, head) in pair {
                beam(out, foot, head, 0.11, WOOD);
            }
            if !lod {
                let (h0, f1) = (pair[0].1, pair[1].0);
                let low = f1.y.max(WATER_LEVEL_M - 1.0) + 0.6;
                beam(
                    out,
                    h0 - Vec3::Y * 0.1,
                    Vec3::new(f1.x, low.min(h0.y - 0.5), f1.z),
                    0.05,
                    WOOD,
                );
            }
        }
        // Split-rail railing on the open sides with a lantern arm at each outer corner.
        let rail_pts = [
            (-l + 0.1, w + 0.1),
            (-l + 0.1, w + d - 0.15),
            (l - 0.1, w + d - 0.15),
            (l - 0.1, w + 0.1),
        ];
        for pair in rail_pts.windows(2) {
            let (p0, p1) = (pair[0], pair[1]);
            let len = ((p1.0 - p0.0).powi(2) + (p1.1 - p0.1).powi(2)).sqrt();
            let n = if lod { 1 } else { (len / 1.6).ceil() as i32 };
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
        let lanterns = self.lanterns();
        for (k, s) in [1.0f32, -1.0].into_iter().enumerate() {
            let corner = self.p(s * (l - 0.1), f, back * (w + d - 0.15));
            beam(out, corner, corner + Vec3::Y * 2.3, 0.07, WOOD);
            let lantern = lanterns[lanterns.len() - 2 + k];
            beam(
                out,
                corner + Vec3::Y * 2.15,
                Vec3::new(lantern.x, f + 2.15, lantern.z),
                0.035,
                WOOD,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cottage(style: Style, deck: f32) -> Cottage {
        let t = Terrain::new(20260927);
        Cottage::plan(&t, 900.0, 1100.0, false, 1.0, deck, style, 7)
    }

    #[test]
    fn the_front_door_is_open_and_the_walls_are_solid() {
        let c = cottage(Style::cottage(), 0.0);
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
        let t = Terrain::new(20260927);
        let mut c = cottage(Style::cottage(), 0.0);
        // Raise the floor well above the ground so a flight is needed.
        c.floor += 1.5;
        c.steps = c.count_steps(&t);
        assert!(c.steps >= 3, "{} steps", c.steps);
        let mut height = c.floor;
        let mut k = 0;
        while let Some(top) = c.step_top(0.0, c.front * (c.half_wid + k as f32 * VOXEL_SIZE + 0.25)) {
            assert!(height - top <= VOXEL_SIZE + 1e-4, "rise {} at step {k}", height - top);
            assert!(top > c.step_ground(&t, k), "step {k} buried");
            height = top;
            k += 1;
        }
        // From the ground past the last step it is one step up.
        assert!(
            height - c.step_ground(&t, k) <= VOXEL_SIZE + 0.1,
            "flight stops at {height}, ground {}",
            c.step_ground(&t, k)
        );
    }

    #[test]
    fn stairs_climb_to_the_upper_floor() {
        for style in [Style::house(), Style::house().jettied().size(5.0, 3.75)] {
            let c = cottage(style, 0.0);
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
    }

    #[test]
    fn a_jettied_storey_oversails_the_ground_floor() {
        let c = cottage(Style::house().jettied(), 0.0);
        let (f, w) = (c.floor, c.half_wid);
        // Outside the ground-floor wall, under the oversail, is open air...
        let under = c.p(-2.0, f + 1.0, -(w + 0.25));
        assert_ne!(c.block(under, f - 1.0), Some(BUILT));
        // ...the upper wall stands out over it, and the room upstairs reaches
        // past the ground-floor wall line.
        assert_eq!(c.block(c.p(-2.0, f + 4.5, -(w + 0.25)), f - 1.0), Some(BUILT));
        assert_eq!(c.block(c.p(-2.0, f + 4.5, w - 0.25), f - 1.0), Some(AIR));
        assert!(c.ridge() > c.top() + c.upper_wid() * c.pitch);
    }

    /// Each window of a model as (centre, lamp level), eight vertices apiece.
    fn windows(m: &MeshData) -> Vec<(Vec3, u32)> {
        let v = &m.vertices;
        let mut out = Vec::new();
        let mut i = 0;
        while i < v.len() {
            if (v[i].data >> 3) & 0xff == WINDOW as u32 {
                let c = v[i..i + 4].iter().map(|v| Vec3::from(v.pos)).sum::<Vec3>() / 4.0;
                let lamp = (v[i].data >> 11) & 3;
                assert!(v[i..i + 8].iter().all(|v| (v.data >> 11) & 3 == lamp));
                out.push((c, lamp));
                i += 8;
            } else {
                i += 1;
            }
        }
        out
    }

    #[test]
    fn windows_burn_unevenly_and_some_are_dark_near_and_far_alike() {
        let t = Terrain::new(20260927);
        let (mut dark, mut lit, mut levels) = (0, 0, [0; 4]);
        for seed in 0..6 {
            let style = Style::house().jettied().with_dormers();
            let c = Cottage::plan(&t, 900.0, 1100.0, false, 1.0, 0.0, style, seed);
            let (mut near, mut far) = (MeshData::default(), MeshData::default());
            c.model(&t, &mut near);
            c.exterior(&t, &mut far);
            let (near, far) = (windows(&near), windows(&far));
            for &(p, lamp) in &near {
                levels[lamp as usize] += 1;
                if lamp == 0 {
                    dark += 1;
                } else {
                    lit += 1;
                }
                let _ = p;
            }
            // Each window far away is lit as the one nearest it close up.
            for &(p, lamp) in &far {
                let (_, near_lamp) = near
                    .iter()
                    .min_by(|a, b| a.0.distance(p).total_cmp(&b.0.distance(p)))
                    .unwrap();
                assert_eq!(lamp, *near_lamp, "window at {p}");
            }
        }
        assert!(dark * 8 > dark + lit && dark * 2 < dark + lit, "{dark} dark, {lit} lit");
        assert!(levels.iter().all(|&n| n > 0), "{levels:?}");
    }

    #[test]
    fn roofs_carry_their_covering() {
        let t = Terrain::new(20260927);
        for (roof, mat, tone) in [
            (Roof::Slate, ROOF, 3),
            (Roof::Tile, ROOF_TILE, 3),
            (Roof::Shingle, ROOF_TILE, 2),
        ] {
            let c = Cottage::plan(&t, 900.0, 1100.0, false, 1.0, 0.0, Style::cottage().roofed(roof), 3);
            let mut m = MeshData::default();
            c.exterior(&t, &mut m);
            let roofing: Vec<_> = m
                .vertices
                .iter()
                .filter(|v| matches!(((v.data >> 3) & 0xff) as Block, ROOF | ROOF_TILE))
                .collect();
            assert!(roofing.len() > 100);
            assert!(roofing
                .iter()
                .all(|v| (v.data >> 3) & 0xff == mat as u32 && (v.data >> 11) & 3 == tone));
        }
    }

    #[test]
    fn a_cottage_model_is_a_few_thousand_triangles() {
        let t = Terrain::new(20260927);
        let big = Style::house().jettied().size(5.0, 3.75).pitch(1.4).with_dormers();
        for style in [Style::cottage(), Style::house(), big] {
            let c = cottage(style, 7.0);
            let mut out = MeshData::default();
            c.model(&t, &mut out);
            let tris = out.indices.len() / 3;
            assert!(tris > 1500 && tris < 30_000, "{tris}");
            // The far-field outside is a small fraction of it.
            let mut far = MeshData::default();
            c.exterior(&t, &mut far);
            let far_tris = far.indices.len() / 3;
            assert!(far_tris < 2_500 && far_tris * 4 < tris, "{far_tris} far of {tris}");
        }
    }
}
