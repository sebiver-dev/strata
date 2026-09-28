//! The watchtower on the knoll above the hamlet, an authored model: a stout
//! shaft of weathered coursed stone, some ten metres square, standing on a
//! battered stone podium with a flight of steps up to its door. Quoined
//! corners, string courses, arrow slits and pairs of arched windows glowing
//! warm; a stone stair winds up inside the thick walls. On top, carried on deep
//! stepped corbels and raking oak struts, a timber lookout projects well past
//! the shaft on every side: a boarded breastwork with tall leaded windows lit
//! from within above it, under a broad hipped roof whose bell-cast eaves
//! overhang the lookout again, with an iron finial and a pennant. Lanterns hang
//! in the lookout, by the door and in the stairwell.
//!
//! As with the cottages, the voxels only carry collision (`BUILT`) and lights
//! (`LANTERN`); the model is what you see.

use crate::block::*;
use crate::mesh::MeshData;
use crate::model::{beam, board, panel, soft_box};
use crate::noise::{hash2, unit};
use crate::terrain::Terrain;
use glam::{Vec2, Vec3};

/// Half width of the hollow inside the shaft: a ring of stairs round a well.
const INNER: f32 = 1.5;
/// Half width of the shaft's collision.
const WALL_OUT: f32 = 5.0;
/// Outer half width of the masonry at the foot of the shaft and at its top.
const FOOT_HALF: f32 = 5.0;
const TOP_HALF: f32 = 4.6;
/// Half width of the podium under the shaft, and of its collision.
const PLINTH_HALF: f32 = 5.9;
const PLINTH_OUT: f32 = 6.0;
/// Height of the podium: the tower's ground floor above the knoll (whole voxels).
const LIFT: f32 = 3.0;
/// Height of the lookout floor above the ground floor (a whole number of voxels).
const SHAFT: f32 = 21.0;
/// Half width of the lookout floor.
const LOOK: f32 = 6.5;
/// Height of the lookout's wall plate above its floor.
const ROOM: f32 = 3.0;
/// Height of the lookout's boarded breastwork, under its windows.
const RAIL: f32 = 1.0;
/// Roof: half width at the eaves, where the bell-cast flare meets the main
/// slope, and the rise per metre of that slope.
const EAVE: f32 = 8.4;
const KICK: f32 = 7.2;
const PITCH: f32 = 0.95;
/// The doorway: half width and height.
const DOOR_HALF_W: f32 = 0.5;
const DOOR_H: f32 = 2.5;
/// Stone courses, matching the masonry shader's joints.
const COURSE: f32 = 0.42;
const STONE_W: f32 = 0.7;
/// Stair steps in one full turn, and the number of steps up to the lookout
/// (the last of which is flush with its floor).
const TURN: i32 = 16;
const STEPS: i32 = (SHAFT / VOXEL_SIZE) as i32 - 1;
/// String courses: mid-height, and under the corbels.
const STRING_Y: f32 = 26.0 * COURSE;
const STRING_TOP: f32 = SHAFT - 2.35;
/// Posts along each side of the lookout, either side of the middle bay.
const POST_T: f32 = 2.2;
/// Where the lookout's corner lanterns hang, in from its corners.
const CORNER_LAMP: f32 = 5.25;
/// Half width of the flight of steps up the front of the podium.
const STEP_HALF_W: f32 = 1.5;

/// The rings of 2x2-voxel blocks the stair climbs through, starting inside the
/// door and turning to the right: (block u, block v, travel along u, along v).
const RING: [(i32, i32, i32, i32); 8] = [
    (1, 0, 1, 0),
    (2, 0, 1, 0),
    (2, 1, 0, 1),
    (2, 2, 0, 1),
    (1, 2, -1, 0),
    (0, 2, -1, 0),
    (0, 1, 0, -1),
    (0, 0, 0, -1),
];

/// Height of the roof's outer surface above the ground floor, `m` metres out
/// from the centre (the larger of the two local distances).
fn roof_y(m: f32) -> f32 {
    let eave = SHAFT + ROOM;
    if m >= KICK {
        eave + 0.35 - (m - KICK) * (0.7 / (EAVE - KICK))
    } else {
        eave + 0.35 + (KICK - m) * PITCH
    }
}

/// Height of the roof's apex above the ground floor.
pub fn apex() -> f32 {
    roof_y(0.0)
}

/// Stair step numbers in interior cell (i, j) (0..6 each) repeat every turn;
/// this is the lowest, if the cell is on the stair ring.
fn step_base(i: i32, j: i32) -> Option<i32> {
    if !(0..6).contains(&i) || !(0..6).contains(&j) {
        return None;
    }
    let (u, v) = (i / 2, j / 2);
    let k = RING.iter().position(|r| r.0 == u && r.1 == v)?;
    let (du, dv) = (RING[k].2, RING[k].3);
    let s = match (du, dv) {
        (1, _) => i % 2,
        (-1, _) => 1 - i % 2,
        (_, 1) => j % 2,
        _ => 1 - j % 2,
    };
    // The block inside the door is the landing at the foot; the stair starts to its right.
    Some(2 * ((k as i32 + 7) % 8) + s)
}

/// Top of step `n` above the ground floor.
fn tread_top(n: i32) -> f32 {
    (n + 1) as f32 * VOXEL_SIZE
}

/// Interior cell of a local position, if inside the shaft's hollow.
fn cell(a: f32, b: f32) -> Option<(i32, i32)> {
    if a.abs() >= INNER || b.abs() >= INNER {
        return None;
    }
    Some((
        ((a + INNER) / VOXEL_SIZE).floor() as i32,
        ((b + INNER) / VOXEL_SIZE).floor() as i32,
    ))
}

/// Whether a cell lies under the hatch in the lookout floor: the top turn's
/// steps just below the floor and the one beside them, so there is headroom.
fn hatch_cell(i: i32, j: i32) -> bool {
    let top = STEPS - 6..STEPS;
    step_base(i, j).is_some_and(|n0| (n0..STEPS).step_by(TURN as usize).any(|n| top.contains(&n)))
}

fn in_voxel(p: Vec3, target: Vec3) -> bool {
    (p / VOXEL_SIZE).floor() == (target / VOXEL_SIZE).floor()
}

#[derive(Clone, Debug)]
pub struct Watchtower {
    /// Centre, on a voxel corner so the shaft covers whole voxels.
    pub c: Vec2,
    /// The knoll's top under the tower, on the voxel grid: the foot of the
    /// podium, whose top (`LIFT` higher) is the ground floor inside.
    pub floor: f32,
    /// Lowest ground under the tower, where its footings start.
    pub base: f32,
    /// Unit axis vector pointing out through the door.
    pub front: Vec2,
    pub seed: u32,
}

impl Watchtower {
    /// A tower on the ground at `site`, its door facing the axis nearest `towards`.
    pub fn plan(t: &Terrain, site: Vec2, towards: Vec2, seed: u32) -> Self {
        let c = (site / VOXEL_SIZE).round() * VOXEL_SIZE;
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        let n = (PLINTH_OUT / 0.5).ceil() as i32 + 1;
        for j in -n..=n {
            for i in -n..=n {
                let h = t.height_at(c.x + i as f32 * 0.5, c.y + j as f32 * 0.5).0;
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        let d = towards - c;
        let front = if d.x.abs() > d.y.abs() {
            Vec2::new(d.x.signum(), 0.0)
        } else {
            Vec2::new(0.0, d.y.signum())
        };
        Self::new(c, (hi / VOXEL_SIZE).ceil() * VOXEL_SIZE, lo, front, seed)
    }

    pub fn new(c: Vec2, floor: f32, base: f32, front: Vec2, seed: u32) -> Self {
        Watchtower {
            c,
            floor,
            base: base.min(floor - 0.5),
            front,
            seed,
        }
    }

    /// Local axes: `a` across the front, `b` from the front towards the back.
    fn ra(&self) -> Vec2 {
        Vec2::new(-self.front.y, self.front.x)
    }

    fn rb(&self) -> Vec2 {
        -self.front
    }

    fn ax(&self) -> Vec3 {
        let r = self.ra();
        Vec3::new(r.x, 0.0, r.y)
    }

    fn bx(&self) -> Vec3 {
        let r = self.rb();
        Vec3::new(r.x, 0.0, r.y)
    }

    /// Walking surface of the ground floor inside, on top of the podium.
    fn gf(&self) -> f32 {
        self.floor + LIFT
    }

    /// World point from local (a, b) and a height above the ground floor.
    fn p(&self, a: f32, dy: f32, b: f32) -> Vec3 {
        let o = self.ra() * a + self.rb() * b;
        Vec3::new(self.c.x + o.x, self.gf() + dy, self.c.y + o.y)
    }

    fn local(&self, p: Vec3) -> (f32, f32) {
        let o = Vec2::new(p.x, p.z) - self.c;
        (o.dot(self.ra()), o.dot(self.rb()))
    }

    /// Outer half width of the shaft's masonry at a height above the floor.
    fn hw(dy: f32) -> f32 {
        FOOT_HALF + (TOP_HALF - FOOT_HALF) * (dy / SHAFT).clamp(0.0, 1.0)
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        // The steps run furthest out, down the front of the podium.
        let e = (PLINTH_OUT + (self.gf() - self.base) + 1.0).max(EAVE + 1.5);
        (
            Vec3::new(self.c.x - e, self.base - 1.5, self.c.y - e),
            Vec3::new(self.c.x + e, self.gf() + apex() + 2.5, self.c.y + e),
        )
    }

    /// Voxel centres of the lanterns: in the lookout (one in the middle and one
    /// in each corner), by the door, in the stairwell.
    fn lanterns(&self) -> [Vec3; 7] {
        let v = |a: f32, dy: f32, b: f32| self.p(a, dy + 0.25, b);
        let (h, c) = (SHAFT + 2.0, CORNER_LAMP);
        [
            v(0.25, h, 0.25),
            v(-c, h, -c),
            v(c, h, -c),
            v(c, h, c),
            v(-c, h, c),
            v(1.25, 2.0, -(WALL_OUT + 0.25)),
            v(-0.25, 2.0, 0.25),
        ]
    }

    /// Top of the stone step under `(a, b)` outside the door, if any. The
    /// first is flush with the plinth; each further one drops half a metre.
    fn step_top(&self, a: f32, b: f32) -> Option<f32> {
        let out = -b - PLINTH_OUT;
        if out < 0.0 || a.abs() >= STEP_HALF_W || out >= 2.0 * (LIFT + 2.0) {
            return None;
        }
        let k = (out / VOXEL_SIZE).floor();
        Some(self.gf() - (k + 1.0) * VOXEL_SIZE).filter(|top| *top > self.base - 0.25)
    }

    /// Whether a stair tread fills the interior voxel at (a, b), dy.
    fn stair_at(&self, a: f32, b: f32, dy: f32) -> bool {
        let Some((i, j)) = cell(a, b) else {
            return false;
        };
        let Some(n0) = step_base(i, j) else {
            return false;
        };
        let mut n = n0;
        while n < STEPS {
            let top = tread_top(n);
            if dy < top && dy >= top - VOXEL_SIZE {
                return true;
            }
            n += TURN;
        }
        false
    }

    fn hatch(&self, a: f32, b: f32) -> bool {
        cell(a, b).is_some_and(|(i, j)| hatch_cell(i, j))
    }

    /// Collision and light voxels, and the air the tower clears around itself.
    pub fn block(&self, p: Vec3, ground: f32) -> Option<Block> {
        if self.lanterns().iter().any(|&v| in_voxel(p, v)) {
            return Some(LANTERN);
        }
        let (a, b) = self.local(p);
        let m = a.abs().max(b.abs());
        let dy = p.y - self.gf();
        if dy < 0.0 {
            if m < PLINTH_OUT {
                return (p.y >= self.base - 1.0).then_some(BUILT);
            }
            if let Some(top) = self.step_top(a, b) {
                if p.y < top && p.y + VOXEL_SIZE > ground {
                    return Some(BUILT);
                }
            }
            // Keep trees and grass off the knoll top round the podium.
            return (m < EAVE + 1.0 && p.y >= ground).then_some(AIR);
        }
        if dy < SHAFT - VOXEL_SIZE {
            if m < INNER {
                return Some(if self.stair_at(a, b, dy) { BUILT } else { AIR });
            }
            if m < WALL_OUT {
                let door = b < 0.0 && a.abs() < DOOR_HALF_W && dy < DOOR_H;
                return Some(if door { AIR } else { BUILT });
            }
        } else if dy < SHAFT {
            if m < LOOK {
                return Some(if self.hatch(a, b) { AIR } else { BUILT });
            }
        } else if dy < SHAFT + ROOM && m < LOOK {
            // Breastwork and glazing close the lookout in all round.
            return Some(if m >= LOOK - VOXEL_SIZE { BUILT } else { AIR });
        }
        if m < EAVE {
            let r = roof_y(m);
            if dy < r && dy >= r - 0.7 {
                return Some(BUILT);
            }
            if dy < r && m < LOOK && dy >= SHAFT {
                return Some(AIR);
            }
        }
        // Clear trees and grass off the knoll top around the foot.
        (m < EAVE + 1.0 && dy < apex() + 1.0 && p.y >= ground - VOXEL_SIZE).then_some(AIR)
    }

    fn rnd(&self, i: i32, j: i32) -> f32 {
        unit(hash2(self.seed, i, j))
    }

    /// A box aligned with the tower, between local corners (heights above the floor).
    #[allow(clippy::too_many_arguments)]
    fn lbox(&self, out: &mut MeshData, a0: f32, a1: f32, y0: f32, y1: f32, b0: f32, b1: f32, mat: Block) {
        let (a0, a1) = (a0.min(a1), a0.max(a1));
        let (b0, b1) = (b0.min(b1), b0.max(b1));
        if a1 - a0 < 1e-3 || b1 - b0 < 1e-3 || y1 - y0 < 1e-3 {
            return;
        }
        soft_box(
            out,
            self.p((a0 + a1) * 0.5, (y0 + y1) * 0.5, (b0 + b1) * 0.5),
            [self.ax(), Vec3::Y, self.bx()],
            Vec3::new((a1 - a0) * 0.5, (y1 - y0) * 0.5, (b1 - b0) * 0.5),
            mat,
        );
    }

    /// Face `k` (0 front, then round to the right): its outward normal and
    /// its tangent in local (a, b).
    fn face(k: usize) -> (Vec2, Vec2) {
        let n = [
            Vec2::new(0.0, -1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(-1.0, 0.0),
        ][k];
        (n, Vec2::new(-n.y, n.x))
    }

    /// A box on face `k`: `t` along it, `d` out from the face at that height
    /// (`d0` < `d1`), heights above the floor. `hw` is the face's distance from the centre.
    #[allow(clippy::too_many_arguments)]
    fn fbox(&self, out: &mut MeshData, k: usize, hw: f32, t: (f32, f32), y: (f32, f32), d: (f32, f32), mat: Block) {
        let (n, tg) = Self::face(k);
        let q0 = n * (hw + d.0) + tg * t.0;
        let q1 = n * (hw + d.1) + tg * t.1;
        self.lbox(out, q0.x, q1.x, y.0, y.1, q0.y, q1.y, mat);
    }

    /// World point on face `k`, `d` out from a face `hw` from the centre.
    fn fp(&self, k: usize, hw: f32, t: f32, dy: f32, d: f32) -> Vec3 {
        let (n, tg) = Self::face(k);
        let q = n * (hw + d) + tg * t;
        self.p(q.x, dy, q.y)
    }

    fn fdirs(&self, k: usize) -> (Vec3, Vec3) {
        let (n, tg) = Self::face(k);
        (self.ax() * n.x + self.bx() * n.y, self.ax() * tg.x + self.bx() * tg.y)
    }

    /// The tower as an authored model, in world space.
    pub fn model(&self, t: &Terrain, out: &mut MeshData) {
        self.plinth(t, out);
        self.shaft(out);
        self.quoins(out);
        self.string_courses(out);
        self.door(out);
        self.windows(out);
        self.stones(out);
        self.interior(out);
        self.corbels(out);
        self.lookout(out);
        self.roof(out);
        self.finial(out);
        self.lantern_irons(out);
    }

    /// The podium: footings from below the lowest ground up to the threshold,
    /// with a battered base course, dressed corners and a projecting cap, and
    /// a broad flight of stone steps down from the door to the knoll.
    fn plinth(&self, t: &Terrain, out: &mut MeshData) {
        let bot = self.base - self.gf() - 1.0;
        let h = PLINTH_HALF;
        self.lbox(out, -h, h, bot, -0.3, -h, h, MASONRY);
        let foot = -LIFT + 0.75;
        let hb = h + 0.22;
        self.lbox(out, -hb, hb, bot, foot - 0.14, -hb, hb, MASONRY);
        let hc = h + 0.1;
        self.lbox(
            out,
            -hb + 0.1,
            hb - 0.1,
            foot - 0.14,
            foot,
            -hb + 0.1,
            hb - 0.1,
            MASONRY,
        );
        self.lbox(out, -hc, hc, -0.34, -0.02, -hc, hc, MASONRY);
        // Long-and-short stones up the podium's corners.
        for (r, y0, y1) in self.rows_from(foot, -0.36) {
            for (ci, (sa, sb)) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
                .iter()
                .enumerate()
            {
                let long = (r + ci as i32) % 2 == 0;
                let (la, lb) = if long { (0.75, 0.4) } else { (0.4, 0.75) };
                let e = h + 0.04;
                self.lbox(
                    out,
                    sa * e,
                    sa * (h - la),
                    y0 + 0.012,
                    y1 - 0.012,
                    sb * e,
                    sb * (h - lb),
                    MASONRY,
                );
            }
        }
        // Chamfered cap stones where the shaft rises.
        for k in 0..4 {
            let (hw, e) = (Self::hw(0.0), h - 0.12);
            let n = (2.0 * e / 0.9).ceil() as i32;
            for s in 0..n {
                let t0 = -e + s as f32 * 2.0 * e / n as f32;
                let t1 = t0 + 2.0 * e / n as f32;
                if k == 0 && t1 > -DOOR_HALF_W - 0.3 && t0 < DOOR_HALF_W + 0.3 {
                    continue;
                }
                let j = self.rnd(k as i32 * 50 + s, 3) * 0.015;
                self.fbox(
                    out,
                    k,
                    hw,
                    (t0 + 0.01, t1 - 0.01),
                    (-0.04, 0.02 + j),
                    (-0.3, e - hw + 0.02),
                    MASONRY,
                );
            }
        }
        // Threshold, and the steps with a low parapet each side.
        self.lbox(
            out,
            -DOOR_HALF_W - 0.25,
            DOOR_HALF_W + 0.25,
            -0.4,
            0.0,
            -PLINTH_OUT,
            -INNER,
            MASONRY,
        );
        let mut last = -PLINTH_OUT;
        for k in 0..(4.0 * (LIFT + 2.0)) as i32 {
            let b1 = -PLINTH_OUT - k as f32 * VOXEL_SIZE;
            let top = -(k as f32 + 1.0) * VOXEL_SIZE;
            let q = self.p(0.0, top, b1 - 0.25);
            let g = t.height_at(q.x, q.z).0 - self.gf();
            if top + self.gf() <= self.base - 0.25 || top <= g - 0.05 {
                break;
            }
            let w = STEP_HALF_W + self.rnd(k, 9) * 0.06;
            self.lbox(
                out,
                -w,
                w,
                g.min(top) - 0.6,
                top,
                b1 - VOXEL_SIZE - 0.02,
                b1 + 0.1,
                MASONRY,
            );
            last = b1 - VOXEL_SIZE;
        }
        for s in [-1.0f32, 1.0] {
            let (a0, a1) = (s * (STEP_HALF_W + 0.08), s * (STEP_HALF_W + 0.5));
            let y_hi = 0.45;
            let n = ((-PLINTH_OUT - last) / 0.7).ceil().max(1.0) as i32;
            for i in 0..n {
                let b0 = -PLINTH_OUT - i as f32 * (-PLINTH_OUT - last) / n as f32;
                let b1 = b0 - (-PLINTH_OUT - last) / n as f32;
                // The coping steps down with the flight.
                let y1 = y_hi - (i as f32 + 0.5) * (LIFT / n as f32);
                self.lbox(out, a0, a1, bot, y1, b1 + 0.01, b0 - 0.01, MASONRY);
            }
        }
    }

    /// The hollow shaft, two courses at a time so it tapers smoothly, with the doorway left open.
    fn shaft(&self, out: &mut MeshData) {
        let top = SHAFT - VOXEL_SIZE;
        let band = 2.0 * COURSE;
        let mut y0 = 0.0;
        while y0 < top - 1e-3 {
            let y1 = (y0 + band).min(top);
            let hw = Self::hw((y0 + y1) * 0.5);
            let i = INNER;
            // Front and back slabs run the full width; the sides fit between them.
            for s in [-1.0f32, 1.0] {
                if s < 0.0 && y0 < DOOR_H + 0.01 {
                    let dh = DOOR_HALF_W;
                    self.lbox(out, -hw, -dh, y0, y1, -hw, -i, MASONRY);
                    self.lbox(out, dh, hw, y0, y1, -hw, -i, MASONRY);
                } else {
                    self.lbox(out, -hw, hw, y0, y1, s * hw, s * i, MASONRY);
                }
                self.lbox(out, s * hw, s * i, y0, y1, -i, i, MASONRY);
            }
            y0 = y1;
        }
    }

    /// Rows of stone in the shader's course grid (world heights), from the
    /// floor up to `top` above it: (row index, bottom, top) above the floor.
    fn rows(&self, top: f32) -> Vec<(i32, f32, f32)> {
        self.rows_from(0.0, top)
    }

    /// Rows of stone between two heights above the floor, as `rows`.
    fn rows_from(&self, bot: f32, top: f32) -> Vec<(i32, f32, f32)> {
        let f = self.gf();
        let first = ((f + bot) / COURSE).ceil() as i32;
        let last = ((f + top) / COURSE).floor() as i32;
        (first..last)
            .map(|r| (r, r as f32 * COURSE - f, (r + 1) as f32 * COURSE - f))
            .collect()
    }

    /// Long-and-short dressed stones up each corner, standing a little proud.
    fn quoins(&self, out: &mut MeshData) {
        for (r, y0, y1) in self.rows(STRING_TOP) {
            let hw = Self::hw((y0 + y1) * 0.5);
            for (ci, (sa, sb)) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
                .iter()
                .enumerate()
            {
                let long = (r + ci as i32) % 2 == 0;
                let j = self.rnd(r, ci as i32) * 0.08;
                let (la, lb) = if long { (0.9 + j, 0.5) } else { (0.5, 0.9 + j) };
                let e = hw + 0.03;
                self.lbox(
                    out,
                    sa * e,
                    sa * (hw - la),
                    y0 + 0.012,
                    y1 - 0.012,
                    sb * e,
                    sb * (hw - lb),
                    MASONRY,
                );
            }
        }
    }

    /// Projecting bands of stone round the shaft.
    fn string_courses(&self, out: &mut MeshData) {
        for y in [STRING_Y, STRING_TOP] {
            let hw = Self::hw(y);
            for k in 0..4 {
                // Each face's band runs past the corners so they overlap cleanly.
                let e = hw + 0.12;
                self.fbox(out, k, hw, (-e, e), (y, y + 0.1), (-0.15, 0.12), MASONRY);
                self.fbox(
                    out,
                    k,
                    hw,
                    (-e + 0.05, e - 0.05),
                    (y + 0.1, y + 0.2),
                    (-0.15, 0.07),
                    MASONRY,
                );
            }
        }
    }

    /// The lit openings on face `k`: (centre along the face, sill height,
    /// height to the springing of the arch, half width, arched).
    fn lights(k: usize) -> Vec<(f32, f32, f32, f32, bool)> {
        let mut o = Vec::new();
        for t in [-2.1f32, 2.1] {
            o.push((t, 8.6, 1.2, 0.45, true));
            o.push((t, SHAFT - 5.2, 1.6, 0.55, true));
        }
        o.push((0.0, 5.6, 1.3, 0.1, false));
        o.push((0.0, 12.9, 1.3, 0.1, false));
        if k != 0 {
            o.push((0.0, 2.3, 1.3, 0.1, false));
        }
        o
    }

    /// Openings on each face, as (t0, t1, y0, y1), for the scattered stones to avoid.
    fn openings(k: usize) -> Vec<(f32, f32, f32, f32)> {
        let mut o: Vec<_> = Self::lights(k)
            .into_iter()
            .map(|(t, y, h, w, arch)| {
                let top = y + h + if arch { w + 0.25 } else { 0.2 };
                (t - w - 0.3, t + w + 0.3, y - 0.2, top + 0.05)
            })
            .collect();
        if k == 0 {
            o.push((-1.2, 1.2, 0.0, 4.1));
            o.push((0.9, 1.6, 2.2, 3.1));
        }
        o
    }

    /// Stones standing proud of the wall here and there, laid in the same
    /// courses and bond as the shader draws, so the wall reads as built.
    fn stones(&self, out: &mut MeshData) {
        let top = STRING_TOP - 0.1;
        for k in 0..4 {
            let (nw, tw) = self.fdirs(k);
            // The shader lays courses along world X or Z, whichever the face runs along.
            let along_x = nw.x.abs() < nw.z.abs();
            let sgn = if along_x { tw.x } else { tw.z };
            let centre = self.fp(k, 0.0, 0.0, 0.0, 0.0);
            let uc = if along_x { centre.x } else { centre.z };
            let open = Self::openings(k);
            for (r, y0, y1) in self.rows(top) {
                if (y0..y1 + 0.3).contains(&STRING_Y) || y1 > STRING_Y && y0 < STRING_Y + 0.25 {
                    continue;
                }
                let hw = Self::hw((y0 + y1) * 0.5);
                let lim = hw - 0.95;
                let shift = r as f32 * 0.5;
                let (u0, u1) = (uc - lim, uc + lim);
                let mut col = (u0 / STONE_W + shift).floor() as i32;
                loop {
                    let su0 = (col as f32 - shift) * STONE_W;
                    if su0 > u1 {
                        break;
                    }
                    let su1 = su0 + STONE_W;
                    col += 1;
                    if self.rnd(r * 7 + k as i32, col) > 0.22 {
                        continue;
                    }
                    let (ta, tb) = ((su0 - uc) * sgn, (su1 - uc) * sgn);
                    let (t0, t1) = (ta.min(tb).max(-lim), ta.max(tb).min(lim));
                    if t1 - t0 < 0.25 {
                        continue;
                    }
                    if open.iter().any(|o| t1 > o.0 && t0 < o.1 && y1 > o.2 && y0 < o.3) {
                        continue;
                    }
                    let jut = 0.012 + 0.03 * self.rnd(r, col * 3 + k as i32);
                    self.fbox(
                        out,
                        k,
                        hw,
                        (t0 + 0.014, t1 - 0.014),
                        (y0 + 0.014, y1 - 0.014),
                        (-0.05, jut),
                        MASONRY,
                    );
                }
            }
        }
    }

    /// The doorway: jambs, a lintel under a relieving arch, and a plank door
    /// standing open against the inside wall.
    fn door(&self, out: &mut MeshData) {
        let hw = Self::hw(1.0);
        let dh = DOOR_HALF_W;
        for (r, y0, y1) in self.rows(DOOR_H) {
            for s in [-1.0f32, 1.0] {
                let w = if r % 2 == 0 { 0.5 } else { 0.3 } + self.rnd(r, 40) * 0.05;
                self.fbox(
                    out,
                    0,
                    hw,
                    (s * dh, s * (dh + w)),
                    (y0 + 0.012, y1 - 0.012),
                    (-(hw - INNER) - 0.01, 0.035),
                    MASONRY,
                );
            }
        }
        self.fbox(
            out,
            0,
            hw,
            (-dh - 0.35, dh + 0.35),
            (DOOR_H, DOOR_H + 0.38),
            (-(hw - INNER), 0.04),
            MASONRY,
        );
        // Relieving arch of wedge-shaped stones.
        let (cy, r0, r1) = (DOOR_H + 0.38, 0.72, 1.02);
        let n = 9;
        for i in 0..n {
            let a0 = std::f32::consts::PI * i as f32 / n as f32;
            let a1 = std::f32::consts::PI * (i + 1) as f32 / n as f32;
            let am = (a0 + a1) * 0.5;
            let rm = (r0 + r1) * 0.5;
            let (nw, tw) = self.fdirs(0);
            let centre = self.fp(0, hw, rm * am.cos(), cy + rm * am.sin(), 0.01);
            let tangent = (tw * -am.sin() + Vec3::Y * am.cos()).normalize();
            let half_w = rm * (a1 - a0) * 0.5 - 0.012;
            board(
                out,
                centre,
                tangent,
                nw,
                Vec3::new(half_w, 0.06, (r1 - r0) * 0.5),
                MASONRY,
            );
        }
        // The door leaf, swung in against the left of the passage.
        let a = -dh - 0.05;
        self.lbox(
            out,
            a - 0.03,
            a + 0.03,
            0.02,
            DOOR_H - 0.06,
            -INNER + 0.05,
            -INNER + 1.0,
            BOARDS,
        );
        for y in [0.4, DOOR_H - 0.5] {
            self.lbox(
                out,
                a + 0.03,
                a + 0.045,
                y,
                y + 0.07,
                -INNER + 0.05,
                -INNER + 0.85,
                IRON,
            );
        }
        self.lbox(out, a + 0.03, a + 0.07, 1.15, 1.25, -INNER + 0.85, -INNER + 0.93, IRON);
    }

    /// A glowing opening on face `k`: arrow slit (`arch` false) or a small
    /// arched window, with its dressed surround.
    #[allow(clippy::too_many_arguments)]
    fn opening(&self, out: &mut MeshData, k: usize, tc: f32, y0: f32, h: f32, w: f32, arch: bool) {
        let hw = Self::hw(y0 + h * 0.5);
        let (nw, tw) = self.fdirs(k);
        let q = |t: f32, y: f32| self.fp(k, hw, tc + t, y, 0.006);
        let mut pts = vec![q(-w, y0), q(w, y0)];
        if arch {
            for i in 0..=8 {
                let a = std::f32::consts::PI * i as f32 / 8.0;
                pts.push(q(w * a.cos(), y0 + h + w * a.sin()));
            }
        } else {
            pts.push(q(w, y0 + h));
            pts.push(q(-w, y0 + h));
        }
        panel(out, &pts, GLOW);
        // A dark reveal behind the glass keeps the inside of the wall from showing.
        let top = y0 + h + if arch { w } else { 0.0 };
        self.fbox(
            out,
            k,
            hw,
            (tc - w - 0.02, tc + w + 0.02),
            (y0 - 0.02, top + 0.02),
            (-0.2, -0.01),
            OAK,
        );
        // Jambs, sill and head.
        let jamb = 0.16;
        let n = if arch { 2 } else { 3 };
        for i in 0..n {
            let (s0, s1) = (y0 + h * i as f32 / n as f32, y0 + h * (i + 1) as f32 / n as f32);
            for s in [-1.0f32, 1.0] {
                let wide = jamb
                    + if (i + (s > 0.0) as usize).is_multiple_of(2) {
                        0.1
                    } else {
                        0.0
                    };
                self.fbox(
                    out,
                    k,
                    hw,
                    (tc + s * w, tc + s * (w + wide)),
                    (s0 + 0.01, s1 - 0.01),
                    (-0.1, 0.04),
                    MASONRY,
                );
            }
        }
        self.fbox(
            out,
            k,
            hw,
            (tc - w - 0.22, tc + w + 0.22),
            (y0 - 0.14, y0),
            (-0.1, 0.08),
            MASONRY,
        );
        if arch {
            let (cy, r0, r1) = (y0 + h, w, w + 0.22);
            let m = 5;
            for i in 0..m {
                let a0 = std::f32::consts::PI * i as f32 / m as f32;
                let a1 = std::f32::consts::PI * (i + 1) as f32 / m as f32;
                let am = (a0 + a1) * 0.5;
                let rm = (r0 + r1) * 0.5;
                let centre = self.fp(k, hw, tc + rm * am.cos(), cy + rm * am.sin(), 0.0);
                let tangent = (tw * -am.sin() + Vec3::Y * am.cos()).normalize();
                board(
                    out,
                    centre,
                    tangent,
                    nw,
                    Vec3::new(rm * (a1 - a0) * 0.5 - 0.01, 0.05, (r1 - r0) * 0.5),
                    MASONRY,
                );
            }
        } else {
            self.fbox(
                out,
                k,
                hw,
                (tc - w - 0.2, tc + w + 0.2),
                (y0 + h, y0 + h + 0.2),
                (-0.1, 0.05),
                MASONRY,
            );
        }
    }

    fn windows(&self, out: &mut MeshData) {
        for k in 0..4 {
            for (t, y, h, w, arch) in Self::lights(k) {
                self.opening(out, k, t, y, h, w, arch);
            }
        }
    }

    /// Flagged floor and the stone stair winding up the inside.
    fn interior(&self, out: &mut MeshData) {
        let i = INNER + 0.02;
        self.lbox(out, -i, i, -0.12, 0.0, -i, i, MASONRY);
        for n in 0..STEPS {
            let n0 = n % TURN;
            let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
            for j in 0..6 {
                for ii in 0..6 {
                    if step_base(ii, j) == Some(n0) {
                        let q = Vec2::new(ii as f32, j as f32) * VOXEL_SIZE - INNER;
                        lo = lo.min(q);
                        hi = hi.max(q + VOXEL_SIZE);
                    }
                }
            }
            let top = tread_top(n);
            // Treads run a little into the wall they are built into.
            let lo_e = |x: f32| if x <= -INNER + 0.01 { x - 0.1 } else { x + 0.005 };
            let hi_e = |x: f32| if x >= INNER - 0.01 { x + 0.1 } else { x - 0.005 };
            self.lbox(
                out,
                lo_e(lo.x),
                hi_e(hi.x),
                if n < 3 { (top - 0.5).max(0.0) } else { top - 0.25 },
                top,
                lo_e(lo.y),
                hi_e(hi.y),
                MASONRY,
            );
        }
    }

    /// Deep stepped stone corbels and raking oak struts carrying the lookout
    /// out past the shaft on every side.
    fn corbels(&self, out: &mut MeshData) {
        let s = SHAFT;
        let hw = TOP_HALF;
        for k in 0..4 {
            for t in [-3.3f32, -1.1, 1.1, 3.3] {
                for (i, (y0, y1, d)) in [
                    (s - 2.0, s - 1.62, 0.35),
                    (s - 1.62, s - 1.24, 0.7),
                    (s - 1.24, s - 0.88, 1.05),
                    (s - 0.88, s - 0.5, 1.45),
                ]
                .iter()
                .enumerate()
                {
                    let w = 0.26 + 0.02 * i as f32;
                    self.fbox(
                        out,
                        k,
                        hw,
                        (t - w, t + w),
                        (*y0 + 0.01, *y1 - 0.01),
                        (-0.1, *d),
                        MASONRY,
                    );
                }
            }
            // Struts from the wall up under the posts.
            for t in [-POST_T, POST_T] {
                let foot = self.fp(k, Self::hw(s - 3.6), t, s - 3.6, -0.05);
                let head = self.fp(k, LOOK - 0.3, t, s - 0.5, 0.0);
                beam(out, foot, head, 0.13, OAK);
                self.fbox(
                    out,
                    k,
                    Self::hw(s - 3.6),
                    (t - 0.2, t + 0.2),
                    (s - 3.9, s - 3.4),
                    (-0.1, 0.12),
                    MASONRY,
                );
            }
        }
        // Diagonal braces from the shaft's corners out to the lookout's.
        for (sa, sb) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let foot = self.p(sa * (hw - 0.05), s - 3.4, sb * (hw - 0.05));
            let head = self.p(sa * (LOOK - 0.2), s - 0.55, sb * (LOOK - 0.2));
            beam(out, foot, head, 0.13, OAK);
        }
    }

    /// The timber lookout: floor, posts, a boarded breastwork, tall leaded
    /// windows lit from within, the wall plate and a propped shutter.
    fn lookout(&self, out: &mut MeshData) {
        let s = SHAFT;
        let e = LOOK;
        let pl = LOOK - 0.12;
        // Sill beams round the edge and joists across.
        for k in 0..4 {
            self.fbox(out, k, pl, (-e, e), (s - 0.5, s - 0.12), (-0.14, 0.14), OAK);
        }
        let mut a = -e + 0.45;
        while a < e - 0.3 {
            self.lbox(out, a - 0.09, a + 0.09, s - 0.48, s - 0.11, -e + 0.05, e - 0.05, OAK);
            a += 0.75;
        }
        // Floor boards, one run per voxel row, leaving the hatch open.
        let rows = (2.0 * e / VOXEL_SIZE) as i32;
        for j in 0..rows {
            let b0 = -e + j as f32 * VOXEL_SIZE;
            let mut run: Option<f32> = None;
            for i in 0..=rows {
                let a0 = -e + i as f32 * VOXEL_SIZE;
                let open = i == rows || self.hatch(a0 + 0.25, b0 + 0.25);
                match (run, open) {
                    (None, false) => run = Some(a0),
                    (Some(st), true) => {
                        for h in 0..2 {
                            let bb = b0 + h as f32 * 0.25;
                            let dip = self.rnd(j * 2 + h, i) * 0.01;
                            self.lbox(
                                out,
                                st + 0.004,
                                a0 - 0.004,
                                s - 0.11 - dip,
                                s - dip,
                                bb + 0.006,
                                bb + 0.244,
                                BOARDS,
                            );
                        }
                        run = None;
                    }
                    _ => {}
                }
            }
        }
        // Posts at the corners and two along each side.
        let top = s + ROOM - 0.12;
        let mut posts = Vec::new();
        for (sa, sb) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            posts.push(Vec2::new(sa * pl, sb * pl));
        }
        for k in 0..4 {
            let (n, tg) = Self::face(k);
            for t in [-POST_T, POST_T] {
                posts.push(n * pl + tg * t);
            }
        }
        for q in &posts {
            self.lbox(out, q.x - 0.15, q.x + 0.15, s - 0.12, top, q.y - 0.15, q.y + 0.15, OAK);
        }
        // Wall plate on the posts, and tie beams across carrying the lantern.
        for k in 0..4 {
            self.fbox(out, k, pl, (-e, e), (top, top + 0.26), (-0.15, 0.15), OAK);
        }
        let ty = s + ROOM + 0.14;
        self.lbox(out, -pl, pl, ty, ty + 0.24, 0.13, 0.37, OAK);
        self.lbox(out, 0.13, 0.37, ty + 0.24, ty + 0.46, -pl, pl, OAK);
        let bays = [(-pl, -POST_T), (-POST_T, POST_T), (POST_T, pl)];
        let (lo, hi) = (s + 0.18, s + RAIL - 0.1);
        for k in 0..4 {
            let (nw, _) = self.fdirs(k);
            // Breastwork: kick board, cap rail, and close boarding between.
            self.fbox(out, k, pl, (-e, e), (s, s + 0.18), (-0.05, 0.06), OAK);
            self.fbox(out, k, pl, (-e, e), (hi, s + RAIL + 0.04), (-0.1, 0.12), OAK);
            for (bi, &(t0, t1)) in bays.iter().enumerate() {
                let (t0, t1) = (t0 + 0.15, t1 - 0.15);
                let n = ((t1 - t0) / 0.3).round().max(1.0) as i32;
                for i in 0..n {
                    let tc = t0 + (i as f32 + 0.5) * (t1 - t0) / n as f32;
                    let hw = (t1 - t0) / n as f32 * 0.5 - 0.01;
                    let sag = self.rnd(k as i32 * 100 + i, bi as i32) * 0.04;
                    let c = self.fp(k, pl, tc, (lo + hi) * 0.5 - sag * 0.5, 0.0);
                    board(
                        out,
                        c,
                        Vec3::Y,
                        nw,
                        Vec3::new((hi - lo) * 0.5 - sag * 0.5, 0.025, hw),
                        BOARDS,
                    );
                }
                // A tall window filling the bay over the breastwork, lit
                // from within, with mullions and a transom.
                let (w0, w1) = (s + RAIL + 0.04, top - 0.02);
                let q = |t: f32, y: f32| self.fp(k, pl, t, y, -0.02);
                panel(out, &[q(t0, w0), q(t1, w0), q(t1, w1), q(t0, w1)], GLOW);
                let m = ((t1 - t0) / 1.2).round().max(1.0) as i32;
                for i in 1..m {
                    let tm = t0 + (t1 - t0) * i as f32 / m as f32;
                    self.fbox(out, k, pl, (tm - 0.05, tm + 0.05), (w0, w1), (-0.04, 0.06), OAK);
                }
                let yt = w0 + (w1 - w0) * 0.68;
                self.fbox(out, k, pl, (t0, t1), (yt - 0.04, yt + 0.04), (-0.04, 0.06), OAK);
                // Knee braces under the wall plate.
                let yb = top - 0.6;
                beam(
                    out,
                    self.fp(k, pl, t0 - 0.05, yb, 0.08),
                    self.fp(k, pl, t0 + 0.55, top, 0.08),
                    0.07,
                    OAK,
                );
                beam(
                    out,
                    self.fp(k, pl, t1 + 0.05, yb, 0.08),
                    self.fp(k, pl, t1 - 0.55, top, 0.08),
                    0.07,
                    OAK,
                );
            }
        }
        // A top-hung shutter propped open over the middle bay at the back.
        let k = 2usize;
        let (nw, tw) = self.fdirs(k);
        let hinge = self.fp(k, pl, 0.0, top - 0.02, 0.12);
        let ang: f32 = 1.05 + 0.2 * self.rnd(k as i32, 77);
        let d = (Vec3::Y * -ang.cos() + nw * ang.sin()).normalize();
        let len = 1.3;
        let normal = d.cross(tw).normalize();
        board(
            out,
            hinge + d * (len * 0.5),
            tw,
            normal,
            Vec3::new(POST_T - 0.2, 0.025, len * 0.5),
            BOARDS,
        );
        for t in [-1.2f32, 0.0, 1.2] {
            let c = hinge + d * (len * 0.5) + tw * t + normal * 0.03;
            board(out, c, d, normal, Vec3::new(len * 0.5 - 0.03, 0.012, 0.05), OAK);
        }
        let tip = hinge + d * len + tw * 1.5;
        beam(out, self.fp(k, pl, 1.5, s + RAIL, 0.1), tip, 0.025, WOOD);
    }

    /// Broad hipped roof with a bell-cast flare out over the lookout: rows of
    /// slate, hip rolls, boarding and rafters underneath, and a fascia along
    /// the eaves.
    fn roof(&self, out: &mut MeshData) {
        let rim = |m: f32| (m, roof_y(m));
        // Row boundaries, from the eaves up to the apex.
        let mut bounds = Vec::new();
        let mut m = EAVE;
        while m > KICK + 1e-3 {
            bounds.push(m);
            m -= 0.24;
        }
        bounds.push(KICK);
        m = KICK;
        let step = 0.27 / (1.0 + PITCH * PITCH).sqrt();
        while m > step * 0.6 {
            m -= step;
            bounds.push(m.max(0.0));
        }
        if *bounds.last().unwrap() > 0.0 {
            bounds.push(0.0);
        }
        let lift = 0.045;
        for k in 0..4 {
            let (n, tg) = Self::face(k);
            let pt = |m: f32, t: f32, up: f32| {
                let q = n * m + tg * t;
                self.p(q.x, roof_y(m) + up, q.y)
            };
            for w in bounds.windows(2) {
                let (m0, m1) = (w[0], w[1]);
                let lo = [pt(m0, -m0, lift), pt(m0, m0, lift)];
                if m1 > 1e-3 {
                    panel(out, &[lo[0], lo[1], pt(m1, m1, 0.0), pt(m1, -m1, 0.0)], ROOF);
                } else {
                    panel(out, &[lo[0], lo[1], pt(0.0, 0.0, 0.0)], ROOF);
                }
                // The slates' lower edges.
                panel(out, &[pt(m0, -m0, 0.0), pt(m0, m0, 0.0), lo[1], lo[0]], ROOF);
            }
            // Boarding seen from inside, and the fascia along the eave.
            let under = -0.14;
            panel(
                out,
                &[
                    pt(EAVE, -EAVE, under),
                    pt(EAVE, EAVE, under),
                    pt(KICK, KICK, under),
                    pt(KICK, -KICK, under),
                ],
                BOARDS,
            );
            panel(
                out,
                &[pt(KICK, -KICK, under), pt(KICK, KICK, under), pt(0.0, 0.0, under)],
                BOARDS,
            );
            let (nw, tw) = self.fdirs(k);
            let (_, ye) = rim(EAVE);
            let fc = self.p(n.x * EAVE, ye - 0.04, n.y * EAVE);
            board(out, fc, tw, nw, Vec3::new(EAVE + 0.02, 0.03, 0.08), OAK);
            // Frieze boards closing the gap over the wall plate.
            let pl = LOOK - 0.12;
            let y0 = SHAFT + ROOM + 0.12;
            let y1 = roof_y(pl) - 0.12;
            self.fbox(out, k, pl, (-pl, pl), (y0, y1), (-0.03, 0.03), BOARDS);
            // Common rafters under the boarding.
            for t in [-4.8f32, -3.6, -2.4, -1.2, 0.0, 1.2, 2.4, 3.6, 4.8] {
                let m1 = t.abs().max(0.25);
                let a = self.p((n * pl + tg * t).x, roof_y(pl) - 0.28, (n * pl + tg * t).y);
                let q = n * m1 + tg * t;
                let b = self.p(q.x, roof_y(m1) - 0.28, q.y);
                beam(out, a, b, 0.07, OAK);
            }
        }
        // Hip rolls from each corner to the apex, and hip rafters beneath.
        for (sa, sb) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let c = |m: f32, up: f32| self.p(sa * m, roof_y(m) + up, sb * m);
            beam(out, c(EAVE + 0.02, 0.07), c(KICK, 0.08), 0.075, ROOF);
            beam(out, c(KICK, 0.08), c(0.0, 0.08), 0.075, ROOF);
            beam(out, c(LOOK - 0.12, -0.28), c(0.1, -0.28), 0.09, OAK);
        }
    }

    /// Lead cap, iron spike with a ball, and a red pennant.
    fn finial(&self, out: &mut MeshData) {
        let top = apex();
        soft_box(
            out,
            self.p(0.0, top + 0.03, 0.0),
            [self.ax(), Vec3::Y, self.bx()],
            Vec3::new(0.3, 0.2, 0.3),
            ROOF,
        );
        beam(
            out,
            self.p(0.0, top - 0.3, 0.0),
            self.p(0.0, top + 2.2, 0.0),
            0.05,
            IRON,
        );
        let ball = self.p(0.0, top + 0.65, 0.0);
        for axes in [
            [self.ax(), Vec3::Y, self.bx()],
            [
                (self.ax() + self.bx()).normalize(),
                Vec3::Y,
                (self.bx() - self.ax()).normalize(),
            ],
        ] {
            soft_box(out, ball, axes, Vec3::splat(0.14), IRON);
        }
        // The pennant streams away from the valley.
        let dir = (self.bx() * 0.8 + self.ax() * 0.6).normalize();
        let base = self.p(0.0, top + 1.35, 0.0);
        let mid = base + dir * 0.7 + Vec3::Y * 0.08 + self.ax() * 0.08;
        let tip = base + dir * 1.45 - Vec3::Y * 0.1;
        let hi = base + Vec3::Y * 0.65;
        panel(out, &[base, mid, hi], CLOTH);
        panel(out, &[mid, tip, hi + Vec3::Y * -0.1 + dir * 0.7], CLOTH);
        panel(out, &[hi, mid, hi + Vec3::Y * -0.1 + dir * 0.7], CLOTH);
    }

    /// Iron arms the lanterns hang from.
    fn lantern_irons(&self, out: &mut MeshData) {
        // In the lookout: rods down from the corner posts' heads.
        let top = SHAFT + ROOM - 0.12;
        let pl = LOOK - 0.12;
        for (sa, sb) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let (a, b) = (sa * (CORNER_LAMP + 0.25), sb * (CORNER_LAMP + 0.25));
            let hang = self.p(a, top - 0.05, b);
            beam(
                out,
                self.p(sa * (pl - 0.15), top - 0.05, sb * (pl - 0.15)),
                hang,
                0.025,
                IRON,
            );
            beam(out, hang, self.p(a, SHAFT + 2.5, b), 0.015, IRON);
        }
        let hook = 2.0 + 0.63;
        // By the door: a bracket out from the wall.
        let hw = Self::hw(hook);
        let wall = self.p(1.25, hook + 0.03, -hw + 0.02);
        let end = self.p(1.25, hook + 0.03, -(WALL_OUT + 0.25));
        beam(out, wall, end, 0.022, IRON);
        beam(
            out,
            self.p(1.25, hook - 0.4, -hw + 0.02),
            self.p(1.25, hook + 0.03, -hw - 0.2),
            0.016,
            IRON,
        );
        self.lbox(out, 1.15, 1.35, hook - 0.5, hook + 0.13, -hw - 0.02, -hw + 0.04, IRON);
        // In the stairwell: an arm from the back wall.
        let arm_a = self.p(-0.25, hook + 0.03, INNER);
        let arm_b = self.p(-0.25, hook + 0.03, 0.25);
        beam(out, arm_a, arm_b, 0.02, IRON);
        beam(
            out,
            self.p(-0.25, hook - 0.35, INNER),
            self.p(-0.25, hook + 0.03, 0.8),
            0.015,
            IRON,
        );
    }

    /// A light stand-in for far away: podium, tapering shaft, corbel band,
    /// the lookout with its lit windows, the flared roof and the lit windows
    /// of the shaft. Every part sits just inside the near model's surfaces,
    /// so where both are drawn for a moment the model wins.
    pub fn far(&self, out: &mut MeshData) {
        let b = |out: &mut MeshData, lo: Vec2, hi: Vec2, y0: f32, y1: f32, mat: Block| {
            let lo = self.p(lo.x, y0, lo.y);
            let hi = self.p(hi.x, y1, hi.y);
            crate::structures::boxed(out, lo.min(hi), lo.max(hi), mat);
        };
        let sq = |h: f32| (Vec2::splat(-h), Vec2::splat(h));
        let (lo, hi) = sq(PLINTH_HALF - 0.05);
        b(out, lo, hi, self.base - self.gf(), -0.05, MASONRY);
        let (lo, hi) = sq(TOP_HALF + 0.3);
        b(out, lo, hi, SHAFT - 1.9, SHAFT - 0.5, MASONRY);
        let (lo, hi) = sq(LOOK - 0.2);
        b(out, lo, hi, SHAFT - 0.45, SHAFT + RAIL - 0.05, BOARDS);
        let pl = LOOK - 0.12;
        let top = SHAFT + ROOM - 0.12;
        let (lo, hi) = sq(pl + 0.1);
        b(out, lo, hi, top + 0.02, top + 0.24, OAK);
        for (sa, sb) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let c = Vec2::new(sa * pl, sb * pl);
            b(out, c - 0.12, c + 0.12, SHAFT, top, OAK);
        }
        let apex_y = apex();
        let e = EAVE - 0.05;
        let (s0, s1) = (0.0, SHAFT - 0.5);
        for k in 0..4 {
            let (n, tg) = Self::face(k);
            // The shaft's face, tapering.
            let (h0, h1) = (Self::hw(s0) - 0.04, Self::hw(s1) - 0.04);
            panel(
                out,
                &[
                    self.fp(k, h0, -h0, s0, 0.0),
                    self.fp(k, h0, h0, s0, 0.0),
                    self.fp(k, h1, h1, s1, 0.0),
                    self.fp(k, h1, -h1, s1, 0.0),
                ],
                MASONRY,
            );
            // The lookout's windows.
            let (w0, w1) = (SHAFT + RAIL, top - 0.02);
            panel(
                out,
                &[
                    self.fp(k, pl, -pl, w0, -0.04),
                    self.fp(k, pl, pl, w0, -0.04),
                    self.fp(k, pl, pl, w1, -0.04),
                    self.fp(k, pl, -pl, w1, -0.04),
                ],
                GLOW,
            );
            let q = |m: f32, t: f32| {
                let v = n * m + tg * t;
                self.p(v.x, roof_y(m) - 0.04, v.y)
            };
            panel(out, &[q(e, -e), q(e, e), q(KICK, KICK), q(KICK, -KICK)], ROOF);
            panel(
                out,
                &[q(KICK, -KICK), q(KICK, KICK), self.p(0.0, apex_y - 0.04, 0.0)],
                ROOF,
            );
            // The arched windows of the shaft.
            for (t, y, h, w, arch) in Self::lights(k) {
                if !arch {
                    continue;
                }
                let hw = Self::hw(y + h * 0.5);
                let pts = [
                    self.fp(k, hw, t - w, y + 0.02, 0.002),
                    self.fp(k, hw, t + w, y + 0.02, 0.002),
                    self.fp(k, hw, t + w, y + h + w * 0.8, 0.002),
                    self.fp(k, hw, t - w, y + h + w * 0.8, 0.002),
                ];
                panel(out, &pts, GLOW);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{chunk_of, local_index, Chunk, CHUNK, CHUNK_VOLUME};
    use crate::player::{MoveInput, Player};
    use crate::world::World;
    use glam::IVec3;

    const G: f32 = 40.0;

    fn tower() -> Watchtower {
        Watchtower::new(Vec2::new(100.0, 100.0), G + 1.0, G, Vec2::new(0.0, -1.0), 7)
    }

    /// A world of flat ground at `G` with just the tower on it.
    fn world(w: &Watchtower) -> World {
        let mut world = World::new(1, 1);
        let (lo, hi) = w.bounds();
        let c0 = chunk_of((lo / VOXEL_SIZE).floor().as_ivec3() - 2);
        let c1 = chunk_of((hi / VOXEL_SIZE).ceil().as_ivec3() + 2);
        for cz in c0.z..=c1.z {
            for cy in c0.y..=c1.y {
                for cx in c0.x..=c1.x {
                    let cpos = IVec3::new(cx, cy, cz);
                    let mut data = Box::new([AIR; CHUNK_VOLUME]);
                    for z in 0..CHUNK {
                        for y in 0..CHUNK {
                            for x in 0..CHUNK {
                                let v = cpos * CHUNK + IVec3::new(x, y, z);
                                let p = (v.as_vec3() + 0.5) * VOXEL_SIZE;
                                let mut b = if p.y < G { STONE } else { AIR };
                                if let Some(s) = w.block(p, G) {
                                    b = s;
                                }
                                data[local_index(x, y, z)] = b;
                            }
                        }
                    }
                    world.chunks.insert(cpos, Chunk::from_dense(data));
                }
            }
        }
        world
    }

    /// Walks the player through `path` (local a, b), steering at each point in turn.
    fn walk(world: &World, w: &Watchtower, start: Vec3, path: &[Vec2]) -> Player {
        let mut p = Player::new(start);
        let mut k = 0;
        for _ in 0..60 * 120 {
            if k >= path.len() {
                break;
            }
            let target = w.p(path[k].x, 0.0, path[k].y);
            let d = Vec2::new(target.x - p.pos.x, target.z - p.pos.z);
            if d.length() < 0.12 {
                k += 1;
                continue;
            }
            p.yaw = d.y.atan2(d.x);
            let input = MoveInput {
                forward: (d.length() * 2.0).min(1.0),
                ..Default::default()
            };
            p.update(world, input, 1.0 / 60.0);
        }
        assert_eq!(
            k,
            path.len(),
            "stuck at {:?} heading for {:?}",
            w.local(p.pos),
            path.get(k)
        );
        p
    }

    /// Centre of the tread of step `n`, kept clear of the walls.
    fn tread(n: i32) -> Vec2 {
        let (mut lo, mut hi) = (Vec2::splat(9.0), Vec2::splat(-9.0));
        for j in 0..6 {
            for i in 0..6 {
                if step_base(i, j) == Some(n % TURN) {
                    let q = Vec2::new(i as f32, j as f32) * VOXEL_SIZE - INNER;
                    lo = lo.min(q);
                    hi = hi.max(q + VOXEL_SIZE);
                }
            }
        }
        ((lo + hi) * 0.5).clamp(Vec2::splat(-1.15), Vec2::splat(1.15))
    }

    #[test]
    fn every_ring_cell_has_one_step_and_the_well_none() {
        let mut seen = [0; TURN as usize];
        for j in 0..6 {
            for i in 0..6 {
                match step_base(i, j) {
                    Some(n) => seen[n as usize] += 1,
                    None => assert!((2..4).contains(&i) && (2..4).contains(&j)),
                }
            }
        }
        assert!(seen.iter().all(|&c| c == 2), "{seen:?}");
    }

    #[test]
    fn the_door_is_open_and_the_walls_are_solid() {
        let w = tower();
        let at = |a: f32, dy: f32, b: f32| w.block(w.p(a, dy + 0.25, b), G);
        assert_eq!(at(0.25, 0.0, -4.75), Some(AIR), "doorway");
        assert_eq!(at(0.25, 0.0, -2.25), Some(AIR), "passage through the wall");
        assert_eq!(at(0.25, 2.0, -1.75), Some(AIR), "doorway head");
        assert_eq!(at(0.25, 2.5, -4.75), Some(BUILT), "over the door");
        assert_eq!(at(1.25, 1.0, -4.75), Some(BUILT), "front wall");
        assert_eq!(at(-4.75, 5.0, 0.25), Some(BUILT), "side wall");
        assert_eq!(at(0.25, 0.0, -0.75), Some(AIR), "landing inside the door");
        assert_eq!(at(0.25, -1.0, -5.75), Some(BUILT), "podium");
        assert_eq!(at(0.25, -1.0, -6.25), Some(BUILT), "top step");
        assert_eq!(at(0.25, 0.0, -6.25), Some(AIR), "over the top step");
        assert_eq!(at(1.75, SHAFT + 0.0, 1.75), Some(AIR), "standing room in the lookout");
        assert_eq!(
            at(LOOK - 0.75, SHAFT - 0.5, 0.25),
            Some(BUILT),
            "lookout floor over the drop"
        );
        assert_eq!(at(LOOK - 0.75, SHAFT - 1.5, 0.25), Some(AIR), "under the overhang");
        assert_eq!(at(LOOK - 0.25, SHAFT + 0.5, 0.25), Some(BUILT), "breastwork");
        assert_eq!(at(LOOK - 0.25, SHAFT + 2.0, 0.25), Some(BUILT), "windows");
        assert_eq!(at(LOOK - 0.75, SHAFT + 2.0, 0.25), Some(AIR), "inside the windows");
        assert_eq!(at(0.25, apex() - 1.0, 0.25), Some(BUILT), "roof");
    }

    #[test]
    fn you_can_climb_the_stair_to_the_lookout() {
        let w = tower();
        let world = world(&w);
        // Up the steps onto the podium, in through the wall, and round.
        let mut path = vec![
            Vec2::new(0.0, -12.0),
            Vec2::new(0.0, -7.0),
            Vec2::new(0.0, -5.5),
            Vec2::new(0.0, -2.0),
            Vec2::new(0.0, -1.0),
        ];
        for n in 0..=STEPS {
            path.push(tread(n));
        }
        // Out to the lookout's windows over the overhang.
        // Round the hatch, which is open over the top turn of the stair.
        path.push(Vec2::new(-1.8, 2.0));
        path.push(Vec2::new(1.8, 2.2));
        path.push(Vec2::new(5.4, 2.2));
        let start = w.p(0.0, 0.0, -13.0);
        let p = walk(&world, &w, Vec3::new(start.x, G, start.z), &path);
        assert!(
            (p.pos.y - (w.gf() + SHAFT)).abs() < 0.05,
            "feet at {}",
            p.pos.y - w.gf()
        );
    }

    #[test]
    fn you_cannot_walk_through_the_walls() {
        let w = tower();
        let world = world(&w);
        // From the podium's ledge, beside the door.
        let mut p = Player::new(w.p(2.0, 0.0, -5.6));
        let target = w.p(2.0, 0.0, 0.0);
        for _ in 0..60 * 6 {
            let d = Vec2::new(target.x - p.pos.x, target.z - p.pos.z);
            p.yaw = d.y.atan2(d.x);
            p.update(
                &world,
                MoveInput {
                    forward: 1.0,
                    ..Default::default()
                },
                1.0 / 60.0,
            );
        }
        let (_, b) = w.local(p.pos);
        assert!(b < -WALL_OUT, "got in through the wall to b = {b}");
    }

    #[test]
    fn the_model_is_a_modest_number_of_triangles() {
        let t = Terrain::new(20260927);
        let w = tower();
        let mut out = MeshData::default();
        w.model(&t, &mut out);
        let tris = out.indices.len() / 3;
        assert!(tris > 3000 && tris < 25000, "{tris} triangles");
        let mut far = MeshData::default();
        w.far(&mut far);
        assert!(far.indices.len() / 3 < 300, "{} far triangles", far.indices.len() / 3);
    }

    #[test]
    fn proportions_match_the_reference() {
        // A stout stone tower, not a spire: ten metres square, about three
        // times as tall to the lookout (podium included) as it is wide.
        let height = LIFT + apex() + 2.2;
        assert!((32.0..=38.0).contains(&height), "{height} m tall");
        assert!(
            (9.5..=12.0).contains(&(2.0 * FOOT_HALF)),
            "shaft {} m wide",
            2.0 * FOOT_HALF
        );
        let stout = (LIFT + SHAFT) / (2.0 * FOOT_HALF);
        assert!((2.0..=2.8).contains(&stout), "shaft {stout} times as tall as wide");
        // The lookout overhangs the shaft clearly, and the roof the lookout.
        const { assert!(LOOK - TOP_HALF >= 1.5, "the lookout projects past the shaft") };
        const { assert!(EAVE - LOOK >= 1.5, "the eaves overhang the lookout") };
        // A broad lookout roof, not a needle: lower than it is wide.
        let rise = apex() - (SHAFT + ROOM);
        assert!(rise < EAVE, "roof rises {rise} m over {} m", 2.0 * EAVE);
        // The podium lifts the door clear of the knoll top.
        const { assert!(LIFT >= 2.5) };
    }
}
