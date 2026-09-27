//! The first view of the world, composed on purpose rather than left to noise.
//!
//! The player arrives on a grassy rise on the east bank, facing upstream into
//! a low sun. Below on the left the river runs in a shallow gorge, stepping
//! down a cascade seen through the arches of the stone bridge, which a lantern
//! lit path curves down to. On the far bank a rounded knoll holds a watchtower
//! in front of rolling wooded hills, and ahead and to the right a sheer rock
//! bluff carries the castle, with thin waterfalls pouring off its faces.
//!
//! Every feature here is a shaping of the ground height that `Terrain::height_at`
//! applies, so columns, far tiles, trees and structures all see it. Each one
//! fades smoothly into the procedural valley and is skipped outright by a cheap
//! distance test away from it.

use crate::block::VOXEL_SIZE;
use crate::far::face;
use crate::mesh::MeshData;
use crate::noise::fbm2;
use crate::terrain::{smooth_water_level, smoothstep, water_level};
use glam::{Vec2, Vec3};

/// Where the player arrives (metres): the middle of the spawn rise's flat top,
/// with `y` its ground height, 8 m above the home reach of the river.
pub const SPAWN: (f32, f32, f32) = (900.0, 32.5, 1024.0);
/// Radius (metres) of the rise's flat top, kept clear for a big tree.
pub const RISE_TOP_R: f32 = 8.5;
/// Radius (metres) where the rise has blended back into the bank.
const RISE_R: f32 = 30.0;

/// Z (metres) of the stone bridge (structures plan it 72 m ahead of spawn).
pub const BRIDGE_Z: f32 = 952.0;

/// Centre of the castle bluff's flat top (x, ground height, z) in metres.
pub const CASTLE_TOP: (f32, f32, f32) = (1058.0, 105.0, 648.0);
/// Half extents (x, z) of the bluff's flat top: room for the castle's bailey
/// and corner towers with a margin.
const CASTLE_HALF: (f32, f32) = (27.0, 21.0);
/// Horizontal depth of the sheer faces towards the river (-X) and the viewer (+Z).
pub const FACE_W: f32 = 7.0;
/// Depth of the steep, ledged faces on the back (+X and -Z).
const BACK_W: f32 = 22.0;

/// Top of the rounded knoll on the west bank that holds the watchtower.
pub const TOWER_KNOLL: (f32, f32, f32) = (860.0, 52.0, 810.0);
/// Radius (metres) of the knoll's rounded top, kept clear for the tower.
const KNOLL_TOP_R: f32 = 7.0;
/// Radius where the knoll meets the valley floor.
const KNOLL_R: f32 = 62.0;

/// A thin waterfall pouring off a face of the castle bluff.
pub struct CliffFall {
    /// Centre of the lip where it leaves the top (metres).
    pub x: f32,
    pub z: f32,
    /// Half its width in metres.
    pub half_w: f32,
    /// The face it pours down, as an index into the mesher's face table:
    /// 4 is the +Z face (towards the viewer), 1 the -X face (towards the river).
    pub face: usize,
}

impl CliffFall {
    /// Outward direction of its face, in XZ.
    fn out(&self) -> Vec2 {
        if self.face == 4 {
            Vec2::Y
        } else {
            Vec2::NEG_X
        }
    }

    /// Where it lands (metres, XZ): just past the foot of the face, in its pool.
    pub fn foot(&self) -> Vec2 {
        Vec2::new(self.x, self.z) + self.out() * (FACE_W + 1.0)
    }
}

/// The bluff's waterfalls: two off the face towards the viewer into a plunge
/// pool that opens into the river, one off the river face.
pub const CLIFF_FALLS: [CliffFall; 3] = [
    CliffFall {
        x: 1044.0,
        z: CASTLE_TOP.2 + CASTLE_HALF.1,
        half_w: 1.25,
        face: 4,
    },
    CliffFall {
        x: 1069.0,
        z: CASTLE_TOP.2 + CASTLE_HALF.1,
        half_w: 1.0,
        face: 4,
    },
    CliffFall {
        x: CASTLE_TOP.0 - CASTLE_HALF.0,
        z: 655.0,
        half_w: 1.25,
        face: 1,
    },
];

/// Plunge pools at the foot of the bluff: capsules (from, to, radius) in metres,
/// dug below the local water level so they fill.
const POOLS: [(Vec2, Vec2, f32); 2] = [
    (Vec2::new(998.0, 680.0), Vec2::new(1074.0, 680.0), 4.5),
    (Vec2::new(1010.0, 655.0), Vec2::new(1025.0, 655.0), 4.0),
];

/// Which way the player faces on arrival (`Player::new`), as a yaw in radians,
/// and the wedge of meadow ahead of the rise kept free of trees.
const VIEW_YAW: f32 = -1.3;
const VIEW_HALF_ANGLE: f32 = 0.42;
const VIEW_CLEAR_M: f32 = 100.0;

/// The path from the spawn rise down to the bridge's east end: its Z range,
/// the X it starts from at the bridge and the sideways swing of its curve.
const PATH_Z: (f32, f32) = (BRIDGE_Z + 1.0, SPAWN.2 - RISE_TOP_R + 1.5);
const PATH_X0: f32 = 903.5;
const PATH_SWING: f32 = 7.0;
/// Half the width of the path's packed surface.
pub const PATH_HALF_WIDTH_M: f32 = 1.2;
/// Lanterns stand this far apart along the path, alternating sides.
pub const PATH_LANTERN_SPACING_M: f32 = 14.0;

/// How strongly a Z lies in the hamlet's reach of the river, where it runs
/// in a gorge (0..1).
fn gorge(z: f32) -> f32 {
    smoothstep(840.0, 880.0, z) * (1.0 - smoothstep(1060.0, 1100.0, z))
}

/// Distances (metres from the river's centre line) over which the river bed
/// blends into the banks: narrow, so the channel walls stand steep, in the gorge.
pub fn channel_edges(z: f32) -> (f32, f32) {
    let g = gorge(z);
    (7.0 + 2.0 * g, 17.0 - 5.0 * g)
}

/// How far (metres) the valley road swings east to pass behind the castle bluff.
pub fn road_detour(z: f32) -> f32 {
    72.0 * smoothstep(555.0, 610.0, z) * (1.0 - smoothstep(690.0, 745.0, z))
}

/// Composed shaping applied to the procedural ground before the river channel
/// is cut: hills behind the far bank, the tower knoll, the gorge's banks and the
/// spawn rise. `rx` is the river's centre X at this Z.
pub fn before_channel(seed: u32, x: f32, z: f32, rx: f32, mut h: f32) -> f32 {
    let d = (x - rx).abs();

    // Rolling hills rise behind the knoll on the west bank.
    let west = rx - x;
    if (100.0..420.0).contains(&west) && (560.0..1000.0).contains(&z) {
        let zone =
            smoothstep(560.0, 640.0, z) * (1.0 - smoothstep(900.0, 1000.0, z)) * (1.0 - smoothstep(340.0, 420.0, west));
        h += 16.0 * smoothstep(100.0, 200.0, west) * zone * fbm2(seed.wrapping_add(64), x / 85.0, z / 85.0, 3);
    }

    // The tower knoll: a rounded dome, grassy all over.
    let (kx, ky, kz) = TOWER_KNOLL;
    let kr = Vec2::new(x - kx, z - kz).length();
    if kr < KNOLL_R {
        let dome = ky - (ky - 30.0) * smoothstep(KNOLL_TOP_R, KNOLL_R, kr);
        h = h.max(dome);
    }

    // In the gorge the banks stand 5 to 6.5 m above the water: a little lower at
    // the lip of the channel walls, so the bridge's ramps meet them.
    let g = gorge(z);
    if g > 0.0 && d > 8.0 && d < 90.0 {
        let out = smoothstep(17.0, 32.0, d);
        let wobble = 0.8 * (fbm2(seed.wrapping_add(65), x / 23.0, z / 23.0, 2) - 0.5);
        let bank = smooth_water_level(z) + 4.7 + (1.8 + wobble) * out;
        h += (bank - h).max(0.0) * g;
    }

    // The spawn rise: a flat top, then a gentle fall back to the bank.
    let (sx, sy, sz) = SPAWN;
    let sr = Vec2::new(x - sx, z - sz).length();
    if sr < RISE_R {
        h += (sy - h).max(0.0) * (1.0 - smoothstep(RISE_TOP_R, RISE_R, sr));
    }
    h
}

/// The castle bluff's height at a point, if the point lies on it, and how far
/// out from the top it is in face depths (0 at the edge of the top, 1 at the foot).
fn bluff(seed: u32, x: f32, z: f32) -> Option<f32> {
    let (cx, top, cz) = CASTLE_TOP;
    let (hx, hz) = CASTLE_HALF;
    let (u, v) = (x - cx, z - cz);
    let reach = BACK_W + 3.0;
    if u.abs() > hx + reach || v.abs() > hz + reach {
        return None;
    }
    // The edges wander a little in plan, but run straight where water pours over.
    let calm = CLIFF_FALLS
        .iter()
        .map(|f| {
            let along = if f.face == 4 { x - f.x } else { z - f.z };
            smoothstep(f.half_w + 1.0, f.half_w + 6.0, along.abs())
        })
        .fold(1.0f32, f32::min);
    let wob = |k: u32, t: f32| 4.0 * (fbm2(seed.wrapping_add(k), t / 16.0, 0.5, 2) - 0.5) * calm;
    let rx = if u < 0.0 {
        (-u - hx + wob(66, v)) / FACE_W
    } else {
        (u - hx + wob(67, v)) / BACK_W
    };
    let rz = if v > 0.0 {
        (v - hz + wob(68, u)) / FACE_W
    } else {
        (-v - hz + wob(69, u)) / BACK_W
    };
    let (a, b) = (rx.max(0.0), rz.max(0.0));
    let r = (a * a + b * b).sqrt() + rx.max(rz).min(0.0);
    if r >= 1.2 {
        return None;
    }
    // Three sheer drops with two narrow ledges between them; the ledges wander
    // up and down a little so they read as rock bands rather than stairs.
    let j = 0.07 * (fbm2(seed.wrapping_add(70), (x + z) / 19.0, (x - z) / 31.0, 2) - 0.5);
    let fall = 0.36 * smoothstep(0.0, 0.16, r)
        + 0.30 * smoothstep(0.38 + j, 0.53 + j, r)
        + 0.34 * smoothstep(0.76 - j, 0.95, r);
    let top_h = top + 0.5 * (fbm2(seed.wrapping_add(71), x / 9.0, z / 9.0, 2) - 0.5);
    let base = smooth_water_level(z) - 4.0;
    Some(top_h - (top_h - base) * fall)
}

/// Whether a point is on the castle bluff, where no snow lies: its top is grass
/// and its faces bare rock.
pub fn snow_free(x: f32, z: f32) -> bool {
    let (cx, _, cz) = CASTLE_TOP;
    let (hx, hz) = CASTLE_HALF;
    (x - cx).abs() < hx + BACK_W + 3.0 && (z - cz).abs() < hz + BACK_W + 3.0
}

/// Composed shaping applied after the river channel is cut: the castle bluff,
/// whose river face drops straight into the water, and the plunge pools at its foot.
pub fn after_channel(seed: u32, x: f32, z: f32, rx: f32, mut h: f32) -> f32 {
    if let Some(b) = bluff(seed, x, z) {
        let d = (x - rx).abs();
        h += (b - h).max(0.0) * smoothstep(9.0, 14.0, d);
    }
    for (a, b, r) in POOLS {
        let p = Vec2::new(x, z);
        let lo = a.min(b) - r - 2.0;
        let hi = a.max(b) + r + 2.0;
        if p.x < lo.x || p.y < lo.y || p.x > hi.x || p.y > hi.y {
            continue;
        }
        let ab = b - a;
        let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
        let dist = p.distance(a + ab * t);
        let bottom = water_level(x, z) - 2.2;
        h += (bottom - h).min(0.0) * (1.0 - smoothstep(r - 1.5, r + 1.5, dist));
    }
    h
}

/// Top (metres) of the water a cliff fall stands in a column, if the column is
/// on one: the sheet hugs the face, filling each drop in front of the rock and
/// running a voxel deep over the ledges and the lip. `ground` is this column's
/// height and `height` the terrain's height function.
pub fn curtain_top(x: f32, z: f32, ground: f32, height: impl Fn(f32, f32) -> f32) -> Option<f32> {
    for f in &CLIFF_FALLS {
        let (along, across, inner) = if f.face == 4 {
            (x - f.x, z - f.z, (x, z - VOXEL_SIZE))
        } else {
            (z - f.z, f.x - x, (x + VOXEL_SIZE, z))
        };
        if along.abs() > f.half_w || !(-2.5..FACE_W + 1.5).contains(&across) {
            continue;
        }
        let inner_h = height(inner.0, inner.1);
        return Some(inner_h.max(ground + VOXEL_SIZE).min(CASTLE_TOP.1 + VOXEL_SIZE));
    }
    None
}

/// Whether any cliff fall's water could reach into a box of metres.
pub fn curtain_touches(lo: Vec3, hi: Vec3) -> bool {
    CLIFF_FALLS.iter().any(|f| {
        let (a, b) = (Vec2::new(f.x, f.z), f.foot());
        let (mn, mx) = (a.min(b) - 3.0, a.max(b) + 3.0);
        mn.x < hi.x && mx.x > lo.x && mn.y < hi.z && mx.y > lo.z && lo.y < CASTLE_TOP.1 + 1.0
    })
}

/// Far away the cliff falls are single sheets standing in front of their faces,
/// from the pool to the lip. Adds those whose lip lies in [lo, hi).
pub fn far_curtains(out: &mut MeshData, lo: Vec2, hi: Vec2) {
    for f in &CLIFF_FALLS {
        if f.x < lo.x || f.z < lo.y || f.x >= hi.x || f.z >= hi.y {
            continue;
        }
        let foot = Vec2::new(f.x, f.z) + f.out() * (FACE_W + 0.6);
        let bottom = water_level(foot.x, foot.y) - 0.06;
        let top = CASTLE_TOP.1;
        let (a, b) = if f.face == 4 {
            (
                Vec3::new(f.x - f.half_w, bottom, foot.y),
                Vec3::new(f.x + f.half_w, top, foot.y),
            )
        } else {
            (
                Vec3::new(foot.x, bottom, f.z - f.half_w),
                Vec3::new(foot.x, top, f.z + f.half_w),
            )
        };
        face(
            &mut out.water_vertices,
            &mut out.water_indices,
            a,
            b,
            f.face,
            crate::block::WATER,
        );
    }
}

/// X of the path's centre line at a Z, if the path reaches that Z.
pub fn path_x(z: f32) -> Option<f32> {
    let (z0, z1) = PATH_Z;
    if !(z0..=z1).contains(&z) {
        return None;
    }
    let t = (z - z0) / (z1 - z0);
    Some(PATH_X0 + (SPAWN.0 - PATH_X0) * t + PATH_SWING * (t * std::f32::consts::PI).sin())
}

/// Roughly how far (metres) a point is from the path's centre line.
pub fn path_distance(x: f32, z: f32) -> f32 {
    let (z0, z1) = PATH_Z;
    let Some(px) = path_x(z.clamp(z0, z1)) else {
        return f32::MAX;
    };
    let t = (z.clamp(z0, z1) - z0) / (z1 - z0);
    let slope =
        ((SPAWN.0 - PATH_X0) + PATH_SWING * std::f32::consts::PI * (t * std::f32::consts::PI).cos()) / (z1 - z0);
    let across = (x - px).abs() / (1.0 + slope * slope).sqrt();
    let beyond = (z0 - z).max(z - z1).max(0.0);
    (across * across + beyond * beyond).sqrt()
}

/// Lantern posts along the path (metres): each post's foot, and which way along
/// X its lantern hangs, towards the path.
pub fn path_lanterns() -> impl Iterator<Item = (f32, f32, f32)> {
    let (z0, z1) = PATH_Z;
    let n = ((z1 - z0) / PATH_LANTERN_SPACING_M) as i32;
    (0..n).filter_map(move |k| {
        let z = z0 + PATH_LANTERN_SPACING_M * (k as f32 + 0.5);
        let side = if k % 2 == 0 { 2.3 } else { -2.3 };
        path_x(z).map(|x| (x + side, z, -side.signum()))
    })
}

/// Whether a tree trunk here would crowd a composed feature kept open for
/// something else: the spawn rise's top (for its big tree), the knoll's top
/// (for the watchtower) and the path.
pub fn keeps_clear(x: f32, z: f32) -> bool {
    let (dx, dz) = (x - SPAWN.0, z - SPAWN.2);
    let r = (dx * dx + dz * dz).sqrt();
    // The meadow ahead of the rise stays open, so the bridge, the knoll and the
    // castle are seen on arrival rather than a wall of nearby crowns.
    let ahead = r > RISE_TOP_R + 1.5 && r < VIEW_CLEAR_M && (dz.atan2(dx) - VIEW_YAW).abs() < VIEW_HALF_ANGLE;
    ahead
        || r < RISE_TOP_R + 1.5
        || Vec2::new(x - TOWER_KNOLL.0, z - TOWER_KNOLL.2).length() < KNOLL_TOP_R + 4.0
        || path_distance(x, z) < 3.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{STONE, WATER};
    use crate::terrain::Terrain;
    use glam::IVec3;

    fn world() -> Terrain {
        Terrain::new(20260927)
    }

    fn block(t: &mut Terrain, v: IVec3) -> crate::block::Block {
        let l = crate::chunk::local_of(v);
        t.generate(crate::chunk::chunk_of(v)).get(l.x, l.y, l.z)
    }

    #[test]
    fn spawn_rise_stands_above_the_river() {
        let t = world();
        let s = t.spawn_point();
        let (x, z) = (SPAWN.0, SPAWN.2);
        assert_eq!((s.x, s.z), (x, z));
        let river = water_level(t.river_x(z), z);
        let top = t.height_at(x, z).0;
        assert!((7.0..=9.0).contains(&(top - river)), "rise {top} over water {river}");
        // The top is flat enough for a big tree.
        for k in 0..16 {
            let a = k as f32 / 16.0 * std::f32::consts::TAU;
            let h = t.height_at(x + a.cos() * RISE_TOP_R, z + a.sin() * RISE_TOP_R).0;
            assert!((h - top).abs() < 0.8, "rise edge at {a}: {h} vs {top}");
        }
        assert!(s.y > top);
    }

    #[test]
    fn gorge_banks_stand_above_the_water_near_the_bridge() {
        let t = world();
        for z in [890.0, 952.0, 980.0, 1050.0] {
            let rx = t.river_x(z);
            let wl = water_level(rx, z);
            for side in [-1.0f32, 1.0] {
                let lip = t.height_at(rx + side * 13.0, z).0 - wl;
                let bank = t.height_at(rx + side * 30.0, z).0 - wl;
                assert!(lip > 4.5, "z {z} side {side}: lip {lip}");
                assert!(bank > 5.5, "z {z} side {side}: bank {bank}");
                // The channel wall between the water and the lip is steep rock.
                let wall = t.column_info(rx + side * 10.75, z);
                assert_eq!(wall.surface, STONE, "z {z} side {side}");
            }
            assert!(t.height_at(rx, z).0 < wl - 1.5);
        }
    }

    #[test]
    fn castle_bluff_towers_over_the_river() {
        let t = world();
        let (x, y, z) = CASTLE_TOP;
        let river = water_level(t.river_x(z), z);
        let top = t.height_at(x, z).0;
        assert!((top - y).abs() < 0.5);
        assert!((60.0..=100.0).contains(&(top - river)), "top {top}, river {river}");
        // The top is flat across the castle.
        for (dx, dz) in [(-25.0, -18.0), (25.0, -18.0), (-25.0, 18.0), (25.0, 18.0)] {
            let h = t.height_at(x + dx, z + dz).0;
            assert!((h - top).abs() < 1.5, "corner {dx},{dz}: {h}");
        }
        // Sheer towards the river and towards the viewer.
        let front = t.height_at(x, z + CASTLE_HALF.1 + FACE_W + 1.0).0;
        let side = t.height_at(x - CASTLE_HALF.0 - FACE_W - 1.0, z).0;
        assert!(top - front > 55.0 && top - side > 55.0, "{front} {side}");
        assert_eq!(t.column_info(x, z).surface, crate::block::GRASS);
        let f = &CLIFF_FALLS[0];
        assert_eq!(t.column_info(f.x, f.z + 0.5).surface, STONE);
        // The castle's spires stay under the top of the world.
        let world_top = (crate::terrain::WORLD_CHUNKS_Y * crate::chunk::CHUNK) as f32 * VOXEL_SIZE;
        for st in t.structures.iter() {
            if let crate::structures::Structure::Castle(c) = st {
                assert!((c.c.x - x).abs() < 0.5 && (c.c.y - z).abs() < 0.5);
                for tw in &c.towers {
                    assert!(tw.top + tw.room + tw.cone_h < world_top - 4.0);
                }
            }
        }
    }

    #[test]
    fn tower_knoll_rises_on_the_west_bank() {
        let t = world();
        let (x, y, z) = TOWER_KNOLL;
        assert!(x < t.river_x(z) - 40.0);
        let river = water_level(t.river_x(z), z);
        let top = t.height_at(x, z).0;
        assert!((top - y).abs() < 1.0);
        assert!((22.0..=28.0).contains(&(top - river)), "knoll {top}, river {river}");
        assert_eq!(t.column_info(x, z).surface, crate::block::GRASS);
    }

    #[test]
    fn cliff_falls_pour_water_down_the_faces() {
        let mut t = world();
        for f in &CLIFF_FALLS {
            let mut wet = 0;
            let mut levels = Vec::new();
            for step in 0..=(FACE_W / VOXEL_SIZE) as i32 {
                let p = Vec2::new(f.x, f.z) + f.out() * (step as f32 * VOXEL_SIZE + 0.25);
                let g = t.height_at(p.x, p.y).0;
                let Some(top) = curtain_top(p.x, p.y, g, |x, z| t.height_at(x, z).0) else {
                    continue;
                };
                // Water just above the ground and at the top of the sheet.
                for y in [g + 0.3, top - 0.3] {
                    let v = (Vec3::new(p.x, y, p.y) / VOXEL_SIZE).floor().as_ivec3();
                    wet += (block(&mut t, v) == WATER) as i32;
                }
                levels.push(top);
            }
            let span = levels.iter().fold(f32::MIN, |a, b| a.max(*b)) - levels.iter().fold(f32::MAX, |a, b| a.min(*b));
            assert!(span > 50.0, "fall at {},{} spans {span} m", f.x, f.z);
            assert!(wet >= 20, "fall at {},{}: {wet} wet samples", f.x, f.z);
        }
    }

    #[test]
    fn far_field_draws_the_cliff_falls() {
        let mut out = MeshData::default();
        far_curtains(&mut out, Vec2::ZERO, Vec2::splat(crate::terrain::WORLD_SIZE_M));
        assert_eq!(out.water_indices.len(), 6 * CLIFF_FALLS.len());
    }

    #[test]
    fn shader_knows_the_cliff_falls() {
        let wgsl = include_str!("shaders/world.wgsl");
        assert!(wgsl.contains(&format!("const CLIFF_FALL_COUNT: u32 = {}u;", CLIFF_FALLS.len())));
        for f in &CLIFF_FALLS {
            let p = f.foot();
            assert!(wgsl.contains(&format!("vec2({:.1}, {:.1})", p.x, p.y)), "fall foot {p}");
        }
    }

    #[test]
    fn landmarks_stand_on_their_anchors() {
        let t = world();
        let mut seen = 0;
        for st in t.structures.iter() {
            match st {
                crate::structures::Structure::Bridge(b) => {
                    assert_eq!(b.z, BRIDGE_Z);
                    // Dry ground at both ends, above the water, and the path meets the east end.
                    for x in [b.x0 - 0.5, b.x1 + 0.5] {
                        assert!(t.height_at(x, b.z).0 > water_level(x, b.z) + 3.0, "bridge end {x}");
                    }
                    assert!((path_x(PATH_Z.0).unwrap() - b.x1).abs() < 2.0);
                    seen += 1;
                }
                crate::structures::Structure::Tower(tw) => {
                    assert!(tw.c.distance(Vec2::new(TOWER_KNOLL.0, TOWER_KNOLL.2)) < 0.5);
                    seen += 1;
                }
                _ => {}
            }
        }
        assert_eq!(seen, 2);
    }

    #[test]
    fn path_runs_from_the_rise_to_the_bridge() {
        let t = world();
        let (z0, z1) = PATH_Z;
        let mut z = z0;
        while z <= z1 {
            let x = path_x(z).unwrap();
            let c = t.column_info(x, z);
            assert_eq!(c.surface, crate::block::PATH, "path at {x},{z}");
            z += 2.0;
        }
        assert!(path_lanterns().count() >= 4);
    }
}
