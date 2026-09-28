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

use crate::block::{VOXEL_SIZE, WATER};
use crate::mesh::{smooth_data, MeshData, Vertex};
use crate::noise::{fbm2, value2};
use crate::terrain::{smooth_water_level, smoothstep, water_level};
use glam::{Vec2, Vec3};
use std::f32::consts::{PI, TAU};

/// The middle of the spawn rise's flat top (metres), with `y` its ground
/// height, 8 m above the home reach of the river. A big tree stands here.
pub const SPAWN: (f32, f32, f32) = (900.0, 32.5, 1024.0);
/// Where on the rise the player arrives (x, z in metres): near the edge of the
/// top on the river side, so the water shows below on the left.
pub const ARRIVAL: (f32, f32) = (893.0, 1024.0);
/// Which way the player faces on arrival, as a yaw in radians (forward is
/// (cos, 0, sin)): upstream, the castle a little right of centre, the low sun
/// and the watchtower ahead-left, the bridge and cascade just left of centre.
pub const SPAWN_YAW: f32 = -1.5;
/// Where the golden-hour sun stands, as a yaw: ahead-left of the arrival view.
pub const SUN_YAW: f32 = SPAWN_YAW - 0.3;
/// Radius (metres) of the rise's flat top, kept clear for a big tree.
pub const RISE_TOP_R: f32 = 8.5;
/// Radius (metres) where the rise has blended back into the bank.
const RISE_R: f32 = 22.0;

/// Z (metres) of the stone bridge (structures plan it 72 m ahead of spawn).
pub const BRIDGE_Z: f32 = 952.0;

/// Centre of the castle bluff's flat top (x, ground height, z) in metres.
pub const CASTLE_TOP: (f32, f32, f32) = (1058.0, 105.0, 642.0);
/// Half extents (x, z) of the terrace the castle needs flat: its bailey and
/// corner towers. The bluff's rim runs a few metres outside it and further on
/// the back, in lobes.
const TERRACE_HALF: (f32, f32) = (27.5, 20.5);
/// The bluff's buttresses: ribs of rock standing out of the faces, as angles
/// around the top's centre (0 is +X, PI/2 is +Z, towards the viewer).
const BUTTRESSES: [f32; 2] = [1.66, 2.35];
/// The far end of the ridge that ties the bluff back into the mountains, and
/// the height of its crest there.
const RIDGE_END: (f32, f32, f32) = (1205.0, 78.0, 568.0);

/// Top of the rounded knoll on the west bank that holds the watchtower.
pub const TOWER_KNOLL: (f32, f32, f32) = (860.0, 52.0, 810.0);
/// Radius (metres) of the knoll's rounded top, kept clear for the tower.
const KNOLL_TOP_R: f32 = 7.0;
/// Radius where the knoll meets the valley floor.
const KNOLL_R: f32 = 62.0;

/// A thin waterfall pouring off a face of the castle bluff.
pub struct CliffFall {
    /// Where it leaves the top, as an angle around the top's centre.
    pub angle: f32,
    /// Half its width in metres.
    pub half_w: f32,
}

impl CliffFall {
    /// Outward direction of its face, in XZ.
    pub fn out(&self) -> Vec2 {
        Vec2::new(self.angle.cos(), self.angle.sin())
    }

    /// Distances (metres) from the top's centre to the lip and to the foot of its face.
    fn reach(&self) -> (f32, f32) {
        let r = rim(self.angle);
        (r, r + face_depth(self.angle))
    }

    /// The middle of its lip (metres, XZ).
    pub fn lip(&self) -> Vec2 {
        centre() + self.out() * self.reach().0
    }

    /// Where it lands (metres, XZ): just past the foot of the face, in its pool.
    pub fn foot(&self) -> Vec2 {
        centre() + self.out() * (self.reach().1 + 1.0)
    }
}

/// The bluff's waterfalls: two off the face towards the viewer, either side of
/// a buttress, and one off the river face.
pub const CLIFF_FALLS: [CliffFall; 3] = [
    CliffFall {
        angle: 1.45,
        half_w: 1.25,
    },
    CliffFall {
        angle: 1.87,
        half_w: 1.0,
    },
    CliffFall {
        angle: 2.95,
        half_w: 1.25,
    },
];
/// Radius (metres) of the plunge pool dug at each fall's foot.
const POOL_R: f32 = 4.5;

/// Trees are kept off the lines of sight from the arrival spot to these (x, z),
/// within this many metres to either side: the bridge, the cascade, the castle.
const SIGHTS: [(f32, f32, f32); 3] = [
    (886.75, 952.0, 6.0),
    (891.0, 931.0, 5.0),
    (CASTLE_TOP.0, CASTLE_TOP.2, 7.0),
];
/// How far out (metres) trees are kept off those lines.
const SIGHT_CLEAR_M: f32 = 170.0;

/// The path from the spawn rise down to the bridge's east end: its Z range,
/// the X it starts from at the bridge, the X it ends at on the rise and the
/// sideways swing of its curve.
/// It starts at the player's feet and bows out around the east side of the
/// cottage whose deck stands over the river, then back to the bridge.
const PATH_Z: (f32, f32) = (BRIDGE_Z + 1.0, ARRIVAL.1 - 2.5);
const PATH_X0: f32 = 903.5;
const PATH_X1: f32 = ARRIVAL.0;
const PATH_SWING: f32 = 16.0;
/// Half the width of the path's packed surface.
pub const PATH_HALF_WIDTH_M: f32 = 1.2;
/// Lanterns stand this far apart along the path, alternating sides.
pub const PATH_LANTERN_SPACING_M: f32 = 14.0;

fn centre() -> Vec2 {
    Vec2::new(CASTLE_TOP.0, CASTLE_TOP.2)
}

/// Difference between two angles, wrapped into -PI..PI.
fn angle_diff(a: f32, b: f32) -> f32 {
    (a - b + PI).rem_euclid(TAU) - PI
}

/// How much an angle around the bluff faces the river (-X), 0..1.
fn river_side(angle: f32) -> f32 {
    smoothstep(0.3, 0.8, -angle.cos())
}

/// How much an angle around the bluff faces the viewer (+Z), 0..1.
fn front_side(angle: f32) -> f32 {
    smoothstep(0.55, 0.9, angle.sin())
}

/// Distance (metres) from the top's centre to the rim at an angle: just outside
/// the castle's terrace towards the river and the viewer, pushed out in broad
/// lobes on the back and in bays along the viewer's face.
fn rim(angle: f32) -> f32 {
    let (c, s) = (angle.cos().abs().max(1e-4), angle.sin().abs().max(1e-4));
    let terrace = (TERRACE_HALF.0 / c).min(TERRACE_HALF.1 / s);
    let back = (1.0 - river_side(angle)) * (1.0 - front_side(angle));
    let lobes = 12.0 * (0.5 + 0.5 * (3.0 * angle + 1.1).sin()) + 6.0 * (0.5 + 0.5 * (5.0 * angle + 0.4).sin());
    let bays = 5.0 * (0.5 + 0.5 * (7.0 * angle + 2.0).sin()) * front_side(angle);
    terrace + 2.0 + lobes * back + bays
}

/// How far (metres) a buttress stands out of the face below the rim at an angle.
fn buttress(angle: f32) -> f32 {
    BUTTRESSES
        .iter()
        .map(|&b| 8.0 * (-(angle_diff(angle, b) / 0.2).powi(2)).exp())
        .sum()
}

/// Horizontal depth (metres) of the near-vertical face below the rim at an angle.
fn face_depth(angle: f32) -> f32 {
    26.0 - 14.0 * river_side(angle) - 7.0 * front_side(angle)
}

/// How strongly a point is away from every cliff fall (0 on one, 1 clear of
/// them): where the water pours, the rim runs straight and the talus gives way
/// to the pool.
fn clear_of_falls(rel: Vec2) -> f32 {
    CLIFF_FALLS
        .iter()
        .map(|f| {
            let d = f.out();
            let lateral = if rel.dot(d) > 0.0 { d.perp_dot(rel).abs() } else { 1e3 };
            smoothstep(f.half_w + 1.5, f.half_w + 7.0, lateral)
        })
        .fold(1.0f32, f32::min)
}

/// Whether a point may lie on the bluff or its ridge: a cheap box test.
fn near_bluff(x: f32, z: f32) -> bool {
    let (cx, _, cz) = CASTLE_TOP;
    let (ex, _, ez) = RIDGE_END;
    x > cx - 80.0 && x < ex + 60.0 && z > ez - 60.0 && z < cz + 100.0
}

/// How strongly a Z lies in the hamlet's reach of the river, where it runs
/// in a gorge (0..1).
fn gorge(z: f32) -> f32 {
    smoothstep(840.0, 880.0, z) * (1.0 - smoothstep(1060.0, 1100.0, z))
}

/// How strongly a point lies on the viewer's own bank below the spawn rise,
/// where the gorge's lip is let down so the water shows from the rise (0..1).
fn viewer_bank(x: f32, z: f32, rx: f32) -> f32 {
    if x < rx {
        return 0.0;
    }
    smoothstep(956.0, 975.0, z) * (1.0 - smoothstep(1032.0, 1052.0, z))
}

/// Distances (metres from the river's centre line) over which the river bed
/// blends into the banks: narrow, so the channel walls stand steep, in the gorge,
/// and a little wider on the bank below the rise, which slopes to the water.
pub fn channel_edges(x: f32, z: f32, rx: f32) -> (f32, f32) {
    let g = gorge(z);
    let v = viewer_bank(x, z, rx);
    (7.0 + 2.0 * g, 17.0 - 5.0 * g + 2.0 * v)
}

/// How far (metres) the valley road swings east to pass behind the castle bluff.
pub fn road_detour(z: f32) -> f32 {
    90.0 * smoothstep(555.0, 610.0, z) * (1.0 - smoothstep(690.0, 745.0, z))
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
        h += 16.0 * smoothstep(100.0, 200.0, west) * zone * fbm2(seed.wrapping_add(64), x / 85.0, z / 85.0, 2);
    }

    // The tower knoll: a rounded dome, grassy all over.
    let (kx, ky, kz) = TOWER_KNOLL;
    let kr = Vec2::new(x - kx, z - kz).length();
    if kr < KNOLL_R {
        let dome = ky - (ky - 30.0) * smoothstep(KNOLL_TOP_R, KNOLL_R, kr);
        h = h.max(dome);
    }

    // In the gorge the banks stand 5 to 6.5 m above the water: a little lower at
    // the lip of the channel walls, so the bridge's ramps meet them. Below the
    // rise the near bank instead slopes down to the water's edge.
    let g = gorge(z);
    if g > 0.0 && d > 8.0 && d < 90.0 {
        let out = smoothstep(17.0, 32.0, d);
        let wobble = 0.8 * (value2(seed.wrapping_add(65), x / 23.0, z / 23.0) - 0.5);
        let wl = smooth_water_level(z);
        let gorge_bank = wl + 4.7 + (1.8 + wobble) * out;
        let v = viewer_bank(x, z, rx);
        let bank = gorge_bank + (wl + 0.9 + 5.2 * smoothstep(11.0, 34.0, d) + wobble - gorge_bank) * v;
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

/// The castle crag's height at a point, if the point lies on it.
///
/// The castle's terrace is a flat bench; around it the top rises into broken
/// rocky crowns and rounds over a ragged rim, notched here and there. Below the
/// rim the faces fall in four sheer drops with narrow grassy ledges between
/// them, the rock fluted and ribbed with buttresses, and at the foot a talus
/// of scree and boulders flares into the valley. Only shapes a few metres
/// across live here, so they hold up in the far field; the shader adds the
/// banding and the fine relief.
fn bluff(seed: u32, x: f32, z: f32) -> Option<f32> {
    let (_, top, _) = CASTLE_TOP;
    let rel = Vec2::new(x, z) - centre();
    if rel.x.abs() > 130.0 || rel.y.abs() > 130.0 {
        return None;
    }
    let calm = clear_of_falls(rel);
    let len = rel.length();
    let dir = rel / len.max(1e-3);
    // Noise running round the crag is sampled on a circle, so it has no seam.
    let ring = centre() + dir * 40.0;
    // The outline wanders outwards (never into the castle's terrace) in noisy
    // bulges, except where water pours over.
    let angle = rel.y.atan2(rel.x);
    let bulge = 12.0 * fbm2(seed.wrapping_add(66), x / 34.0, z / 34.0, 2) * calm;
    let mut s = len - rim(angle) - bulge;
    // Vertical flutes a couple of metres apart, and buttresses: ribs that lean
    // out of the face as it falls, two placed for the view and more from noise.
    s += 1.6 * (value2(seed.wrapping_add(74), ring.x / 2.6, ring.y / 2.6) - 0.5) * calm;
    let noisy = 7.0 * smoothstep(0.55, 0.85, value2(seed.wrapping_add(73), ring.x / 9.0, ring.y / 9.0));
    let rib = (buttress(angle) + noisy) * calm;
    if s > 0.0 && rib > 0.0 {
        s -= rib * smoothstep(0.0, 2.0 * rib, s);
    }
    let depth = face_depth(angle);
    let river = river_side(angle);
    // The talus: tall and wide below the back and the viewer's face, a mere
    // apron along the river, and none where the falls land in their pools.
    let talus_w = (34.0 - 29.0 * river - 12.0 * front_side(angle)) * (0.4 + 0.6 * calm);
    let talus_h = (30.0 - 27.0 * river - 14.0 * front_side(angle)) * calm + 1.0;
    if s > depth + talus_w {
        return None;
    }
    let base = smooth_water_level(z) - 3.0;
    let foot = smooth_water_level(z) + talus_h;
    // Metres outside the terrace (and a margin round it), where the castle stands.
    let out = (rel.abs() - Vec2::new(TERRACE_HALF.0 + 1.5, TERRACE_HALF.1 + 1.5))
        .max(Vec2::ZERO)
        .length();
    // Rocky crowns rise beyond it, leaving a saddle for the approach behind the gate.
    let saddle = if rel.y < 0.0 {
        smoothstep(4.0, 9.0, (rel.x - 8.0).abs())
    } else {
        1.0
    };
    let crown =
        smoothstep(3.0, 12.0, out) * saddle * (1.5 + 8.0 * fbm2(seed.wrapping_add(75), x / 16.0, z / 16.0, 2)) * calm;
    let top_h = top + crown + 0.5 * (value2(seed.wrapping_add(71), x / 9.0, z / 9.0) - 0.5);
    // The rim rounds over in a band of varying width and is notched in places.
    let rough = value2(seed.wrapping_add(76), ring.x / 7.0, ring.y / 7.0);
    let notch = 3.5 * smoothstep(0.6, 0.85, rough) * calm;
    let lip = top_h - (1.2 + notch) * smoothstep(0.5, 2.0, out);
    Some(if s < 0.0 {
        top_h - (top_h - lip) * smoothstep(-2.0 - 3.0 * rough, 0.0, s)
    } else if s < depth {
        // Four sheer drops, each from a sharp brow, with narrow ledges between
        // them that tilt outwards and wander up and down, so they read as rock
        // bands rather than stairs.
        let u = s / depth;
        let j = 0.06 * (value2(seed.wrapping_add(70), (x + z) / 13.0, (x - z) / 23.0) - 0.5);
        let drop = |a: f32, b: f32| {
            let t = ((u - a) / (b - a)).clamp(0.0, 1.0);
            1.0 - (1.0 - t).powf(2.5)
        };
        let steps = 0.30 * drop(0.0, 0.1)
            + 0.26 * drop(0.28 + j, 0.36 + j)
            + 0.24 * drop(0.55 - j, 0.63 - j)
            + 0.20 * drop(0.8 + 0.5 * j, 0.9);
        lip - (lip - foot) * (0.94 * steps + 0.06 * u)
    } else {
        // Scree flaring out from the foot of the face, strewn with boulders.
        let v = 1.0 - (s - depth) / talus_w;
        let boulders = 2.4 * smoothstep(0.6, 0.9, value2(seed.wrapping_add(77), x / 3.2, z / 3.2)) * calm;
        base + (foot - base) * v * v + boulders * v.sqrt()
    })
}

/// The ridge that runs from the back of the bluff into the mountains: a steep,
/// rocky crest falling towards its far end, cut through where the valley road
/// passes under the castle.
fn ridge(seed: u32, x: f32, z: f32) -> Option<f32> {
    let (a, b) = (centre(), Vec2::new(RIDGE_END.0, RIDGE_END.2));
    let p = Vec2::new(x, z);
    let ab = b - a;
    let along = (p - a).dot(ab) / ab.length_squared();
    if !(0.0..1.2).contains(&along) {
        return None;
    }
    let t = along.min(1.0);
    let perp = p.distance(a + ab * t);
    if perp > 60.0 {
        return None;
    }
    let crest = CASTLE_TOP.1 - 10.0
        + (RIDGE_END.1 - CASTLE_TOP.1 + 10.0) * t
        + 8.0 * (fbm2(seed.wrapping_add(72), x / 30.0, z / 30.0, 2) - 0.5);
    let p = perp.min(58.0);
    // It grows out of the back of the bluff and sinks into the mountains past its end.
    let fade = smoothstep(0.1, 0.3, along) * (1.0 - smoothstep(1.0, 1.2, along));
    Some(crest - (1.35 * p - 0.0115 * p * p) - (1.0 - fade) * 60.0)
}

/// Whether a point is on the castle bluff, where no snow lies: its top is grass
/// and its faces bare rock.
pub fn snow_free(x: f32, z: f32) -> bool {
    Vec2::new(x, z).distance(centre()) < SNOW_FREE_R
}

/// Radius (metres) around the castle bluff's centre where no snow lies. The
/// shader's snow cover keeps the same rule (`SNOW_FREE` in world.wgsl).
pub const SNOW_FREE_R: f32 = 130.0;

/// Composed shaping applied after the river channel is cut: the castle bluff,
/// whose river face drops straight into the water, its ridge and the plunge
/// pools at its foot. `road` gives the distance to the nearest road, asked only
/// where the ridge rises above the ground.
pub fn after_channel(seed: u32, x: f32, z: f32, rx: f32, mut h: f32, road: impl Fn() -> f32) -> f32 {
    if !near_bluff(x, z) {
        return h;
    }
    let d = (x - rx).abs();
    let guard = smoothstep(9.0, 14.0, d);
    if let Some(b) = bluff(seed, x, z) {
        h += (b - h).max(0.0) * guard;
    }
    if let Some(r) = ridge(seed, x, z) {
        if r > h {
            h += (r - h) * guard * smoothstep(6.0, 18.0, road());
        }
    }
    for f in &CLIFF_FALLS {
        let dist = Vec2::new(x, z).distance(f.foot());
        if dist < POOL_R + 2.0 {
            let bottom = water_level(x, z) - 2.2;
            h += (bottom - h).min(0.0) * (1.0 - smoothstep(POOL_R - 1.5, POOL_R + 1.5, dist));
        }
    }
    h
}

/// Top (metres) of the water a cliff fall stands in a column, if the column is
/// on one: the sheet hugs the face, filling each drop in front of the rock and
/// running a voxel deep over the ledges and the lip. `ground` is this column's
/// height and `height` the terrain's height function.
pub fn curtain_top(x: f32, z: f32, ground: f32, height: impl Fn(f32, f32) -> f32) -> Option<f32> {
    let rel = Vec2::new(x, z) - centre();
    for f in &CLIFF_FALLS {
        let d = f.out();
        let (along, lateral) = (rel.dot(d), d.perp_dot(rel));
        let (lip, foot) = f.reach();
        if lateral.abs() > f.half_w || along < lip - 2.5 || along > foot + 1.5 {
            continue;
        }
        let inner = Vec2::new(x, z) - d * VOXEL_SIZE;
        let inner_h = height(inner.x, inner.y);
        return Some(inner_h.max(ground + VOXEL_SIZE).min(CASTLE_TOP.1 + VOXEL_SIZE));
    }
    None
}

/// Whether any cliff fall's water could reach into a box of metres.
pub fn curtain_touches(lo: Vec3, hi: Vec3) -> bool {
    CLIFF_FALLS.iter().any(|f| {
        let (a, b) = (f.lip() - f.out() * 3.0, f.foot());
        let (mn, mx) = (a.min(b) - 3.0, a.max(b) + 3.0);
        mn.x < hi.x && mx.x > lo.x && mn.y < hi.z && mx.y > lo.z && lo.y < CASTLE_TOP.1 + 1.0
    })
}

/// Far away each cliff fall is one ribbon of water from the top, over the lip
/// and down its face into the pool. It follows the face's outer hull, leaping
/// from each ledge's brow to the next, so the coarse far ground (which never
/// stands above that hull) cannot swallow it. Adds those whose lip lies in
/// [lo, hi); `height` is the terrain's height function.
pub fn far_curtains(out: &mut MeshData, lo: Vec2, hi: Vec2, height: impl Fn(f32, f32) -> f32) {
    for f in &CLIFF_FALLS {
        let lip = f.lip();
        if lip.x < lo.x || lip.y < lo.y || lip.x >= hi.x || lip.y >= hi.y {
            continue;
        }
        let d = f.out();
        let (r0, r1) = f.reach();
        let at = |r: f32| centre() + d * r;
        let pool = water_level(f.foot().x, f.foot().y) - 0.06;
        // The profile down the fall's line, then its upper hull.
        let mut hull: Vec<Vec2> = Vec::new();
        let mut r = r0 - 1.5;
        while r <= r1 + 0.6 {
            let q = at(r);
            let p = Vec2::new(r, height(q.x, q.y).max(pool));
            while hull.len() >= 2 {
                let (a, b) = (hull[hull.len() - 2], hull[hull.len() - 1]);
                if (b - a).perp_dot(p - a) >= 0.0 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(p);
            r += 1.0;
        }
        // The water ends in the pool.
        if let Some(last) = hull.last_mut() {
            last.y = pool;
        }
        // Across the fall, so that across x down-the-ribbon points out of the face.
        let across = Vec3::new(d.y, 0.0, -d.x) * f.half_w;
        let seg_n = |k: usize| {
            let (a, b) = (hull[k], hull[k + 1]);
            Vec2::new(a.y - b.y, b.x - a.x).normalize_or(Vec2::Y)
        };
        let start = out.water_vertices.len() as u32;
        for k in 0..hull.len() {
            let nr = match k {
                0 => seg_n(0),
                k if k == hull.len() - 1 => seg_n(k - 1),
                k => (seg_n(k - 1) + seg_n(k)).normalize_or(Vec2::Y),
            };
            let normal = Vec3::new(d.x * nr.x, nr.y, d.y * nr.x);
            let q = at(hull[k].x);
            let c = Vec3::new(q.x, hull[k].y + 0.35, q.y) + normal * 0.25;
            for p in [c - across, c + across] {
                out.water_vertices.push(Vertex {
                    pos: p.to_array(),
                    data: smooth_data(WATER, 3, normal),
                });
            }
        }
        for k in 0..hull.len() as u32 - 1 {
            let (tm, tp, bm, bp) = (start + 2 * k, start + 2 * k + 1, start + 2 * k + 2, start + 2 * k + 3);
            out.water_indices.extend_from_slice(&[bm, bp, tp, bm, tp, tm]);
        }
    }
}

/// X of the path's centre line at a Z, if the path reaches that Z.
pub fn path_x(z: f32) -> Option<f32> {
    let (z0, z1) = PATH_Z;
    if !(z0..=z1).contains(&z) {
        return None;
    }
    let t = (z - z0) / (z1 - z0);
    Some(PATH_X0 + (PATH_X1 - PATH_X0) * t + PATH_SWING * (t * PI).sin())
}

/// Roughly how far (metres) a point is from the path's centre line; past its
/// ends, the distance to the end, so the ends are round.
pub fn path_distance(x: f32, z: f32) -> f32 {
    let (z0, z1) = PATH_Z;
    let zc = z.clamp(z0, z1);
    let Some(px) = path_x(zc) else {
        return f32::MAX;
    };
    let t = (zc - z0) / (z1 - z0);
    let slope = ((PATH_X1 - PATH_X0) + PATH_SWING * PI * (t * PI).cos()) / (z1 - z0);
    let across = (x - px).abs() / (1.0 + slope * slope).sqrt();
    let beyond = (z0 - z).max(z - z1).max(0.0);
    (across * across + beyond * beyond).sqrt()
}

/// Whether a column is on the path, given the same 0..1 fray the road's
/// ragged edges use. The path narrows to a rounded end where it starts on the rise.
pub fn on_path(x: f32, z: f32, fray: f32) -> bool {
    let d = path_distance(x, z);
    if d > PATH_HALF_WIDTH_M + 1.0 {
        return false;
    }
    let taper = 0.55 + 0.45 * smoothstep(0.0, 7.0, PATH_Z.1 - z);
    d < PATH_HALF_WIDTH_M * taper + 0.5 * fray
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
/// (for the watchtower), the path, and the lines of sight from the arrival spot
/// to the bridge, the cascade and the castle.
pub fn keeps_clear(x: f32, z: f32) -> bool {
    let from = Vec2::new(ARRIVAL.0, ARRIVAL.1);
    let rel = Vec2::new(x, z) - from;
    let r = rel.length();
    let in_sight = r < SIGHT_CLEAR_M
        && SIGHTS.iter().any(|&(tx, tz, w)| {
            let to = Vec2::new(tx, tz) - from;
            let d = to.normalize();
            let along = rel.dot(d);
            along > 0.0 && along < to.length() && d.perp_dot(rel).abs() < w
        });
    in_sight
        || Vec2::new(x - SPAWN.0, z - SPAWN.2).length() < RISE_TOP_R + 1.5
        || Vec2::new(x - TOWER_KNOLL.0, z - TOWER_KNOLL.2).length() < KNOLL_TOP_R + 4.0
        || path_distance(x, z) < 3.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{GRASS, PATH, STONE};
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
        let (x, z) = (SPAWN.0, SPAWN.2);
        let river = water_level(t.river_x(z), z);
        let top = t.height_at(x, z).0;
        assert!((7.0..=9.0).contains(&(top - river)), "rise {top} over water {river}");
        // The top is flat enough for a big tree.
        for k in 0..16 {
            let a = k as f32 / 16.0 * TAU;
            let h = t.height_at(x + a.cos() * RISE_TOP_R, z + a.sin() * RISE_TOP_R).0;
            assert!((h - top).abs() < 0.8, "rise edge at {a}: {h} vs {top}");
        }
        // The player arrives on the top, standing on it.
        let s = t.spawn_point();
        assert_eq!((s.x, s.z), ARRIVAL);
        assert!(Vec2::new(s.x - x, s.z - z).length() < RISE_TOP_R);
        assert!(s.y > top && s.y < top + 1.5);
    }

    #[test]
    fn river_shows_below_the_arrival_view() {
        let t = world();
        let s = t.spawn_point();
        let eye = Vec3::new(s.x, s.y + 0.6, s.z);
        // Water points on the left half of the view, under 30 degrees down,
        // with nothing in the way.
        let mut seen = 0;
        for k in 0..40 {
            let yaw = SPAWN_YAW - 0.9 * k as f32 / 40.0;
            for step in 8..80 {
                let dist = step as f32;
                let p = Vec2::new(eye.x + yaw.cos() * dist, eye.z + yaw.sin() * dist);
                let (h, _) = t.height_at(p.x, p.y);
                let wl = water_level(p.x, p.y);
                if h > wl {
                    continue;
                }
                let target = Vec3::new(p.x, wl, p.y);
                let dir = target - eye;
                if -dir.y / dir.length() > 0.5 {
                    continue;
                }
                let clear = (1..40).all(|i| {
                    let q = eye + dir * (i as f32 / 40.0);
                    t.height_at(q.x, q.z).0 < q.y - 0.05
                });
                if clear {
                    seen += 1;
                    break;
                }
            }
        }
        assert!(seen >= 8, "water seen along {seen} of 40 lines of sight");
    }

    #[test]
    fn gorge_banks_stand_above_the_water_near_the_bridge() {
        let t = world();
        for (z, sides) in [(890.0, [-1.0f32, 1.0]), (952.0, [-1.0, 1.0]), (1000.0, [-1.0, -1.0])] {
            let rx = t.river_x(z);
            let wl = water_level(rx, z);
            for side in sides {
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
        // The terrace is flat across the castle, and grassy.
        for (dx, dz) in [(-25.5, -18.5), (25.5, -18.5), (-25.5, 18.5), (25.5, 18.5), (0.0, 0.0)] {
            let h = t.height_at(x + dx, z + dz).0;
            assert!((h - top).abs() < 1.0, "terrace corner {dx},{dz}: {h}");
        }
        assert_eq!(t.column_info(x, z).surface, GRASS);
        // Sheer towards the viewer and the river: the ground drops most of the way
        // within a few metres of the rim.
        for a in [1.2f32, 1.66, 2.1, 2.7, 3.1] {
            let d = Vec2::new(a.cos(), a.sin());
            let at = |r: f32| t.height_at(x + d.x * r, z + d.y * r).0;
            let lip = (0..400).map(|k| k as f32 * 0.25).find(|&r| at(r) < top - 4.0).unwrap();
            let below = at(lip + face_depth(a) + buttress(a) + 2.0);
            assert!(top - below > 35.0, "face at {a}: lip at {lip} m, {below} below it");
        }
        // Not a box: the rim's distance from the centre varies a lot around it.
        let rims: Vec<f32> = (0..32).map(|k| rim(k as f32 / 32.0 * TAU)).collect();
        let (lo, hi) = rims
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), r| (a.min(*r), b.max(*r)));
        assert!(hi - lo > 15.0, "rim {lo}..{hi}");
        // The ridge ties it back into the mountains.
        let mid = centre().lerp(Vec2::new(RIDGE_END.0, RIDGE_END.2), 0.5);
        assert!(t.height_at(mid.x, mid.y).0 > 80.0);
        // The castle's spires stay under the top of the world.
        let world_top = (crate::terrain::WORLD_CHUNKS_Y * crate::chunk::CHUNK) as f32 * VOXEL_SIZE;
        let mut castles = 0;
        for st in t.structures.iter() {
            if let crate::structures::Structure::Castle(c) = st {
                assert!((c.c.x - x).abs() < 0.5 && (c.c.y - z).abs() < 0.5);
                assert!(c.floor - c.ground_lo < 3.0, "terrace {} over {}", c.floor, c.ground_lo);
                assert!(c.tip() < world_top - 4.0);
                castles += 1;
            }
        }
        assert_eq!(castles, 1);
    }

    #[test]
    fn valley_road_passes_under_the_castle() {
        let t = world();
        for k in 0..60 {
            let z = 540.0 + k as f32 * 4.0;
            let x = t.road_x(z);
            let (h, _) = t.height_at(x, z);
            let (hn, _) = t.height_at(x, z + 2.0);
            assert!(h < 75.0 && (hn - h).abs() < 3.0, "road at {x},{z}: {h} then {hn}");
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
        assert_eq!(t.column_info(x, z).surface, GRASS);
    }

    #[test]
    fn cliff_falls_pour_down_real_faces_into_pools() {
        let mut t = world();
        for f in &CLIFF_FALLS {
            let (lip, foot) = f.reach();
            let mut wet = 0;
            let mut levels = Vec::new();
            let steps = ((foot - lip) / VOXEL_SIZE) as i32;
            for step in 0..=steps {
                let p = centre() + f.out() * (lip + step as f32 * VOXEL_SIZE + 0.25);
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
            let (lo, hi) = levels
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), l| (a.min(*l), b.max(*l)));
            assert!(hi - lo > 50.0, "fall at {}: spans {} m", f.angle, hi - lo);
            assert!(wet >= steps, "fall at {}: {wet} wet samples", f.angle);
            // It lands in a pool of standing water.
            let p = f.foot();
            let wl = water_level(p.x, p.y);
            assert!(t.height_at(p.x, p.y).0 < wl - 1.0, "no pool at fall {}", f.angle);
            let v = (Vec3::new(p.x, wl - 0.4, p.y) / VOXEL_SIZE).floor().as_ivec3();
            assert_eq!(block(&mut t, v), WATER, "pool at fall {}", f.angle);
        }
    }

    #[test]
    fn far_field_draws_the_cliff_falls() {
        let t = world();
        let height = |x: f32, z: f32| t.height_at(x, z).0;
        let mut out = MeshData::default();
        far_curtains(&mut out, Vec2::ZERO, Vec2::splat(crate::terrain::WORLD_SIZE_M), height);
        let (v, idx) = (&out.water_vertices, &out.water_indices);
        let tris: Vec<[Vec3; 3]> = idx
            .chunks(3)
            .map(|c| [0, 1, 2].map(|i| Vec3::from(v[c[i] as usize].pos)))
            .collect();
        for f in &CLIFF_FALLS {
            // This fall's triangles: those near its line.
            let mine: Vec<&[Vec3; 3]> = tris
                .iter()
                .filter(|tr| {
                    let q = Vec2::new(tr[0].x, tr[0].z) - f.foot();
                    f.out().perp_dot(q).abs() < f.half_w + 0.5 && q.length() < 60.0
                })
                .collect();
            assert!(!mine.is_empty(), "fall at {}", f.angle);
            let (lo, hi) = mine
                .iter()
                .flat_map(|tr| tr.iter())
                .fold((f32::MAX, f32::MIN), |(a, b), p| (a.min(p.y), b.max(p.y)));
            assert!(hi - lo > 50.0, "fall at {}: spans {lo}..{hi}", f.angle);
            // The sheet faces out of its face (or up, over the top), and it
            // stands clear of the ground everywhere.
            for tr in &mine {
                let n = (tr[1] - tr[0]).cross(tr[2] - tr[0]).normalize();
                assert!(
                    Vec2::new(n.x, n.z).dot(f.out()) > -0.05 && n.y > -0.05,
                    "fall at {}: {n}",
                    f.angle
                );
                for p in tr.iter() {
                    assert!(p.y > height(p.x, p.z) - 0.1, "fall at {} buried at {p}", f.angle);
                }
            }
        }
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
    fn shader_keeps_snow_off_the_bluff() {
        let wgsl = include_str!("shaders/world.wgsl");
        let (x, _, z) = CASTLE_TOP;
        assert!(wgsl.contains(&format!(
            "const SNOW_FREE: vec3<f32> = vec3({x:.1}, {z:.1}, {SNOW_FREE_R:.1});"
        )));
        let t = world();
        for (dx, dz) in [(0.0, 0.0), (20.0, -15.0), (-24.0, 18.0), (60.0, -40.0)] {
            assert_ne!(t.column_info(x + dx, z + dz).surface, crate::block::SNOW);
        }
    }

    #[test]
    fn landmarks_stand_on_their_anchors() {
        let t = world();
        let mut seen = 0;
        for st in t.structures.iter() {
            match st {
                crate::structures::Structure::Bridge(b) => {
                    // The bridge centres on whole voxels, so it may sit a quarter metre off.
                    assert!((b.z - BRIDGE_Z).abs() <= 0.25, "bridge z {}", b.z);
                    // Dry ground at both ends, above the water, and the path meets the east end.
                    for x in [b.x0 - 0.5, b.x1 + 0.5] {
                        assert!(t.height_at(x, b.z).0 > water_level(x, b.z) + 3.0, "bridge end {x}");
                    }
                    // The path meets the east end, or the paved approach bridges the gap.
                    let px = path_x(PATH_Z.0).unwrap();
                    assert!(px >= b.x1 - 2.0 && px - b.x1 < 8.0, "path {px}, bridge end {}", b.x1);
                    let mut x = b.x1;
                    while x < px {
                        assert!(
                            t.structures.path_at(&t, x, b.z) || path_distance(x, b.z) < 2.0,
                            "unpaved at {x}"
                        );
                        x += 0.5;
                    }
                    seen += 1;
                }
                crate::structures::Structure::Watchtower(tw) => {
                    assert!(tw.c.distance(Vec2::new(TOWER_KNOLL.0, TOWER_KNOLL.2)) < 0.5);
                    // It stands on the knoll's flat top, not on stilts or sunk into it.
                    assert!((tw.floor - TOWER_KNOLL.1).abs() < 1.0, "tower floor {}", tw.floor);
                    assert!(tw.floor - tw.base < 1.5, "footings {} deep", tw.floor - tw.base);
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
        // It never runs through a building, and stays west of the road's fence.
        use crate::structures::Structure;
        let buildings: Vec<(Vec3, Vec3)> = t
            .structures
            .iter()
            .filter_map(|st| match st {
                Structure::Cottage(c) => Some(c.bounds()),
                Structure::Watchtower(w) => Some(w.bounds()),
                Structure::Castle(c) => Some(c.bounds()),
                _ => None,
            })
            .collect();
        let mut z = PATH_Z.0 + 2.0;
        while z < PATH_Z.1 {
            let x = path_x(z).unwrap();
            let (x0, x1) = (x - PATH_HALF_WIDTH_M, x + PATH_HALF_WIDTH_M);
            for (lo, hi) in &buildings {
                let inside = lo.x < x1 && hi.x > x0 && lo.z < z + 0.25 && hi.z > z - 0.25;
                assert!(!inside, "path at {x},{z} runs into a building");
            }
            // The west fence stands 3.6 m off the road's centre line.
            assert!(x1 < t.road_x(z) - 3.6 - 0.5, "path at {x},{z} crowds the fence");
            z += 0.5;
        }
        let (z0, z1) = PATH_Z;
        let mut z = z0;
        while z <= z1 - 1.0 {
            let x = path_x(z).unwrap();
            let c = t.column_info(x, z);
            assert_eq!(c.surface, PATH, "path at {x},{z}");
            z += 2.0;
        }
        // It ends in a rounded tip on the rise, not a square edge.
        let end = path_x(z1).unwrap();
        assert!(on_path(end, z1, 0.0) && !on_path(end + 1.2, z1, 0.0) && !on_path(end, z1 + 1.5, 0.0));
        assert!(Vec2::new(end - SPAWN.0, z1 - SPAWN.2).length() <= RISE_TOP_R + 0.5);
        assert!(path_lanterns().count() >= 4);
    }

    #[test]
    fn sight_lines_stay_open_and_the_meadow_keeps_its_trees() {
        let t = world();
        // Nothing stands on the line of sight to the castle near the rise.
        let from = Vec2::new(ARRIVAL.0, ARRIVAL.1);
        let to = centre();
        for k in 1..40 {
            let p = from.lerp(to, k as f32 / 40.0 * SIGHT_CLEAR_M / from.distance(to));
            assert!(keeps_clear(p.x, p.y));
        }
        // But the meadows around the hamlet keep their trees.
        let mut trees = 0;
        for gz in 95..115 {
            for gx in 90..110 {
                trees += t.tree_in_cell(gx, gz).is_some() as i32;
            }
        }
        assert!(trees > 30, "{trees} trees around the hamlet");
    }

    /// Where `p` lands in the first view, as fractions of the frame's width
    /// and height from its top-left corner, or None behind the eye. This is
    /// the camera Sebastian sees on opening (`?cam=893,33.5,1024,-1.5,-0.15`):
    /// feet on the arrival spot, eye 1.6 m up, the game's 70 degree field of
    /// view at the side-by-side's 1280 by 760.
    fn first_view(p: glam::Vec3) -> Option<(f32, f32)> {
        use glam::{Mat4, Vec3};
        let (yaw, pitch) = (SPAWN_YAW, -0.15f32);
        let eye = Vec3::new(ARRIVAL.0, 33.5 + 1.6, ARRIVAL.1);
        let dir = Vec3::new(yaw.cos() * pitch.cos(), pitch.sin(), yaw.sin() * pitch.cos());
        let vp = Mat4::perspective_infinite_reverse_rh(70f32.to_radians(), 1280.0 / 760.0, 0.05)
            * Mat4::look_to_rh(eye, dir, Vec3::Y);
        let c = vp * p.extend(1.0);
        (c.w > 0.0).then(|| ((c.x / c.w + 1.0) * 0.5, (1.0 - c.y / c.w) * 0.5))
    }

    #[test]
    fn great_oak_frames_the_left_of_the_first_view() {
        let t = world();
        let oak = t.hero_tree.expect("the arrival has a great oak");
        // Nothing of the tree reaches right of 45% of the width, where the
        // valley, the bridge, the castle crag (about yaw -1.16, pitch 0.17 to
        // 0.35) and the peaks must stay open, and its crown stays in the
        // top-left quarter.
        let mut mesh = crate::mesh::MeshData::default();
        oak.mesh(&mut mesh);
        let mut crown = 0;
        let (mut trunk, mut trunk_edge) = (0, 0f32);
        for v in &mesh.vertices {
            let p = glam::Vec3::from(v.pos);
            let Some((x, y)) = first_view(p) else { continue };
            if !(0.0..1.0).contains(&x) || !(0.0..1.0).contains(&y) {
                continue;
            }
            assert!(x <= 0.45, "oak at {x:.2}, {y:.2} of the frame, over the valley");
            if p.y - oak.base.y < 3.0 {
                trunk += 1;
                trunk_edge = trunk_edge.max(x);
            }
            if p.y - oak.base.y > 0.3 * oak.height {
                assert!(
                    y <= 0.5,
                    "crown at {x:.2}, {y:.2} of the frame, below the top-left quarter"
                );
                crown += 1;
            }
        }
        // The trunk's near side frames the left edge, and no wider than that.
        assert!(trunk > 10, "trunk not in the view");
        assert!(trunk_edge < 0.15, "trunk reaches {trunk_edge:.2} of the width");
        assert!(crown > 50, "the crown should hang into the top-left corner");
    }
}
