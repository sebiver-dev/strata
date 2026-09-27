//! Procedural world generation: a river valley between ridged mountains,
//! with caves, beaches, snow caps and forests. Pure functions of the seed.

use crate::block::*;
use crate::chunk::{local_index, Chunk, CHUNK, CHUNK_VOLUME};
use crate::noise::{fbm2, hash2, ridged2, unit, value3};
use crate::structures::Structures;
use crate::trees::{Kind as TreeKind, Tree};
use glam::{IVec2, IVec3, Vec2, Vec3};
use std::collections::HashMap;
use std::sync::Arc;

/// Side length of the playable world in metres.
pub const WORLD_SIZE_M: f32 = 2048.0;
/// Horizontal extent in chunks (each chunk is 16 m wide).
pub const WORLD_CHUNKS_XZ: i32 = (WORLD_SIZE_M / VOXEL_SIZE) as i32 / CHUNK;
/// Vertical extent in chunks (192 m of height).
pub const WORLD_CHUNKS_Y: i32 = 12;
/// Height of the river and lake surfaces in metres along the home reach,
/// the calm stretch around spawn where the village stands. Upstream the
/// river sits higher, one step per fall, and downstream lower.
pub const WATER_LEVEL_M: f32 = 24.5;
/// The home reach holds this Z (metres).
const HOME_Z_M: f32 = WORLD_SIZE_M * 0.5;
/// Where the river drops as it runs towards +Z: the Z (metres) of each fall's
/// lip, the drop in metres (whole voxels), and how many metres the valley floor
/// takes to follow it down (short makes a cliff). A few small drops close
/// together make a cascade. Keep Z and drop in step with fall() in world.wgsl.
pub const FALLS: [(f32, f32, f32); 10] = [
    (380.0, 1.0, 50.0),
    (388.0, 1.0, 50.0),
    (396.0, 1.5, 50.0),
    (690.0, 8.0, 6.0),
    (1290.0, 1.0, 50.0),
    (1297.0, 1.0, 50.0),
    (1304.0, 1.0, 50.0),
    (1540.0, 4.0, 60.0),
    (1790.0, 1.5, 50.0),
    (1798.0, 1.5, 50.0),
];

// Trees are sized like real ones (broadleaf 9 to 14 m, conifers 15 to 23 m)
// so a 1.75 m player reads at the right scale against them.
/// Trees sit on a jittered grid with this spacing in metres.
pub const TREE_CELL_M: f32 = 9.0;
/// How far a trunk can lean away from its foot, in metres.
const TREE_REACH_M: f32 = 2.5;
const TREE_MAX_HEIGHT_M: f32 = 26.0;
/// Height of the rock bands where mountain slopes break into cliffs.
const CLIFF_STEP_M: f32 = 10.0;

/// Half the width of a road's packed surface in metres.
pub const ROAD_HALF_WIDTH_M: f32 = 1.6;
/// Lantern posts stand this far apart along each road, alternating sides.
const LANTERN_SPACING_M: f32 = 18.0;
/// Z positions (metres) where side roads leave the valley road for the hills.
const SIDE_ROADS_Z: [f32; 4] = [300.0, 780.0, 1290.0, 1760.0];
/// How far side roads climb away from the valley road.
const SIDE_ROAD_LENGTH_M: f32 = 560.0;
/// Voxels in a lantern post below the lantern itself.
const LANTERN_POST: i32 = 6;

/// Z (metres) of a fall's lip at a given X: the lip bows a little across the river.
pub fn fall_z(z0: f32, x_m: f32) -> f32 {
    z0 + 3.0 * (x_m * 0.21 + z0 * 0.01).sin()
}

/// How far below the home reach the river ends up after the falls downstream of it.
fn drop_below_home() -> f32 {
    FALLS.iter().filter(|f| f.0 > HOME_Z_M).map(|f| f.1).sum()
}

/// Height of still water (river and lakes) at a point, in metres.
pub fn water_level(x_m: f32, z_m: f32) -> f32 {
    WATER_LEVEL_M - drop_below_home()
        + FALLS
            .iter()
            .filter(|f| z_m < fall_z(f.0, x_m))
            .map(|f| f.1)
            .sum::<f32>()
}

/// How far the valley floor is raised (or lowered) at a point so it keeps
/// above the river's steps: it follows each fall down just below the lip. A
/// fall that makes a cliff still lets a road ramp down beside it.
fn floor_rise(z_m: f32, road_m: f32) -> f32 {
    let ramp = 1.0 - smoothstep(6.0, 30.0, road_m);
    FALLS
        .iter()
        .map(|&(z0, d, len)| {
            let len = len + (60.0 - len).max(0.0) * ramp;
            d * (1.0 - smoothstep(z0 + 1.0, z0 + 1.0 + len, z_m))
        })
        .sum::<f32>()
        - drop_below_home()
}

/// Distance (metres) downstream of the nearest fall's lip, if within `reach`.
pub fn below_fall(x_m: f32, z_m: f32, reach: f32) -> Option<f32> {
    FALLS
        .iter()
        .map(|f| z_m - fall_z(f.0, x_m))
        .filter(|d| (0.0..reach).contains(d))
        .reduce(f32::min)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// What an untouched column of terrain looks like from above.
#[derive(Clone, Copy)]
pub struct ColumnInfo {
    pub height_m: f32,
    pub surface: Block,
    pub subsurface: Block,
}

pub struct Terrain {
    pub seed: u32,
    /// Bridge, cottages, fences, watchtower and castle.
    pub structures: Structures,
    /// The great oak that frames the view from the start.
    pub hero_tree: Option<Tree>,
    columns: HashMap<IVec2, Arc<Vec<ColumnInfo>>>,
}

impl Terrain {
    pub fn new(seed: u32) -> Self {
        let mut t = Self {
            seed,
            structures: Structures::default(),
            hero_tree: None,
            columns: HashMap::new(),
        };
        t.structures = Structures::plan(&t);
        t.hero_tree = t.plan_hero_tree();
        t
    }

    /// X coordinate (metres) of the river's centre line at a given Z.
    pub fn river_x(&self, z_m: f32) -> f32 {
        let s = self.seed;
        WORLD_SIZE_M * 0.5
            + 170.0 * (z_m / 270.0 + 0.7).sin()
            + 90.0 * (fbm2(s.wrapping_add(1), z_m / 420.0, 0.5, 3) * 2.0 - 1.0)
    }

    /// X coordinate (metres) of the valley road's centre line at a given Z. It
    /// follows the east bank of the river, swinging a little on its own.
    pub fn road_x(&self, z_m: f32) -> f32 {
        self.river_x(z_m) + 40.0 + 8.0 * (z_m / 130.0 + 1.3).sin()
    }

    /// Z coordinate of side road `k` at a given X, if that X is on it.
    fn side_road_z(&self, k: usize, x_m: f32) -> Option<f32> {
        let z0 = SIDE_ROADS_Z[k];
        let start = self.road_x(z0);
        if x_m < start || x_m > start + SIDE_ROAD_LENGTH_M {
            return None;
        }
        Some(z0 + 25.0 * ((x_m - start) / 110.0).sin())
    }

    /// Roughly how far (metres) a point is from the centre of the nearest road.
    pub fn road_distance(&self, x_m: f32, z_m: f32) -> f32 {
        let mut d = (x_m - self.road_x(z_m)).abs();
        for k in 0..SIDE_ROADS_Z.len() {
            if let Some(z) = self.side_road_z(k, x_m) {
                d = d.min((z_m - z).abs());
            }
        }
        d
    }

    /// Ground positions (voxel coordinates) of the lantern posts whose base lies
    /// within the given rectangle of metres, each with the direction (one voxel
    /// step) towards its road, where its lantern hangs.
    fn lanterns_in(&self, x0: f32, x1: f32, z0: f32, z1: f32) -> Vec<(IVec3, IVec3)> {
        let mut out = Vec::new();
        let mut add = |x: f32, z: f32, towards: IVec3| {
            if x < x0 || x >= x1 || z < z0 || z >= z1 {
                return;
            }
            let (h, _) = self.height_at(x, z);
            if h > water_level(x, z) + 0.8 {
                let base = IVec3::new(
                    (x / VOXEL_SIZE).floor() as i32,
                    (h / VOXEL_SIZE).floor() as i32,
                    (z / VOXEL_SIZE).floor() as i32,
                );
                out.push((base, towards));
            }
        };
        let side = |k: i32| if k % 2 == 0 { 2.6 } else { -2.6 };
        let towards = |off: f32, axis: IVec3| if off > 0.0 { -axis } else { axis };
        let s = LANTERN_SPACING_M;
        for k in (z0 / s).floor() as i32 - 1..=(z1 / s).ceil() as i32 {
            let z = (k as f32 + 0.5) * s;
            add(self.road_x(z) + side(k), z, towards(side(k), IVec3::X));
        }
        for (r, z0) in SIDE_ROADS_Z.iter().enumerate() {
            let start = self.road_x(*z0);
            // Side-road lanterns start a little way up so they do not crowd the junction.
            for k in 1..(SIDE_ROAD_LENGTH_M / s) as i32 {
                let x = start + k as f32 * s;
                if x + 3.0 < x0 || x - 3.0 > x1 {
                    continue;
                }
                if let Some(z) = self.side_road_z(r, x) {
                    add(x, z + side(k), towards(side(k), IVec3::Z));
                }
            }
        }
        out
    }

    /// Ground height in metres, plus how strongly this point is river channel (0..1).
    pub fn height_at(&self, x_m: f32, z_m: f32) -> (f32, f32) {
        let s = self.seed;
        let d = (x_m - self.river_x(z_m)).abs();
        // Steep valley walls rise to tall peaks within a few hundred metres.
        let valley = smoothstep(30.0, 280.0, d);

        let floor = 26.0 + 2.5 * fbm2(s.wrapping_add(2), x_m / 70.0, z_m / 70.0, 3);
        let hills = 14.0 * fbm2(s.wrapping_add(5), x_m / 110.0, z_m / 110.0, 4);
        let ridge = ridged2(s.wrapping_add(3), x_m / 420.0, z_m / 420.0, 5);
        let mountains = 38.0 + 115.0 * ridge * ridge + 22.0 * fbm2(s.wrapping_add(4), x_m / 90.0, z_m / 90.0, 4);

        let edge = x_m.min(z_m).min(WORLD_SIZE_M - x_m).min(WORLD_SIZE_M - z_m);
        let edge_rise = (1.0 - smoothstep(0.0, 260.0, edge)) * 70.0;

        // In places the slopes break into ledges and sheer rock bands.
        let mut rise = mountains * valley.powf(1.2);
        let band = CLIFF_STEP_M * (0.7 + 0.8 * fbm2(s.wrapping_add(9), x_m / 180.0, z_m / 180.0, 2));
        let k = rise / band;
        let ledges = (k.floor() + smoothstep(0.45, 0.9, k.fract())) * band;
        let cliffy = smoothstep(0.4, 0.65, fbm2(s.wrapping_add(8), x_m / 260.0, z_m / 260.0, 2));
        rise += (ledges - rise) * cliffy * 0.7;

        let mut h = floor + hills * valley.sqrt() + rise + edge_rise;
        // The valley floor climbs with the river's steps; the mountains far from
        // it barely need to, and must stay under the sky limit.
        h += floor_rise(z_m, self.road_distance(x_m, z_m)) * (1.0 - 0.85 * valley);

        let channel = 1.0 - smoothstep(7.0, 17.0, d);
        let bed = water_level(x_m, z_m) - 2.8 + 0.8 * fbm2(s.wrapping_add(6), x_m / 12.0, z_m / 12.0, 2);
        h += (bed - h) * channel;

        (
            h.clamp(3.0, (WORLD_CHUNKS_Y * CHUNK) as f32 * VOXEL_SIZE - 6.0),
            channel,
        )
    }

    pub fn column_info(&self, x_m: f32, z_m: f32) -> ColumnInfo {
        let (h, channel) = self.height_at(x_m, z_m);
        let (hx, _) = self.height_at(x_m + 1.0, z_m);
        let (hz, _) = self.height_at(x_m, z_m + 1.0);
        let slope = (hx - h).abs().max((hz - h).abs());
        // Snow lies above a ragged snowline, lower on flat ledges, never on steep rock.
        let snow_line = 104.0 + 22.0 * fbm2(self.seed.wrapping_add(7), x_m / 60.0, z_m / 60.0, 3)
            - 14.0 * (1.0 - smoothstep(0.3, 0.9, slope));

        let water = water_level(x_m, z_m);
        let (surface, subsurface) = if slope > 1.3 && h > water - 1.0 {
            // Steep banks, such as the cliffs beside a fall, are bare rock.
            (STONE, STONE)
        } else if h < water + 1.0 {
            if channel > 0.4 {
                (GRAVEL, GRAVEL)
            } else {
                (SAND, SAND)
            }
        } else if h > snow_line && slope < 1.0 {
            (SNOW, STONE)
        } else if slope > 1.05 {
            (STONE, STONE)
        } else {
            (GRASS, DIRT)
        };
        // Roads: packed earth with slightly ragged edges, only on dry ground.
        let (surface, subsurface) = if h > water + 0.6 && slope < 1.3 && {
            let d = self.road_distance(x_m, z_m);
            let fray = unit(hash2(
                self.seed.wrapping_add(31),
                (x_m * 2.0) as i32,
                (z_m * 2.0) as i32,
            ));
            d < ROAD_HALF_WIDTH_M + 0.5 * fray || self.structures.path_at(self, x_m, z_m)
        } {
            (PATH, DIRT)
        } else {
            (surface, subsurface)
        };
        ColumnInfo {
            height_m: h,
            surface,
            subsurface,
        }
    }

    /// Height info for every voxel column of one chunk column, cached.
    fn columns(&mut self, cx: i32, cz: i32) -> Arc<Vec<ColumnInfo>> {
        let key = IVec2::new(cx, cz);
        if let Some(c) = self.columns.get(&key) {
            return c.clone();
        }
        if self.columns.len() > 4096 {
            self.columns.clear();
        }
        let mut v = Vec::with_capacity((CHUNK * CHUNK) as usize);
        for z in 0..CHUNK {
            for x in 0..CHUNK {
                let xm = ((cx * CHUNK + x) as f32 + 0.5) * VOXEL_SIZE;
                let zm = ((cz * CHUNK + z) as f32 + 0.5) * VOXEL_SIZE;
                v.push(self.column_info(xm, zm));
            }
        }
        let arc = Arc::new(v);
        self.columns.insert(key, arc.clone());
        arc
    }

    /// The tree standing in a cell of the tree grid, if any.
    pub fn tree_in_cell(&self, gx: i32, gz: i32) -> Option<Tree> {
        let h = hash2(self.seed.wrapping_add(20), gx, gz);
        let xm = (gx as f32 + 0.15 + 0.7 * unit(h)) * TREE_CELL_M;
        let zm = (gz as f32 + 0.15 + 0.7 * unit(h.rotate_left(11))) * TREE_CELL_M;
        let density = fbm2(self.seed.wrapping_add(21), xm / 160.0, zm / 160.0, 3);
        let chance = smoothstep(0.42, 0.65, density) * 0.85;
        if unit(h.rotate_left(22)) >= chance {
            return None;
        }
        let info = self.column_info(xm, zm);
        if info.surface != GRASS || info.height_m < water_level(xm, zm) + 1.5 || info.height_m > 104.0 {
            return None;
        }
        // Keep roads and buildings clear of trunks, and room around the great oak.
        if self.road_distance(xm, zm) < 5.0 || self.structures.blocks_tree(xm, zm) {
            return None;
        }
        if let Some(hero) = &self.hero_tree {
            if Vec2::new(xm - hero.base.x, zm - hero.base.z).length() < 12.0 {
                return None;
            }
        }
        // Broadleaf trees fill the valley; conifers take over up the slopes
        // and gather round the watchtower.
        let slope_line = 50.0 + 14.0 * fbm2(self.seed.wrapping_add(25), xm / 90.0, zm / 90.0, 2);
        let by_tower = self.structures.iter().any(|st| match st {
            crate::structures::Structure::Tower(t) => t.c.distance(Vec2::new(xm, zm)) < 70.0,
            _ => false,
        });
        let conifer =
            info.height_m > slope_line || (by_tower && unit(h.rotate_left(5)) < 0.7) || unit(h.rotate_left(5)) < 0.1;
        let size = unit(h.rotate_left(17));
        Some(Tree {
            base: Vec3::new(xm, info.height_m, zm),
            height: if conifer { 14.0 + size * 9.0 } else { 9.0 + size * 5.5 },
            kind: if conifer {
                TreeKind::Conifer
            } else {
                TreeKind::Broadleaf
            },
            seed: h,
            hero: None,
        })
    }

    /// Every tree whose trunk stands inside a box of metres (x and z).
    pub fn trees_in(&self, lo: Vec2, hi: Vec2) -> Vec<Tree> {
        let first = (lo / TREE_CELL_M).floor().as_ivec2();
        let last = (hi / TREE_CELL_M).floor().as_ivec2();
        let mut out = Vec::new();
        for gz in first.y..=last.y {
            for gx in first.x..=last.x {
                if let Some(t) = self.tree_in_cell(gx, gz) {
                    if t.base.x >= lo.x && t.base.x < hi.x && t.base.z >= lo.y && t.base.z < hi.y {
                        out.push(t);
                    }
                }
            }
        }
        if let Some(t) = self.hero_tree {
            if t.base.x >= lo.x && t.base.x < hi.x && t.base.z >= lo.y && t.base.z < hi.y {
                out.push(t);
            }
        }
        out
    }

    /// A great oak just left of the start, reaching a bough out over the
    /// player's head so it frames the first view down the valley.
    fn plan_hero_tree(&self) -> Option<Tree> {
        let spawn = self.spawn_point();
        let yaw = crate::structures::SPAWN_YAW;
        let fwd = Vec2::new(yaw.cos(), yaw.sin());
        let right = Vec2::new(-fwd.y, fwd.x);
        for (ahead, left) in [(4.0, 4.5), (3.0, 5.5), (5.0, 6.0), (2.0, 6.5), (4.0, 8.0), (6.0, 9.0)] {
            let p = Vec2::new(spawn.x, spawn.z) + fwd * ahead - right * left;
            let info = self.column_info(p.x, p.y);
            if info.surface != GRASS
                || info.height_m < water_level(p.x, p.y) + 1.0
                || self.road_distance(p.x, p.y) < 3.0
                || self.structures.blocks_tree(p.x, p.y)
            {
                continue;
            }
            return Some(Tree {
                base: Vec3::new(p.x, info.height_m, p.y),
                height: 15.0,
                kind: TreeKind::Broadleaf,
                seed: self.seed ^ 0x0a4_7ee,
                // The long bough reaches across the top of the first view.
                hero: Some((right + fwd * 0.6).normalize()),
            });
        }
        None
    }

    /// Generates the voxels of one chunk.
    pub fn generate(&mut self, cpos: IVec3) -> Chunk {
        let outside = cpos.x < 0
            || cpos.z < 0
            || cpos.x >= WORLD_CHUNKS_XZ
            || cpos.z >= WORLD_CHUNKS_XZ
            || cpos.y >= WORLD_CHUNKS_Y;
        if cpos.y < 0 {
            return Chunk::Uniform(STONE);
        }
        if outside {
            return Chunk::Uniform(AIR);
        }
        let cols = self.columns(cpos.x, cpos.z);
        let origin = cpos * CHUNK;
        let chunk_bottom_m = origin.y as f32 * VOXEL_SIZE;
        let chunk_top_m = (origin.y + CHUNK) as f32 * VOXEL_SIZE;

        let max_h = cols.iter().map(|c| c.height_m).fold(f32::MIN, f32::max);
        let min_h = cols.iter().map(|c| c.height_m).fold(f32::MAX, f32::min);
        let tree_top = max_h + TREE_MAX_HEIGHT_M;
        let (ox, oz) = (origin.x as f32 * VOXEL_SIZE, origin.z as f32 * VOXEL_SIZE);
        let side = CHUNK as f32 * VOXEL_SIZE;
        let lo_m = origin.as_vec3() * VOXEL_SIZE;
        let built = self.structures.touches(lo_m, lo_m + side);
        // The fall lip bows by a few metres, so look a little upstream for the highest water.
        let top_water = water_level(ox, oz - 4.0).max(water_level(ox + side, oz - 4.0));
        if chunk_bottom_m > tree_top.max(top_water) && !built {
            return Chunk::Uniform(AIR);
        }

        let mut data = Box::new([AIR; CHUNK_VOLUME]);
        let s = self.seed;
        for z in 0..CHUNK {
            for x in 0..CHUNK {
                let col = cols[(z * CHUNK + x) as usize];
                let wx = origin.x + x;
                let wz = origin.z + z;
                let (xm, zm) = ((wx as f32 + 0.5) * VOXEL_SIZE, (wz as f32 + 0.5) * VOXEL_SIZE);
                let mut water = water_level(xm, zm);
                // Just below a fall's lip the upper reach pours over in a
                // curtain that stands on the pool below.
                let upper = water_level(xm, zm - VOXEL_SIZE);
                if upper > water && self.height_at(xm, zm - VOXEL_SIZE).0 < upper - VOXEL_SIZE {
                    water = upper;
                }
                for y in 0..CHUNK {
                    let wy = origin.y + y;
                    let ym = (wy as f32 + 0.5) * VOXEL_SIZE;
                    let b = if wy == 0 {
                        STONE
                    } else if ym <= col.height_m {
                        let depth = col.height_m - ym;
                        let dry = col.height_m > water + 2.0;
                        if depth > 3.0 && dry && ym > 2.0 && is_cave(s, wx, wy, wz) {
                            AIR
                        } else if depth < VOXEL_SIZE {
                            col.surface
                        } else if depth < 2.0 {
                            col.subsurface
                        } else {
                            STONE
                        }
                    } else if ym < water {
                        WATER
                    } else if col.surface == GRASS
                        && ym - VOXEL_SIZE <= col.height_m
                        && tall_grass(s, wx, wy, wz, crate::plants::verge(self, xm, zm))
                    {
                        TALL_GRASS
                    } else {
                        AIR
                    };
                    data[local_index(x, y, z)] = b;
                }
            }
        }

        if chunk_top_m >= min_h && chunk_bottom_m <= tree_top {
            // Trees are models; their trunks collide through hidden voxels.
            let reach = TREE_REACH_M;
            let lo = Vec2::new(ox - reach, oz - reach);
            for tree in self.trees_in(lo, lo + Vec2::splat(side + 2.0 * reach)) {
                tree.trunk_cells(|p| {
                    let l = p - origin;
                    if l.cmpge(IVec3::ZERO).all() && l.cmplt(IVec3::splat(CHUNK)).all() {
                        let i = local_index(l.x, l.y, l.z);
                        if data[i] == AIR || data[i] == TALL_GRASS {
                            data[i] = BUILT;
                        }
                    }
                });
            }
            // Boulders are models; their voxels are hidden collision.
            crate::rocks::stamp(self, origin, &mut data);
            // A lantern hangs a voxel beside its post, so look a little past the chunk.
            let m = VOXEL_SIZE * 2.0;
            for (base, towards) in self.lanterns_in(ox - m, ox + side + m, oz - m, oz + side + m) {
                stamp_lantern(base, towards, origin, &mut data);
            }
        }

        if built {
            self.structures
                .stamp(origin, |x, z| cols[(z * CHUNK + x) as usize].height_m, &mut data);
        }

        Chunk::from_dense(data)
    }

    /// A pleasant starting spot on the river bank near the middle of the world.
    pub fn spawn_point(&self) -> glam::Vec3 {
        let z = WORLD_SIZE_M * 0.5;
        let rx = self.river_x(z);
        for off in [26.0, 34.0, 44.0, -26.0, -34.0, -44.0, 60.0] {
            let x = rx + off;
            let (h, _) = self.height_at(x, z);
            if h > water_level(x, z) + 0.8 {
                return glam::Vec3::new(x, h + 1.0, z);
            }
        }
        glam::Vec3::new(rx + 30.0, 60.0, z)
    }
}

/// A square wooden post with an arm at the top, reaching over the road, and
/// a lantern hanging from the arm's end (the mesher draws the arm and hook).
fn stamp_lantern(base: IVec3, towards: IVec3, origin: IVec3, data: &mut [Block; CHUNK_VOLUME]) {
    let mut put = |p: IVec3, b: Block| {
        let l = p - origin;
        if l.cmpge(IVec3::ZERO).all() && l.cmplt(IVec3::splat(CHUNK)).all() {
            data[local_index(l.x, l.y, l.z)] = b;
        }
    };
    for dy in 0..=LANTERN_POST {
        put(base + IVec3::Y * dy, POST);
    }
    put(base + towards + IVec3::Y * (LANTERN_POST - 1), LANTERN);
}

/// How much a spot (in metres) is meadow, 0..1: grass grows thick there.
pub fn meadow(seed: u32, xm: f32, zm: f32) -> f32 {
    smoothstep(0.38, 0.62, fbm2(seed.wrapping_add(30), xm / 38.0, zm / 38.0, 3))
}

/// How much a spot (in metres) is tall-grass field, 0..1: dense grass that
/// stands chest- to head-high, in wide patches of its own.
pub fn tall_meadow(seed: u32, xm: f32, zm: f32) -> f32 {
    smoothstep(0.56, 0.68, fbm2(seed.wrapping_add(34), xm / 64.0, zm / 64.0, 3))
}

/// Whether the air voxel resting on a grass block holds tall grass: dense in
/// meadows, tall-grass fields and along road verges and river banks (`verge`,
/// 0..1), sparse between them.
fn tall_grass(seed: u32, x: i32, y: i32, z: i32, verge: f32) -> bool {
    let (xm, zm) = (x as f32 * VOXEL_SIZE, z as f32 * VOXEL_SIZE);
    let chance = (0.06 + 0.8 * meadow(seed, xm, zm))
        .max(0.97 * tall_meadow(seed, xm, zm))
        .max(0.85 * verge);
    unit(crate::noise::hash3(seed.wrapping_add(32), x, y, z)) < chance
}

/// Spaghetti caves: tunnels where two independent noise fields both cross their midpoint.
fn is_cave(seed: u32, x: i32, y: i32, z: i32) -> bool {
    let (xm, ym, zm) = (x as f32 * VOXEL_SIZE, y as f32 * VOXEL_SIZE, z as f32 * VOXEL_SIZE);
    let a = value3(seed.wrapping_add(10), xm / 22.0, ym / 14.0, zm / 22.0) - 0.5;
    if a.abs() > 0.045 {
        return false;
    }
    let b = value3(seed.wrapping_add(11), xm / 22.0, ym / 14.0, zm / 22.0) - 0.5;
    b.abs() < 0.045
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        let mut a = Terrain::new(42);
        let mut b = Terrain::new(42);
        let p = IVec3::new(64, 1, 64);
        let (ca, cb) = (a.generate(p), b.generate(p));
        for i in 0..CHUNK {
            assert_eq!(ca.get(i, i, i), cb.get(i, i, i));
            assert_eq!(ca.get(i, 0, 31 - i), cb.get(i, 0, 31 - i));
        }
    }

    #[test]
    fn valley_has_tall_grass_fields() {
        let seed = 20260927;
        let mut field = 0;
        for i in 0..400 {
            let (x, z) = (600.0 + (i % 20) as f32 * 40.0, 700.0 + (i / 20) as f32 * 40.0);
            field += (tall_meadow(seed, x, z) > 0.9) as i32;
        }
        assert!((10..300).contains(&field), "{field} of 400 samples in tall grass");
    }

    #[test]
    fn valley_road_has_lantern_posts() {
        let mut t = Terrain::new(20260927);
        let z = 1024.0;
        let x = t.road_x(z);
        assert!(t.road_distance(x, z) < 0.01);
        assert_eq!(t.column_info(x, z).surface, PATH);
        let posts = t.lanterns_in(x - 8.0, x + 8.0, z - 40.0, z + 40.0);
        assert!(posts.len() >= 3, "{posts:?}");
        // The lantern sits on top of its post in the generated chunk.
        // The lantern hangs beside the top of its post, towards the road.
        let (base, towards) = posts[0];
        let lamp = base + towards + IVec3::Y * (LANTERN_POST - 1);
        let c = t.generate(crate::chunk::chunk_of(lamp));
        let l = crate::chunk::local_of(lamp);
        assert_eq!(c.get(l.x, l.y, l.z), LANTERN);
        let lamp_m = (lamp.as_vec3() + 0.5) * VOXEL_SIZE;
        let base_m = (base.as_vec3() + 0.5) * VOXEL_SIZE;
        assert!(t.road_distance(lamp_m.x, lamp_m.z) < t.road_distance(base_m.x, base_m.z));
    }

    #[test]
    fn river_holds_water_and_spawn_is_dry() {
        let t = Terrain::new(42);
        let z = 1000.0;
        let (h, channel) = t.height_at(t.river_x(z), z);
        assert!(channel > 0.99);
        assert!(h < water_level(t.river_x(z), z));
        let s = t.spawn_point();
        assert!(s.y > water_level(s.x, s.z));
    }

    #[test]
    fn river_steps_down_at_each_fall() {
        let mut t = Terrain::new(20260927);
        for (z0, drop, _) in FALLS {
            let x = t.river_x(z0);
            let lip = fall_z(z0, x);
            let (above, below) = (water_level(x, lip - 1.0), water_level(x, lip + 1.0));
            assert_eq!(above - below, drop);
            // Real water on both sides of the lip, at those levels (boulders
            // at the foot may fill a few columns).
            for (side, level) in [(-1.5, above), (1.5, below)] {
                let y = (level / VOXEL_SIZE) as i32 - 1;
                let mut wet = 0;
                for dx in -8..=8 {
                    let xm = x + dx as f32 * VOXEL_SIZE;
                    let z = fall_z(z0, xm) + side;
                    let v = IVec3::new((xm / VOXEL_SIZE) as i32, y, (z / VOXEL_SIZE) as i32);
                    let at = |t: &mut Terrain, v: IVec3| {
                        let l = crate::chunk::local_of(v);
                        t.generate(crate::chunk::chunk_of(v)).get(l.x, l.y, l.z)
                    };
                    wet += (at(&mut t, v) == WATER) as i32;
                    assert_ne!(at(&mut t, v + IVec3::Y), WATER, "fall at {z0}, z {z}");
                }
                assert!(wet >= 8, "fall at {z0}, side {side}: {wet} water columns");
            }
        }
        // The home reach, where the village stands, is at the base level.
        assert_eq!(water_level(t.river_x(HOME_Z_M), HOME_Z_M), WATER_LEVEL_M);
    }

    #[test]
    fn shader_knows_the_falls() {
        let wgsl = include_str!("shaders/world.wgsl");
        for (z0, drop, _) in FALLS {
            assert!(wgsl.contains(&format!("vec2({z0:.1}, {drop:.1})")), "fall at {z0}");
        }
        assert!(wgsl.contains(&format!("const WATER_LEVEL: f32 = {WATER_LEVEL_M:.1};")));
        assert!(wgsl.contains(&format!("const HOME_Z: f32 = {HOME_Z_M:.1};")));
    }

    #[test]
    fn valley_floor_stays_above_the_river() {
        let t = Terrain::new(20260927);
        let mut flooded = 0;
        for i in 0..2000 {
            let z = 280.0 + i as f32 * 0.8;
            let x = t.road_x(z);
            flooded += (t.height_at(x, z).0 < water_level(x, z) + 0.6) as i32;
        }
        assert_eq!(flooded, 0, "road samples under water");
    }

    #[test]
    fn sky_and_bedrock_are_uniform() {
        let mut t = Terrain::new(1);
        assert!(t.generate(IVec3::new(10, WORLD_CHUNKS_Y - 1, 10)).is_uniform(AIR));
        assert!(t.generate(IVec3::new(10, -1, 10)).is_uniform(STONE));
    }
}
