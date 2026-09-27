//! Procedural world generation: a river valley between ridged mountains,
//! with caves, beaches, snow caps and forests. Pure functions of the seed.

use crate::block::*;
use crate::chunk::{local_index, Chunk, CHUNK, CHUNK_VOLUME};
use crate::noise::{fbm2, hash2, ridged2, unit, value3};
use crate::structures::Structures;
use glam::{IVec2, IVec3, Vec3};
use std::collections::HashMap;
use std::sync::Arc;

/// Side length of the playable world in metres.
pub const WORLD_SIZE_M: f32 = 2048.0;
/// Horizontal extent in chunks (each chunk is 16 m wide).
pub const WORLD_CHUNKS_XZ: i32 = (WORLD_SIZE_M / VOXEL_SIZE) as i32 / CHUNK;
/// Vertical extent in chunks (128 m of height).
pub const WORLD_CHUNKS_Y: i32 = 8;
/// Height of the river and lake surfaces in metres.
pub const WATER_LEVEL_M: f32 = 24.5;

// Trees are sized like real ones (broadleaf 9 to 14 m, conifers 15 to 23 m)
// so a 1.75 m player reads at the right scale against them.
/// Trees sit on a jittered grid with this spacing in metres.
pub const TREE_CELL_M: f32 = 9.0;
const TREE_REACH_M: f32 = 7.5;
/// Grid cell (metres) that holds at most one boulder.
const BOULDER_CELL_M: f32 = 6.0;
/// Largest boulder radius in metres.
const BOULDER_MAX_R_M: f32 = 2.4;
const TREE_MAX_HEIGHT_M: f32 = 26.0;

/// Half the width of a road's packed surface in metres.
const ROAD_HALF_WIDTH_M: f32 = 1.6;
/// Lantern posts stand this far apart along each road, alternating sides.
const LANTERN_SPACING_M: f32 = 18.0;
/// Z positions (metres) where side roads leave the valley road for the hills.
const SIDE_ROADS_Z: [f32; 4] = [300.0, 780.0, 1290.0, 1760.0];
/// How far side roads climb away from the valley road.
const SIDE_ROAD_LENGTH_M: f32 = 560.0;
/// Voxels in a lantern post below the lantern itself.
const LANTERN_POST: i32 = 6;

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

struct Tree {
    base: IVec3,
    trunk_voxels: i32,
    canopy_r: f32,
    conifer: bool,
}

pub struct Terrain {
    pub seed: u32,
    /// Bridge, cottages, fences, watchtower and castle.
    pub structures: Structures,
    columns: HashMap<IVec2, Arc<Vec<ColumnInfo>>>,
}

impl Terrain {
    pub fn new(seed: u32) -> Self {
        let mut t = Self {
            seed,
            structures: Structures::default(),
            columns: HashMap::new(),
        };
        t.structures = Structures::plan(&t);
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
            if h > WATER_LEVEL_M + 0.8 {
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
        let valley = smoothstep(28.0, 460.0, d);

        let floor = 26.0 + 2.5 * fbm2(s.wrapping_add(2), x_m / 70.0, z_m / 70.0, 3);
        let hills = 14.0 * fbm2(s.wrapping_add(5), x_m / 110.0, z_m / 110.0, 4);
        let mountains = 88.0 * ridged2(s.wrapping_add(3), x_m / 420.0, z_m / 420.0, 5)
            + 16.0 * fbm2(s.wrapping_add(4), x_m / 90.0, z_m / 90.0, 4);

        let edge = x_m.min(z_m).min(WORLD_SIZE_M - x_m).min(WORLD_SIZE_M - z_m);
        let edge_rise = (1.0 - smoothstep(0.0, 260.0, edge)) * 55.0;

        let mut h = floor + hills * valley.sqrt() + mountains * valley.powf(1.5) + edge_rise;

        let channel = 1.0 - smoothstep(7.0, 17.0, d);
        let bed = WATER_LEVEL_M - 2.8 + 0.8 * fbm2(s.wrapping_add(6), x_m / 12.0, z_m / 12.0, 2);
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
        let snow_line = 84.0 + 8.0 * fbm2(self.seed.wrapping_add(7), x_m / 40.0, z_m / 40.0, 2);

        let (surface, subsurface) = if h < WATER_LEVEL_M + 1.0 {
            if channel > 0.4 {
                (GRAVEL, GRAVEL)
            } else {
                (SAND, SAND)
            }
        } else if h > snow_line && slope < 1.4 {
            (SNOW, STONE)
        } else if slope > 1.05 {
            (STONE, STONE)
        } else {
            (GRASS, DIRT)
        };
        // Roads: packed earth with slightly ragged edges, only on dry ground.
        let (surface, subsurface) = if h > WATER_LEVEL_M + 0.6 && slope < 1.3 && {
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

    fn tree_in_cell(&self, gx: i32, gz: i32) -> Option<Tree> {
        let h = hash2(self.seed.wrapping_add(20), gx, gz);
        let xm = (gx as f32 + 0.15 + 0.7 * unit(h)) * TREE_CELL_M;
        let zm = (gz as f32 + 0.15 + 0.7 * unit(h.rotate_left(11))) * TREE_CELL_M;
        let density = fbm2(self.seed.wrapping_add(21), xm / 160.0, zm / 160.0, 3);
        let chance = smoothstep(0.42, 0.65, density) * 0.85;
        if unit(h.rotate_left(22)) >= chance {
            return None;
        }
        let info = self.column_info(xm, zm);
        if info.surface != GRASS || info.height_m < WATER_LEVEL_M + 1.5 || info.height_m > 78.0 {
            return None;
        }
        // Keep roads and buildings clear of trunks.
        if self.road_distance(xm, zm) < 5.0 || self.structures.blocks_tree(xm, zm) {
            return None;
        }
        let conifer = info.height_m > 44.0 || unit(h.rotate_left(5)) < 0.25;
        let base = IVec3::new(
            (xm / VOXEL_SIZE) as i32,
            (info.height_m / VOXEL_SIZE).floor() as i32,
            (zm / VOXEL_SIZE) as i32,
        );
        let size = unit(h.rotate_left(17));
        Some(Tree {
            base,
            trunk_voxels: if conifer {
                30 + (size * 14.0) as i32
            } else {
                14 + (size * 8.0) as i32
            },
            canopy_r: if conifer { 3.0 + size * 1.2 } else { 3.8 + size * 2.0 },
            conifer,
        })
    }

    /// A rounded, half-buried boulder: common along the river banks and in the
    /// shallows, scattered sparsely over the meadows, never on roads.
    pub fn boulder_in_cell(&self, gx: i32, gz: i32) -> Option<Boulder> {
        let h = hash2(self.seed.wrapping_add(40), gx, gz);
        let xm = (gx as f32 + 0.1 + 0.8 * unit(h)) * BOULDER_CELL_M;
        let zm = (gz as f32 + 0.1 + 0.8 * unit(h.rotate_left(9))) * BOULDER_CELL_M;
        let info = self.column_info(xm, zm);
        let above_water = info.height_m - WATER_LEVEL_M;
        let chance = if (-1.2..2.5).contains(&above_water) {
            0.5
        } else if info.surface == GRASS && info.height_m < 60.0 {
            0.05
        } else {
            0.0
        };
        if unit(h.rotate_left(19)) >= chance || self.road_distance(xm, zm) < 3.5 {
            return None;
        }
        let size = unit(h.rotate_left(27));
        let r = 0.6 + (BOULDER_MAX_R_M - 0.6) * size * size;
        let squash = 0.65 + 0.25 * unit(h.rotate_left(4));
        Some(Boulder {
            centre: Vec3::new(xm, info.height_m - r * 0.3, zm),
            radii: Vec3::new(r * (0.95 + 0.3 * unit(h.rotate_left(13))), r * squash, r),
            seed: h,
        })
    }

    fn stamp_tree(&self, tree: &Tree, origin: IVec3, data: &mut [Block; CHUNK_VOLUME]) {
        let mut put = |p: IVec3, b: Block, replace_solid: bool| {
            let l = p - origin;
            if l.cmplt(IVec3::ZERO).any() || l.cmpge(IVec3::splat(CHUNK)).any() {
                return;
            }
            let i = local_index(l.x, l.y, l.z);
            if replace_solid || data[i] == AIR {
                data[i] = b;
            }
        };
        let r_vox = tree.canopy_r / VOXEL_SIZE;
        let top = tree.base.y + tree.trunk_voxels;
        let ri = (r_vox * 1.2).ceil() as i32 + 1;
        // Broadleaf crowns sit partly around the upper trunk, like real ones.
        let crown = top - (r_vox * 0.35) as i32;
        let (y0, y1) = if tree.conifer {
            (tree.base.y + tree.trunk_voxels / 3, top + 3)
        } else {
            (crown - ri + 1, crown + ri)
        };
        // Broadleaf crowns: clump centres (voxels, relative to the trunk at crown
        // height) and radii.
        let h = crate::noise::hash3(self.seed.wrapping_add(24), tree.base.x, tree.base.y, tree.base.z);
        let hr = |k: u32| unit(h.rotate_left(k));
        let mut clumps = vec![(Vec3::new(0.0, r_vox * 0.2, 0.0), r_vox * 0.66)];
        let n = 5 + (hr(1) * 3.0) as u32;
        for k in 0..n {
            let a = (k as f32 + 0.35 * hr(3 + k)) * std::f32::consts::TAU / n as f32;
            let out = r_vox * (0.5 + 0.15 * hr(9 + k));
            let lift = r_vox * (-0.2 + 0.5 * hr(17 + k));
            clumps.push((
                Vec3::new(a.cos() * out, lift, a.sin() * out),
                r_vox * (0.4 + 0.14 * hr(25 + k)),
            ));
        }
        // Only visit the part of the crown that falls inside this chunk.
        let lo = origin - IVec3::new(tree.base.x, 0, tree.base.z);
        let hi = lo + IVec3::splat(CHUNK - 1);
        let (dx0, dx1) = ((-ri).max(lo.x), ri.min(hi.x));
        let (dz0, dz1) = ((-ri).max(lo.z), ri.min(hi.z));
        for y in y0.max(origin.y)..=y1.min(origin.y + CHUNK - 1) {
            for dz in dz0..=dz1 {
                for dx in dx0..=dx1 {
                    let p = IVec3::new(tree.base.x + dx, y, tree.base.z + dz);
                    let (fx, fz) = (dx as f32 + 0.5, dz as f32 + 0.5);
                    let inside = if tree.conifer {
                        let t = (y - y0) as f32 / (y1 - y0) as f32;
                        (fx * fx + fz * fz).sqrt() < r_vox * (1.0 - t) + 0.6
                    } else {
                        // A cloud of rounded clumps around a central one, each a
                        // little lumpy, so the crown reads as masses of foliage.
                        let q = Vec3::new(fx, (y - crown) as f32 + 0.5, fz);
                        let lump = value3(
                            self.seed.wrapping_add(23),
                            p.x as f32 * 0.3,
                            p.y as f32 * 0.3,
                            p.z as f32 * 0.3,
                        ) - 0.5;
                        clumps.iter().any(|(c, r)| q.distance(*c) < r * (1.0 + 0.3 * lump))
                    };
                    if inside {
                        put(p, LEAVES, false);
                    }
                }
            }
        }
        if !tree.conifer {
            // Boughs from the upper trunk out into the outer clumps.
            for (c, _) in clumps.iter().skip(1) {
                let from = Vec3::new(0.0, (top - 3 - crown) as f32, 0.0);
                let to = *c * 0.8;
                let steps = (to - from).length().ceil() as i32 * 2;
                for k in 0..=steps {
                    let v = from.lerp(to, k as f32 / steps as f32);
                    let cell = IVec3::new(tree.base.x, crown, tree.base.z) + v.floor().as_ivec3();
                    put(cell, WOOD, true);
                }
            }
        }
        for y in tree.base.y..top {
            for (dx, dz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                put(IVec3::new(tree.base.x + dx - 1, y, tree.base.z + dz - 1), WOOD, true);
            }
        }
        if !tree.conifer {
            // Roots flare out around the foot of the trunk.
            for (dx, dz) in [(-2, -1), (-2, 0), (1, -1), (1, 0), (-1, -2), (0, -2), (-1, 1), (0, 1)] {
                if hr((dx * 7 + dz * 3 + 40) as u32) < 0.7 {
                    put(IVec3::new(tree.base.x + dx, tree.base.y, tree.base.z + dz), WOOD, true);
                }
            }
        }
    }

    /// Rough boxes (min, max in metres) around the canopy and trunk of the tree in a
    /// tree-grid cell, if it has one. Used to draw forests far away.
    pub fn tree_boxes(&self, gx: i32, gz: i32) -> Option<[(Vec3, Vec3); 2]> {
        let t = self.tree_in_cell(gx, gz)?;
        let r = t.canopy_r / VOXEL_SIZE;
        let top = (t.base.y + t.trunk_voxels) as f32;
        let (cx, cz) = (t.base.x as f32, t.base.z as f32);
        let (canopy_lo, canopy_hi) = if t.conifer {
            let y0 = (t.base.y + t.trunk_voxels / 3) as f32;
            let h = r * 0.62;
            (Vec3::new(cx - h, y0, cz - h), Vec3::new(cx + h, top + 4.0, cz + h))
        } else {
            // Matches the crown in `stamp_tree`: centred below the trunk top.
            let crown = top - (r * 0.35).floor();
            let (h, v) = (r * 0.86, r / 1.15 * 0.86);
            (
                Vec3::new(cx - h, crown + 0.5 - v, cz - h),
                Vec3::new(cx + h, crown + 0.5 + v, cz + h),
            )
        };
        let trunk_lo = Vec3::new(cx - 1.0, t.base.y as f32, cz - 1.0);
        let trunk_hi = Vec3::new(cx + 1.0, canopy_lo.y, cz + 1.0);
        Some([
            (canopy_lo * VOXEL_SIZE, canopy_hi * VOXEL_SIZE),
            (trunk_lo * VOXEL_SIZE, trunk_hi * VOXEL_SIZE),
        ])
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
        let side = CHUNK as f32 * VOXEL_SIZE;
        let lo_m = origin.as_vec3() * VOXEL_SIZE;
        let built = self.structures.touches(lo_m, lo_m + side);
        if chunk_bottom_m > tree_top.max(WATER_LEVEL_M) && !built {
            return Chunk::Uniform(AIR);
        }

        let mut data = Box::new([AIR; CHUNK_VOLUME]);
        let s = self.seed;
        for z in 0..CHUNK {
            for x in 0..CHUNK {
                let col = cols[(z * CHUNK + x) as usize];
                let wx = origin.x + x;
                let wz = origin.z + z;
                for y in 0..CHUNK {
                    let wy = origin.y + y;
                    let ym = (wy as f32 + 0.5) * VOXEL_SIZE;
                    let b = if wy == 0 {
                        STONE
                    } else if ym <= col.height_m {
                        let depth = col.height_m - ym;
                        let dry = col.height_m > WATER_LEVEL_M + 2.0;
                        if depth > 3.0 && dry && ym > 2.0 && is_cave(s, wx, wy, wz) {
                            AIR
                        } else if depth < VOXEL_SIZE {
                            col.surface
                        } else if depth < 2.0 {
                            col.subsurface
                        } else {
                            STONE
                        }
                    } else if ym < WATER_LEVEL_M {
                        WATER
                    } else if col.surface == GRASS && ym - VOXEL_SIZE <= col.height_m && tall_grass(s, wx, wy, wz) {
                        TALL_GRASS
                    } else {
                        AIR
                    };
                    data[local_index(x, y, z)] = b;
                }
            }
        }

        if chunk_top_m >= min_h && chunk_bottom_m <= tree_top {
            let reach = TREE_REACH_M;
            let x0 = ((origin.x as f32 * VOXEL_SIZE - reach) / TREE_CELL_M).floor() as i32;
            let x1 = (((origin.x + CHUNK) as f32 * VOXEL_SIZE + reach) / TREE_CELL_M).floor() as i32;
            let z0 = ((origin.z as f32 * VOXEL_SIZE - reach) / TREE_CELL_M).floor() as i32;
            let z1 = (((origin.z + CHUNK) as f32 * VOXEL_SIZE + reach) / TREE_CELL_M).floor() as i32;
            for gz in z0..=z1 {
                for gx in x0..=x1 {
                    if let Some(tree) = self.tree_in_cell(gx, gz) {
                        self.stamp_tree(&tree, origin, &mut data);
                    }
                }
            }
            let (ox, oz) = (origin.x as f32 * VOXEL_SIZE, origin.z as f32 * VOXEL_SIZE);
            let reach = BOULDER_MAX_R_M * 1.3;
            let cell = |m: f32| (m / BOULDER_CELL_M).floor() as i32;
            for gz in cell(oz - reach)..=cell(oz + side + reach) {
                for gx in cell(ox - reach)..=cell(ox + side + reach) {
                    if let Some(b) = self.boulder_in_cell(gx, gz) {
                        stamp_boulder(&b, s, origin, &mut data);
                    }
                }
            }
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
            if h > WATER_LEVEL_M + 0.8 {
                return glam::Vec3::new(x, h + 1.0, z);
            }
        }
        glam::Vec3::new(rx + 30.0, 60.0, z)
    }
}

pub struct Boulder {
    /// Centre in metres.
    pub centre: Vec3,
    /// Half extents in metres.
    pub radii: Vec3,
    seed: u32,
}

/// Fills a lumpy ellipsoid of stone; the smooth mesher rounds it off.
fn stamp_boulder(b: &Boulder, seed: u32, origin: IVec3, data: &mut [Block; CHUNK_VOLUME]) {
    let lo = ((b.centre - b.radii * 1.2) / VOXEL_SIZE).floor().as_ivec3().max(origin);
    let hi = ((b.centre + b.radii * 1.2) / VOXEL_SIZE)
        .ceil()
        .as_ivec3()
        .min(origin + IVec3::splat(CHUNK - 1));
    for y in lo.y..=hi.y {
        for z in lo.z..=hi.z {
            for x in lo.x..=hi.x {
                let p = (IVec3::new(x, y, z).as_vec3() + 0.5) * VOXEL_SIZE;
                let d = ((p - b.centre) / b.radii).length();
                let lump = value3(seed.wrapping_add(41) ^ b.seed, p.x * 0.9, p.y * 0.9, p.z * 0.9) - 0.5;
                if d < 1.0 + lump * 0.35 {
                    let l = IVec3::new(x, y, z) - origin;
                    data[local_index(l.x, l.y, l.z)] = STONE;
                }
            }
        }
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
/// meadows and tall-grass fields, sparse between them.
fn tall_grass(seed: u32, x: i32, y: i32, z: i32) -> bool {
    let (xm, zm) = (x as f32 * VOXEL_SIZE, z as f32 * VOXEL_SIZE);
    let chance = (0.06 + 0.8 * meadow(seed, xm, zm)).max(0.97 * tall_meadow(seed, xm, zm));
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
    fn river_banks_have_boulders_and_roads_do_not() {
        let t = Terrain::new(20260927);
        let (mut bank, mut road) = (0, 0);
        for gz in 150..190 {
            for gx in 120..180 {
                if let Some(b) = t.boulder_in_cell(gx, gz) {
                    let above = t.column_info(b.centre.x, b.centre.z).height_m - WATER_LEVEL_M;
                    bank += (above < 2.5) as i32;
                    road += (t.road_distance(b.centre.x, b.centre.z) < 3.5) as i32;
                    assert!(b.radii.max_element() <= BOULDER_MAX_R_M * 1.25);
                }
            }
        }
        assert!(bank > 10, "{bank} boulders by the river");
        assert_eq!(road, 0);
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
        assert!(h < WATER_LEVEL_M);
        let s = t.spawn_point();
        assert!(s.y > WATER_LEVEL_M);
    }

    #[test]
    fn sky_and_bedrock_are_uniform() {
        let mut t = Terrain::new(1);
        assert!(t.generate(IVec3::new(10, WORLD_CHUNKS_Y - 1, 10)).is_uniform(AIR));
        assert!(t.generate(IVec3::new(10, -1, 10)).is_uniform(STONE));
    }
}
