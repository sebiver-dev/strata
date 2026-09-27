//! Procedural world generation: a river valley between ridged mountains,
//! with caves, beaches, snow caps and forests. Pure functions of the seed.

use crate::block::*;
use crate::chunk::{local_index, Chunk, CHUNK, CHUNK_VOLUME};
use crate::noise::{fbm2, hash2, ridged2, unit, value3};
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

/// Trees sit on a jittered grid with this spacing in metres.
pub const TREE_CELL_M: f32 = 7.0;
const TREE_REACH_M: f32 = 4.5;

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
    columns: HashMap<IVec2, Arc<Vec<ColumnInfo>>>,
}

impl Terrain {
    pub fn new(seed: u32) -> Self {
        Self {
            seed,
            columns: HashMap::new(),
        }
    }

    /// X coordinate (metres) of the river's centre line at a given Z.
    pub fn river_x(&self, z_m: f32) -> f32 {
        let s = self.seed;
        WORLD_SIZE_M * 0.5
            + 170.0 * (z_m / 270.0 + 0.7).sin()
            + 90.0 * (fbm2(s.wrapping_add(1), z_m / 420.0, 0.5, 3) * 2.0 - 1.0)
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
                12 + (size * 8.0) as i32
            } else {
                8 + (size * 6.0) as i32
            },
            canopy_r: 2.4 + size * 1.4,
            conifer,
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
        let ri = r_vox.ceil() as i32 + 1;
        let (y0, y1) = if tree.conifer {
            (tree.base.y + tree.trunk_voxels / 3, top + 3)
        } else {
            (top - ri + 1, top + ri)
        };
        for y in y0..=y1 {
            for dz in -ri..=ri {
                for dx in -ri..=ri {
                    let p = IVec3::new(tree.base.x + dx, y, tree.base.z + dz);
                    let (fx, fz) = (dx as f32 + 0.5, dz as f32 + 0.5);
                    let inside = if tree.conifer {
                        let t = (y - y0) as f32 / (y1 - y0) as f32;
                        (fx * fx + fz * fz).sqrt() < r_vox * (1.0 - t) + 0.6
                    } else {
                        let fy = (y - top) as f32 * 1.25;
                        (fx * fx + fy * fy + fz * fz).sqrt() < r_vox
                    };
                    let hole = unit(crate::noise::hash3(self.seed, p.x, p.y, p.z)) < 0.12;
                    if inside && !hole {
                        put(p, LEAVES, false);
                    }
                }
            }
        }
        for y in tree.base.y..top {
            for (dx, dz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                put(IVec3::new(tree.base.x + dx - 1, y, tree.base.z + dz - 1), WOOD, true);
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
            let (h, v) = (r * 0.86, r / 1.25 * 0.86);
            (
                Vec3::new(cx - h, top + 0.5 - v, cz - h),
                Vec3::new(cx + h, top + 0.5 + v, cz + h),
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
        let tree_top = max_h + 12.0;
        if chunk_bottom_m > tree_top.max(WATER_LEVEL_M) {
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
