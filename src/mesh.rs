//! Turns a chunk of voxels into triangles. Solid ground, rock, trees and built
//! pieces are drawn as one smooth surface (surface nets over a lightly blurred
//! occupancy field), so nothing in the world reads as a cube. Water, lantern
//! posts, lanterns and grass blades keep their own shapes.

use crate::block::*;
use crate::chunk::CHUNK;
use crate::world::World;
use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec3};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    /// World position in metres.
    pub pos: [f32; 3],
    /// Bits 0..3 face direction (7: smooth surface), 3..11 material, 11..13 ambient
    /// occlusion (0 darkest), and for smooth surfaces 13..31 the normal in
    /// octahedral form (9 bits per axis). Grass blades use the occlusion bits for
    /// height along the blade (3 at the tip).
    pub data: u32,
}

#[derive(Default)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub water_vertices: Vec<Vertex>,
    pub water_indices: Vec<u32>,
}

impl MeshData {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() && self.water_indices.is_empty()
    }
}

pub(crate) struct Face {
    pub n: IVec3,
    pub u: IVec3,
    pub v: IVec3,
}

/// The six face directions in the order the shader's normal table uses.
pub(crate) const FACES: [Face; 6] = [
    Face {
        n: IVec3::X,
        u: IVec3::Y,
        v: IVec3::Z,
    },
    Face {
        n: IVec3::NEG_X,
        u: IVec3::Z,
        v: IVec3::Y,
    },
    Face {
        n: IVec3::Y,
        u: IVec3::Z,
        v: IVec3::X,
    },
    Face {
        n: IVec3::NEG_Y,
        u: IVec3::X,
        v: IVec3::Z,
    },
    Face {
        n: IVec3::Z,
        u: IVec3::X,
        v: IVec3::Y,
    },
    Face {
        n: IVec3::NEG_Z,
        u: IVec3::Y,
        v: IVec3::X,
    },
];

/// Voxels of border gathered around a chunk: one for the blur, one for the
/// surface cells that straddle the chunk's edge.
const BORDER: i32 = 2;
const P: i32 = CHUNK + 2 * BORDER;

/// Face value marking a vertex of the smooth surface.
pub const SMOOTH_FACE: u32 = 7;

/// A copy of the chunk plus a two-voxel border from its neighbours.
struct Padded {
    data: Vec<Block>,
}

impl Padded {
    fn gather(world: &World, cpos: IVec3) -> Self {
        let origin = cpos * CHUNK - IVec3::splat(BORDER);
        let mut data = vec![AIR; (P * P * P) as usize];
        // Look up the 27 chunks once instead of hashing every voxel.
        let mut near: [Option<&crate::chunk::Chunk>; 27] = [None; 27];
        for (i, slot) in near.iter_mut().enumerate() {
            let d = IVec3::new(i as i32 % 3, i as i32 / 3 % 3, i as i32 / 9) - IVec3::ONE;
            *slot = world.chunks.get(&(cpos + d));
        }
        for y in 0..P {
            for z in 0..P {
                for x in 0..P {
                    let v = origin + IVec3::new(x, y, z);
                    let d = crate::chunk::chunk_of(v) - cpos + IVec3::ONE;
                    let b = if v.y < 0 {
                        STONE
                    } else {
                        match near[(d.x + d.y * 3 + d.z * 9) as usize] {
                            Some(c) => {
                                let l = crate::chunk::local_of(v);
                                c.get(l.x, l.y, l.z)
                            }
                            None => AIR,
                        }
                    };
                    data[((y * P + z) * P + x) as usize] = b;
                }
            }
        }
        Self { data }
    }

    /// Local coordinates in -BORDER..CHUNK + BORDER.
    #[inline]
    fn get(&self, p: IVec3) -> Block {
        let q = p + IVec3::splat(BORDER);
        self.data[((q.y * P + q.z) * P + q.x) as usize]
    }
}

/// Blocks drawn as part of the smooth surface.
#[inline]
fn is_smooth(b: Block) -> bool {
    is_solid(b) && b != LANTERN && b != POST
}

pub fn build(world: &World, cpos: IVec3) -> MeshData {
    let air_beyond = |d: IVec3| world.chunks.get(&(cpos + d)).is_none_or(|n| n.is_uniform(AIR));
    match world.chunks.get(&cpos) {
        None => return MeshData::default(),
        // An empty chunk still owns the surface against solid chunks on its +X, +Y and +Z sides.
        Some(c) if c.is_uniform(AIR) && [IVec3::X, IVec3::Y, IVec3::Z].into_iter().all(air_beyond) => {
            return MeshData::default()
        }
        Some(c) => {
            if let crate::chunk::Chunk::Uniform(b) = c {
                let buried = [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z]
                    .iter()
                    .all(|d| match world.chunks.get(&(cpos + *d)) {
                        Some(crate::chunk::Chunk::Uniform(n)) => is_opaque(*n),
                        None => cpos.y + d.y < 0,
                        _ => false,
                    });
                if is_opaque(*b) && buried {
                    return MeshData::default();
                }
            }
        }
    }
    let pad = Padded::gather(world, cpos);
    let mut out = MeshData::default();
    let origin = cpos * CHUNK;

    for y in 0..CHUNK {
        for z in 0..CHUNK {
            for x in 0..CHUNK {
                let p = IVec3::new(x, y, z);
                let b = pad.get(p);
                if b == AIR {
                    continue;
                }
                if b == TALL_GRASS {
                    grass_blades(&mut out, origin, p, world.terrain.seed);
                    continue;
                }
                if b == POST {
                    post(&mut out, &pad, origin, p);
                    continue;
                }
                if b == LANTERN {
                    lantern(&mut out, origin, p, pad.get(p - IVec3::Y) == POST);
                    continue;
                }
                if is_smooth(b) {
                    continue;
                }
                for (fi, f) in FACES.iter().enumerate() {
                    let nb = pad.get(p + f.n);
                    if b == WATER {
                        if nb == AIR || nb == TALL_GRASS {
                            emit(
                                &pad,
                                &mut out.water_vertices,
                                &mut out.water_indices,
                                origin,
                                p,
                                fi,
                                f,
                                b,
                                true,
                            );
                        }
                    } else if !is_opaque(nb) || is_smooth(nb) {
                        emit(&pad, &mut out.vertices, &mut out.indices, origin, p, fi, f, b, false);
                    }
                }
            }
        }
    }
    smooth_surface(&pad, origin, &mut out);
    out
}

/// Packs a smooth-surface vertex's material, occlusion and normal.
pub fn smooth_data(mat: Block, ao: u32, n: Vec3) -> u32 {
    let n = n / (n.x.abs() + n.y.abs() + n.z.abs()).max(1e-6);
    let (mut u, mut v) = (n.x, n.z);
    if n.y < 0.0 {
        let sign = |a: f32| if a >= 0.0 { 1.0 } else { -1.0 };
        (u, v) = ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v));
    }
    let q = |a: f32| ((a * 0.5 + 0.5) * 511.0).round().clamp(0.0, 511.0) as u32;
    SMOOTH_FACE | ((mat as u32) << 3) | (ao.min(3) << 11) | (q(u) << 13) | (q(v) << 22)
}

/// Surface nets over the chunk. Occupancy is blurred with a 1-2-1 kernel so
/// steps and corners round off, but every solid voxel keeps at least a small
/// rounded body and every empty one stays open, so thin trunks, posts and
/// single placed voxels never vanish or fuse shut.
fn smooth_surface(pad: &Padded, origin: IVec3, out: &mut MeshData) {
    let idx = |x: i32, y: i32, z: i32| ((y * P + z) * P + x) as usize;
    // The foot of a post stands in the top ground voxel, so it counts as ground there.
    let stride_y = (P * P) as usize;
    let mut a: Vec<f32> = (0..pad.data.len())
        .map(|i| {
            let b = pad.data[i];
            let ground_post = b == POST && i >= stride_y && is_smooth(pad.data[i - stride_y]);
            (is_smooth(b) || ground_post) as u8 as f32
        })
        .collect();
    if a.iter().all(|v| *v == 0.0) {
        return;
    }
    let solid: Vec<bool> = a.iter().map(|v| *v > 0.5).collect();
    // Separable blur; the outermost ring keeps its raw value and is never read as a cell corner.
    let mut b = a.clone();
    for axis in 0..3 {
        // idx() lays voxels out x fastest, then z, then y.
        let step = [1, P * P, P][axis] as usize;
        for y in 0..P {
            for z in 0..P {
                for x in 0..P {
                    let c = [x, y, z][axis];
                    if c == 0 || c == P - 1 {
                        continue;
                    }
                    let i = idx(x, y, z);
                    b[i] = (a[i - step] + 2.0 * a[i] + a[i + step]) * 0.25;
                }
            }
        }
        std::mem::swap(&mut a, &mut b);
    }
    let d: Vec<f32> = a
        .iter()
        .zip(&solid)
        .map(|(v, s)| if *s { v.max(0.56) } else { v.min(0.44) })
        .collect();

    // One vertex per cell (a cube between eight voxel centres) that the surface crosses.
    // Cells start one voxel before the chunk so quads on its low faces can reach them.
    const C: i32 = CHUNK + 1;
    let cell_of = |c: IVec3| ((c.y + 1) * C * C + (c.z + 1) * C + (c.x + 1)) as usize;
    let mut cell_vertex = vec![u32::MAX; (C * C * C) as usize];
    let at = |v: IVec3| {
        let q = v + IVec3::splat(BORDER);
        idx(q.x, q.y, q.z)
    };
    // Cell vertices are made on first use, so cells no quad touches cost nothing.
    let mut vertex = |c: IVec3, out: &mut MeshData| -> u32 {
        let slot = &mut cell_vertex[cell_of(c)];
        if *slot != u32::MAX {
            return *slot;
        }
        let mut corner = [0.0f32; 8];
        for (k, v) in corner.iter_mut().enumerate() {
            let o = IVec3::new(k as i32 & 1, (k as i32 >> 1) & 1, k as i32 >> 2);
            *v = d[at(c + o)];
        }
        // Average of where the surface crosses the cell's twelve edges.
        let mut sum = Vec3::ZERO;
        let mut crossings = 0.0;
        for (e0, e1) in CELL_EDGES {
            let (v0, v1) = (corner[e0], corner[e1]);
            if (v0 > 0.5) != (v1 > 0.5) {
                let t = (v0 - 0.5) / (v0 - v1);
                sum += corner_pos(e0).lerp(corner_pos(e1), t);
                crossings += 1.0;
            }
        }
        let l = sum / crossings;
        // Density gradient by trilinear differences; the surface faces down it.
        let lerp2 = |a: f32, b: f32, c: f32, d: f32, s: f32, t: f32| {
            (a * (1.0 - s) + b * s) * (1.0 - t) + (c * (1.0 - s) + d * s) * t
        };
        let k = &corner;
        let gx = lerp2(k[1] - k[0], k[3] - k[2], k[5] - k[4], k[7] - k[6], l.y, l.z);
        let gy = lerp2(k[2] - k[0], k[3] - k[1], k[6] - k[4], k[7] - k[5], l.x, l.z);
        let gz = lerp2(k[4] - k[0], k[5] - k[1], k[6] - k[2], k[7] - k[3], l.x, l.y);
        let g = Vec3::new(gx, gy, gz);
        let n = if g.length_squared() > 1e-8 {
            -g.normalize()
        } else {
            Vec3::Y
        };
        // Material of the highest solid corner, so grass tops win over dirt below.
        let mut mat = STONE;
        let mut best = i32::MIN;
        for (k, v) in corner.iter().enumerate() {
            let o = IVec3::new(k as i32 & 1, (k as i32 >> 1) & 1, k as i32 >> 2);
            if *v > 0.5 && o.y > best {
                best = o.y;
                mat = pad.get(c + o);
                if mat == POST {
                    // The foot of a lantern post counts as the ground it stands in.
                    mat = pad.get(c + o - IVec3::Y);
                }
            }
        }
        // More solid around the cell means a more enclosed, darker spot.
        let fill = corner.iter().sum::<f32>() / 8.0;
        let ao = ((1.0 - (fill - 0.5) * 2.6).clamp(0.0, 1.0) * 3.0).round() as u32;
        let pos = ((origin + c).as_vec3() + 0.5 + l) * VOXEL_SIZE;
        *slot = out.vertices.len() as u32;
        out.vertices.push(Vertex {
            pos: pos.to_array(),
            data: smooth_data(mat, ao, n),
        });
        *slot
    };

    // One quad for every voxel edge the surface crosses, owned by the edge's lower end.
    // (u, v) are chosen so u x v points along the axis, which makes the quad
    // counter-clockwise seen from outside when the lower voxel is the solid one.
    const AXES: [(IVec3, IVec3, IVec3); 3] = [
        (IVec3::X, IVec3::Y, IVec3::Z),
        (IVec3::Y, IVec3::Z, IVec3::X),
        (IVec3::Z, IVec3::X, IVec3::Y),
    ];
    for y in 0..CHUNK {
        for z in 0..CHUNK {
            for x in 0..CHUNK {
                let p = IVec3::new(x, y, z);
                let here = d[at(p)] > 0.5;
                for (axis, u, v) in AXES {
                    if here == (d[at(p + axis)] > 0.5) {
                        continue;
                    }
                    let [a, b, c, e] = [p - u - v, p - v, p, p - u].map(|c| vertex(c, out));
                    if here {
                        out.indices.extend_from_slice(&[a, b, c, a, c, e]);
                    } else {
                        out.indices.extend_from_slice(&[a, c, b, a, e, c]);
                    }
                }
            }
        }
    }
}

/// Corner k of a cell: bit 0 is +X, bit 1 is +Y, bit 2 is +Z.
fn corner_pos(k: usize) -> Vec3 {
    Vec3::new((k & 1) as f32, ((k >> 1) & 1) as f32, (k >> 2) as f32)
}

const CELL_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// Blades per tall grass voxel.
const BLADES: u32 = 5;

/// A tuft of thin tapered blades rising from the floor of voxel `p`. Heights
/// vary in soft patches, from ankle-high to hip-high, and tall-grass fields
/// stand chest- to head-high. Each blade leans a little its own way; tall ones
/// get a joint halfway so they bend in a curve. Both windings are emitted so
/// blades show from either side.
fn grass_blades(out: &mut MeshData, origin: IVec3, p: IVec3, seed: u32) {
    let w = origin + p;
    // Rooted a little below the voxel floor, since the smooth ground can dip there.
    let floor = w.as_vec3() * VOXEL_SIZE - Vec3::Y * 0.12;
    let patch = crate::noise::fbm2(97, w.x as f32 / 9.0, w.z as f32 / 9.0, 2);
    let field = crate::terrain::tall_meadow(seed, w.x as f32 * VOXEL_SIZE, w.z as f32 * VOXEL_SIZE);
    let tall = (0.22 + 0.95 * patch * patch) * (1.0 - field) + (1.25 + 0.6 * patch) * field;
    let blades = BLADES + (field * 3.0).round() as u32;
    let data = |ao: u32| 2 | ((TALL_GRASS as u32) << 3) | (ao << 11);
    for k in 0..blades {
        let h = crate::noise::hash3(0x5eed + k, w.x, w.y, w.z);
        let r = |shift: u32| crate::noise::unit(h.rotate_left(shift));
        let base = floor + Vec3::new(0.05 + 0.4 * r(0), 0.0, 0.05 + 0.4 * r(8));
        let height = tall * (0.55 + 0.6 * r(16));
        let angle = r(24) * std::f32::consts::TAU;
        let side = Vec3::new(angle.cos(), 0.0, angle.sin()) * (0.035 + 0.012 * field);
        let lean = Vec3::new(r(4) - 0.5, 0.0, r(12) - 0.5) * height * 0.5;
        let tip = base + lean + Vec3::Y * height;
        let start = out.vertices.len() as u32;
        let mut push = |pos: Vec3, ao: u32| {
            out.vertices.push(Vertex {
                pos: pos.to_array(),
                data: data(ao),
            })
        };
        push(base - side, 0);
        push(base + side, 0);
        if height > 0.7 {
            // Joint halfway up, leaning less than the tip so the blade curves.
            let mid = base + lean * 0.3 + Vec3::Y * height * 0.55;
            push(mid - side * 0.7, 1);
            push(mid + side * 0.7, 1);
            push(tip, 3);
            let (a, b, c, d, t) = (start, start + 1, start + 2, start + 3, start + 4);
            out.indices
                .extend_from_slice(&[a, b, d, a, d, b, a, d, c, a, c, d, c, d, t, c, t, d]);
        } else {
            push(tip, 3);
            out.indices
                .extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 1]);
        }
    }
    flowers(out, floor + Vec3::Y * 0.1, w, seed);
}

/// Adds a double-sided triangle for flowers and stems; `ao` is the height
/// along the plant that the wind bends (3 at the top).
fn plant_tri(out: &mut MeshData, pts: [(Vec3, u32); 3], mat: Block) {
    let start = out.vertices.len() as u32;
    for (pos, ao) in pts {
        out.vertices.push(Vertex {
            pos: pos.to_array(),
            data: 2 | ((mat as u32) << 3) | (ao << 11),
        });
    }
    out.indices
        .extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 1]);
}

/// Wildflowers among the grass of voxel `w`: purple lupin spikes in patches,
/// and daisies scattered through the meadows.
fn flowers(out: &mut MeshData, floor: Vec3, w: IVec3, seed: u32) {
    let h = crate::noise::hash3(0xf10e, w.x, w.y, w.z);
    let r = |shift: u32| crate::noise::unit(h.rotate_left(shift));
    let (xm, zm) = (w.x as f32 * VOXEL_SIZE, w.z as f32 * VOXEL_SIZE);
    let lupins = crate::noise::fbm2(seed.wrapping_add(50), xm / 16.0, zm / 16.0, 2);
    let meadow = crate::terrain::meadow(seed, xm, zm);
    let at = floor + Vec3::new(0.1 + 0.3 * r(3), -0.1, 0.1 + 0.3 * r(7));
    let angle = r(11) * std::f32::consts::TAU;
    let across = Vec3::new(angle.cos(), 0.0, angle.sin());
    let across2 = Vec3::new(-across.z, 0.0, across.x);
    if lupins > 0.55 && r(0) < 0.6 {
        // A stem, then a tapering spike of florets on its top half, as two crossed planes.
        let height = 0.7 + 0.55 * r(15);
        let lean = Vec3::new(r(19) - 0.5, 0.0, r(23) - 0.5) * 0.12;
        let spike = at + lean * 0.5 + Vec3::Y * height * 0.45;
        let top = at + lean + Vec3::Y * height;
        plant_tri(
            out,
            [(at - across * 0.012, 0), (at + across * 0.012, 0), (spike, 1)],
            TALL_GRASS,
        );
        let wide = 0.07 + 0.03 * r(27);
        for side in [across, across2] {
            plant_tri(
                out,
                [(spike - side * wide, 1), (spike + side * wide, 1), (top, 3)],
                LUPIN,
            );
        }
    } else if r(0) < 0.03 + 0.1 * meadow {
        // A daisy: a thin stem with a white flower head facing the sky.
        let height = 0.22 + 0.25 * r(15);
        let top = at + Vec3::new(r(19) - 0.5, 0.0, r(23) - 0.5) * 0.08 + Vec3::Y * height;
        plant_tri(
            out,
            [(at - across * 0.008, 0), (at + across * 0.008, 0), (top, 3)],
            TALL_GRASS,
        );
        let ring = |k: u32, rad: f32, y: f32| {
            let a = angle + k as f32 * std::f32::consts::TAU / 6.0;
            top + Vec3::new(a.cos() * rad, y, a.sin() * rad)
        };
        let petals = 0.045 + 0.02 * r(29);
        for k in 0..6 {
            plant_tri(
                out,
                [(top, 3), (ring(k, petals, 0.0), 3), (ring(k + 1, petals, 0.0), 3)],
                DAISY,
            );
        }
        for k in 0..6 {
            let y = 0.006;
            plant_tri(
                out,
                [
                    (top + Vec3::Y * y, 3),
                    (ring(k, 0.016, y), 3),
                    (ring(k + 1, 0.016, y), 3),
                ],
                DAISY_HEART,
            );
        }
    }
}

/// Adds a flat convex polygon (a triangle or quad) facing away from `centre`.
fn flat(out: &mut MeshData, pts: &[Vec3], centre: Vec3, mat: Block) {
    let mut n = (pts[1] - pts[0]).cross(pts[2] - pts[0]).normalize_or_zero();
    let mid = pts.iter().copied().sum::<Vec3>() / pts.len() as f32;
    let flip = n.dot(mid - centre) < 0.0;
    if flip {
        n = -n;
    }
    let start = out.vertices.len() as u32;
    for q in pts {
        out.vertices.push(Vertex {
            pos: q.to_array(),
            data: smooth_data(mat, 3, n),
        });
    }
    for i in 1..pts.len() as u32 - 1 {
        if flip {
            out.indices.extend_from_slice(&[start, start + i + 1, start + i]);
        } else {
            out.indices.extend_from_slice(&[start, start + i, start + i + 1]);
        }
    }
}

/// A closed box from `lo` to `hi`.
fn cuboid(out: &mut MeshData, lo: Vec3, hi: Vec3, mat: Block) {
    let c = (lo + hi) * 0.5;
    let v = |x: bool, y: bool, z: bool| {
        Vec3::new(
            if x { hi.x } else { lo.x },
            if y { hi.y } else { lo.y },
            if z { hi.z } else { lo.z },
        )
    };
    let (f, t) = (false, true);
    for face in [
        [v(f, f, f), v(t, f, f), v(t, t, f), v(f, t, f)],
        [v(f, f, t), v(t, f, t), v(t, t, t), v(f, t, t)],
        [v(f, f, f), v(f, t, f), v(f, t, t), v(f, f, t)],
        [v(t, f, f), v(t, t, f), v(t, t, t), v(t, f, t)],
        [v(f, f, f), v(t, f, f), v(t, f, t), v(f, f, t)],
        [v(f, t, f), v(t, t, f), v(t, t, t), v(f, t, t)],
    ] {
        flat(out, &face, c, mat);
    }
}

/// Half the thickness of a lantern post in metres.
const POST_RADIUS: f32 = 0.08;

/// A straight timber of square section `half` from `a` to `b`.
fn beam(out: &mut MeshData, a: Vec3, b: Vec3, half: f32, mat: Block) {
    let dir = (b - a).normalize_or_zero();
    let side = if dir.y.abs() > 0.99 {
        Vec3::X
    } else {
        dir.cross(Vec3::Y).normalize()
    };
    let up = side.cross(dir);
    let (s, u) = (side * half, up * half);
    let corner = |e: Vec3, k: usize| e + [s + u, -s + u, -s - u, s - u][k];
    let centre = (a + b) * 0.5;
    for k in 0..4 {
        let j = (k + 1) % 4;
        flat(
            out,
            &[corner(a, k), corner(a, j), corner(b, j), corner(b, k)],
            centre,
            mat,
        );
    }
    flat(
        out,
        &[corner(a, 0), corner(a, 1), corner(a, 2), corner(a, 3)],
        centre,
        mat,
    );
    flat(
        out,
        &[corner(b, 0), corner(b, 1), corner(b, 2), corner(b, 3)],
        centre,
        mat,
    );
}

/// One voxel's length of a square wooden lantern post. The top voxel gets a
/// cap, and when a lantern hangs beside the voxel below it, an arm reaching
/// out over it with a diagonal brace.
fn post(out: &mut MeshData, pad: &Padded, origin: IVec3, p: IVec3) {
    let centre = (origin + p).as_vec3() * VOXEL_SIZE + Vec3::new(VOXEL_SIZE * 0.5, 0.0, VOXEL_SIZE * 0.5);
    let r = POST_RADIUS;
    let above = pad.get(p + IVec3::Y);
    let top = above != POST && above != LANTERN;
    let height = if top { ARM_HEIGHT + 0.08 } else { VOXEL_SIZE };
    cuboid(
        out,
        centre + Vec3::new(-r, 0.0, -r),
        centre + Vec3::new(r, height, r),
        WOOD,
    );
    if !top {
        return;
    }
    let cap = r + 0.025;
    cuboid(
        out,
        centre + Vec3::new(-cap, height, -cap),
        centre + Vec3::new(cap, height + 0.04, cap),
        WOOD,
    );
    for d in [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
        if pad.get(p + d - IVec3::Y) != LANTERN {
            continue;
        }
        let out_dir = d.as_vec3();
        let arm_y = centre + Vec3::Y * ARM_HEIGHT;
        let end = arm_y + out_dir * (VOXEL_SIZE + 0.1);
        beam(out, arm_y - out_dir * r, end, 0.045, WOOD);
        // Brace from lower on the post up to the middle of the arm.
        let low = centre + Vec3::Y * (ARM_HEIGHT - 0.42) + out_dir * r * 0.5;
        beam(out, low, arm_y + out_dir * 0.36 - Vec3::Y * 0.03, 0.03, WOOD);
    }
}

/// Height above the top post voxel's floor at which the lantern arm runs.
const ARM_HEIGHT: f32 = 0.2;

/// A lantern: a glass body with an iron frame (drawn by the shader) under a
/// small pointed iron roof. Standing on a post it gets a collar that grips
/// the post; otherwise it hangs by a hook from the arm above.
fn lantern(out: &mut MeshData, origin: IVec3, p: IVec3, on_post: bool) {
    let floor = (origin + p).as_vec3() * VOXEL_SIZE + Vec3::new(VOXEL_SIZE * 0.5, 0.0, VOXEL_SIZE * 0.5);
    let at = |dx: f32, y: f32, dz: f32| floor + Vec3::new(dx, y, dz);
    let (b0, b1) = LANTERN_BODY;
    let hw = LANTERN_HALF_WIDTH;
    if on_post {
        // Collar: slightly wider than the post and overlapping its top.
        let r = POST_RADIUS + 0.03;
        cuboid(out, at(-r, -0.04, -r), at(r, b0, r), WOOD);
    } else {
        // Base plate, then the hook up to the arm in the voxel above.
        cuboid(
            out,
            at(-hw - 0.015, b0 - 0.025, -hw - 0.015),
            at(hw + 0.015, b0, hw + 0.015),
            LANTERN,
        );
        beam(
            out,
            at(0.0, b1 + 0.15, 0.0),
            at(0.0, VOXEL_SIZE + ARM_HEIGHT, 0.0),
            0.012,
            LANTERN,
        );
    }
    cuboid(out, at(-hw, b0, -hw), at(hw, b1, hw), LANTERN);
    // Roof: a low pyramid with an overhang.
    let o = hw + 0.05;
    let eave = b1 + 0.02;
    let apex = at(0.0, b1 + 0.16, 0.0);
    let corners = [at(-o, eave, -o), at(o, eave, -o), at(o, eave, o), at(-o, eave, o)];
    let centre = at(0.0, eave + 0.04, 0.0);
    for k in 0..4 {
        flat(out, &[corners[k], corners[(k + 1) % 4], apex], centre, LANTERN);
    }
    flat(out, &corners, centre, LANTERN);
    // Thin iron plate joining the body to the roof.
    cuboid(
        out,
        at(-hw - 0.01, b1, -hw - 0.01),
        at(hw + 0.01, eave, hw + 0.01),
        LANTERN,
    );
}

/// The lantern body's bottom and top above its voxel floor, in metres.
pub const LANTERN_BODY: (f32, f32) = (0.04, 0.32);
/// Half the width of the lantern body in metres.
pub const LANTERN_HALF_WIDTH: f32 = 0.13;

#[allow(clippy::too_many_arguments)]
fn emit(
    pad: &Padded,
    verts: &mut Vec<Vertex>,
    idx: &mut Vec<u32>,
    origin: IVec3,
    p: IVec3,
    fi: usize,
    f: &Face,
    b: Block,
    water: bool,
) {
    let base = if f.n.cmpgt(IVec3::ZERO).any() { p + f.n } else { p };
    let front = p + f.n;
    let occ = |q: IVec3| is_opaque(pad.get(q)) as u32;
    let start = verts.len() as u32;
    let mut ao = [3u32; 4];
    for (i, (cu, cv)) in [(0, 0), (1, 0), (1, 1), (0, 1)].into_iter().enumerate() {
        let su = if cu == 1 { f.u } else { -f.u };
        let sv = if cv == 1 { f.v } else { -f.v };
        if !water {
            let (s1, s2, c) = (occ(front + su), occ(front + sv), occ(front + su + sv));
            ao[i] = if s1 == 1 && s2 == 1 { 0 } else { 3 - (s1 + s2 + c) };
        }
        let corner = base + f.u * cu + f.v * cv;
        let mut pos = (origin + corner).as_vec3() * VOXEL_SIZE;
        if water && f.n == IVec3::Y {
            pos.y -= 0.06;
        } else if water && f.n.y == 0 {
            // A falling sheet meets the surfaces above and below it, which sit
            // a little under the voxel tops.
            let top = corner.y > p.y;
            if (top && pad.get(p + IVec3::Y) != WATER) || (!top && pad.get(front - IVec3::Y) == WATER) {
                pos.y -= 0.06;
            }
        }
        let data = fi as u32 | ((b as u32) << 3) | (ao[i] << 11);
        verts.push(Vertex {
            pos: pos.to_array(),
            data,
        });
    }
    // Split the quad along the brighter diagonal so occlusion gradients stay smooth.
    if ao[0] + ao[2] >= ao[1] + ao[3] {
        idx.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
    } else {
        idx.extend_from_slice(&[start + 1, start + 2, start + 3, start + 1, start + 3, start]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;

    #[test]
    fn single_voxel_is_a_small_rounded_body() {
        let mut w = World::new(1, 1);
        let mut c = Chunk::default();
        c.set(5, 5, 5, STONE);
        w.chunks.insert(IVec3::ZERO, c);
        let m = build(&w, IVec3::ZERO);
        // Eight surface cells around it, one quad per face.
        assert_eq!(m.vertices.len(), 8);
        assert_eq!(m.indices.len(), 36);
        let centre = glam::Vec3::splat(5.5 * VOXEL_SIZE);
        for v in &m.vertices {
            assert_eq!(v.data & 7, SMOOTH_FACE);
            // Corners are pulled in from the cube's corners, so it reads as round.
            let r = (glam::Vec3::from(v.pos) - centre).abs();
            assert!(r.max_element() < 0.25 && r.min_element() > 0.02, "{r:?}");
        }
    }

    #[test]
    fn smooth_normals_round_trip_through_the_packing() {
        for n in [Vec3::Y, Vec3::NEG_Y, Vec3::X, Vec3::new(0.3, -0.8, 0.5).normalize()] {
            let d = smooth_data(STONE, 3, n);
            let q = |s: u32| ((d >> s) & 511) as f32 / 511.0 * 2.0 - 1.0;
            let (u, v) = (q(13), q(22));
            let mut m = Vec3::new(u, 1.0 - u.abs() - v.abs(), v);
            if m.y < 0.0 {
                let sign = |a: f32| if a >= 0.0 { 1.0 } else { -1.0 };
                (m.x, m.z) = ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v));
            }
            assert!(m.normalize().dot(n) > 0.995, "{n:?} -> {m:?}");
        }
    }

    #[test]
    fn flat_ground_is_flat_and_faces_up() {
        let mut w = World::new(1, 1);
        let mut c = Chunk::default();
        for z in 0..CHUNK {
            for x in 0..CHUNK {
                for y in 0..4 {
                    c.set(x, y, z, GRASS);
                }
            }
        }
        w.chunks.insert(IVec3::ZERO, c);
        let m = build(&w, IVec3::ZERO);
        // Away from the chunk's open edges, every vertex sits on the top of the
        // fourth voxel layer with an upward normal.
        for v in m
            .vertices
            .iter()
            .filter(|v| v.pos[0] > 3.0 && v.pos[0] < 12.0 && v.pos[2] > 3.0 && v.pos[2] < 12.0)
        {
            assert!((v.pos[1] - 2.0).abs() < 0.02, "{:?}", v.pos);
            assert_eq!((v.data >> 3) & 255, GRASS as u32);
            assert_eq!(v.data >> 13 & 511, 256);
        }
    }

    #[test]
    fn tall_grass_is_blades_over_smooth_ground() {
        let mut w = World::new(1, 1);
        let mut c = Chunk::default();
        c.set(5, 5, 5, GRASS);
        c.set(5, 6, 5, TALL_GRASS);
        w.chunks.insert(IVec3::ZERO, c);
        let m = build(&w, IVec3::ZERO);
        let ground = m
            .vertices
            .iter()
            .filter(|v| (v.data >> 3) & 255 == GRASS as u32)
            .count();
        assert_eq!(ground, 8);
        let blades = m
            .vertices
            .iter()
            .filter(|v| (v.data >> 3) & 255 == TALL_GRASS as u32)
            .count();
        assert!(
            (3 * BLADES as usize..=5 * (BLADES as usize + 3)).contains(&blades),
            "{blades}"
        );
    }

    #[test]
    fn lantern_sits_on_its_post() {
        let mut w = World::new(1, 1);
        let mut c = Chunk::default();
        c.set(5, 4, 5, STONE);
        for y in 5..8 {
            c.set(5, y, 5, POST);
        }
        c.set(5, 8, 5, LANTERN);
        w.chunks.insert(IVec3::ZERO, c);
        let m = build(&w, IVec3::ZERO);
        let floor = 8.0 * VOXEL_SIZE;
        let mat = |v: &Vertex| (v.data >> 3) & 255;
        let post_top = m
            .vertices
            .iter()
            .filter(|v| mat(v) == WOOD as u32 && v.pos[1] <= floor + 1e-4)
            .map(|v| v.pos[1])
            .fold(f32::MIN, f32::max);
        assert!((post_top - floor).abs() < 1e-4, "post ends at {post_top}");
        // The lantern's collar reaches down over the post and its body starts right above.
        let lowest = m
            .vertices
            .iter()
            .filter(|v| v.pos[1] > floor - 0.2 && mat(v) != STONE as u32 && v.pos[1] < floor)
            .count();
        assert!(lowest > 0);
        // The ground around the post's foot keeps the ground's material.
        assert!(m.vertices.iter().all(|v| mat(v) != POST as u32));
        let body = m.vertices.iter().filter(|v| mat(v) == LANTERN as u32);
        let bottom = body.map(|v| v.pos[1]).fold(f32::MAX, f32::min);
        assert!((bottom - floor - LANTERN_BODY.0).abs() < 1e-4);
        // Nothing post-shaped is wider than the post itself below the lantern.
        let centre = 5.5 * VOXEL_SIZE;
        for v in m
            .vertices
            .iter()
            .filter(|v| mat(v) == WOOD as u32 && v.pos[1] < floor - 0.05)
        {
            let r = (v.pos[0] - centre).abs().max((v.pos[2] - centre).abs());
            assert!(r <= POST_RADIUS + 1e-4, "{r}");
        }
    }

    #[test]
    fn hanging_lantern_hooks_onto_the_arm() {
        let mut w = World::new(1, 1);
        let mut c = Chunk::default();
        c.set(5, 4, 5, STONE);
        for y in 5..=10 {
            c.set(5, y, 5, POST);
        }
        c.set(6, 9, 5, LANTERN);
        w.chunks.insert(IVec3::ZERO, c);
        let m = build(&w, IVec3::ZERO);
        let mat = |v: &Vertex| (v.data >> 3) & 255;
        let arm_y = 10.0 * VOXEL_SIZE + ARM_HEIGHT;
        let lamp_x = 6.5 * VOXEL_SIZE;
        // The arm reaches past the lantern's centre at arm height.
        let reach = m
            .vertices
            .iter()
            .filter(|v| mat(v) == WOOD as u32 && (v.pos[1] - arm_y).abs() < 0.06)
            .map(|v| v.pos[0])
            .fold(f32::MIN, f32::max);
        assert!(reach > lamp_x, "arm ends at {reach}");
        // The hook climbs from the lantern roof to the arm.
        let hook_top = m
            .vertices
            .iter()
            .filter(|v| mat(v) == LANTERN as u32)
            .map(|v| v.pos[1])
            .fold(f32::MIN, f32::max);
        assert!((hook_top - arm_y).abs() < 1e-3, "hook reaches {hook_top}");
    }

    #[test]
    fn hidden_faces_are_culled() {
        let mut w = World::new(1, 1);
        let mut c = Chunk::default();
        c.set(5, 5, 5, STONE);
        c.set(6, 5, 5, STONE);
        w.chunks.insert(IVec3::ZERO, c);
        let m = build(&w, IVec3::ZERO);
        assert_eq!(m.indices.len(), 10 * 6);
    }

    #[test]
    fn faces_wind_counter_clockwise_outward() {
        let mut w = World::new(1, 1);
        let mut c = Chunk::default();
        c.set(5, 5, 5, STONE);
        w.chunks.insert(IVec3::ZERO, c);
        let m = build(&w, IVec3::ZERO);
        let centre = glam::Vec3::splat(5.5 * VOXEL_SIZE);
        for tri in m.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|i| glam::Vec3::from(m.vertices[tri[i] as usize].pos));
            let normal = (b - a).cross(c - a);
            let outward = (a + b + c) / 3.0 - centre;
            assert!(normal.dot(outward) > 0.0);
        }
    }
}
