//! Authored procedural models: buildings and fences are put together from
//! shaped parts (beams, boards, panels, tiles) rather than sculpted from
//! voxels. Every part is emitted straight into a `MeshData` with smooth
//! normals. Box edges carry normals that lean towards their neighbours, so the
//! lighting rolls softly over each edge and nothing reads as a hard cube.

use crate::block::Block;
use crate::mesh::{smooth_data, MeshData, Vertex};
use glam::Vec3;

/// How far a box corner's normal leans towards its neighbouring faces; larger
/// values read as a rounder edge.
const EDGE_SOFTNESS: f32 = 0.45;

/// How far corner normals lean for a face this many metres from centre to edge.
fn lean(half: f32) -> f32 {
    EDGE_SOFTNESS * (0.15 / half.max(1e-3)).min(1.0)
}

/// An oriented box. `axes` are three unit vectors (right-handed), `half` the
/// half extents along them.
pub fn soft_box(out: &mut MeshData, centre: Vec3, axes: [Vec3; 3], half: Vec3, mat: Block) {
    let h = [half.x, half.y, half.z];
    for (i, &n) in axes.iter().enumerate() {
        let (u, v) = (axes[(i + 1) % 3], axes[(i + 2) % 3]);
        let (hu, hv) = (h[(i + 1) % 3], h[(i + 2) % 3]);
        for s in [1.0f32, -1.0] {
            let face_c = centre + n * h[i] * s;
            let start = out.vertices.len() as u32;
            for (cu, cv) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let pos = face_c + u * hu * cu + v * hv * cv;
                // Thin parts round off fully; broad faces stay nearly flat.
                let (ku, kv) = (lean(hu), lean(hv));
                let normal = (n * s + u * cu * ku + v * cv * kv).normalize();
                out.vertices.push(Vertex {
                    pos: pos.to_array(),
                    data: smooth_data(mat, 3, normal),
                });
            }
            // u x v = n, so (0,1,2) winds counter-clockwise seen from +n.
            if s > 0.0 {
                out.indices
                    .extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
            } else {
                out.indices
                    .extend_from_slice(&[start, start + 2, start + 1, start, start + 3, start + 2]);
            }
        }
    }
}

/// An axis-aligned soft box between two corners.
pub fn block(out: &mut MeshData, lo: Vec3, hi: Vec3, mat: Block) {
    soft_box(
        out,
        (lo + hi) * 0.5,
        [Vec3::X, Vec3::Y, Vec3::Z],
        (hi - lo).abs() * 0.5,
        mat,
    );
}

/// A square-section beam of the given half thickness from `a` to `b`.
pub fn beam(out: &mut MeshData, a: Vec3, b: Vec3, half: f32, mat: Block) {
    let d = b - a;
    let len = d.length();
    if len < 1e-4 {
        return;
    }
    let x = d / len;
    let up = if x.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
    let z = x.cross(up).normalize();
    let y = z.cross(x);
    soft_box(out, (a + b) * 0.5, [x, y, z], Vec3::new(len * 0.5, half, half), mat);
}

/// A flat board of `half` extents lying along `along` (its length) with its
/// face towards `normal`.
pub fn board(out: &mut MeshData, centre: Vec3, along: Vec3, normal: Vec3, half: Vec3, mat: Block) {
    let x = along.normalize();
    let y = normal.normalize();
    let z = x.cross(y).normalize();
    let y = z.cross(x);
    soft_box(out, centre, [x, y, z], half, mat);
}

/// A flat convex polygon drawn from both sides.
pub fn panel(out: &mut MeshData, pts: &[Vec3], mat: Block) {
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

/// Appends `src` to `dst`.
pub fn append(dst: &mut MeshData, src: &MeshData) {
    let base = dst.vertices.len() as u32;
    dst.vertices.extend_from_slice(&src.vertices);
    dst.indices.extend(src.indices.iter().map(|i| i + base));
    let base = dst.water_vertices.len() as u32;
    dst.water_vertices.extend_from_slice(&src.water_vertices);
    dst.water_indices.extend(src.water_indices.iter().map(|i| i + base));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_boxes_are_closed_and_face_outward() {
        let mut out = MeshData::default();
        let axes = [Vec3::new(0.6, 0.8, 0.0), Vec3::new(-0.8, 0.6, 0.0), Vec3::Z];
        soft_box(&mut out, Vec3::new(1.0, 2.0, 3.0), axes, Vec3::new(0.5, 0.2, 1.0), 1);
        assert_eq!(out.indices.len(), 36);
        for tri in out.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|k| Vec3::from(out.vertices[tri[k] as usize].pos));
            let normal = (b - a).cross(c - a);
            assert!(normal.dot((a + b + c) / 3.0 - Vec3::new(1.0, 2.0, 3.0)) > 0.0);
        }
    }

    #[test]
    fn beams_reach_both_ends() {
        let mut out = MeshData::default();
        beam(&mut out, Vec3::ZERO, Vec3::new(0.0, 3.0, 0.0), 0.1, 1);
        let ys: Vec<f32> = out.vertices.iter().map(|v| v.pos[1]).collect();
        assert!(ys.iter().any(|y| (y - 3.0).abs() < 1e-4));
        assert!(ys.iter().any(|y| y.abs() < 1e-4));
    }
}
