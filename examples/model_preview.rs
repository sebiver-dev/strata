//! Renders tree models on the CPU to a PPM image, for tuning shapes without
//! a GPU. Run with: cargo run --release --example model_preview -- out.ppm

use glam::{Mat4, Vec2, Vec3, Vec4};
use strata::mesh::MeshData;
use strata::trees::{Kind, Tree};

const W: usize = 1200;
const H: usize = 700;

fn normal_of(d: u32) -> Vec3 {
    let q = |s: u32| ((d >> s) & 511) as f32 / 511.0 * 2.0 - 1.0;
    let (u, v) = (q(13), q(22));
    let mut m = Vec3::new(u, 1.0 - u.abs() - v.abs(), v);
    if m.y < 0.0 {
        let sign = |a: f32| if a >= 0.0 { 1.0 } else { -1.0 };
        (m.x, m.z) = ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v));
    }
    m.normalize()
}

fn colour(mat: u32) -> Vec3 {
    match mat {
        27 => Vec3::new(0.30, 0.23, 0.17),
        28 => Vec3::new(0.18, 0.34, 0.09),
        29 => Vec3::new(0.09, 0.2, 0.11),
        _ => Vec3::new(0.6, 0.6, 0.6),
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or("preview.ppm".into());
    let mut mesh = MeshData::default();
    let trees = [
        (Kind::Broadleaf, 10.0, 0.0, 1u32, None),
        (Kind::Broadleaf, 13.0, 16.0, 7, None),
        (Kind::Conifer, 18.0, 32.0, 3, None),
        (Kind::Broadleaf, 18.0, 52.0, 11, Some(Vec2::new(-1.0, 0.3).normalize())),
        (Kind::Conifer, 14.0, 68.0, 9, None),
    ];
    for (kind, height, x, seed, hero) in trees {
        let t = Tree {
            base: Vec3::new(x, 0.0, 0.0),
            height,
            kind,
            seed,
            hero,
        };
        if std::env::args().any(|a| a == "far") {
            t.far(&mut mesh, 0);
        } else {
            t.mesh(&mut mesh);
        }
    }
    println!("{} triangles", mesh.indices.len() / 3);
    // `close` looks up at the hero oak from where a player would stand.
    let close = std::env::args().any(|a| a == "close");
    let (eye, at) = if close {
        (Vec3::new(47.0, 1.6, 11.0), Vec3::new(52.0, 7.0, 0.0))
    } else {
        (Vec3::new(34.0, 9.0, 62.0), Vec3::new(34.0, 7.0, 0.0))
    };
    let view = Mat4::look_at_rh(eye, at, Vec3::Y);
    let proj = Mat4::perspective_rh(0.9, W as f32 / H as f32, 0.5, 500.0);
    let vp = proj * view;
    let sun = Vec3::new(-0.5, 0.45, 0.6).normalize();
    let mut depth = vec![f32::INFINITY; W * H];
    let mut img = vec![Vec3::new(0.55, 0.62, 0.75); W * H];
    for y in 0..H {
        let t = y as f32 / H as f32;
        for x in 0..W {
            img[y * W + x] = Vec3::new(0.95, 0.7, 0.5).lerp(Vec3::new(0.45, 0.55, 0.75), 1.0 - t);
        }
    }
    let project = |p: Vec3| {
        let c = vp * Vec4::new(p.x, p.y, p.z, 1.0);
        let n = c.truncate() / c.w;
        Vec3::new((n.x * 0.5 + 0.5) * W as f32, (0.5 - n.y * 0.5) * H as f32, c.w)
    };
    for tri in mesh.indices.chunks(3) {
        let v = [0, 1, 2].map(|k| mesh.vertices[tri[k] as usize]);
        let s = v.map(|v| project(Vec3::from(v.pos)));
        if s.iter().any(|p| p.z < 0.5) {
            continue;
        }
        let area = (s[1].x - s[0].x) * (s[2].y - s[0].y) - (s[2].x - s[0].x) * (s[1].y - s[0].y);
        // Counter-clockwise in world is clockwise on screen (y down): cull the rest.
        if area >= 0.0 {
            continue;
        }
        let lo = s.iter().fold(Vec2::splat(f32::MAX), |m, p| m.min(Vec2::new(p.x, p.y)));
        let hi = s.iter().fold(Vec2::splat(f32::MIN), |m, p| m.max(Vec2::new(p.x, p.y)));
        let shade = v.map(|v| {
            let n = normal_of(v.data);
            let mat = (v.data >> 3) & 255;
            let ao = ((v.data >> 11) & 3) as f32 / 3.0;
            let wrap = if mat == 28 || mat == 29 { 0.35 } else { 0.0 };
            let ndl = ((n.dot(sun) + wrap) / (1.0 + wrap)).max(0.0);
            let sky = 0.5 + 0.5 * n.y;
            colour(mat) * (Vec3::new(1.0, 0.8, 0.6) * 2.2 * ndl + Vec3::new(0.4, 0.45, 0.6) * sky * (0.4 + 0.6 * ao))
        });
        for py in lo.y.max(0.0) as usize..(hi.y.ceil() as usize).min(H) {
            for px in lo.x.max(0.0) as usize..(hi.x.ceil() as usize).min(W) {
                let p = Vec2::new(px as f32 + 0.5, py as f32 + 0.5);
                let e = |a: Vec3, b: Vec3| (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
                let w0 = e(s[1], s[2]) / area;
                let w1 = e(s[2], s[0]) / area;
                let w2 = e(s[0], s[1]) / area;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = w0 * s[0].z + w1 * s[1].z + w2 * s[2].z;
                let i = py * W + px;
                if z < depth[i] {
                    depth[i] = z;
                    img[i] = shade[0] * w0 + shade[1] * w1 + shade[2] * w2;
                }
            }
        }
    }
    let mut out = format!("P6 {W} {H} 255\n").into_bytes();
    for c in img {
        let m = c / (c + 1.0) * 1.6;
        out.extend(m.to_array().map(|x| (x.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8));
    }
    std::fs::write(path, out).unwrap();
}
