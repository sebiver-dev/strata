//! Measures how long generating and meshing the area around spawn takes, and
//! how long the far field for the whole valley takes to build.
//! Run with: cargo run --release --example worldgen_bench

use strata::world::World;

fn main() {
    let mut world = World::new(20260927, 12);
    let spawn = world.terrain.spawn_point();
    let t = std::time::Instant::now();
    world.stream(spawn, f64::INFINITY);
    let gen = t.elapsed();
    let chunks = world.chunks.len();

    let ready: Vec<_> = world
        .dirty
        .iter()
        .copied()
        .filter(|c| world.neighbourhood_ready(*c))
        .collect();
    let t = std::time::Instant::now();
    let (mut tris, mut meshed) = (0, 0);
    for c in &ready {
        let m = strata::mesh::build(&world, *c);
        tris += (m.indices.len() + m.water_indices.len()) / 3;
        meshed += !m.is_empty() as usize;
    }
    let mesh = t.elapsed();
    println!("spawn at {spawn:.1}");
    println!(
        "generated {chunks} chunks in {gen:.2?} ({:.3} ms each)",
        gen.as_secs_f64() * 1000.0 / chunks as f64
    );
    println!(
        "meshed {} chunks ({meshed} non-empty, {tris} triangles) in {mesh:.2?} ({:.3} ms each)",
        ready.len(),
        mesh.as_secs_f64() * 1000.0 / ready.len() as f64
    );

    let t = std::time::Instant::now();
    let mut far = strata::far::FarField::default();
    let (mut far_tris, mut tiles) = (0, 0);
    far.update(&world.terrain, spawn, f64::INFINITY, |_, m| {
        far_tris += (m.indices.len() + m.water_indices.len()) / 3;
        tiles += 1;
    });
    let far_time = t.elapsed();
    println!(
        "built {tiles} far tiles ({far_tris} triangles) in {far_time:.2?} ({:.3} ms each)",
        far_time.as_secs_f64() * 1000.0 / tiles as f64
    );
}
