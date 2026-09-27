//! Measures how long generating and meshing the area around spawn takes.
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
}
