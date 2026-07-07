use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion};
use neo_frizbee::Match;
use rand::RngExt;

fn bench_with_scale(c: &mut Criterion, scale: usize, batch_size: BatchSize) {
    let mut rng = rand::rng();
    let matches = (0..16 * scale)
        .map(|index| Match {
            score: rng.random(),
            index: index as u32,
            exact: rng.random_bool(0.5),
            #[cfg(feature = "match_end_col")]
            end_col: 0,
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("Sort");

    let input = Vec::from(&matches[..scale]);

    group.throughput(criterion::Throughput::Elements(scale as u64));

    group.bench_with_input(BenchmarkId::new("std", scale), &input, |b, i| {
        b.iter_batched(
            || i.clone(),
            |mut d| {
                d.sort_by_key(|m| m.score);
                d
            },
            batch_size,
        )
    });

    group.bench_with_input(BenchmarkId::new("radix", scale), &input, |b, i| {
        b.iter_batched(
            || i.clone(),
            |mut m| {
                neo_frizbee::radix_sort_matches(&mut m);
                m
            },
            batch_size,
        )
    });

    group.finish();
}

fn bench(c: &mut Criterion) {
    bench_with_scale(c, 10, BatchSize::SmallInput);
    bench_with_scale(c, 100, BatchSize::SmallInput);
    bench_with_scale(c, 1000, BatchSize::LargeInput);
    bench_with_scale(c, 10000, BatchSize::LargeInput);
    bench_with_scale(c, 100000, BatchSize::LargeInput);
}

criterion::criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(500))
        .with_plots();
    targets = bench
}
criterion::criterion_main!(benches);
