use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::Matcher;
use super::resolved::ResolvedScratch;
use crate::k_merge::{
    k_merge_matches_by_index_asc, k_merge_matches_by_index_desc,
    k_merge_matches_by_score_then_index_asc, k_merge_matches_by_score_then_index_desc,
};
use crate::sort::radix_sort_matches;
use crate::{Match, SortStrategy};

const ITEMS_PER_THREAD: usize = if cfg!(miri) { 4 } else { 2000 };
const CHUNK_SIZE: usize = if cfg!(miri) { 8 } else { 2048 };

impl Matcher {
    /// Matches a list of haystacks in parallel on multiple real threads,
    /// returning a list of [`Match`] values.
    ///
    /// If `threads == 0`, the matcher will default to available CPU cores - 2.
    ///
    /// This API provides the most performant path when matching on lists.
    pub fn match_list_parallel<S: AsRef<str> + Sync>(
        &mut self,
        haystacks: &[S],
        threads: usize,
    ) -> Vec<Match> {
        Self::guard_against_haystack_overflow(haystacks.len(), 0);

        // If threads == 0, default to available cpu cores
        let mut threads = threads;
        if threads == 0 {
            threads = std::thread::available_parallelism()
                .map(|n| n.get().saturating_sub(2))
                .unwrap_or(1)
                .max(1);
        }

        // Limit threads based on the number of haystacks
        let threads = threads
            .min(haystacks.len().div_ceil(ITEMS_PER_THREAD))
            .max(1);

        if haystacks.is_empty() || self.patterns.is_empty() || threads == 1 {
            return self.match_list(haystacks);
        }

        // Smaller chunks enable better load balancing via stealing
        // but too small increases atomic contention
        let num_chunks = haystacks.len().div_ceil(CHUNK_SIZE);
        let next_chunk = AtomicUsize::new(0);

        let matcher = &*self;

        thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut local_matches = Vec::new();
                        let mut matcher = matcher.clone();

                        loop {
                            // Claim next available chunk
                            let chunk_idx = next_chunk.fetch_add(1, Ordering::Relaxed);
                            if chunk_idx >= num_chunks {
                                break;
                            }

                            let start = chunk_idx * CHUNK_SIZE;
                            let end = (start + CHUNK_SIZE).min(haystacks.len());
                            let haystacks_chunk = &haystacks[start..end];

                            matcher.match_list_into(
                                haystacks_chunk,
                                start as u32,
                                &mut local_matches,
                            );
                        }

                        // Each thread sorts so that we can perform k-way merge
                        if matcher.config.sort.is_reversed() {
                            local_matches.reverse();
                        }
                        if matcher.config.sort.is_by_score() {
                            radix_sort_matches(&mut local_matches);
                        }

                        local_matches
                    })
                })
                .collect();

            let matches = handles.into_iter().map(|h| h.join().unwrap()).collect();
            match matcher.config.sort {
                SortStrategy::ScoreThenIndexAsc => k_merge_matches_by_score_then_index_asc(matches),
                SortStrategy::ScoreThenIndexDesc => {
                    k_merge_matches_by_score_then_index_desc(matches)
                }
                SortStrategy::IndexAsc => k_merge_matches_by_index_asc(matches),
                SortStrategy::IndexDesc => k_merge_matches_by_index_desc(matches),
            }
        })
    }
    /// Matches items in parallel on multiple real threads, resolving each
    /// item's haystack bytes through the `resolve` callback, returning a list
    /// of [`Match`] values ordered by the configured [`SortStrategy`].
    ///
    /// If `threads == 0`, the matcher will default to available CPU cores - 2.
    ///
    /// See [`Matcher::match_list_resolved_into`] for the resolver contract.
    pub fn match_list_parallel_resolved<T, F, const N: usize>(
        &mut self,
        items: &[T],
        resolve: &F,
        threads: usize,
    ) -> Vec<Match>
    where
        T: Sync,
        F: Fn(&T, &mut [*const u8; N]) -> Option<(usize, u16)> + Sync,
    {
        Self::guard_against_haystack_overflow(items.len(), 0);

        // If threads == 0, default to available cpu cores
        let mut threads = threads;
        if threads == 0 {
            threads = std::thread::available_parallelism()
                .map(|n| n.get().saturating_sub(2))
                .unwrap_or(1)
                .max(1);
        }

        // Limit threads based on the number of items
        let threads = threads.min(items.len().div_ceil(ITEMS_PER_THREAD)).max(1);

        if items.is_empty() || self.patterns.is_empty() || threads == 1 {
            return self.match_list_resolved(items, resolve);
        }

        let num_chunks = items.len().div_ceil(CHUNK_SIZE);
        let next_chunk = AtomicUsize::new(0);

        let matcher = &*self;

        thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut local_matches = Vec::new();
                        let mut matcher = matcher.clone();
                        let mut scratch = ResolvedScratch::default();

                        loop {
                            // Claim next available chunk
                            let chunk_idx = next_chunk.fetch_add(1, Ordering::Relaxed);
                            if chunk_idx >= num_chunks {
                                break;
                            }

                            let start = chunk_idx * CHUNK_SIZE;
                            let end = (start + CHUNK_SIZE).min(items.len());

                            matcher.match_list_resolved_into_with(
                                &items[start..end],
                                start as u32,
                                resolve,
                                &mut local_matches,
                                &mut scratch,
                            );
                        }

                        // Each thread sorts so that we can perform k-way merge
                        if matcher.config.sort.is_reversed() {
                            local_matches.reverse();
                        }
                        if matcher.config.sort.is_by_score() {
                            radix_sort_matches(&mut local_matches);
                        }

                        local_matches
                    })
                })
                .collect();

            let matches = handles.into_iter().map(|h| h.join().unwrap()).collect();
            match matcher.config.sort {
                SortStrategy::ScoreThenIndexAsc => k_merge_matches_by_score_then_index_asc(matches),
                SortStrategy::ScoreThenIndexDesc => {
                    k_merge_matches_by_score_then_index_desc(matches)
                }
                SortStrategy::IndexAsc => k_merge_matches_by_index_asc(matches),
                SortStrategy::IndexDesc => k_merge_matches_by_index_desc(matches),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::CHUNK_SIZE;
    use crate::{Config, Matcher};

    fn thread_counts() -> &'static [usize] {
        if cfg!(miri) {
            &[2]
        } else {
            &[1, 2, 3, 4, 5, 6, 7, 8]
        }
    }

    #[test]
    fn sorted_matches_sequential_across_chunk_boundaries() {
        let mut haystacks = (0..2 * CHUNK_SIZE + 5)
            .map(|index| format!("nomatch-{index}"))
            .collect::<Vec<_>>();
        for (index, value) in [
            (0, "abc"),
            (CHUNK_SIZE - 1, "xabc"),
            (CHUNK_SIZE, "abxc"),
            (CHUNK_SIZE + 1, "alpha/beta/abc"),
            (2 * CHUNK_SIZE - 1, "ABC"),
            (2 * CHUNK_SIZE, "a_b_c"),
            (2 * CHUNK_SIZE + 4, "zabc"),
        ] {
            haystacks[index] = value.to_string();
        }

        let config = Config::default();
        let sequential = Matcher::new("abc", &config).match_list(&haystacks);
        assert!(sequential.is_sorted());

        for &threads in thread_counts() {
            let parallel = Matcher::new("abc", &config).match_list_parallel(&haystacks, threads);
            assert_eq!(&parallel, &sequential, "threads={threads}");
            assert!(parallel.is_sorted(), "threads={threads}");
        }
    }

    #[test]
    fn zero_threads_uses_available_parallelism() {
        let haystacks = ["abc", "xabc", "zzz"];
        let mut matcher = Matcher::new("abc", &Config::default());
        let sequential = matcher.match_list(&haystacks);
        assert_eq!(matcher.match_list_parallel(&haystacks, 0), sequential);
    }

    #[test]
    fn multi_pattern_matches_sequential_across_chunk_boundaries() {
        use crate::{Matcher, Pattern, SortStrategy};

        let mut haystacks = (0..2 * CHUNK_SIZE + 5)
            .map(|index| format!("nomatch-{index}"))
            .collect::<Vec<_>>();
        for (index, value) in [
            (0, "abc"),
            (1, "abcxyz"),
            (CHUNK_SIZE - 1, "xabc"),
            (CHUNK_SIZE, "abxc"),
            (CHUNK_SIZE + 1, "alpha/beta/abc"),
            (CHUNK_SIZE + 2, "xyz/abc"),
            (2 * CHUNK_SIZE - 1, "ABC"),
            (2 * CHUNK_SIZE, "a_b_c"),
            (2 * CHUNK_SIZE + 4, "zabc"),
        ] {
            haystacks[index] = value.to_string();
        }

        for query in ["abc !xyz", "abc a", "!abc !xyz"] {
            for sort in [SortStrategy::ScoreThenIndexAsc, SortStrategy::IndexAsc] {
                let config = Config::default().sort(sort);
                let mut matcher = Matcher::from_patterns(&Pattern::parse_query(query), &config);
                let sequential = matcher.match_list(&haystacks);

                for &threads in thread_counts() {
                    let parallel = matcher.match_list_parallel(&haystacks, threads);
                    assert_eq!(
                        &parallel, &sequential,
                        "query={query:?}, sort={sort:?}, threads={threads}"
                    );
                }
            }
        }
    }

    #[test]
    fn resolved_parallel_matches_sequential_across_chunk_boundaries() {
        use crate::matcher::resolved::tests::{ChunkItem, resolve_chunks, string_to_chunks};
        use crate::{Pattern, SortStrategy};

        let mut haystacks = (0..2 * CHUNK_SIZE + 5)
            .map(|index| format!("nomatch-{index}"))
            .collect::<Vec<_>>();
        for (index, value) in [
            (0, "abc"),
            (CHUNK_SIZE - 1, "xabc"),
            (CHUNK_SIZE, "abxc"),
            (CHUNK_SIZE + 1, "alpha/beta/abc"),
            (2 * CHUNK_SIZE - 1, "ABC"),
            (2 * CHUNK_SIZE, "a_b_c"),
            (2 * CHUNK_SIZE + 4, "zabc"),
        ] {
            haystacks[index] = value.to_string();
        }
        let chunk_data: Vec<ChunkItem> = haystacks.iter().map(|s| string_to_chunks(s)).collect();

        for query in ["abc", "abc !xyz"] {
            for sort in [
                SortStrategy::ScoreThenIndexAsc,
                SortStrategy::ScoreThenIndexDesc,
                SortStrategy::IndexAsc,
                SortStrategy::IndexDesc,
            ] {
                let config = Config::default().sort(sort);
                let mut matcher = Matcher::from_patterns(&Pattern::parse_query(query), &config);
                let sequential = matcher.match_list(&haystacks);
                assert!(!sequential.is_empty());

                for &threads in thread_counts() {
                    let parallel = matcher.match_list_parallel_resolved(
                        &chunk_data,
                        &resolve_chunks::<2>,
                        threads,
                    );
                    assert_eq!(
                        &parallel, &sequential,
                        "query={query:?}, sort={sort:?}, threads={threads}"
                    );
                }
            }
        }
    }

    #[test]
    fn resolved_parallel_skips_none_items() {
        use crate::matcher::resolved::tests::{ChunkItem, resolve_chunks, string_to_chunks};

        let present = string_to_chunks("hello_world");
        let items: Vec<Option<ChunkItem>> = (0..2 * CHUNK_SIZE)
            .map(|i| (i % 3 != 1).then(|| present.clone()))
            .collect();
        let resolve =
            |item: &Option<ChunkItem>, ptrs_buf: &mut [*const u8; 4]| -> Option<(usize, u16)> {
                item.as_ref()
                    .and_then(|item| resolve_chunks(item, ptrs_buf))
            };
        let expected: Vec<u32> = (0..2 * CHUNK_SIZE as u32).filter(|i| i % 3 != 1).collect();

        for needle in ["hw", ""] {
            let mut matcher = Matcher::new(needle, &Config::default());
            for &threads in thread_counts() {
                let matches = matcher.match_list_parallel_resolved(&items, &resolve, threads);
                assert_eq!(
                    matches.iter().map(|m| m.index).collect::<Vec<_>>(),
                    expected,
                    "needle={needle:?} threads={threads}"
                );
            }
        }
    }
}
