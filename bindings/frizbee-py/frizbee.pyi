"""Fast typo-resistant fuzzy matching via SIMD smith waterman, similar algorithm to FZF."""

from collections.abc import Iterable, Sequence
from typing import Literal, final

__version__: str

Casing = Literal["ignore", "smart", "respect"]
Unicode = Literal["ignore", "smart", "always"]
Matching = Literal["fuzzy", "exact", "prefix", "suffix", "substring"]
Sort = Literal[
    "score_then_index_asc", "score_then_index_desc", "index_asc", "index_desc", "unsorted"
]

@final
class Haystacks:
    """Incrementally updatable Haystack copied into Rust memory to avoid
    per-call overhead and locking the GIL.

    Match indices refer to this list's order.
    """

    def __init__(self, items: Iterable[str] | None = None) -> None: ...
    def append(self, item: str) -> None:
        """Appends a single haystack."""

    def extend(self, items: Iterable[str]) -> None:
        """Appends every haystack in the iterable (list/tuple are the fast paths)."""

    def clear(self) -> None:
        """Empties the list, keeping the allocation for reuse."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> str:
        """Returns a haystack by index"""

@final
class Scoring:
    """Controls the scoring used by the smith waterman algorithm.

    Mirrors the core defaults when a field is omitted. Pay close attention to
    the documentation for each property, as small changes can lead to poor
    matching.
    """

    def __init__(
        self,
        *,
        match_score: int = 12,
        mismatch_penalty: int = 6,
        gap_open_penalty: int = 5,
        gap_extend_penalty: int = 1,
        prefix_bonus: int = 12,
        capitalization_bonus: int = 4,
        matching_case_bonus: int = 4,
        exact_match_bonus: int = 8,
        delimiter_bonus: int = 4,
    ) -> None: ...
    @property
    def match_score(self) -> int:
        """Score for a matching character between needle and haystack."""

    @property
    def mismatch_penalty(self) -> int:
        """Penalty for a mismatch (substitution)."""

    @property
    def gap_open_penalty(self) -> int:
        """Penalty for opening a gap (deletion/insertion)."""

    @property
    def gap_extend_penalty(self) -> int:
        """Penalty for extending a gap (deletion/insertion)."""

    @property
    def prefix_bonus(self) -> int:
        """Bonus for matching the first character of the haystack (e.g. "h" on "hello_world")."""

    @property
    def capitalization_bonus(self) -> int:
        """Bonus for matching a capital letter after a lowercase letter.

        E.g. "b" on "fooBar" will receive a bonus on "B".
        """

    @property
    def matching_case_bonus(self) -> int:
        """Bonus for matching the case of the needle.

        E.g. "WorLd" on "WoRld" will receive a bonus on "W", "o", "d".
        """

    @property
    def exact_match_bonus(self) -> int:
        """Bonus for matching the exact needle (e.g. "foo" on "foo" will receive the bonus)."""

    @property
    def delimiter_bonus(self) -> int:
        """Bonus for matching *after* a delimiter character.

        E.g. "hw" on "hello_world" will give a bonus on "w".
        """

    @property
    def max_needle_len(self) -> int:
        """Needle length up to which scores are guaranteed to fit within the
        16-bit score.

        Longer needles still match, but their scores may saturate at ``0xffff``.
        """

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Match:
    """Result of a fuzzy match, containing the score and index in the haystack. Supports ``==`` and ``hash()``."""

    @property
    def score(self) -> int:
        """Score of the match, higher is better."""

    @property
    def index(self) -> int:
        """Index of the match in the original list of haystacks."""

    @property
    def exact(self) -> bool:
        """Matched the needle exactly (e.g. "foo" on "foo")."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class MatchIndices:
    """Like `Match` but includes the indices of the chars in the haystack that
    matched the needle in ascending order."""

    @property
    def score(self) -> int:
        """Score of the match, higher is better."""

    @property
    def index(self) -> int:
        """Index of the match in the original list of haystacks."""

    @property
    def exact(self) -> bool:
        """Matched the needle exactly (e.g. "foo" on "foo")."""

    @property
    def indices(self) -> tuple[int, ...]:
        """Ascending Python character indices in the haystack that matched the needle."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Pattern:
    """A single pattern to match.

    ``needle`` is read-only (make a new ``Pattern`` instead).
    """

    def __init__(
        self,
        needle: str,
        *,
        negated: bool = False,
        matching: Matching | None = None,
        max_typos: int | None = None,
        casing: Casing | None = None,
        unicode: Unicode | None = None,
        scoring: Scoring | None = None,
    ) -> None: ...
    @property
    def needle(self) -> str:
        """Text to match with the syntax stripped, e.g. ``foo``."""

    @property
    def negated(self) -> bool:
        """Haystacks matching this atom are excluded."""

    @negated.setter
    def negated(self, value: bool) -> None: ...
    @property
    def matching(self) -> Matching | None:
        """Per-pattern override for ``matching``; ``None`` inherits it."""

    @matching.setter
    def matching(self, value: Matching | None) -> None: ...
    @property
    def max_typos(self) -> int | None:
        """Per-pattern override for ``max_typos``; ``None`` inherits the matcher
        config.

        The matcher's ``max_typos`` is itself optional, so there is no way to
        request unlimited typos for a single pattern while the matcher's config
        sets a limit. Instead, just set it to ``0xFFFF``.
        """

    @max_typos.setter
    def max_typos(self, value: int | None) -> None: ...
    @property
    def casing(self) -> Casing | None:
        """Per-pattern override for ``casing``; ``None`` inherits the matcher
        config."""

    @casing.setter
    def casing(self, value: Casing | None) -> None: ...
    @property
    def unicode(self) -> Unicode | None:
        """Per-pattern override for ``unicode``; ``None`` inherits the matcher
        config."""

    @unicode.setter
    def unicode(self, value: Unicode | None) -> None: ...
    @property
    def scoring(self) -> Scoring | None:
        """Per-pattern override for ``scoring``; ``None`` inherits the matcher
        config."""

    @scoring.setter
    def scoring(self, value: Scoring | None) -> None: ...
    def __eq__(self, other: object) -> bool: ...

def parse_query(query: str) -> list[Pattern]:
    """Parses a query of whitespace separated atoms.

    See `Matcher.from_query` for the atom syntax. E.g. ``foo !^bar`` matches
    haystacks that fuzzy match ``foo`` and don't start with ``bar``. Escape
    a literal space with a backslash, e.g. ``foo\\ bar`` is a single atom.
    Atoms with an empty needle, e.g. ``!`` or ``^$``, are dropped.

    The returned patterns carry only the ``matching`` mode derived from the
    syntax. Any other per-pattern override is left as ``None`` and inherits
    the matcher's config. Set fields on the results to override per-pattern.
    For example, setting the max typos based on needle length::

        patterns = parse_query("foo longerneedle")
        for p in patterns:
            p.max_typos = len(p.needle) // 4
        matcher = Matcher(patterns)
    """

@final
class Matcher:
    """Primary entrypoint for fuzzy matching.

    `Matcher` compiles the patterns once, allocates memory for the Smith
    Waterman matrix, and reuses the selected SIMD backend.

    Ideally, only construct these at most once per list. They're cheap to
    construct, but end up being expensive if you construct them for each item
    in your list.
    """

    def __init__(
        self,
        patterns: str | Pattern | Sequence[str | Pattern],
        *,
        max_typos: int | None = 0,
        limit: int | None = None,
        casing: Casing | None = None,
        unicode: Unicode | None = None,
        matching: Matching | None = None,
        sort: Sort | None = None,
        scoring: Scoring | None = None,
    ) -> None:
        """Creates a matcher from one or more patterns (string or `Pattern`),
        matched independently.

        Strings convert into patterns that match literally. Use `from_query`
        for query syntax. A haystack matches when all of the patterns match,
        where the score is the sum of each pattern's score.
        """

    @classmethod
    def from_query(
        cls,
        query: str,
        *,
        max_typos: int | None = 0,
        limit: int | None = None,
        casing: Casing | None = None,
        unicode: Unicode | None = None,
        matching: Matching | None = None,
        sort: Sort | None = None,
        scoring: Scoring | None = None,
    ) -> Matcher:
        """Parses whitespace-separated query atoms, where special syntax
        changes each atom's matching mode:

        - ``foo`` - defers to matcher config, which defaults to fuzzy
        - ``^foo`` - prefix
        - ``foo$`` - suffix
        - ``'foo`` - substring
        - ``^foo$`` - exact
        - ``!foo`` - negated, substring unless combined with the syntax above

        Any special character can be escaped with a backslash, e.g. ``\\!foo``,
        ``\\^foo``, ``foo\\$`` or ``\\'foo`` match the literal leading/trailing
        character, and ``foo\\ bar`` matches the literal space.
        """

    def match(
        self,
        haystacks: Iterable[str] | Haystacks,
        *,
        threads: int | None = None,
    ) -> list[Match]:
        """Matches a list of haystacks.

        When ``threads`` is set, matches in parallel on multiple real threads.
        When ``threads == 0``, the matcher will default to available CPU cores -
        2.

        This API provides the most performant path when matching on lists.
        Ordinary iterables are copied into Rust memory before parallel matching;
        matching then runs with the GIL released for every input type. Sequential
        matching releases the GIL when passed a `Haystacks`.
        """

    def match_indices(
        self, haystacks: Iterable[str] | Haystacks
    ) -> list[MatchIndices]:
        """Matches a list of haystacks, returning a list of `MatchIndices`
        which are equivalent to `Match` except they include the indices of the
        matched characters in the haystack.

        This API has not been optimized for performance, and should only be
        used on small lists or after matching a list of haystacks with
        `match`. Useful for displaying matched indices in the UI.

        The GIL is released while matching if passed a `Haystacks`.
        """

    def match_one(self, haystack: str, index: int = 0) -> Match | None:
        """Matches a single haystack, returning its `Match` if it passes.

        This API performs much slower than the `match` API with
        `Haystacks`.
        """

    def match_one_indices(
        self, haystack: str, index: int = 0
    ) -> MatchIndices | None:
        """Like `match_one` but includes the indices of the chars in the
        haystack that matched the needle. Useful for displaying matched indices
        in the UI."""

    def update(
        self,
        *,
        patterns: str | Pattern | Sequence[str | Pattern] | None = None,
        max_typos: int | None = ...,
        limit: int | None = ...,
        casing: Casing | None = None,
        unicode: Unicode | None = None,
        matching: Matching | None = None,
        sort: Sort | None = None,
        scoring: Scoring | None = None,
    ) -> None:
        """Updates any combination of patterns, matcher options, and result limit.

        Omitted values remain unchanged. ``None`` means unlimited for
        ``max_typos`` and ``limit``.
        """

    @property
    def patterns(self) -> list[Pattern]:
        """The current patterns returned as copies (mutate them freely)"""

    @property
    def max_typos(self) -> int | None:
        """The maximum number of characters missing from the needle, before an item in the
        haystack is filtered out. ``None`` means unlimited."""

    @property
    def limit(self) -> int | None:
        """Max matches returned from the ``match`` APIs. ``None`` means unlimited."""

    @property
    def casing(self) -> Casing:
        """How case sensitivity is handled while matching"""

    @property
    def unicode(self) -> Unicode:
        """How unicode is handled while matching"""

    @property
    def matching(self) -> Matching:
        """The matching algorithm: fuzzy (Smith-Waterman) or one of the literal modes
        (exact, prefix, suffix, substring). Literal modes require the needle to appear
        as a contiguous run of characters and do not support typos (``max_typos`` is ignored)."""

    @property
    def sort(self) -> Sort:
        """How results are ordered"""

    @property
    def scoring(self) -> Scoring:
        """The scoring used by the smith waterman algorithm. Pay close attention to the
        documentation for each property, as small changes can lead to poor matching."""
