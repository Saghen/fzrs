//! JS objects to Rust structs, through `js_sys` directly to avoid
//! `serde`/`f64::fmt` size overhead

use js_sys::Reflect;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(typescript_custom_section)]
const TYPESCRIPT_TYPES: &'static str = r#"
export type CaseMatching = 'ignore' | 'smart' | 'respect';
export type UnicodeMatching = 'ignore' | 'smart' | 'always';
export type Matching = 'fuzzy' | 'exact' | 'prefix' | 'suffix' | 'substring';
export type SortStrategy = 'scoreThenIndexAsc' | 'scoreThenIndexDesc' | 'indexAsc' | 'indexDesc';

/**
 * Controls the scoring used by the smith waterman algorithm. Pay close
 * attention to the documentation for each property, as small changes can lead
 * to poor matching.
 */
export interface Scoring {
  /** Score for a matching character between needle and haystack */
  matchScore?: number;
  /** Penalty for a mismatch (substitution) */
  mismatchPenalty?: number;
  /** Penalty for opening a gap (deletion/insertion) */
  gapOpenPenalty?: number;
  /** Penalty for extending a gap (deletion/insertion) */
  gapExtendPenalty?: number;
  /** Bonus for matching the first character of the haystack (e.g. "h" on "hello_world") */
  prefixBonus?: number;
  /**
   * Bonus for matching a capital letter after a lowercase letter
   * (e.g. "b" on "fooBar" will receive a bonus on "B")
   */
  capitalizationBonus?: number;
  /**
   * Bonus for matching the case of the needle (e.g. "WorLd" on "WoRld" will receive
   * a bonus on "W", "o", "d")
   */
  matchingCaseBonus?: number;
  /** Bonus for matching the exact needle (e.g. "foo" on "foo" will receive the bonus) */
  exactMatchBonus?: number;
  /**
   * Bonus for matching *after* a delimiter character (e.g. "hw" on "hello_world"
   * will give a bonus on "w")
   */
  delimiterBonus?: number;
}

export interface Config {
  /**
   * The maximum number of characters missing from the needle, before an item in the
   * haystack is filtered out. `Infinity` = unlimited, `undefined` = the core default
   * (`0`, typo resistance disabled)
   */
  maxTypos?: number;
  /**
   * Maximum number of matches returned from the `matchList` APIs, applied after sorting
   * (so with the default sort, the best `limit` matches are returned). `Infinity`
   * and `undefined` mean unlimited
   */
  limit?: number;
  /** Controls how case sensitivity/insensitivity is handled while matching */
  casing?: CaseMatching;
  /** Controls how unicode is handled while matching */
  unicode?: UnicodeMatching;
  /**
   * Selects the matching algorithm: fuzzy (Smith-Waterman) or one of the literal
   * modes (exact, prefix, suffix, substring). Literal modes require the needle to
   * appear as a contiguous run of characters and do not support typos
   * (`maxTypos` is ignored)
   */
  matching?: Matching;
  /** Controls how results are ordered */
  sort?: SortStrategy;
  /** Controls the scoring used by the smith waterman algorithm */
  scoring?: Scoring;
}
"#;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "Config")]
    pub type ConfigObject;
    #[wasm_bindgen(typescript_type = "Scoring")]
    pub type ScoringObject;
}

/// Reads `obj[key]`, mapping `undefined`/`null` to `None` (both mean "use the
/// default", like serde's treatment of `Option`)
pub(crate) fn get_opt(obj: &JsValue, key: &str) -> Option<JsValue> {
    // Reflect::get only throws when obj is not an object; every caller checks first
    let value = Reflect::get(obj, &JsValue::from_str(key)).unwrap_throw();
    (!value.is_undefined() && !value.is_null()).then_some(value)
}

pub(crate) fn opt_f64(obj: &JsValue, key: &str) -> Result<Option<f64>, JsError> {
    get_opt(obj, key)
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| JsError::new(&format!("invalid {key}: expected a number")))
        })
        .transpose()
}

fn opt_u16(obj: &JsValue, key: &str) -> Result<Option<u16>, JsError> {
    opt_f64(obj, key)?
        .map(|value| {
            if value.fract() == 0.0 && (0.0..=f64::from(u16::MAX)).contains(&value) {
                Ok(value as u16)
            } else {
                Err(JsError::new(&format!(
                    "invalid {key}: expected an integer between 0 and {}",
                    u16::MAX
                )))
            }
        })
        .transpose()
}

pub(crate) fn opt_bool(obj: &JsValue, key: &str) -> Result<Option<bool>, JsError> {
    get_opt(obj, key)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| JsError::new(&format!("invalid {key}: expected a boolean")))
        })
        .transpose()
}

pub(crate) fn opt_object(obj: &JsValue, key: &str) -> Result<Option<JsValue>, JsError> {
    get_opt(obj, key)
        .map(|value| {
            value
                .is_object()
                .then_some(value)
                .ok_or_else(|| JsError::new(&format!("invalid {key}: expected an object")))
        })
        .transpose()
}

/// Generates a mirror of the same-named core enum with its JS string names, the
/// `From` conversions in both directions, and string parsing/formatting for the
/// boundary
macro_rules! mirror_enum {
    ($name:ident { $($variant:ident = $string:literal),+ $(,)? }) => {
        #[doc = concat!("Mirror of [`frizbee::", stringify!($name), "`] crossing the boundary as a string")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name {
            $($variant,)+
        }

        impl $name {
            // only enums that appear in parseQuery output are serialized back to JS
            #[allow(dead_code)]
            pub(crate) fn as_str(self) -> &'static str {
                match self {
                    $($name::$variant => $string,)+
                }
            }

            /// Reads `obj[key]` as this enum, erroring on non-strings and unknown values
            pub(crate) fn opt_from_js(obj: &JsValue, key: &str) -> Result<Option<Self>, JsError> {
                get_opt(obj, key)
                    .map(|value| {
                        match value.as_string().as_deref() {
                            $(Some($string) => Ok($name::$variant),)+
                            _ => Err(JsError::new(&format!(
                                "invalid {key}: expected one of {}",
                                [$(concat!("'", $string, "'")),+].join(", ")
                            ))),
                        }
                    })
                    .transpose()
            }
        }

        impl From<$name> for frizbee::$name {
            fn from(value: $name) -> Self {
                match value {
                    $($name::$variant => frizbee::$name::$variant,)+
                }
            }
        }

        impl From<frizbee::$name> for $name {
            fn from(value: frizbee::$name) -> Self {
                match value {
                    $(frizbee::$name::$variant => $name::$variant,)+
                }
            }
        }
    };
}

mirror_enum!(CaseMatching { Ignore = "ignore", Smart = "smart", Respect = "respect" });
mirror_enum!(UnicodeMatching { Ignore = "ignore", Smart = "smart", Always = "always" });
mirror_enum!(Matching {
    Fuzzy = "fuzzy",
    Exact = "exact",
    Prefix = "prefix",
    Suffix = "suffix",
    Substring = "substring",
});
mirror_enum!(SortStrategy {
    ScoreThenIndexAsc = "scoreThenIndexAsc",
    ScoreThenIndexDesc = "scoreThenIndexDesc",
    IndexAsc = "indexAsc",
    IndexDesc = "indexDesc",
    Unsorted = "unsorted",
});

/// Mirror of [`frizbee::Scoring`] with all-optional camelCase fields; missing
/// fields fall back to the core defaults
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Scoring {
    pub match_score: Option<u16>,
    pub mismatch_penalty: Option<u16>,
    pub gap_open_penalty: Option<u16>,
    pub gap_extend_penalty: Option<u16>,
    pub prefix_bonus: Option<u16>,
    pub capitalization_bonus: Option<u16>,
    pub matching_case_bonus: Option<u16>,
    pub exact_match_bonus: Option<u16>,
    pub delimiter_bonus: Option<u16>,
}

impl Scoring {
    pub(crate) fn from_js(obj: &JsValue) -> Result<Self, JsError> {
        Ok(Self {
            match_score: opt_u16(obj, "matchScore")?,
            mismatch_penalty: opt_u16(obj, "mismatchPenalty")?,
            gap_open_penalty: opt_u16(obj, "gapOpenPenalty")?,
            gap_extend_penalty: opt_u16(obj, "gapExtendPenalty")?,
            prefix_bonus: opt_u16(obj, "prefixBonus")?,
            capitalization_bonus: opt_u16(obj, "capitalizationBonus")?,
            matching_case_bonus: opt_u16(obj, "matchingCaseBonus")?,
            exact_match_bonus: opt_u16(obj, "exactMatchBonus")?,
            delimiter_bonus: opt_u16(obj, "delimiterBonus")?,
        })
    }

    pub(crate) fn to_core(&self) -> frizbee::Scoring {
        let default = frizbee::Scoring::default();
        frizbee::Scoring {
            match_score: self.match_score.unwrap_or(default.match_score),
            mismatch_penalty: self.mismatch_penalty.unwrap_or(default.mismatch_penalty),
            gap_open_penalty: self.gap_open_penalty.unwrap_or(default.gap_open_penalty),
            gap_extend_penalty: self
                .gap_extend_penalty
                .unwrap_or(default.gap_extend_penalty),
            prefix_bonus: self.prefix_bonus.unwrap_or(default.prefix_bonus),
            capitalization_bonus: self
                .capitalization_bonus
                .unwrap_or(default.capitalization_bonus),
            matching_case_bonus: self
                .matching_case_bonus
                .unwrap_or(default.matching_case_bonus),
            exact_match_bonus: self.exact_match_bonus.unwrap_or(default.exact_match_bonus),
            delimiter_bonus: self.delimiter_bonus.unwrap_or(default.delimiter_bonus),
        }
    }

    pub(crate) fn from_core(scoring: &frizbee::Scoring) -> Self {
        Self {
            match_score: Some(scoring.match_score),
            mismatch_penalty: Some(scoring.mismatch_penalty),
            gap_open_penalty: Some(scoring.gap_open_penalty),
            gap_extend_penalty: Some(scoring.gap_extend_penalty),
            prefix_bonus: Some(scoring.prefix_bonus),
            capitalization_bonus: Some(scoring.capitalization_bonus),
            matching_case_bonus: Some(scoring.matching_case_bonus),
            exact_match_bonus: Some(scoring.exact_match_bonus),
            delimiter_bonus: Some(scoring.delimiter_bonus),
        }
    }
}

/// `Infinity` means unlimited typos, mirroring core's `max_typos: None`.
pub(crate) fn parse_max_typos(value: f64) -> Result<Option<u16>, JsError> {
    if value.is_infinite() && value.is_sign_positive() {
        return Ok(None);
    }
    if value.fract() == 0.0 && (0.0..=f64::from(u16::MAX)).contains(&value) {
        return Ok(Some(value as u16));
    }
    // Don't include the value because it pulls in ~15 KB of float-formatting
    Err(JsError::new(&format!(
        "invalid maxTypos: expected an integer between 0 and {} or Infinity for unlimited",
        u16::MAX
    )))
}

/// `Infinity` means unlimited
fn parse_limit(value: f64) -> Result<Option<u32>, JsError> {
    if value.is_infinite() && value.is_sign_positive() {
        return Ok(None);
    }
    if value.fract() == 0.0 && (0.0..=u32::MAX as f64).contains(&value) {
        return Ok(Some(value as u32));
    }
    Err(JsError::new(
        "invalid limit: expected a non-negative integer or Infinity for unlimited",
    ))
}

/// Returns the core config plus the binding-level `limit` (`None` =
/// unlimited)
pub(crate) fn config_from_js(
    config: Option<ConfigObject>,
) -> Result<(frizbee::Config, Option<u32>), JsError> {
    let default = frizbee::Config::default();
    let Some(config) = config else {
        return Ok((default, None));
    };
    let config: JsValue = config.into();
    if !config.is_object() {
        return Err(JsError::new("invalid config: expected an object"));
    }

    let limit = opt_f64(&config, "limit")?
        .map(parse_limit)
        .transpose()?
        .flatten();
    let config = frizbee::Config {
        max_typos: match opt_f64(&config, "maxTypos")? {
            Some(value) => parse_max_typos(value)?,
            None => default.max_typos,
        },
        casing: CaseMatching::opt_from_js(&config, "casing")?
            .map(Into::into)
            .unwrap_or(default.casing),
        unicode: UnicodeMatching::opt_from_js(&config, "unicode")?
            .map(Into::into)
            .unwrap_or(default.unicode),
        matching: Matching::opt_from_js(&config, "matching")?
            .map(Into::into)
            .unwrap_or(default.matching),
        sort: SortStrategy::opt_from_js(&config, "sort")?
            .map(Into::into)
            .unwrap_or(default.sort),
        scoring: opt_object(&config, "scoring")?
            .map(|scoring| Scoring::from_js(&scoring))
            .transpose()?
            .map(|scoring| scoring.to_core())
            .unwrap_or(default.scoring),
    };
    Ok((config, limit))
}

/// Needle length up to which scores are guaranteed to fit within the 16-bit
/// score (the core default `Scoring` when omitted). Longer needles still match,
/// but their scores may saturate at `0xffff`
#[wasm_bindgen(js_name = maxNeedleLen)]
pub fn max_needle_len(scoring: Option<ScoringObject>) -> Result<f64, JsError> {
    let scoring = match scoring {
        Some(scoring) => {
            let scoring: JsValue = scoring.into();
            if !scoring.is_object() {
                return Err(JsError::new("invalid scoring: expected an object"));
            }
            Scoring::from_js(&scoring)?.to_core()
        }
        None => frizbee::Scoring::default(),
    };
    Ok(scoring.max_needle_len() as f64)
}
