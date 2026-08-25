"""Time-series step 2 capsule: clean readings -> plot-ready analysis CSVs.

Code Ocean concepts demonstrated here:

* Input data assets are mounted READ-ONLY under ``/data/<mount>/...``, and the
  mount name is chosen by whoever attaches the asset. So this capsule never
  hardcodes a path: it scans ``/data`` recursively for CSVs (extension matched
  ignoring case, because pathlib's glob is case-sensitive on POSIX and Code
  Ocean's filesystem is too), prefers ``clean_readings.csv`` (what step 1
  writes), then ``readings.csv``, then the largest remaining candidate — which
  means it also runs straight on the raw readings asset with step 1 skipped.
  That is STEP 1'S ordering, deliberately: the two capsules picking different
  files out of one mount is a defect all of its own, and every candidate not
  chosen is named in the manifest by both of them. See ``pick_readings``.
* Everything written under ``/results/`` is captured when the computation
  finishes and can be turned into a data asset, so the four files below are
  the whole contract with the next stage.
* Capsules get no network at runtime and must not fail the pipeline on a typo:
  every run parameter is optional, a bad value logs a warning and falls back to
  its default, and even an empty input produces valid (header-only) CSVs and
  exits 0. "Bad value" covers more than it looks: ``float()`` accepts ``nan``,
  ``inf``, ``-inf``, ``Infinity`` and overflows ``1e400`` to inf, and a finite
  but enormous ``1e40`` only detonates later inside pandas. See
  ``parse_positive_int`` for how each of those is caught before it can end the
  run with an empty ``/results``.
* The manifest must never CONTRADICT the caller. This capsule exists to
  demonstrate provenance, so a manifest that reads "you sent anomaly_z=3" when
  the operator sent 1.5 is the worst defect available here — silent, and baked
  into the artifact everyone trusts afterwards. So ``manifest.json`` records
  the raw argument list verbatim under ``parameters_source``, and every value
  the run altered (rejected, clamped, or truncated from a fraction) leaves a
  ``parameter_warnings`` entry behind. The invariant that follows from it, and
  that ``parse_parameters`` enforces: ``parameters_supplied`` must never name a
  parameter whose recorded value is its default-because-we-could-not-use-it.
* The input's headers are trimmed and lower-cased so ``Timestamp`` is accepted
  as ``timestamp`` — which means two distinct columns can collapse onto one
  name (``Reading,reading``). That used to end the run with exit 1 and an empty
  ``/results``; now the FIRST column of each name wins, the duplicates are
  dropped at read time, and the choice is logged, warned about and listed in
  the manifest. See ``drop_duplicate_columns``.
* Timestamps are the other column that could end the run with nothing written,
  and for two unrelated reasons: a column of MIXED UTC offsets has no single
  dtype (``pd.to_datetime`` hands back an OBJECT index, which cannot be
  resampled), and a single implausibly old instant makes any span overflow
  int64 nanoseconds. Both are DATA, so neither is fatal now: the column is
  parsed to one tz-naive ``datetime64`` dtype and every span is computed in a
  unit that cannot overflow. See ``parse_timestamp_series`` and
  ``timestamps_as_days``, and ``manifest.json``'s
  ``timestamps_normalized_to_utc`` / ``rows_implausible_timestamp``.
* Progress lines printed to stdout show up in the Code Ocean run log.

This capsule emits **DATA, NOT PICTURES**. No plotting library is imported and
no image or HTML is written — the orchestrator app downloads these CSVs and
draws interactive charts from them. That keeps the capsule tiny, keeps the
visualization genuinely interactive, and keeps the compute environment to
``pandas`` + ``numpy``.

Outputs (written to ``/results``):
  resampled.csv           timestamp, instrument_id, mean, min, max, n, rolling_mean
  instrument_summary.csv  instrument_id, n, mean, sd, min, max, slope_per_day,
                          r2, first_timestamp, last_timestamp, n_anomalies,
                          mean_vs_baseline
  anomalies.csv           timestamp, instrument_id, reading, expected, residual, robust_z
  manifest.json           what this step consumed, produced and was asked to do

Local-test override: ``DATA_DIR`` replaces ``/data`` and ``RESULTS_DIR``
replaces ``/results``, e.g.

    DATA_DIR=/tmp/in RESULTS_DIR=/tmp/out bash code/run --resample_interval=1D
"""

import argparse
import json
import os
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Locations (Code Ocean conventions, overridable for local testing)
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))

# Exact output schemas from the build contract. Written even when there is
# nothing to put in them, so a downstream reader never has to special-case an
# empty run — it gets a real CSV with real headers and zero rows.
RESAMPLED_COLUMNS = [
    "timestamp", "instrument_id", "mean", "min", "max", "n", "rolling_mean",
]
SUMMARY_COLUMNS = [
    "instrument_id", "n", "mean", "sd", "min", "max",
    "slope_per_day", "r2", "first_timestamp", "last_timestamp", "n_anomalies",
    # mean_vs_baseline is what makes --baseline_instrument a REAL parameter:
    # change the baseline and this column changes. A parameter that only ever
    # showed up in the manifest would make the "add a parameter and watch the
    # result change" story a lie.
    "mean_vs_baseline",
]
ANOMALY_COLUMNS = [
    "timestamp", "instrument_id", "reading", "expected", "residual", "robust_z",
]

# The three columns this capsule actually needs from the input CSV.
REQUIRED_INPUT_COLUMNS = ["timestamp", "instrument_id", "reading"]

# Filenames that identify the readings table by NAME, best first, and the
# reason this list exists at all: step 1 has the same list (in its own order,
# raw readings first) and the two capsules must not disagree about which file
# they are looking at. They used to. Step 2 took the FIRST CSV in sorted-path
# order that had the three columns above, so an ``archive_2019_readings.csv``
# sitting beside ``readings.csv`` won on the alphabet alone: step 1 QC'd 48
# rows of live data while step 2 analysed one stale archived row, both exited
# 0, and neither said a word about the file it had passed over. Preferring a
# canonical name — this step's own input first, then the raw readings it can
# also run on directly — and falling back to largest-wins is exactly step 1's
# rule, so the same mount now resolves to the same file in both capsules.
CANONICAL_READINGS_NAMES = ["clean_readings.csv", "readings.csv"]

# Columns that identify a CSV as some previous QC run's OUTPUT (step 1's
# qc_flags.csv) rather than readings to analyse. Such a file holds every input
# row INCLUDING the ones QC dropped, so analysing it would quietly re-admit
# rejected readings; it is therefore used only when nothing else has the
# required columns, and the choice is recorded. Step 1 skips the same files for
# the same reason.
QC_OUTPUT_COLUMNS = ["qc_status", "qc_reason"]

# The label a row with a MISSING instrument_id is filed under. It is the same
# literal step 1 uses for the same rows, deliberately: run either capsule on
# the same raw file and both report the same instrument list. Relabelling
# rather than dropping keeps the row (losing data is worse than labelling it),
# and the count reaches the run log and the manifest, because a substituted id
# is still an altered value.
UNKNOWN_INSTRUMENT = "(unknown)"

# Values that mean "this row has no instrument_id". "nan" has to be in here:
# `astype(str)` turns a genuinely missing cell into that literal string, which
# is why a bare `!= ""` guard never fired and an instrument called "nan"
# appeared in the manifest.
MISSING_INSTRUMENT_TOKENS = ["", "nan"]

# argparse's end-of-options marker. Everything after one is forced to be a
# POSITIONAL argument, and this capsule defines no positionals — so argparse
# reports a successful parse while quietly handing back "--anomaly_z=1.5" as
# unrecognized. See resolve_argv for why that is the worst possible outcome
# here and what happens instead.
END_OF_OPTIONS = "--"

# Robust z-score scale factor: for normally distributed data the MAD is
# ~0.6745 sigma, so 0.6745 * (x - median) / MAD is on the same scale as a
# classic z-score but is not dragged around by the outliers we are hunting.
MAD_TO_SIGMA = 0.6745

# Floats are rounded before writing so the CSVs stay readable in the app's
# tables; 6 decimals is far more precision than any instrument reading needs.
ROUND_TO = 6

# The magnitude at which pandas' "multiply by 10**ROUND_TO, round, divide"
# implementation of Series.round would overflow a float64 to inf. Values at or
# above this are written unrounded — at 1e302 a double has no fractional digits
# left, so there is nothing to round away. See round_without_overflow.
ROUND_LIMIT = float(np.finfo(np.float64).max) / (10.0 ** ROUND_TO)


def log(msg):
    # type: (str) -> None
    """Print a progress line (shows up in the Code Ocean run log)."""
    print("[ts-step2] {}".format(msg), flush=True)


# ---------------------------------------------------------------------------
# Run parameters (Code Ocean App Panel)
# ---------------------------------------------------------------------------
# One entry per parameter in .codeocean/app-panel.json. Keep the two in sync:
# the panel defines the FORM, this table defines how the capsule reads it.
#   (param_name / argument key, label shown on the panel, default value)
PARAM_SPECS = [
    ("resample_interval", "Resample interval", "6H"),
    ("rolling_window", "Rolling window (buckets)", "4"),
    ("anomaly_z", "Anomaly threshold (robust z)", "3"),
    ("top_n_anomalies", "Top N anomalies", "20"),
    ("baseline_instrument", "Baseline instrument", ""),
]
PARAM_LABELS = dict((name, label) for name, label, _ in PARAM_SPECS)
PARAM_DEFAULTS = dict((name, default) for name, _, default in PARAM_SPECS)

# The panel offers these four intervals and nothing else (type "list"), but the
# capsule still validates: parameters can also arrive from the API, where
# nobody is holding the dropdown. Each entry maps the token the panel shows to
# the pandas frequency aliases that mean it, most-modern first — pandas 2.2
# deprecated the upper-case "H"/"T" aliases in favour of "h"/"min", so we probe
# at runtime and use whatever the INSTALLED pandas accepts without warning.
INTERVAL_ALIASES = [
    ("1H", ("1h", "1H")),
    ("6H", ("6h", "6H")),
    ("12H", ("12h", "12H")),
    ("1D", ("1D", "1d")),
]
VALID_INTERVALS = [token for token, _ in INTERVAL_ALIASES]
ALIAS_CANDIDATES = dict(INTERVAL_ALIASES)

# Fallback defaults used when a supplied value is unusable.
DEFAULT_INTERVAL = "6H"
DEFAULT_ROLLING_WINDOW = 4
DEFAULT_ANOMALY_Z = 3.0
DEFAULT_TOP_N = 20

# Upper clamps for the numeric parameters. Rejecting nan/inf is not enough on
# its own: 1e40 is perfectly finite, sails through every "is it a number?"
# check, and then detonates deep inside pandas ("Python int too large to
# convert to C long") — which fails the run after writing nothing at all.
# Every value here is far past anything a real analysis would ask for.
MAX_ROLLING_WINDOW = 100000  # buckets; more than any instrument will produce
MAX_TOP_N = 1000000          # rows in anomalies.csv
MAX_ANOMALY_Z = 1000.0       # nothing real scores a robust z anywhere near this


def build_parser():
    # type: () -> argparse.ArgumentParser
    """The argparse parser for the App Panel's arguments."""
    parser = argparse.ArgumentParser(
        add_help=False,      # a stray -h must not short-circuit the analysis
        allow_abbrev=False,  # only exact --param_name keys, no fuzzy matching
    )
    for param_name, label, default in PARAM_SPECS:
        parser.add_argument("--" + param_name, default=default, help=label)
    return parser


def format_argv(argv):
    # type: (List[str]) -> str
    """Render an argument list for a log line, unambiguously.

    Tokens that are empty or contain spaces are quoted, so a reader can tell
    one token containing a space from two separate tokens.
    """
    if not argv:
        return "(empty)"
    return " ".join(
        '"{}"'.format(token) if (token == "" or " " in token) else token
        for token in argv)


def split_recoverable_tokens(argv):
    # type: (List[str]) -> Tuple[List[str], List[str]]
    """Split a malformed argument list into re-parseable tokens and the rest.

    Only reached after the whole list has already failed to parse. Because the
    App Panel sends every value as ONE token shaped ``--param_name=value``, a
    token of exactly that shape cannot be the thing argparse choked on — so
    re-parsing just those recovers every well-formed value the operator sent.
    Everything else (a flag with no value, a bare word, a ``--name value`` pair
    whose two halves can no longer be told apart from a stray positional) is
    handed back as un-parseable and reported by name.
    """
    keep = []  # type: List[str]
    dropped = []  # type: List[str]
    for token in argv:
        key = token[2:].split("=", 1)[0] if token.startswith("--") else None
        if "=" in token and key in PARAM_DEFAULTS:
            keep.append(token)
        else:
            dropped.append(token)
    return keep, dropped


def malformed_argv_warning(argv, honoured, ignored):
    # type: (List[str], List[str], List[str]) -> str
    """The parameter_warnings entry for an argument list that was not usable.

    It quotes the RAW argument list, because the whole point is that the
    manifest must not be readable as "these are the values you sent".
    """
    recovered = sorted(supplied_param_names(honoured))
    if recovered:
        return (
            "the argument list could not be used exactly as given ({}) — the "
            "well-formed --name=value token(s) were still honoured ({}), but "
            "these token(s) were discarded and had NO effect on this run: "
            "{}".format(format_argv(argv), ", ".join(recovered),
                        format_argv(ignored)))
    return (
        "the argument list could not be used as given ({}) and no parameter "
        "value could be recovered from it — EVERY supplied value was ignored "
        "and every parameter fell back to its App Panel default; the discarded "
        "token(s) were: {}".format(format_argv(argv), format_argv(ignored)))


def remove_tokens(tokens, unwanted):
    # type: (List[str], List[str]) -> List[str]
    """``tokens`` minus ONE occurrence of each entry in ``unwanted``."""
    remaining = list(unwanted)
    kept = []  # type: List[str]
    for token in tokens:
        if token in remaining:
            remaining.remove(token)
        else:
            kept.append(token)
    return kept


def parse_dropping_unconsumed(parser, tokens):
    # type: (argparse.ArgumentParser, List[str]) -> Tuple[Optional[argparse.Namespace], List[str], List[str]]
    """Parse ``tokens``, dropping whatever argparse refuses to consume.

    ``parse_known_args`` hands back the tokens it could not place, and those
    tokens had NO effect on the namespace it returned — so they are removed and
    the remainder is re-parsed until argparse consumes everything it is given.
    Each pass removes at least one token, so the loop always terminates.

    Returns (namespace, tokens used, tokens dropped), or ``(None, [], tokens)``
    when argparse rejected the list outright (a flag with no value, which
    raises ``SystemExit``).
    """
    used = list(tokens)
    dropped = []  # type: List[str]
    while True:
        try:
            args, unconsumed = parser.parse_known_args(used)
        except SystemExit:
            return None, [], list(tokens)
        if not unconsumed:
            return args, used, dropped
        dropped.extend(unconsumed)
        used = remove_tokens(used, unconsumed)


def resolve_argv(parser, argv):
    # type: (argparse.ArgumentParser, List[str]) -> Tuple[argparse.Namespace, List[str], List[str], bool]
    """Work out which argv tokens this run can actually HONOUR.

    Returns (namespace, honoured tokens, ignored tokens, used_as_given).

    ``used_as_given`` is true only when argparse consumed the whole list
    exactly as it arrived; that is what the manifest reports as
    ``argv_parsed``. Anything less is false — a rejected list, an unknown
    parameter, or a value stranded behind an end-of-options marker.

    That last one is the subtle case and the reason this function exists.
    ``["--", "--anomaly_z=1.5"]`` does NOT raise: argparse accepts it, demotes
    everything after ``--`` to a positional, finds no positionals to fill and
    returns the token as "unrecognized". Treating that as a clean parse
    produced precisely the manifest this capsule must never write — ``anomaly_z
    "3"`` (the default), ``parameter_warnings []``, ``argv_parsed true``, and
    ``parameters_supplied ["anomaly_z"]`` claiming the operator's 1.5 was used.
    So the markers are stripped up front and the values behind them are parsed
    like any other, while the markers themselves are reported as ignored.
    """
    ignored = [t for t in argv if t == END_OF_OPTIONS]
    args, honoured, dropped = parse_dropping_unconsumed(
        parser, [t for t in argv if t != END_OF_OPTIONS])
    if args is not None:
        ignored = ignored + dropped
        return args, honoured, ignored, not ignored

    # argparse rejected the list outright. Keep only the tokens that cannot be
    # what it choked on: one self-contained --known=value token each.
    keep, junk = split_recoverable_tokens(argv)
    args, honoured, dropped = parse_dropping_unconsumed(parser, keep)
    if args is None:
        # Unreachable in principle (every kept token is --known=value), but a
        # capsule whose job is provenance does not gamble on "should".
        return argparse.Namespace(**PARAM_DEFAULTS), [], list(argv), False
    return args, honoured, list(junk) + list(dropped), False


def parse_parameters(argv, warns):
    # type: (List[str], List[str]) -> Tuple[Dict[str, str], set, Dict[str, Any]]
    """Parse the App Panel values Code Ocean appended to ``code/run``.

    Named parameters arrive as single argv tokens (``--resample_interval=1D``),
    which argparse understands out of the box — a hand-rolled ``--name value``
    parser would silently see nothing.

    Anything argparse cannot USE must not fail the run, and it must not quietly
    become "the operator sent the defaults" either. There are two ways a token
    ends up unused and they used to be handled very differently:

    * a MALFORMED list (``--anomaly_z=1.5 --top_n_anomalies`` — a flag with no
      value) makes argparse raise ``SystemExit``;
    * an ACCEPTED list can still leave tokens unconsumed — an unknown
      parameter, a stray word, or anything after an end-of-options ``--``.

    The second kind was the dangerous one, because argparse reported success:
    the value was dropped, the manifest recorded the DEFAULT in its place with
    an empty ``parameter_warnings``, and ``parameters_supplied`` still named the
    parameter — a manifest that simultaneously claims the value was supplied and
    records the default. Both kinds now go down the same road (``resolve_argv``):

    1. the tokens argparse can actually use are re-parsed and honoured, so the
       operator's real intent survives one bad token elsewhere in the list;
    2. one ``parameter_warnings`` entry quotes the raw argument list and names
       what was discarded;
    3. ``ignored_tokens`` lists the discarded tokens and ``argv_parsed`` goes
       false, so the manifest says the list was not used exactly as sent;
    4. ``parameters_supplied`` is derived from the HONOURED tokens only.

    That last point is the invariant: ``parameters_supplied`` must never name a
    parameter whose recorded value is its default-because-we-could-not-use-it.

    Returns (raw string values, names actually honoured, source dict).
    """
    parser = build_parser()
    args, honoured, ignored, used_as_given = resolve_argv(parser, argv)
    superseded = find_superseded_tokens(honoured)
    source = {
        "argv": list(argv),
        "argv_parsed": bool(used_as_given),
        "parameters_supplied": [],
        "ignored_tokens": list(ignored),
        # Tokens argparse DID understand but a later token of the same name
        # overrode. They are not `ignored_tokens` (the list parsed fine) and
        # they were not honoured either, so without their own key the manifest
        # positively claimed that nothing had been dropped.
        "superseded_tokens": list(superseded),
    }  # type: Dict[str, Any]
    if not used_as_given:
        warns.append(malformed_argv_warning(argv, honoured, ignored))
        log("warning: {}".format(warns[-1]))
    if superseded:
        warns.append(superseded_tokens_warning(superseded, args))
        log("warning: {}".format(warns[-1]))

    values = {}
    for param_name, _label, _default in PARAM_SPECS:
        value = getattr(args, param_name, None)
        values[param_name] = "" if value is None else str(value)
    # Derived from the tokens that were actually USED, not from the raw list:
    # a discarded token must never be recorded as a parameter that was supplied.
    supplied = supplied_param_names(honoured)
    source["parameters_supplied"] = sorted(supplied)
    return values, supplied, source


def supplied_param_names(argv):
    # type: (List[str]) -> set
    """Which parameters were actually passed, vs. left at their default."""
    known = set(PARAM_DEFAULTS)
    supplied = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0]
        if key in known:
            supplied.add(key)
    return supplied


def find_superseded_tokens(honoured):
    # type: (List[str]) -> List[str]
    """Tokens for a parameter that a LATER token of the same name overrode.

    ``--anomaly_z=1 --anomaly_z=9`` parses perfectly: argparse takes the last
    value and never mentions the first. That left the manifest saying
    ``anomaly_z "9"``, ``argv_parsed true`` and — the false part —
    ``ignored_tokens []``, which positively claims nothing was dropped when a
    value the operator sent had in fact been discarded. Every superseded token
    is returned here so the caller can name it in ``parameter_warnings`` and
    record it in ``parameters_source``.
    """
    last_index = {}  # type: Dict[str, int]
    for index, token in enumerate(honoured):
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0]
        if key in PARAM_DEFAULTS:
            last_index[key] = index
    superseded = []  # type: List[str]
    for index, token in enumerate(honoured):
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0]
        if key in PARAM_DEFAULTS and last_index.get(key) != index:
            superseded.append(token)
    return superseded


def superseded_tokens_warning(superseded, args):
    # type: (List[str], argparse.Namespace) -> str
    """The parameter_warnings entry for values a later token overrode."""
    names = sorted(set(t[2:].split("=", 1)[0] for t in superseded))
    winners = ", ".join(
        "--{}={}".format(name, getattr(args, name, "")) for name in names)
    return (
        "{} parameter(s) were given more than once ({}); the LAST value of "
        "each won ({}) and the earlier token(s) had NO effect on this run: "
        "{}".format(len(names), ", ".join(names), winners,
                    format_argv(superseded)))


def parse_interval(raw, warns, notes):
    # type: (str, List[str], List[str]) -> str
    """Validate the resample interval against the panel's four options.

    Matching is deliberately forgiving — ``1d`` and ``" 6h "`` are obviously
    meant to be ``1D`` and ``6H``, and the value can arrive from the API where
    nobody is holding the dropdown. But a forgiving match still CHANGES the
    value: ``parameters`` records ``"1d"`` while ``effective_parameters``
    records ``"1D"``, and a reader owed an explanation for the difference used
    to get none at all. So a value that only matched after case-folding or
    trimming leaves a note, the same way ``baseline_instrument`` already notes
    a case-insensitive instrument match.
    """
    text = (raw or "").strip().upper()
    if not text:
        return DEFAULT_INTERVAL
    if text in VALID_INTERVALS:
        if text != (raw or ""):
            notes.append(
                "{} \"{}\" was read as {} (interval names are matched "
                "case-insensitively, ignoring surrounding whitespace)".format(
                    PARAM_LABELS["resample_interval"], raw, text))
        return text
    warns.append(
        "{} \"{}\" is not one of {} — using {}".format(
            PARAM_LABELS["resample_interval"], raw,
            "/".join(VALID_INTERVALS), DEFAULT_INTERVAL)
    )
    return DEFAULT_INTERVAL


def parse_positive_int(raw, param_name, default, warns, maximum=None):
    # type: (str, str, int, List[str], Optional[int]) -> int
    """Parse a whole-number parameter; anything unusable falls back or clamps.

    Two traps live in the one-line version of this function, and both end the
    run with an exit code of 1 and an empty ``/results``:

    * ``int(float(text))`` inside ``except ValueError`` is not enough cover.
      ``float("inf")``, ``float("-inf")``, ``float("Infinity")`` and
      ``float("1e400")`` (which overflows to inf) all succeed, and the
      ``int()`` that follows raises ``OverflowError``, which that clause does
      not catch. So non-finite input is rejected explicitly, before conversion.
    * ``numpy.isfinite`` does not save us from ``1e40``. That value is finite,
      becomes a legal Python int, and only blows up later inside pandas
      ``rolling()`` as "Python int too large to convert to C long". Hence the
      ``maximum`` clamp — a separate defence for a separate failure.

    Either way the run keeps going, with a warning naming the value.

    A FRACTION (``--top_n_anomalies=1.9``) is the quiet one: ``int()`` truncates
    it towards zero and the run carries on with 1, which used to happen with no
    message at all. Every other coercion in this capsule warns, so an operator
    reasonably reads an empty ``parameter_warnings`` as "nothing was changed" —
    exactly the wrong conclusion. It now warns too, naming both the value sent
    and the value used. The warning is emitted only when the truncated value is
    what the run actually USES: ``0.5`` truncates to 0, which then fails the
    "must be positive" check and falls back to the default, and that message
    already tells the whole story.
    """
    text = (raw or "").strip()
    if not text:
        return default
    try:
        number = float(text)
    except ValueError:
        warns.append(
            "{} \"{}\" is not a whole number — using {}".format(
                PARAM_LABELS[param_name], raw, default)
        )
        return default
    if not np.isfinite(number):
        warns.append(
            "{} \"{}\" is not a finite number — using {}".format(
                PARAM_LABELS[param_name], raw, default)
        )
        return default
    value = int(number)          # truncates towards zero: 1.9 -> 1
    truncated = value != number
    if value < 1:
        warns.append(
            "{} \"{}\" is not a positive whole number — using {}".format(
                PARAM_LABELS[param_name], raw, default)
        )
        return default
    if maximum is not None and value > maximum:
        warns.append(
            "{} \"{}\" is above the maximum {} — clamped to {}".format(
                PARAM_LABELS[param_name], raw, maximum, maximum)
        )
        return maximum
    if truncated:
        warns.append(
            "{} \"{}\" is not a whole number — the fractional part was "
            "discarded and {} was used".format(
                PARAM_LABELS[param_name], raw, value)
        )
    return value


def parse_positive_float(raw, param_name, default, warns, maximum=None):
    # type: (str, str, float, List[str], Optional[float]) -> float
    """Parse a positive numeric parameter; anything unusable falls back.

    Non-finite input gets its own message rather than being folded into the
    "greater than zero" case: ``inf`` IS greater than zero, and telling the
    operator otherwise would send them hunting for the wrong mistake. An
    absurdly large but finite threshold is clamped, because a threshold no
    residual can ever reach switches anomaly detection off — and a rule that
    is off has to say so.
    """
    text = (raw or "").strip()
    if not text:
        return default
    try:
        value = float(text)
    except ValueError:
        warns.append(
            "{} \"{}\" is not a number — using {}".format(
                PARAM_LABELS[param_name], raw, default)
        )
        return default
    if not np.isfinite(value):
        warns.append(
            "{} \"{}\" is not a finite number — using {}".format(
                PARAM_LABELS[param_name], raw, default)
        )
        return default
    if value <= 0:
        warns.append(
            "{} \"{}\" must be greater than zero — using {}".format(
                PARAM_LABELS[param_name], raw, default)
        )
        return default
    if maximum is not None and value > maximum:
        warns.append(
            "{} \"{}\" is above the maximum {} — clamped to {}".format(
                PARAM_LABELS[param_name], raw, maximum, maximum)
        )
        return float(maximum)
    return value


def resolve_freq_alias(interval):
    # type: (str) -> str
    """Pick a pandas frequency alias for ``interval`` that this pandas likes.

    pandas 2.2 deprecated the upper-case "H" alias (FutureWarning now, removal
    later) while older pandas predates the lower-case spelling. Rather than
    pin a version, try the modern alias first and only fall back if this
    interpreter's pandas rejects it or warns about it — ``simplefilter("error")``
    turns the deprecation into an exception we can catch.
    """
    candidates = ALIAS_CANDIDATES.get(interval, (interval,))
    for alias in candidates:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                pd.tseries.frequencies.to_offset(alias)
                return alias
            except Exception:  # noqa: BLE001 - deprecated or unknown alias
                continue
    return candidates[-1]


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------
def find_csvs(data_dir):
    # type: (Path) -> List[Path]
    """Recursively find CSV files under the input tree, extension case-INSENSITIVELY.

    Recursive because the mount name is not load-bearing: whoever attaches the
    asset picks it, and the capsule must not care. Dotfiles/dot-directories
    and Office lock files are skipped.

    The extension is matched case-insensitively because ``rglob("*.csv")`` is
    not: pathlib's pattern matching is case-SENSITIVE on POSIX whatever the
    filesystem underneath does, and Code Ocean runs these capsules on a
    case-sensitive one. A file exported as ``INSTRUMENTS.CSV`` or
    ``Readings.Csv`` was therefore invisible — absent from ``input_files``,
    never considered as the readings file, and with nothing in the log or the
    manifest to say a CSV had been passed over. Worse, it could go either way
    depending on the machine: the same asset would be seen on a developer's
    case-insensitive laptop and skipped in the cloud. Matching on
    ``suffix.lower()`` makes the answer the same everywhere.
    """
    if not data_dir.is_dir():
        return []
    found = []
    for p in sorted(data_dir.rglob("*")):
        if p.suffix.lower() != ".csv":
            continue
        if any(part.startswith(".") for part in p.relative_to(data_dir).parts):
            continue
        if p.name.startswith("~$"):
            continue
        if p.is_file():
            found.append(p)
    return found


def scrub_non_finite(value):
    # type: (Any) -> Any
    """Replace nan/inf floats anywhere inside a decoded JSON value with None.

    Python's ``json`` is asymmetric: it happily READS the bare ``NaN`` and
    ``Infinity`` literals that stricter parsers reject, but this capsule writes
    its manifest with ``allow_nan=False``. So an older upstream manifest that
    contains one would be copied into ``upstream_manifests`` and then explode
    at write time — a crash caused entirely by someone else's file. Dropping
    the offending values on the way in keeps this step's own manifest valid
    without making a legacy input fatal.
    """
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return dict((k, scrub_non_finite(v)) for k, v in value.items())
    if isinstance(value, list):
        return [scrub_non_finite(v) for v in value]
    return value


def find_manifests(data_dir, notes):
    # type: (Path, List[str]) -> List[Tuple[str, Dict[str, Any]]]
    """Find and parse every manifest.json under the input tree.

    Upstream capsules (step 1's QC filter) describe what they did in a
    manifest; carrying it forward makes this step's manifest a full chain.
    Unparseable files are ignored — provenance is nice-to-have, never required.

    All THREE of the things that can go wrong here EDIT the provenance chain,
    so all three are recorded in ``notes`` rather than only printed to the run
    log. ``upstream_manifests`` is the copy people read afterwards; a value
    that was silently substituted, or a link that silently went missing, is
    invisible in exactly the artifact that is supposed to be trustworthy.

    The third one was the quiet one: a ``manifest.json`` that parses as valid
    JSON but is not an OBJECT (a list, a bare string, ``null``) fell through
    the ``isinstance(data, dict)`` test with no ``except`` to catch it and no
    ``else`` to report it, so the link vanished without even the "could not be
    parsed" note its malformed sibling gets.
    """
    manifests = []  # type: List[Tuple[str, Dict[str, Any]]]
    if not data_dir.is_dir():
        return manifests
    for p in sorted(data_dir.rglob("manifest.json")):
        if any(part.startswith(".") for part in p.relative_to(data_dir).parts):
            continue
        rel = str(p.relative_to(data_dir))
        try:
            with open(str(p), "r") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                cleaned = scrub_non_finite(data)
                if cleaned != data:
                    notes.append(
                        "upstream manifest {} contains bare NaN/Infinity "
                        "literals, which are not valid JSON — those values are "
                        "carried forward in upstream_manifests as null, so the "
                        "copy recorded here differs from the file on "
                        "disk".format(rel))
                    log("note: {}".format(notes[-1]))
                manifests.append((rel, cleaned))
            else:
                notes.append(
                    "upstream manifest {} is valid JSON but not a JSON object "
                    "(it is a {}) — a manifest has to be an object, so it is "
                    "NOT included in upstream_manifests and the provenance "
                    "chain recorded here is incomplete".format(
                        rel, type(data).__name__))
                log("note: {}".format(notes[-1]))
        except Exception as exc:  # noqa: BLE001 - provenance is optional
            notes.append(
                "upstream manifest {} could not be parsed ({}) — it is NOT "
                "included in upstream_manifests, so the provenance chain "
                "recorded here is incomplete".format(rel, exc))
            log("note: {}".format(notes[-1]))
    return manifests


def normalize_header(name):
    # type: (Any) -> str
    """The canonical form of one column name: trimmed and lower-cased."""
    return str(name).strip().lower()


def drop_duplicate_columns(df):
    # type: (pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]
    """Keep the FIRST column of each normalized name, drop the later ones.

    Trimming and lower-casing headers is what lets this capsule accept
    ``Timestamp`` as ``timestamp`` — but it can also MERGE two distinct
    columns into one name. ``Reading,reading``, ``Timestamp,timestamp``,
    ``reading ,reading`` and ``Instrument_ID,instrument_id`` are all pairs that
    pandas keeps apart (they are different strings) and this capsule then
    collapses. ``df["reading"]`` is a DataFrame rather than a Series once that
    happens, and ``pd.to_numeric`` on a DataFrame raises ``TypeError`` — which
    ended the run with exit 1 and a COMPLETELY EMPTY ``/results``. Worse, the
    file is reachable through the normal pipeline: step 1 accepts such a header
    and carries both columns into ``clean_readings.csv``.

    So the collision is resolved here, at read time, where it can still be
    explained: the first column of each name wins (it is the one a reader
    scanning the header left-to-right would expect), the rest are dropped
    before anything reads them, and the choice is returned so the caller can
    log it, warn about it and record it in the manifest. It is never silent,
    and it never ends the run.

    Returns (frame with unique normalized names, one record per dropped column).
    """
    seen = {}  # type: Dict[str, str]
    keep_positions = []  # type: List[int]
    dropped = []  # type: List[Dict[str, Any]]
    for position, name in enumerate(df.columns):
        key = normalize_header(name)
        if key in seen:
            dropped.append({
                "column": str(name),
                "position": position,
                "normalized_name": key,
                "kept_column": seen[key],
            })
        else:
            seen[key] = str(name)
            keep_positions.append(position)
    if not dropped:
        return df, []
    # .iloc, not [], because the labels are exactly what is ambiguous here.
    return df.iloc[:, keep_positions].copy(), dropped


def duplicate_columns_warning(source_name, dropped):
    # type: (str, List[Dict[str, Any]]) -> str
    """The warning text for columns that collided once headers were normalized."""
    pairs = "; ".join(
        "'{}' (column {}) duplicates '{}' as '{}'".format(
            d["column"], d["position"] + 1, d["kept_column"], d["normalized_name"])
        for d in dropped)
    return (
        "{} has {} column name(s) that collide once trimmed and lower-cased "
        "({}) — the FIRST column of each name was kept and the duplicate(s) "
        "were dropped before any analysis read them".format(
            source_name, len(dropped), pairs))


def read_csv_safely(path):
    # type: (Path) -> Tuple[Optional[pd.DataFrame], List[Dict[str, Any]], Optional[str]]
    """Read one CSV, lower-casing its headers. Unreadable -> ``(None, [], "why")``.

    A zero-byte file (or garbage) is a skip, not a crash: the input asset can
    legitimately contain files this capsule has no opinion about.

    Headers are de-duplicated BEFORE they are normalized, because normalizing
    is what creates the duplicates (see ``drop_duplicate_columns``). The second
    return value lists whatever was dropped, so the caller can report it.

    The THIRD return value is why the file could not be read, and it exists
    because returning a bare ``None`` made a whole input file disappear in
    silence. On an input holding nothing but an unreadable ``readings.csv``,
    this logged one line, ``pick_readings`` moved on, and the manifest then
    said ``source_file: null``, ``rows_in: 0``, ``instruments: []`` and
    ``data_warnings: []`` while ``input_files`` still listed the file — the
    manifest asserting the input contained no readings when the truth is that
    the readings could not be READ. Step 1 already hands this reason back and
    records it under ``unreadable_input_files``; this is the same fix, with the
    same key name and the same warning shape, so one reader handles both.
    """
    try:
        df = pd.read_csv(str(path))
    except Exception as exc:  # noqa: BLE001 - keep scanning the other files
        log("warning: could not read {} ({})".format(path.name, exc))
        return None, [], "{}: {}".format(type(exc).__name__, exc)
    df, dropped = drop_duplicate_columns(df)
    df.columns = [normalize_header(c) for c in df.columns]
    return df, dropped, None


def qc_columns_in(df):
    # type: (pd.DataFrame) -> List[str]
    """Which QC verdict columns (if any) this frame already carries."""
    cols = set(normalize_header(c) for c in df.columns)
    return [c for c in QC_OUTPUT_COLUMNS if c in cols]


def choose_by_name_then_size(candidates):
    # type: (List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]
    """A canonical filename wins; largest-file-wins is only the tie-breaker.

    Identical in shape to step 1's function of the same name, deliberately: the
    two capsules have to resolve the same mount to the same file, and the only
    difference between them is which canonical name each one prefers first
    (step 1 wants the raw readings, step 2 wants step 1's clean output).

    Returns (chosen entry, why it was chosen) — the reason is quoted in the log
    and in the manifest record of every candidate that lost, because "which
    file did these numbers come from, and what else was there" is a question
    the artifact has to answer on its own.
    """
    for name in CANONICAL_READINGS_NAMES:
        for candidate in candidates:
            if candidate["path"].name.lower() == name:
                return candidate, "canonical name {}".format(name)
    ordered = sorted(candidates,
                     key=lambda c: (c["path"].stat().st_size, str(c["path"])))
    return ordered[-1], "no canonically named readings file ({}), so the " \
        "largest file won".format("/".join(CANONICAL_READINGS_NAMES))


def pick_readings(csvs, data_dir, notes):
    # type: (List[Path], Path, List[str]) -> Dict[str, Any]
    """Choose the readings table out of everything mounted under /data.

    The selection order is STEP 1'S, because the two capsules disagreeing about
    which file is the readings file is a defect in its own right:

    1. a CSV must be readable, and must have all of ``REQUIRED_INPUT_COLUMNS``
       to be a candidate at all (a ``reading`` column alone is not enough here —
       this step buckets by time and groups by instrument);
    2. candidates that already carry ``qc_status``/``qc_reason`` are held back:
       those are a previous QC run's OUTPUT and contain every input row
       including the ones QC DROPPED, so analysing one silently re-admits
       rejected readings;
    3. a canonical name wins: ``clean_readings.csv``, then ``readings.csv``;
    4. only then, largest-file-wins.

    What it used to do was take the FIRST file in sorted-path order with the
    three columns, and record nothing about the rest. An
    ``archive_2019_readings.csv`` holding one stale row, sitting beside a live
    48-row ``readings.csv``, therefore won on the alphabet: step 1 filtered the
    real file, step 2 analysed the archive, both exited 0, and no artifact
    anywhere mentioned that a second readings-shaped file existed. Two capsules
    in one pipeline silently describing different data is the worst outcome
    available to a provenance demo, so now every readings-shaped file that was
    NOT chosen is named in the manifest with the reason it lost.

    Three other outcomes are recorded rather than dropped on the floor:

      unreadable   a CSV pandas could not parse at all. Logged AND returned
                   (see ``read_csv_safely``): with only an unreadable
                   readings.csv in the input, this step used to report
                   ``source_file: null``/``rows_in: 0``/``instruments: []``
                   with no warnings, which reads as "the input had no
                   readings" rather than "the readings could not be read".
      rejected     a CSV with a ``reading`` column but missing ``timestamp``
                   and/or ``instrument_id``. Reachable straight down the
                   pipeline: step 1 tolerates a readings file with no
                   ``timestamp`` and writes a ``clean_readings.csv`` that
                   inherits the hole.
      not_chosen   a fully-qualified candidate that lost the selection above.

    Returns a dict: ``path``, ``frame``, ``dropped_columns`` (of the chosen
    file only — the others were never read into the results), ``rejected``,
    ``not_chosen``, ``unreadable``, and ``chosen_reason``.
    """
    candidates = []  # type: List[Dict[str, Any]]
    qc_outputs = []  # type: List[Dict[str, Any]]
    rejected = []  # type: List[Dict[str, Any]]
    unreadable = []  # type: List[Dict[str, Any]]

    # Every CSV is examined before anything is chosen. The old early return
    # meant a file sorting after the winner was never even looked at, so it
    # could not be reported either.
    for path in csvs:
        rel = str(path.relative_to(data_dir))
        df, dropped, error = read_csv_safely(path)
        if df is None:
            unreadable.append({"file": rel, "error": error or "unreadable"})
            continue
        if "reading" not in df.columns:
            continue  # e.g. instruments.csv — a lookup table, not readings
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
        if missing:
            log("note: {} has a 'reading' column but is missing {} — "
                "skipping".format(rel, ", ".join(missing)))
            rejected.append({"file": rel, "missing_columns": list(missing),
                             "columns_present": [str(c) for c in df.columns],
                             "rows": int(len(df))})
            continue
        entry = {"path": path, "file": rel, "frame": df, "dropped": dropped,
                 "rows": int(len(df)), "qc_columns": qc_columns_in(df)}
        if entry["qc_columns"]:
            log("holding back {} as the readings file: it already carries {} — "
                "that is a previous QC run's output, and it still contains the "
                "rows QC dropped".format(rel, "/".join(entry["qc_columns"])))
            qc_outputs.append(entry)
        else:
            candidates.append(entry)

    pool = candidates if candidates else qc_outputs
    if not pool:
        return {"path": None, "frame": None, "dropped_columns": [],
                "rejected": rejected, "not_chosen": [], "unreadable": unreadable,
                "chosen_reason": None}

    if not candidates:
        # Last resort, exactly as in step 1: analysing a QC output is a worse
        # answer than analysing nothing only if nobody is told. This step reads
        # three columns and ignores the verdict columns, so the risk is not a
        # stale verdict but stale ROWS — the readings a previous QC run threw
        # away are analysed here as if they had passed.
        notes.append(
            "every CSV with the required columns ({}) already carries QC "
            "verdict columns ({}); {} was analysed anyway rather than emitting "
            "nothing, so any row a previous QC run marked qc_status=dropped is "
            "included in these results".format(
                ", ".join(REQUIRED_INPUT_COLUMNS),
                "/".join(QC_OUTPUT_COLUMNS), pool[0]["file"]))
        log("warning: {}".format(notes[-1]))

    chosen, reason = choose_by_name_then_size(pool)
    log("readings file chosen by {}: {}".format(reason, chosen["file"]))

    not_chosen = []  # type: List[Dict[str, Any]]
    for entry in candidates + qc_outputs:
        if entry["path"] == chosen["path"]:
            continue
        is_qc_output = bool(entry["qc_columns"]) and bool(candidates)
        if is_qc_output:
            why = ("it carries {} — a previous QC run's output, which still "
                   "holds the rows QC dropped".format(
                       "/".join(entry["qc_columns"])))
        else:
            why = "{} was preferred ({})".format(chosen["file"], reason)
        not_chosen.append({"file": entry["file"], "rows": entry["rows"],
                           "reason": why,
                           # true = the expected sibling of a captured step-1
                           # result, not a rival readings file. The caller uses
                           # this to decide warning vs note; the record is the
                           # same either way.
                           "is_qc_output": is_qc_output})

    return {"path": chosen["path"], "frame": chosen["frame"],
            "dropped_columns": chosen["dropped"], "rejected": rejected,
            "not_chosen": not_chosen, "unreadable": unreadable,
            "chosen_reason": reason}


def rejected_readings_warning(rejected):
    # type: (List[Dict[str, Any]]) -> str
    """The data_warnings entry for readings files this step would not take."""
    listed = "; ".join(
        "{} ({} row(s), missing {})".format(
            r["file"], r["rows"], ", ".join(r["missing_columns"]))
        for r in rejected)
    return (
        "{} file(s) with a 'reading' column were REFUSED because they are "
        "missing column(s) this step requires ({}): {}. Their rows were not "
        "analysed and are not counted in rows_in — so the empty outputs and "
        "the empty instruments list below mean \"nothing this step could "
        "read\", NOT \"no readings in the input\"".format(
            len(rejected), ", ".join(REQUIRED_INPUT_COLUMNS), listed))


def not_chosen_readings_warning(source_file, not_chosen):
    # type: (Optional[str], List[Dict[str, Any]]) -> str
    """The data_warnings entry naming every readings file that LOST the pick.

    An unselected file is not an error and nothing about it is wrong — but the
    numbers in these outputs came from one file out of several, and a reader
    who cannot see which ones were passed over cannot tell this run from a run
    over completely different data. Step 1 records the same thing about the
    same mount, so the two manifests can be compared directly.
    """
    listed = "; ".join(
        "{} ({} row(s)) — {}".format(r["file"], r["rows"], r["reason"])
        for r in not_chosen)
    return (
        "{} other file(s) in the input have all the columns this step requires "
        "and were NOT analysed: {}. Every number in these outputs comes from "
        "{} alone".format(len(not_chosen), listed, source_file))


def unreadable_files_warning(entries):
    # type: (List[Dict[str, Any]]) -> str
    """The data_warnings entry naming every input CSV that could not be read.

    Deliberately the same shape as step 1's warning of the same name, and it
    feeds the same manifest key (``unreadable_input_files``), so one reader
    handles both capsules. The one difference is the last clause: step 1 copies
    the files it cannot parse into its results and reports whether that worked,
    while this step copies no input anywhere, so there is nothing to report.
    """
    listed = "; ".join(
        "{} ({})".format(e["file"], e["error"]) for e in entries)
    return (
        "{} input CSV file(s) could not be read and contributed NOTHING to "
        "this run's analysis — they are listed in input_files because they "
        "were found, and in unreadable_input_files with the reason: {}. If the "
        "readings were in one of them, the empty/short outputs below mean "
        "\"this step could not read the data\", NOT \"there was no "
        "data\"".format(len(entries), listed))


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
# Timestamps arrive as text and are never trusted. Two things go wrong with
# them often enough to have ended this capsule with exit 1 and a COMPLETELY
# EMPTY /results, and both of them are DATA problems, so neither may stop the
# run:
#
#   MIXED UTC OFFSETS. A real export can carry ``+00:00`` on one row and
#   ``+05:30`` on the next. Plain ``pd.to_datetime(errors="coerce")`` cannot
#   choose one dtype for that and hands back an OBJECT-dtype Index of per-row
#   datetimes — a FutureWarning, not an error. ``errors="coerce"`` is no
#   protection here: it coerces per-ELEMENT parse failures, it says nothing
#   about the dtype of the RESULT. An object index has no ``.dt`` accessor and
#   cannot be resampled, so ``resample_instrument`` raised "Only valid with
#   DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an instance of
#   'Index'". The file is reachable through the ordinary pipeline: step 1
#   accepts a mixed-offset readings.csv, exits 0, and copies those offsets
#   verbatim into clean_readings.csv. Parsing with ``utc=True`` and then
#   dropping the zone puts every row on one timeline in a single tz-naive
#   ``datetime64`` dtype; an all-naive column round-trips unchanged, and two
#   columns that mean the same instants bucket identically however they spell
#   the zone. Step 1's ``qc_filter.py`` does exactly the same thing, and so
#   does batch 1's ``make_report.py``.
#
#   IMPLAUSIBLE INSTANTS. ``1700-01-01T00:00:00`` is inside pandas'
#   representable range, so it parses cleanly — and then any span measured
#   against ordinary 2026 data is ~326 years, which overflows an int64 count of
#   NANOSECONDS at ~292 years. That is what made ``fit_trend`` raise
#   "OverflowError: Overflow in int64 addition" AFTER the resample and the
#   anomaly scoring had already succeeded and been logged. Such a value is
#   still data, so it is never dropped in silence: it is treated exactly like
#   an unparseable timestamp (see ``normalize_readings``), counted, named in
#   the run log, and reported in the manifest as
#   ``rows_implausible_timestamp``.
#
# Every span computation is ALSO made arithmetically overflow-proof (see
# ``timestamps_as_days``), because a guard that merely filters the inputs we
# thought of is not the same as arithmetic that cannot fail.
PLAUSIBLE_MIN_TIMESTAMP = pd.Timestamp("1900-01-01")
PLAUSIBLE_MAX_TIMESTAMP = pd.Timestamp("2200-01-01")

# A trailing UTC offset ("+05:30", "-0400") or the "Z" zone marker — how a raw
# value announces that it carries a zone at all. Matched against the ORIGINAL
# text, because after parsing there is nothing left to count.
UTC_OFFSET_PATTERN = r"(?:[Zz]|[+-]\d{2}:?\d{2})\s*$"

# How a timestamp value is SHAPED, which decides how it gets interpreted. Only
# the first of these is unambiguous; see describe_timestamp_format for what the
# other two silently commit the run to.
ISO_TIMESTAMP_PATTERN = (
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
    r"\s*(?:[Zz]|[+-]\d{2}:?\d{2})?$")
NUMERIC_TIMESTAMP_PATTERN = r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$"
DAY_MONTH_TIMESTAMP_PATTERN = r"^\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"


def _mixed_timestamp_format_supported():
    # type: () -> bool
    """Can this pandas parse a column of per-row date formats (``format="mixed"``)?

    Feature-probed rather than version-sniffed, and probed with the exact shape
    that matters: one offset-AWARE value next to one offset-NAIVE value.

    ``format="mixed"`` arrived in pandas 2.0. On pandas 1.x the keyword exists
    but "mixed" is taken as a literal strftime pattern, which with
    ``errors="coerce"`` would quietly return an all-NaT column — so the probe
    uses ``errors="raise"``, where 1.x fails loudly and we fall back.
    """
    try:
        pd.to_datetime(
            pd.Series(["2025-01-02T00:00:00+00:00", "2025-01-03"]),
            format="mixed", utc=True, errors="raise",
        )
    except Exception:  # noqa: BLE001 - any failure means "not supported here"
        return False
    return True


#: Probed once at import; used by _to_utc_datetime().
MIXED_TIMESTAMP_FORMAT_SUPPORTED = _mixed_timestamp_format_supported()


def _to_utc_datetime(values):
    # type: (pd.Series) -> pd.Series
    """``pd.to_datetime(values, utc=True)``, per-row format where pandas allows it.

    ``format="mixed"`` makes pandas infer a format for EVERY element instead of
    inferring one from the first and coercing everything that disagrees. On a
    pandas too old to support it (probed above) this degrades to the plain
    single-format parse — ``utc=True`` is the part that guarantees one dtype,
    and it is present either way.
    """
    if MIXED_TIMESTAMP_FORMAT_SUPPORTED:
        try:
            return pd.to_datetime(
                values, errors="coerce", utc=True, format="mixed")
        except Exception:  # noqa: BLE001 - fall through to the plain parse
            pass
    return pd.to_datetime(values, errors="coerce", utc=True)


def utc_offsets_in(values):
    # type: (pd.Series) -> Tuple[int, List[str]]
    """How many RAW values carry an explicit UTC offset, and which offsets.

    Counted before parsing, because ``utc=True`` is precisely what erases the
    evidence. More than one distinct entry here is the mixed-offset case that
    used to end the run.
    """
    tz = getattr(values.dtype, "tz", None)
    if tz is not None:  # already tz-aware (not reachable from read_csv)
        return int(values.notna().sum()), [str(tz)]
    if values.dtype != object:
        return 0, []
    text = values.astype(str).str.strip()
    found = text.str.extract("(" + UTC_OFFSET_PATTERN + ")", expand=False)
    found = found.dropna().str.strip().str.upper()
    return int(len(found)), sorted(set(found.tolist()))


def empty_timestamp_format():
    # type: () -> Dict[str, Any]
    """The "no timestamp column was examined" shape of the format report.

    ``unambiguous_iso8601`` is null rather than true on purpose: with nothing
    to look at, "the column was unambiguous" would be an assertion this run is
    in no position to make.
    """
    return {
        "unambiguous_iso8601": None,
        "n_iso8601": 0,
        "n_numeric": 0,
        "n_day_month_ambiguous": 0,
        "n_other_text": 0,
        "n_blank": 0,
        "examples": {},
        "examples_read_as": {},
        "per_element_format_inference": MIXED_TIMESTAMP_FORMAT_SUPPORTED,
    }


def describe_timestamp_format(values):
    # type: (pd.Series) -> Dict[str, Any]
    """Classify how the RAW timestamp text will be INTERPRETED, before parsing.

    Parsing a timestamp column is a reading of the data, and when the text is
    not unambiguous ISO-8601 that reading is a guess this capsule makes on the
    operator's behalf — silently, until now. Two guesses in particular change
    every number downstream:

      * a BARE NUMBER. ``1780000000`` is a perfectly ordinary epoch-SECONDS
        value, and ``pd.to_datetime(..., utc=True)`` reads it as NANOSECONDS:
        the whole series lands in the first two seconds of 1970. Every bucket
        collapses into one, coverage is measured over a span of milliseconds,
        and nothing anywhere says the column was read that way.
      * a DAY/MONTH/YEAR-style date. ``format="mixed"`` infers the format per
        ELEMENT, so ``01/02/2026`` (ambiguous, read month-first) and
        ``13/02/2026`` (unambiguous, read day-first) in the SAME column are
        read with DIFFERENT conventions. That is not a parse failure — it
        succeeds, and hands back dates in two different calendars.

    Neither is something this capsule can fix (the file does not say which
    convention it meant), and neither is a reason to fail: they are data. What
    they are is a fact about the result that has to be recorded, so the counts
    and a few example values go into the manifest under
    ``timestamp_interpretation`` and into a warning in words.

    Blank/NaT-ish values are counted separately and do NOT make the column
    ambiguous: they are already accounted for as unusable timestamps.
    """
    fmt = empty_timestamp_format()
    if len(values) == 0:
        fmt["unambiguous_iso8601"] = True
        return fmt
    text = values.astype(str).str.strip()
    blank = (values.isna() | text.eq("")
             | text.str.lower().isin(["nan", "nat", "none", "null", "na"]))
    iso = ~blank & text.str.match(ISO_TIMESTAMP_PATTERN)
    numeric = ~blank & ~iso & text.str.match(NUMERIC_TIMESTAMP_PATTERN)
    day_month = (~blank & ~iso & ~numeric
                 & text.str.match(DAY_MONTH_TIMESTAMP_PATTERN))
    other = ~blank & ~iso & ~numeric & ~day_month

    def examples(mask):
        # type: (pd.Series) -> List[str]
        return [str(v) for v in text[mask].drop_duplicates().head(3).tolist()]

    fmt["n_iso8601"] = int(iso.sum())
    fmt["n_numeric"] = int(numeric.sum())
    fmt["n_day_month_ambiguous"] = int(day_month.sum())
    fmt["n_other_text"] = int(other.sum())
    fmt["n_blank"] = int(blank.sum())
    fmt["unambiguous_iso8601"] = bool(
        fmt["n_numeric"] == 0 and fmt["n_day_month_ambiguous"] == 0
        and fmt["n_other_text"] == 0)
    for key, mask in (("numeric", numeric),
                      ("day_month_ambiguous", day_month),
                      ("other_text", other)):
        found = examples(mask)
        if found:
            fmt["examples"][key] = found
    return fmt


def annotate_parsed_examples(fmt, values, parsed):
    # type: (Dict[str, Any], pd.Series, pd.Series) -> Dict[str, Any]
    """Record what each example value was ACTUALLY read as.

    Worked out from this run's own parsed column rather than re-parsed here,
    because the answer depends on the column's dtype as well as on the text: a
    column of integers is read as nanoseconds since the epoch, while the same
    digits as strings among other text are read as NaT. An illustration that
    guessed would be one more claim the manifest cannot support.
    """
    text = values.astype(str).str.strip()
    read_as = {}  # type: Dict[str, List[str]]
    for kind, samples in fmt["examples"].items():
        rendered = []
        for sample in samples:
            hits = parsed[text == sample].dropna()
            rendered.append("\"{}\" -> {}".format(
                sample,
                pd.Timestamp(hits.iloc[0]).isoformat() if len(hits)
                else "NaT (unusable, dropped)"))
        read_as[kind] = rendered
    fmt["examples_read_as"] = read_as
    return fmt


def timestamp_format_warning(fmt, consequence):
    # type: (Dict[str, Any], str) -> str
    """The warning for a timestamp column that is not unambiguous ISO-8601."""
    read_as = fmt.get("examples_read_as", {})
    parts = []  # type: List[str]
    if fmt["n_numeric"]:
        parts.append(
            "{} value(s) are bare numbers, which a numeric column reads as "
            "NANOSECONDS since 1970-01-01, so an epoch value in seconds or "
            "milliseconds lands in 1970 rather than where it was meant to "
            "({})".format(fmt["n_numeric"],
                          "; ".join(read_as.get("numeric", []))))
    if fmt["n_day_month_ambiguous"]:
        parts.append(
            "{} value(s) are day/month/year-style dates whose day-and-month "
            "order is inferred PER VALUE, so \"01/02/2026\" and \"13/02/2026\" "
            "in one column are read with DIFFERENT conventions — month-first "
            "for the first, day-first for the second ({})".format(
                fmt["n_day_month_ambiguous"],
                "; ".join(read_as.get("day_month_ambiguous", []))))
    if fmt["n_other_text"]:
        parts.append(
            "{} value(s) are some other text, left to pandas' general date "
            "parser and read as NaT if it cannot make sense of them "
            "({})".format(fmt["n_other_text"],
                          "; ".join(read_as.get("other_text", []))))
    return (
        "the timestamp column is NOT unambiguous ISO-8601, so how it was read "
        "is an interpretation and not a fact about the file: {}. {} The counts "
        "and examples are in the manifest under timestamp_interpretation; "
        "supply ISO-8601 timestamps (2026-06-01T00:00:00) if the reading above "
        "is not the one you meant".format("; ".join(parts), consequence))


def empty_timestamp_info():
    # type: () -> Dict[str, Any]
    """The "nothing to report" shape of ``parse_timestamp_series``'s info dict.

    So the manifest carries the same keys whether or not there were any
    timestamps to parse — a reader must never have to tell "none of this
    happened" from "this run did not report it".
    """
    return {
        "n_with_offset": 0,
        "offsets": [],
        "n_implausible": 0,
        "implausible": None,
        "dtype_fallback": None,
        "format": empty_timestamp_format(),
    }


def parse_timestamp_series(values):
    # type: (pd.Series) -> Tuple[pd.Series, Dict[str, Any]]
    """Parse a timestamp column to ONE tz-naive ``datetime64`` dtype.

    Returns ``(parsed, info)``. ``info`` reports everything the parse changed
    or noticed, so the caller can put it in the run log AND the manifest:

      n_with_offset   raw values that carried an explicit UTC offset or "Z"
      offsets         the distinct offsets seen (>1 entry = the mixed case)
      n_implausible   parsed values outside [1900-01-01, 2200-01-01)
      implausible     the boolean mask for those rows, aligned to ``values``
      dtype_fallback  set only if the parse somehow did not yield a datetime
                      dtype, in which case the column degrades to all-NaT
      format          how the raw text was INTERPRETED, and whether that was
                      an interpretation at all (see describe_timestamp_format)

    The dtype check is not decoration. ``errors="coerce"`` protects against a
    single unreadable ELEMENT; it does not promise a datetime-typed RESULT, and
    an object-dtype column reaches ``.resample`` and ``.dt`` looking fine and
    then raises. Checking here means the rest of the capsule can treat the
    column as ``datetime64`` without another guard — and if a future pandas
    ever breaks that promise, the run degrades to "no usable timestamps" with a
    reported reason instead of exiting 1 with nothing written.
    """
    info = empty_timestamp_info()
    info["n_with_offset"], info["offsets"] = utc_offsets_in(values)
    # How the text will be READ, worked out before parsing erases the evidence.
    # A column that is not unambiguous ISO-8601 is interpreted, not merely
    # read, and the interpretation belongs in the manifest.
    info["format"] = describe_timestamp_format(values)

    parsed = _to_utc_datetime(values)
    if getattr(parsed.dtype, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        info["dtype_fallback"] = str(parsed.dtype)
        parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    # What each example value was actually read AS — from this run's own parse,
    # so the manifest illustrates the interpretation instead of guessing it.
    info["format"] = annotate_parsed_examples(info["format"], values, parsed)

    implausible = parsed.notna() & (
        (parsed < PLAUSIBLE_MIN_TIMESTAMP) | (parsed >= PLAUSIBLE_MAX_TIMESTAMP))
    info["implausible"] = implausible
    info["n_implausible"] = int(implausible.sum())
    return parsed, info


def timestamps_as_days(timestamps):
    # type: (pd.Series) -> Optional[np.ndarray]
    """Each timestamp as DAYS since the earliest one, and it CANNOT overflow.

    The obvious ``(timestamps - timestamps.min()).dt.total_seconds()`` raises
    ``OverflowError: Overflow in int64 addition`` as soon as the span passes
    ~292 years, even though BOTH endpoints are perfectly representable — a
    single 1700-01-01 sentinel among 2026 data is enough, and it took the run
    down after the resample and the anomaly scoring had already been logged.

    The fix is to do the subtraction in MICROSECONDS. An int64 count of
    microseconds spans +-292,471 years, so no pair of instants pandas can
    represent at all (1677..2262) can overflow it, while nanoseconds overflow
    inside pandas' own representable range. Truncating below a microsecond is
    irrelevant to a slope quoted per DAY. NaT positions come back as NaN rather
    than the int64 sentinel, and an all-NaT (or non-datetime) column returns
    None so the caller can degrade instead of dividing by nothing.
    """
    series = pd.Series(timestamps)
    if not pd.api.types.is_datetime64_any_dtype(series):
        return None
    missing = series.isna().to_numpy()
    if missing.all():
        return None
    micros = series.to_numpy(dtype="datetime64[us]").astype("int64")
    base = int(micros[~missing].min())
    # NaT's int64 sentinel must never enter the arithmetic, so it is replaced
    # by the base (difference 0) and blanked out again afterwards.
    offsets = (np.where(missing, base, micros) - base).astype("float64")
    offsets[missing] = np.nan
    return offsets / 1e6 / 86400.0


def normalize_readings(df):
    # type: (pd.DataFrame) -> Tuple[pd.DataFrame, int, int, Dict[str, Any], int, int]
    """Coerce the three required columns and drop rows that cannot be used.

    Returns the usable frame, how many rows were dropped in total, how many of
    those were dropped for a NON-FINITE reading, what parsing the timestamp
    column changed or noticed, how many rows were relabelled to
    ``UNKNOWN_INSTRUMENT``, and how many rows already carried that exact id in
    the input — so the run log and the manifest can both account for every
    input row and name the reason.

    That second count exists because ``pd.to_numeric`` is not the filter it
    looks like: it turns ``"abc"`` into NaN (dropped by ``dropna`` below) but
    turns ``"inf"``, ``"-inf"``, ``"Infinity"`` and ``"1e400"`` into a real
    ``inf`` float, which ``dropna`` keeps. An ``inf`` reading then flows
    straight through the analysis into ``resampled.csv`` (an ``inf`` bucket
    mean) and ``instrument_summary.csv`` (an ``inf`` mean/max, a blank ``sd``
    and a blank ``mean_vs_baseline`` for EVERY instrument, because the
    baseline mean is no longer usable) — numbers no reader can act on, with
    nothing anywhere saying where they came from. A reading that is not a
    finite number is not analyzable, so it is dropped here, at the one place
    that already counts and reports dropped rows.

    The timestamp column gets the same treatment for the same reason. It is
    parsed to ONE tz-naive ``datetime64`` dtype (see ``parse_timestamp_series``
    for why the dtype, not just the elements, is what matters), and a value
    outside the plausible range is then blanked so it is dropped exactly like
    an unparseable one. That is a deliberate choice, not an oversight: a
    1700-01-01 sentinel sitting among 2026 readings is not an instant this
    analysis can use — it turns a 12-hour series into a 326-year one, which
    buckets into nothing and fits a meaningless trend even where the
    arithmetic no longer overflows. It is DATA, so it is counted and reported
    (``rows_implausible_timestamp``) rather than discarded quietly, and every
    other row of the same instrument is analysed normally.
    """
    out = df[REQUIRED_INPUT_COLUMNS].copy()
    stamps, ts_info = parse_timestamp_series(out["timestamp"])
    if ts_info["n_implausible"]:
        stamps = stamps.mask(ts_info["implausible"])
    out["timestamp"] = stamps
    out["reading"] = pd.to_numeric(out["reading"], errors="coerce")
    out["instrument_id"] = out["instrument_id"].astype(str).str.strip()
    before = len(out)
    # NaN is already unusable; this only adds +-inf. Counted before the drop so
    # the manifest can report the subset.
    numeric = pd.to_numeric(out["reading"], errors="coerce")
    non_finite = numeric.notna() & ~np.isfinite(numeric.to_numpy(dtype=float))
    n_non_finite = int(non_finite.sum())
    if n_non_finite:
        out.loc[non_finite, "reading"] = np.nan
    out = out.dropna(subset=REQUIRED_INPUT_COLUMNS)
    # A MISSING instrument_id used to invent an instrument called "nan": the
    # `astype(str)` two lines up runs BEFORE this guard, so a blank cell (a
    # float NaN out of read_csv) had already become the perfectly non-empty
    # string "nan" and `!= ""` never fired. The manifest then listed "nan"
    # among `instruments`, instrument_summary.csv gained a row for it, and
    # `rows_unusable` stayed 0 — a fabricated instrument, reported as real.
    # Step 1 relabels exactly these rows to UNKNOWN_INSTRUMENT, so this does
    # the same thing to the same rows: label, never invent, never drop, and
    # count so the caller can say it happened.
    missing_id = out["instrument_id"].isin(MISSING_INSTRUMENT_TOKENS)
    n_unknown_instrument = int(missing_id.sum())
    # ...and the label can COLLIDE with the data. Nothing stops a file from
    # carrying the literal string "(unknown)" as a real instrument_id, and when
    # it does, the relabelled rows and the real ones merge into a single group
    # that no output can tell apart — while the warning about the relabelling
    # states, falsely, that this id is the capsule's label rather than a value
    # from the file. The rows are counted here so the caller can say what
    # actually happened instead.
    n_literal_unknown = int((out["instrument_id"] == UNKNOWN_INSTRUMENT).sum())
    if n_unknown_instrument:
        out.loc[missing_id, "instrument_id"] = UNKNOWN_INSTRUMENT
    usable = out.sort_values(["instrument_id", "timestamp"]).reset_index(drop=True)
    return (usable, before - len(usable), n_non_finite, ts_info,
            n_unknown_instrument, n_literal_unknown)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def iso(value):
    # type: (Any) -> str
    """One timestamp as an ISO-8601 string (matches the input's own format)."""
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat()


def resample_instrument(group, freq, rolling_window):
    # type: (pd.DataFrame, str, int) -> pd.DataFrame
    """Bucket one instrument's readings and add the smoothed trend line.

    Buckets containing no readings (a coverage gap, e.g. an instrument that
    was offline) are dropped rather than emitted as empty rows: every row of
    ``resampled.csv`` therefore has ``n >= 1`` and a real mean.

    ``resample`` accepts nothing but a DatetimeIndex — on anything else it
    raises ``TypeError: Only valid with DatetimeIndex ... got an instance of
    'Index'``, which is how a mixed-offset timestamp column used to end the run
    with a completely empty ``/results``. ``normalize_readings`` now guarantees
    a tz-naive ``datetime64`` column, so this check should never fire; it stays
    because "should never" is not a contract, and returning no buckets for one
    instrument is a far better answer than writing nothing for any of them.
    """
    series = group.set_index("timestamp")["reading"].sort_index()
    if not isinstance(series.index, pd.DatetimeIndex):
        return pd.DataFrame(
            columns=["timestamp", "mean", "min", "max", "n", "rolling_mean"])
    buckets = series.resample(freq).agg(["mean", "min", "max", "count"])
    buckets = buckets.rename(columns={"count": "n"})
    buckets = buckets[buckets["n"] > 0]
    # min_periods=1 so the first buckets get a rolling mean too (an average of
    # however few points exist yet) instead of a run of NaNs at the start.
    buckets["rolling_mean"] = (
        buckets["mean"].rolling(window=rolling_window, min_periods=1).mean()
    )
    return buckets.reset_index()


def robust_z(values):
    # type: (pd.Series) -> pd.Series
    """Robust z-score of a series via the median absolute deviation.

    A plain z-score uses the mean and standard deviation, both of which are
    inflated by the very outliers we want to find. The median and the MAD are
    not. A zero MAD (a perfectly flat or single-point series) has no scale to
    divide by, so everything scores 0 — no anomalies, no ZeroDivisionError.
    """
    if len(values) == 0:
        return pd.Series([], dtype=float)
    median = values.median()
    mad = (values - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return MAD_TO_SIGMA * (values - median) / mad


def has_robust_scale(values):
    # type: (pd.Series) -> bool
    """Is there a non-zero MAD here, i.e. can ``robust_z`` actually score this?

    Exactly the test ``robust_z`` applies before dividing, asked in advance.
    ``robust_z`` answers a zero MAD with a column of zeros, which is the right
    arithmetic — but a column of zeros is indistinguishable from "measured, and
    nothing stood out", when what really happened is that the rule had no scale
    to measure against and COULD NOT flag anything. The caller uses this to say
    which of the two it was.
    """
    if len(values) == 0:
        return False
    median = values.median()
    mad = (values - median).abs().median()
    return bool(np.isfinite(mad) and mad != 0)


def anomaly_rule_skipped_warning(skipped, rolling_window, interval, n_scored,
                                 n_instruments):
    # type: (List[Dict[str, Any]], int, str, int, int) -> str
    """The data_warnings entry for instruments the anomaly rule could not score.

    This is the difference between "no anomalies" and "no anomaly test", and
    the manifest used to state the first while meaning the second. Raising
    `rolling_window` past an instrument's bucket count — 100 against the demo's
    84/80/78 buckets, an ordinary value nowhere near the clamp, so no clamp
    warning fires — excluded EVERY bucket from scoring: n_anomalies_flagged 0,
    a header-only anomalies.csv, and parameter_warnings, data_warnings and
    notes all empty. A presenter nudging the smoothing window up in the GUI
    would watch the anomaly table empty itself with nothing to explain it.
    """
    listed = "; ".join(
        "{} ({} bucket(s); {})".format(
            s["instrument_id"], s["n_buckets"], s["reason"]) for s in skipped)
    warm_up = max(rolling_window - 1, 0)
    warmup_caused = any(s["n_buckets_scored"] == 0 for s in skipped)
    explanation = ""
    if warmup_caused:
        explanation = (
            " The first {} bucket(s) of every instrument are excluded from "
            "scoring as the rolling-window warm-up (rolling_window={}), "
            "because their rolling mean is built from fewer points and their "
            "residuals are small BY CONSTRUCTION; the warm-up is NOT clamped "
            "to fit a short series, since scoring those buckets would flag a "
            "steadily drifting instrument's legitimate trend as a run of "
            "anomalies. To score a short series, lower rolling_window below "
            "the bucket count(s) above, or choose a resample_interval finer "
            "than {} so there are more buckets.".format(
                warm_up, rolling_window, interval))
    return (
        "the anomaly rule could NOT be applied to {} of {} instrument(s): {}. "
        "{} in this run, so a 0 in n_anomalies for {} means \"the rule "
        "produced no verdict here\", NOT \"nothing unusual happened\".{} "
        "resampled.csv, its rolling_mean column and instrument_summary.csv are "
        "unaffected — only the anomaly scoring is".format(
            len(skipped), n_instruments, listed,
            "NO bucket was scored at all" if n_scored == 0
            else "{} bucket(s) were scored".format(n_scored),
            "them" if len(skipped) > 1 else "it",
            explanation))


def fit_trend(timestamps, readings):
    # type: (pd.Series, pd.Series) -> Tuple[float, float]
    """Least-squares fit of reading vs. time in DAYS -> (slope_per_day, r2).

    Time is expressed in days so the slope reads as "degrees per day" no matter
    what the sampling interval is. ``r2`` is the squared Pearson correlation.
    Degenerate inputs return NaN instead of raising, because a single reading
    or a stuck sensor is data, not an error:
      * fewer than 2 points, or every point at the same instant -> NaN/NaN
      * a perfectly flat series -> slope 0.0 and r2 0.0 (a horizontal line
        fits perfectly but explains no variance, so 0 is the honest answer)
      * no usable timestamp at all, or a column that is not datetime-typed
        -> NaN/NaN

    That promise used to be false for one input: an ordinary series with one
    implausibly old timestamp in it. ``(timestamps - timestamps.min())``
    overflows int64 nanoseconds past a ~292-year span and raised
    ``OverflowError``, killing a run whose resample and anomaly scoring had
    already succeeded. ``timestamps_as_days`` does the same subtraction in a
    unit that cannot overflow, so the arithmetic is safe no matter what reaches
    it — independently of ``normalize_readings`` filtering such rows out first.
    """
    if len(readings) < 2:
        return float("nan"), float("nan")
    days = timestamps_as_days(timestamps)
    if days is None:
        return float("nan"), float("nan")
    y = readings.to_numpy(dtype=float)
    # NaT/NaN in either column has no place in a least-squares fit; np.polyfit
    # would return all-NaN coefficients rather than say so.
    usable = np.isfinite(days) & np.isfinite(y)
    if int(usable.sum()) < 2:
        return float("nan"), float("nan")
    days = days[usable]
    y = y[usable]
    if days.std() == 0:
        return float("nan"), float("nan")
    if y.std() == 0:
        return 0.0, 0.0
    slope, _intercept = np.polyfit(days, y, 1)
    r = float(np.corrcoef(days, y)[0, 1])
    return float(slope), r * r


def summarize_instrument(instrument_id, group, n_anomalies):
    # type: (str, pd.DataFrame, int) -> Dict[str, Any]
    """One row of instrument_summary.csv, computed from the RAW readings.

    Deliberately raw rather than resampled: the mean/sd/min/max and the trend
    fit then describe the instrument itself and do not shift when the user
    changes the bucket size. ``n_anomalies`` does come from the bucket level —
    it is how many rows this instrument contributed to anomalies.csv.
    """
    readings = group["reading"]
    timestamps = group["timestamp"]
    slope, r2 = fit_trend(timestamps, readings)
    return {
        "instrument_id": instrument_id,
        "n": int(len(group)),
        "mean": float(readings.mean()),
        # ddof=1 (sample sd) needs 2+ points; pandas already returns NaN for 1.
        "sd": float(readings.std(ddof=1)),
        "min": float(readings.min()),
        "max": float(readings.max()),
        "slope_per_day": slope,
        "r2": r2,
        "first_timestamp": iso(timestamps.min()),
        "last_timestamp": iso(timestamps.max()),
        "n_anomalies": int(n_anomalies),
    }


def summarize_degenerate_instruments(summary_rows):
    # type: (List[Dict[str, Any]]) -> List[str]
    """Notes for the instruments whose summary row has unexplained blanks.

    Every other value in instrument_summary.csv is explained somewhere — a
    skipped rule in ``parameter_warnings``, a dropped row in ``rows_unusable``,
    a substituted id in ``data_warnings``. The blanks in ``sd``,
    ``slope_per_day`` and ``r2`` were the exception: an instrument with one
    reading, or one whose readings all share a single timestamp, produced empty
    cells with nothing anywhere saying why. That reads as a broken capsule
    rather than as a property of the data, so each case gets a note naming the
    instrument and the reason.
    """
    single_reading = [r["instrument_id"] for r in summary_rows if r["n"] < 2]
    # n >= 2 but no time axis: every reading landed on the same instant, so
    # fit_trend has nothing to regress against (sd is still computable).
    single_instant = [
        r["instrument_id"] for r in summary_rows
        if r["n"] >= 2 and r["first_timestamp"] == r["last_timestamp"]]
    notes = []  # type: List[str]
    if single_reading:
        notes.append(
            "{} instrument(s) have a single usable reading ({}) — their sd, "
            "slope_per_day and r2 are BLANK in instrument_summary.csv because "
            "a sample sd needs two readings and a trend needs two points, not "
            "because the values were lost".format(
                len(single_reading), ", ".join(single_reading)))
    if single_instant:
        notes.append(
            "{} instrument(s) have every reading at one single timestamp ({}) "
            "— their slope_per_day and r2 are BLANK in instrument_summary.csv "
            "because there is no time axis to fit a trend against; sd, mean, "
            "min and max are still computed from all their readings".format(
                len(single_instant), ", ".join(single_instant)))
    return notes


def resolve_baseline(raw, instruments, warns):
    # type: (str, List[str], List[str]) -> str
    """Pick the instrument every other instrument's mean is compared against.

    Blank means "the first one alphabetically". A name that is not in the data
    is a typo, not a failure: warn, fall back, carry on. Whichever id wins is
    recorded in manifest.json AND drives the ``mean_vs_baseline`` column of
    instrument_summary.csv, so the choice is both provenance and result.
    """
    if not instruments:
        return ""
    text = (raw or "").strip()
    if not text:
        return instruments[0]
    for instrument in instruments:
        if instrument.lower() == text.lower():
            return instrument
    warns.append(
        "{} \"{}\" is not in the data ({}) — using {}".format(
            PARAM_LABELS["baseline_instrument"], text,
            ", ".join(instruments), instruments[0])
    )
    return instruments[0]


def add_baseline_comparison(summary_rows, baseline, notes):
    # type: (List[Dict[str, Any]], str, List[str]) -> Optional[float]
    """Fill in ``mean_vs_baseline`` on every summary row, in place.

    The value is this instrument's mean minus the baseline instrument's mean,
    and exactly ``0`` on the baseline row itself — so the column reads as
    "how far above/below the reference this instrument sits". Changing
    ``--baseline_instrument`` re-centres the whole column, which is the point:
    the parameter has to change an OUTPUT, not just a manifest line.

    Returns the baseline's own mean (None when there is no usable baseline —
    an empty input, in which case every value is NaN rather than a crash).

    A baseline row that EXISTS but whose mean is not finite is the awkward
    case, and it used to pass in silence: ``manifest.json`` recorded
    ``baseline_mean: null``, every ``mean_vs_baseline`` cell came out blank,
    and nothing said why — which defeats the stated purpose of the key, that
    the column can be re-derived from the manifest alone. It now leaves a note.
    (``normalize_readings`` drops non-finite readings, so this should no longer
    be reachable from the data; it stays because a null that explains itself is
    the whole contract, and a guard that reports is worth more than one that
    assumes.)
    """
    baseline_mean = None  # type: Optional[float]
    found = False
    for row in summary_rows:
        if row["instrument_id"] == baseline:
            baseline_mean = row["mean"]
            found = True
            break
    usable = baseline_mean is not None and np.isfinite(baseline_mean)
    if found and not usable:
        notes.append(
            "the baseline instrument {} has a mean that is not a finite number "
            "({}), so it cannot be used as a reference — manifest baseline_mean "
            "is null and every mean_vs_baseline in instrument_summary.csv is "
            "blank".format(baseline, baseline_mean))
        log("note: {}".format(notes[-1]))
    for row in summary_rows:
        if not usable:
            row["mean_vs_baseline"] = float("nan")
        elif row["instrument_id"] == baseline:
            row["mean_vs_baseline"] = 0.0  # the reference is zero by definition
        else:
            row["mean_vs_baseline"] = float(row["mean"] - baseline_mean)
    return baseline_mean if usable else None


def round_without_overflow(series):
    # type: (pd.Series) -> pd.Series
    """``series.round(ROUND_TO)`` that cannot manufacture an infinity.

    pandas implements ``.round(6)`` as multiply by 1e6, round, divide by 1e6.
    That is fine for a temperature and catastrophic for a large one: a finite
    reading of 1e308 overflows to ``inf`` AT WRITE TIME, so ``resampled.csv``
    and ``instrument_summary.csv`` ended up holding ``inf`` while the manifest,
    computed from the un-rounded numbers, reported a finite ``baseline_mean``
    and ``rows_non_finite_reading: 0``. The file and the manifest disagreed,
    and the only warning anywhere was a numpy RuntimeWarning on stderr.

    Anything at or above ROUND_LIMIT would overflow the multiply, and at that
    magnitude a double has no fractional part left to round anyway, so those
    cells are passed through untouched. Everything else rounds as before —
    ordinary values are bit-for-bit unchanged.
    """
    values = series.to_numpy(dtype=float, copy=True)
    safe = np.isfinite(values) & (np.abs(values) < ROUND_LIMIT)
    if safe.all():
        return series.round(ROUND_TO)
    rounded = values.copy()
    rounded[safe] = np.round(values[safe], ROUND_TO)
    return pd.Series(rounded, index=series.index)


def write_csv(df, columns, name, non_finite_cells=None):
    # type: (pd.DataFrame, List[str], str, Optional[List[Dict[str, Any]]]) -> int
    """Write one output with EXACTLY the contract's columns, in order.

    Called even when ``df`` is empty, which is the point: a header-only CSV is
    a valid answer that downstream code can read without special cases.

    Two things happen to the float columns on the way out, and both exist
    because non-finite values were reaching these files while the manifest
    denied it:

    * rounding is done so that it cannot CREATE an infinity (see
      ``round_without_overflow``);
    * whatever is still ``inf``/``-inf`` after that — a variance or a slope
      computed from enormous-but-finite readings overflows all on its own — is
      written as an empty cell, the same way this capsule already writes a
      ``sd`` it could not compute, and recorded in ``non_finite_cells`` so the
      caller can warn about it and list it in the manifest. Substituting a
      value is allowed here; substituting it silently is not. ``NaN`` is left
      alone: a blank ``sd`` or ``slope_per_day`` is the documented answer for a
      single-reading instrument, not a surprise.
    """
    if df is None or len(df) == 0:
        out = pd.DataFrame(columns=columns)
    else:
        out = df.reindex(columns=columns)
        for col in out.columns:
            if pd.api.types.is_float_dtype(out[col]):
                rounded = round_without_overflow(out[col])
                infinite = np.isinf(rounded.to_numpy(dtype=float))
                n_infinite = int(infinite.sum())
                if n_infinite:
                    rounded = rounded.mask(infinite)
                    if non_finite_cells is not None:
                        non_finite_cells.append({
                            "file": name, "column": str(col),
                            "n_cells": n_infinite,
                        })
                out[col] = rounded
    path = RESULTS_DIR / name
    out.to_csv(str(path), index=False)
    log("wrote {} ({} rows)".format(path, len(out)))
    return len(out)


def non_finite_cells_warning(cells):
    # type: (List[Dict[str, Any]]) -> str
    """The data_warnings entry for inf cells that were written as blank."""
    listed = "; ".join(
        "{} column '{}' ({} cell(s))".format(c["file"], c["column"], c["n_cells"])
        for c in cells)
    return (
        "{} output column(s) contained an infinite value and were written as "
        "BLANK rather than as 'inf': {}. An infinity here comes from arithmetic "
        "on readings that are finite but enormous (a variance or a trend fit "
        "that overflows a double), so the input rows are not counted in "
        "rows_non_finite_reading — see non_finite_cells_blanked for exactly "
        "which cells were substituted".format(len(cells), listed))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    # type: (Optional[List[str]]) -> int
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Run parameters, straight off the command line ---------------------
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    warns = []  # type: List[str]
    # Warnings about the DATA rather than about a parameter. They are kept
    # apart because `parameter_warnings` is the contract's block — one entry
    # per parameter value that was rejected, clamped or truncated — and a
    # duplicated column header is not a parameter problem. Both lists reach the
    # manifest and the run log.
    data_warnings = []  # type: List[str]
    # `notes` is for things that changed the run but rejected nothing (a blank
    # value falling back to its default, no data to pick a baseline from). They
    # are not warnings, but they still explain a manifest value that would
    # otherwise look unexplained, so they are recorded rather than dropped.
    notes = []  # type: List[str]

    raw_params, supplied, param_source = parse_parameters(argv, warns)
    # parse_parameters logs anything it appends, so the loop further down must
    # start after those or the same text would print twice.
    n_logged = len(warns)

    # `supplied` counts only the tokens that were actually understood, so this
    # line can no longer claim a value was supplied right after announcing that
    # the argument list was discarded.
    if supplied:
        log("run parameters honoured: {}".format(", ".join(sorted(supplied))))
    elif argv:
        log("no usable run parameters in the argument list — using the App "
            "Panel defaults")
    else:
        log("no run parameters supplied — using the App Panel defaults")

    # A blank value is not a rejection, so it is not a warning — but it does
    # make `effective_parameters` differ from what `parameters` records, and
    # every such difference owes the reader an explanation somewhere.
    for param_name, label, default in PARAM_SPECS:
        if default and not raw_params[param_name].strip():
            notes.append("{} was left blank — using the default {}".format(
                label, default))
            log(notes[-1])

    interval = parse_interval(raw_params["resample_interval"], warns, notes)
    rolling_window = parse_positive_int(
        raw_params["rolling_window"], "rolling_window", DEFAULT_ROLLING_WINDOW, warns,
        maximum=MAX_ROLLING_WINDOW)
    anomaly_z = parse_positive_float(
        raw_params["anomaly_z"], "anomaly_z", DEFAULT_ANOMALY_Z, warns,
        maximum=MAX_ANOMALY_Z)
    top_n = parse_positive_int(
        raw_params["top_n_anomalies"], "top_n_anomalies", DEFAULT_TOP_N, warns,
        maximum=MAX_TOP_N)
    freq = resolve_freq_alias(interval)
    log("interval={} (pandas freq '{}'), rolling_window={}, anomaly_z={}, top_n={}".format(
        interval, freq, rolling_window, anomaly_z, top_n))
    # Say what was rejected right here, next to the values actually in force,
    # rather than only at the end of the run: a parameter that was thrown out
    # has to be visible before its effect is. `n_logged` marks how far the log
    # has got, so the later warnings (a baseline instrument that is not in the
    # data) print exactly once. All of them land in parameter_warnings.
    for warning in warns[n_logged:]:
        log("warning: {}".format(warning))
    n_logged = len(warns)

    # --- Find the readings -------------------------------------------------
    log("scanning {} recursively for CSV files...".format(DATA_DIR))
    csvs = find_csvs(DATA_DIR)
    manifests = find_manifests(DATA_DIR, notes)
    log("found {} CSV file(s), {} upstream manifest(s)".format(len(csvs), len(manifests)))

    picked = pick_readings(csvs, DATA_DIR, notes)
    source_path = picked["path"]
    raw_df = picked["frame"]
    dropped_columns = picked["dropped_columns"]
    rejected = picked["rejected"]
    not_chosen = picked["not_chosen"]
    unreadable = picked["unreadable"]
    rows_in = 0
    rows_unusable = 0
    rows_non_finite_reading = 0
    rows_unknown_instrument = 0
    rows_literal_unknown = 0
    ts_info = empty_timestamp_info()
    # An input CSV nobody could read used to leave this run without a trace:
    # still in `input_files` (so it looked consumed), absent from every count,
    # and with `data_warnings` empty. Step 1 has carried these under
    # `unreadable_input_files` since an earlier fix; this is the same key, the
    # same record shape and the same warning shape, one step later.
    if unreadable:
        data_warnings.append(unreadable_files_warning(unreadable))
        log("warning: {}".format(data_warnings[-1]))
    # A file with a `reading` column that this step REFUSED is not the same
    # thing as no readings at all, and saying nothing about it turned "I would
    # not take your data" into "there was no data" — header-only outputs,
    # source_file null, instruments []. Name it, name what it was missing.
    if rejected:
        data_warnings.append(rejected_readings_warning(rejected))
        log("warning: {}".format(data_warnings[-1]))
    # ...and a file that had everything this step needs but LOST the selection
    # is the same defect in a quieter form: the outputs describe one file and
    # nothing says which of several it was. See pick_readings.
    #
    # Warning or note depends on what lost. A rival READINGS file is the
    # dangerous case (that is how this step came to analyse a stale archive
    # while step 1 filtered the live file), so it warns. Step 1's own
    # qc_flags.csv sitting beside its clean_readings.csv is not a rival at all
    # — it is the expected shape of a captured step-1 result, and every chained
    # run in this demo mounts one — so it is recorded as a note. Both end up in
    # the run log and in `readings_candidates_not_chosen` either way; only the
    # severity differs, because a warning that fires on every healthy run
    # teaches the operator to ignore warnings.
    if not_chosen:
        message = not_chosen_readings_warning(
            str(source_path.relative_to(DATA_DIR)) if source_path else None,
            not_chosen)
        rivals = [r for r in not_chosen if not r["is_qc_output"]]
        if rivals:
            data_warnings.append(message)
            log("warning: {}".format(data_warnings[-1]))
        else:
            notes.append(message)
            log("note: {}".format(notes[-1]))
    if raw_df is None:
        log("no CSV with a 'reading' column found — writing empty (header-only) outputs")
        # The manifest owes the reader this sentence too. `source_file: null`
        # with `rows_in: 0` and an empty `instruments` list is indistinguishable
        # from "there was nothing in the input", which is exactly the wrong
        # conclusion when the readings file was merely unreadable or refused.
        notes.append(
            "no CSV with the required columns ({}) could be used from {} — "
            "every output below is header-only{}{}".format(
                ", ".join(REQUIRED_INPUT_COLUMNS), DATA_DIR,
                "" if not unreadable else
                "; note that {} input CSV file(s) could not be READ at all, so "
                "the readings may well be among them — see "
                "unreadable_input_files".format(len(unreadable)),
                "" if not rejected else
                "; {} file(s) with a 'reading' column were refused for missing "
                "columns — see rejected_readings_files".format(len(rejected))))
        log("note: {}".format(notes[-1]))
        readings = pd.DataFrame(columns=REQUIRED_INPUT_COLUMNS)
    else:
        log("readings file: {} ({} rows)".format(
            source_path.relative_to(DATA_DIR), len(raw_df)))
        # Colliding headers are resolved, not fatal — but the choice is one the
        # operator has to be able to see, so it is logged here, warned about in
        # the manifest, and listed column by column under
        # `dropped_duplicate_columns`.
        if dropped_columns:
            data_warnings.append(duplicate_columns_warning(
                str(source_path.relative_to(DATA_DIR)), dropped_columns))
            log("warning: {}".format(data_warnings[-1]))
        rows_in = len(raw_df)
        (readings, rows_unusable, rows_non_finite_reading, ts_info,
         rows_unknown_instrument,
         rows_literal_unknown) = normalize_readings(raw_df)
        if rows_unusable:
            # `notes`, not just `log`: `rows_unusable` is a bare number in the
            # manifest, and a dropped row that only the stdout log accounts for
            # is not accounted for in the artifact anyone reads afterwards.
            notes.append(
                "dropped {} of {} input row(s) with an unparseable or "
                "implausible timestamp, or an unusable reading or "
                "instrument_id — they are counted in rows_unusable and are NOT "
                "in resampled.csv, instrument_summary.csv or "
                "anomalies.csv".format(rows_unusable, rows_in))
            log("note: {}".format(notes[-1]))
        # Timestamps that carried a zone were moved onto one UTC timeline. That
        # SHIFTS every bucket boundary relative to the original local text, so
        # it is a change to the result and cannot be left to a code comment.
        if ts_info["n_with_offset"]:
            data_warnings.append(
                "{} timestamp(s) carried an explicit UTC offset ({}) — every "
                "timestamp was converted to UTC and the zone dropped so the "
                "whole column shares one tz-naive timeline. The bucket "
                "boundaries in resampled.csv are therefore UTC, not the "
                "original local times. Without this, a column of MIXED offsets "
                "parses to an object dtype that cannot be resampled at "
                "all.".format(
                    ts_info["n_with_offset"], ", ".join(ts_info["offsets"])))
            log("warning: {}".format(data_warnings[-1]))
        # How the timestamp text was INTERPRETED, when that was a choice rather
        # than a reading. Bare numbers are read as nanoseconds since the epoch
        # and slash-dates have their day/month order inferred per value — both
        # move every bucket boundary, every coverage figure and every slope in
        # the outputs, and both used to happen with nothing said anywhere.
        if ts_info["format"]["unambiguous_iso8601"] is False:
            data_warnings.append(timestamp_format_warning(
                ts_info["format"],
                "Every bucket boundary in resampled.csv, every timestamp in "
                "anomalies.csv and every slope_per_day in "
                "instrument_summary.csv follows from it."))
            log("warning: {}".format(data_warnings[-1]))
        if ts_info["dtype_fallback"]:
            data_warnings.append(
                "the timestamp column could not be parsed to a datetime dtype "
                "(got {}) — every timestamp is treated as unusable, so no row "
                "could be bucketed. Nothing was written from that column; the "
                "outputs below are header-only rather than "
                "missing".format(ts_info["dtype_fallback"]))
            log("warning: {}".format(data_warnings[-1]))
        # A SUBSET of rows_unusable: an instant so far from the rest that it is
        # a sentinel or a typo, not a measurement. Dropping it silently would
        # be indistinguishable from the crash it replaces.
        if ts_info["n_implausible"]:
            data_warnings.append(
                "{} of those row(s) have a timestamp outside the plausible "
                "range [{} .. {}) — a sentinel or a typo rather than a "
                "measurement, and one of them stretches an instrument's span "
                "across centuries. They are dropped like an unparseable "
                "timestamp and counted here and in the manifest as "
                "rows_implausible_timestamp. Every OTHER row of the same "
                "instrument is analysed normally, so an instrument disappears "
                "from the outputs only if it has no usable timestamp "
                "left.".format(
                    ts_info["n_implausible"],
                    PLAUSIBLE_MIN_TIMESTAMP.isoformat(),
                    PLAUSIBLE_MAX_TIMESTAMP.isoformat()))
            log("warning: {}".format(data_warnings[-1]))
        # A subset of rows_unusable, called out separately: "inf" survives
        # pd.to_numeric as a real float, so without this guard it would reach
        # resampled.csv and instrument_summary.csv as an unusable `inf` and
        # blank out mean_vs_baseline for every instrument. Dropping it silently
        # would be the same defect one step earlier, so it is warned about.
        if rows_non_finite_reading:
            data_warnings.append(
                "{} of those row(s) had a reading of inf/-inf rather than a "
                "finite number — they are dropped like a blank reading, so no "
                "non-finite value reaches resampled.csv or "
                "instrument_summary.csv".format(rows_non_finite_reading))
            log("warning: {}".format(data_warnings[-1]))
        # Rows with no instrument_id used to invent an instrument called "nan"
        # and report it as real. They are relabelled instead — the same label
        # step 1 uses for the same rows — and the substitution is said out loud.
        if rows_unknown_instrument:
            data_warnings.append(
                "{} row(s) have no instrument_id — they were NOT dropped and "
                "they are NOT an instrument called \"nan\": they were "
                "relabelled to \"{}\" and analysed as one instrument, the same "
                "label step 1 gives the same rows. {}".format(
                    rows_unknown_instrument, UNKNOWN_INSTRUMENT,
                    "That id in `instruments` and in instrument_summary.csv is "
                    "this capsule's label, not a value from the input file"
                    if not rows_literal_unknown else
                    "CAREFUL: {} row(s) of the input already carried the "
                    "literal id \"{}\" themselves, so that label does NOT mean "
                    "\"this capsule's substitute\" here — the two groups "
                    "COLLIDED and were analysed as one instrument of {} row(s), "
                    "and no output can tell them apart. See "
                    "n_rows_with_literal_unknown_instrument_id".format(
                        rows_literal_unknown, UNKNOWN_INSTRUMENT,
                        rows_literal_unknown + rows_unknown_instrument)))
            log("warning: {}".format(data_warnings[-1]))
        elif rows_literal_unknown:
            # No substitution happened, but `unknown_instrument_label` in the
            # manifest would otherwise invite the opposite misreading: that
            # this capsule invented the instrument. It did not — the file did.
            notes.append(
                "{} row(s) carry the literal instrument_id \"{}\", which is "
                "also the label this capsule gives rows with a MISSING "
                "instrument_id. No row was relabelled in this run "
                "(rows_unknown_instrument is 0), so that instrument is the "
                "input's own value".format(
                    rows_literal_unknown, UNKNOWN_INSTRUMENT))
            log("note: {}".format(notes[-1]))

    instruments = sorted(readings["instrument_id"].unique().tolist()) if len(readings) else []
    baseline = resolve_baseline(raw_params["baseline_instrument"], instruments, warns)

    # `baseline_instrument` is the one parameter whose effective value routinely
    # differs from the string that was sent (blank means "first alphabetically",
    # and an empty input means "none at all"). Say which happened, so no reader
    # has to guess whether the operator chose this instrument or the code did.
    raw_baseline = raw_params["baseline_instrument"].strip()
    baseline_note = None  # type: Optional[str]
    if not raw_baseline:
        baseline_note = (
            "{} was left blank — using the first instrument alphabetically "
            "({})".format(PARAM_LABELS["baseline_instrument"], baseline)
            if baseline else
            "{} was left blank and there are no usable readings — the "
            "effective baseline_instrument is null".format(
                PARAM_LABELS["baseline_instrument"]))
    elif not baseline:
        baseline_note = (
            "there are no usable readings, so {} \"{}\" could not be used — "
            "the effective baseline_instrument is null".format(
                PARAM_LABELS["baseline_instrument"], raw_baseline))
    elif baseline != raw_baseline and baseline.lower() == raw_baseline.lower():
        baseline_note = (
            "{} \"{}\" matched instrument {} (instrument ids are compared "
            "case-insensitively)".format(
                PARAM_LABELS["baseline_instrument"], raw_baseline, baseline))
    if baseline_note:
        notes.append(baseline_note)
        log(baseline_note)

    # --- Resample + smooth, per instrument ---------------------------------
    bucket_frames = []  # type: List[pd.DataFrame]
    # Instruments the anomaly rule could not be applied to at all, and how many
    # buckets it WAS applied to across the run. Both exist because "0 anomalies
    # because nothing stood out" and "0 anomalies because nothing was ever
    # scored" used to be the same manifest, and the second one is silent rule
    # suppression — the contract violation this pipeline cares most about.
    anomaly_skipped = []  # type: List[Dict[str, Any]]
    n_buckets_scored = 0
    for instrument_id, group in readings.groupby("instrument_id", sort=True):
        buckets = resample_instrument(group, freq, rolling_window)
        buckets["instrument_id"] = instrument_id
        # residual: how far this bucket sits from its own smoothed trend line.
        buckets["residual"] = buckets["mean"] - buckets["rolling_mean"]
        # ...scored per instrument, so a noisy instrument is judged against its
        # own noise rather than against the quietest one in the fleet.
        #
        # The first rolling_window-1 buckets are scored as NaN (never flagged).
        # min_periods=1 gives them a rolling mean built from fewer points, so
        # their residual is small BY CONSTRUCTION, not because the instrument
        # behaved. On a steadily drifting instrument every later residual sits
        # at the same non-zero trend lag, which would make those warm-up rows
        # look like enormous outliers — the demo's drifting instrument is
        # legitimate signal and must surface as a slope, not as an anomaly.
        # resampled.csv still carries the full-length rolling mean; only the
        # anomaly scoring ignores the warm-up.
        #
        # What the warm-up must never do is disable the rule in silence. If
        # `rolling_window` is as long as the instrument's whole series — 100
        # against 84 buckets, an ordinary value far below MAX_ROLLING_WINDOW,
        # so no clamp warning fires — then EVERY bucket is warm-up and the
        # instrument is never scored at all. That produced a header-only
        # anomalies.csv with n_anomalies_flagged 0 and not one warning or note
        # anywhere: a rule silently switched off, which this pipeline treats as
        # a contract violation rather than untidiness. So the case is detected
        # here, per instrument, named in the run log, and carried into the
        # manifest as `anomaly_rule_skipped_instruments`.
        #
        # Clamping the warm-up to fit a short series was considered and
        # rejected: it would score buckets whose rolling mean is an average of
        # however few points came before them, and on the demo's own drifting
        # instrument that manufactures a run of large residuals out of
        # legitimate signal — turning "the rule could not run" into "the rule
        # ran and lied". Not scoring, loudly, is the honest answer.
        warm_up = max(rolling_window - 1, 0)
        scored = buckets.index >= warm_up
        n_scored = int(scored.sum())
        buckets["robust_z"] = np.nan
        skip_reason = None  # type: Optional[str]
        if n_scored:
            residuals = buckets.loc[scored, "residual"]
            buckets.loc[scored, "robust_z"] = robust_z(residuals)
            if not has_robust_scale(residuals):
                # robust_z answers a zero MAD with a column of zeros, so this
                # instrument can never reach any positive anomaly_z. Same
                # distinction as above with a different cause: no scale rather
                # than no buckets. Step 1 reports the same thing about its own
                # spike rule under `spike_rule_skipped_instruments`.
                skip_reason = (
                    "its {} scored bucket(s) have a residual MAD of 0, so no "
                    "robust z exists to compare against anomaly_z".format(
                        n_scored))
        elif len(buckets):
            skip_reason = (
                "the rolling-window warm-up excludes the first {} bucket(s) "
                "and this instrument has only {}".format(warm_up, len(buckets)))
        if skip_reason is not None:
            anomaly_skipped.append({
                "instrument_id": instrument_id,
                "n_buckets": int(len(buckets)),
                "n_buckets_scored": n_scored,
                "warmup_buckets_excluded": warm_up,
                "reason": skip_reason,
            })
            log("{}: anomaly rule skipped — {}".format(
                instrument_id, skip_reason))
        n_buckets_scored += n_scored
        bucket_frames.append(buckets)
        log("{}: {} reading(s) -> {} bucket(s) of {} ({} anomaly-scored)".format(
            instrument_id, len(group), len(buckets), interval, n_scored))

    if bucket_frames:
        resampled = pd.concat(bucket_frames, ignore_index=True)
        resampled = resampled.sort_values(["instrument_id", "timestamp"]).reset_index(drop=True)
        resampled["n"] = resampled["n"].astype(int)
    else:
        resampled = pd.DataFrame(
            columns=RESAMPLED_COLUMNS + ["residual", "robust_z"])

    # An instrument the rule could not be applied to is announced BEFORE the
    # anomaly count below, so "0 bucket(s) at |robust z| >= 3.0" is never read
    # as a measurement when it is really the absence of one.
    if anomaly_skipped:
        data_warnings.append(anomaly_rule_skipped_warning(
            anomaly_skipped, rolling_window, interval, n_buckets_scored,
            len(instruments)))
        log("warning: {}".format(data_warnings[-1]))

    # --- Anomalies: biggest departures from the smoothed trend -------------
    if len(resampled):
        flagged = resampled[resampled["robust_z"].abs() >= anomaly_z].copy()
        n_flagged = len(flagged)
        # "Top N" is global: the N largest departures anywhere in the dataset,
        # not N per instrument, so the table the app renders is a leaderboard.
        flagged = flagged.reindex(
            flagged["robust_z"].abs().sort_values(ascending=False).index)
        anomalies = flagged.head(top_n).copy()
    else:
        n_flagged = 0
        anomalies = pd.DataFrame(columns=RESAMPLED_COLUMNS + ["residual", "robust_z"])
    log("anomalies: {} bucket(s) at |robust z| >= {} out of {} scored bucket(s) "
        "of {} (the rest are rolling-window warm-up), keeping the top "
        "{}".format(n_flagged, anomaly_z, n_buckets_scored, len(resampled),
                    min(top_n, n_flagged)))

    # anomalies.csv reports the bucket mean as `reading` and the rolling mean
    # as `expected` — "what we saw" vs "what the trend said to expect".
    if len(anomalies):
        anomalies_out = pd.DataFrame({
            "timestamp": anomalies["timestamp"].map(iso),
            "instrument_id": anomalies["instrument_id"],
            "reading": anomalies["mean"],
            "expected": anomalies["rolling_mean"],
            "residual": anomalies["residual"],
            "robust_z": anomalies["robust_z"],
        })
        kept_counts = anomalies["instrument_id"].value_counts().to_dict()
    else:
        anomalies_out = pd.DataFrame(columns=ANOMALY_COLUMNS)
        kept_counts = {}

    # --- Per-instrument summary --------------------------------------------
    summary_rows = []  # type: List[Dict[str, Any]]
    for instrument_id, group in readings.groupby("instrument_id", sort=True):
        summary_rows.append(summarize_instrument(
            instrument_id, group, kept_counts.get(instrument_id, 0)))
    # The baseline parameter earns its place here: it re-centres mean_vs_baseline.
    baseline_mean = add_baseline_comparison(summary_rows, baseline, notes)
    if summary_rows:
        log("baseline instrument: {} (mean {}) — mean_vs_baseline is measured "
            "from it".format(
                baseline or "(none)",
                "n/a" if baseline_mean is None else round(baseline_mean, ROUND_TO)))
    # instrument_summary.csv explains every other value it prints, so the blank
    # cells owe the reader the same courtesy. An instrument with ONE reading
    # has no sample sd and no line to fit; an instrument whose readings all
    # share one timestamp has no time axis to fit against. Both are ordinary
    # data, both come out as empty cells, and neither used to say anything at
    # all — which reads as a bug in the capsule rather than a property of the
    # input.
    degenerate = summarize_degenerate_instruments(summary_rows)
    for entry in degenerate:
        notes.append(entry)
        log("note: {}".format(entry))
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    for row in summary_rows:
        log("{}: slope {:+.4f}/day (r2 {:.3f}), {} anomaly row(s), "
            "mean_vs_baseline {:+.4f}".format(
                row["instrument_id"], row["slope_per_day"], row["r2"],
                row["n_anomalies"], row["mean_vs_baseline"]))

    # --- Write the deliverables --------------------------------------------
    if len(resampled):
        resampled_out = resampled.copy()
        resampled_out["timestamp"] = resampled_out["timestamp"].map(iso)
    else:
        resampled_out = pd.DataFrame(columns=RESAMPLED_COLUMNS)
    # Filled in by write_csv: any inf cell it had to substitute for a blank.
    non_finite_cells = []  # type: List[Dict[str, Any]]
    n_buckets = write_csv(resampled_out, RESAMPLED_COLUMNS, "resampled.csv",
                          non_finite_cells)
    write_csv(summary, SUMMARY_COLUMNS, "instrument_summary.csv",
              non_finite_cells)
    n_anomalies = write_csv(anomalies_out, ANOMALY_COLUMNS, "anomalies.csv",
                            non_finite_cells)
    # The manifest is computed from the UN-rounded numbers, so it used to say
    # baseline_mean 1.67e307 while the CSV beside it said inf. Now the CSV
    # never holds an infinity and this says which cells were changed to keep
    # that true.
    if non_finite_cells:
        data_warnings.append(non_finite_cells_warning(non_finite_cells))
        log("warning: {}".format(data_warnings[-1]))

    for warning in warns[n_logged:]:
        log("warning: {}".format(warning))

    # --- Manifest: what this step consumed, produced and was asked to do ---
    # The standard three-key parameter block, identical in every capsule of
    # this demo, so one reader handles them all:
    #   "parameters"           — exactly what this run RECEIVED, raw strings,
    #                            including the defaults nobody touched. Matches
    #                            the computation's own `app_parameters`.
    #   "effective_parameters" — what the run actually USED after coercion, so
    #                            a machine can see that anomaly_z "abc" became
    #                            3.0, or that a value was skipped (null),
    #                            rather than assuming the typo was obeyed.
    #   "parameter_warnings"   — one entry per rejected, clamped or truncated
    #                            value, the same text the run log printed. That
    #                            agreement between log, manifest and
    #                            app_parameters IS the provenance this demo is
    #                            selling.
    # Plus a fourth key that keeps the other three honest:
    #   "parameters_source"    — the RAW argument list, verbatim, whether or not
    #                            it parsed (argv / argv_parsed /
    #                            parameters_supplied / ignored_tokens).
    #                            `parameters` alone cannot express "you sent
    #                            --anomaly_z=1.5 and I could not use it",
    #                            because it holds the DEFAULT in that slot; a
    #                            reader would take that default for the
    #                            operator's choice. For a provenance demo, a
    #                            manifest that quietly contradicts the caller is
    #                            worse than a crash.
    # ...and `notes`, for the things that changed the run but rejected nothing.
    effective_parameters = {
        "resample_interval": interval,
        "rolling_window": rolling_window,
        "anomaly_z": anomaly_z,
        "top_n_anomalies": top_n,
        # null when there was no data to pick a baseline from.
        "baseline_instrument": baseline or None,
    }
    manifest = {
        "step": "analysis",
        "parameters": dict(raw_params),
        "effective_parameters": effective_parameters,
        "parameter_warnings": list(warns),
        # Warnings about the input data rather than about a parameter value —
        # currently only colliding column headers, which are resolved rather
        # than fatal but must never be resolved silently.
        "data_warnings": list(data_warnings),
        # One entry per column dropped because its name collided with an
        # earlier one after trimming/lower-casing. Empty on a normal run.
        "dropped_duplicate_columns": list(dropped_columns),
        "parameters_source": {
            "argv": list(param_source.get("argv", [])),
            "argv_parsed": bool(param_source.get("argv_parsed", True)),
            "parameters_supplied": list(param_source.get("parameters_supplied", [])),
            "ignored_tokens": list(param_source.get("ignored_tokens", [])),
            # Tokens argparse understood but a later token of the same name
            # overrode. `ignored_tokens []` used to be the only statement on
            # the subject, and it read as "nothing was dropped".
            "superseded_tokens": list(param_source.get("superseded_tokens", [])),
        },
        "notes": list(notes),
        "parameters_supplied": sorted(supplied),
        "pandas_freq_alias": freq,
        # The rolling mean's warm-up buckets are smoothed but not anomaly-scored
        # (see resample_instrument) — recorded so the numbers are reproducible.
        "warmup_buckets_excluded_per_instrument": max(rolling_window - 1, 0),
        # How many buckets the anomaly rule actually SCORED, across every
        # instrument. This is the key that makes `n_anomalies_flagged: 0`
        # readable: 0 flagged out of 233 scored is a measurement, 0 flagged out
        # of 0 scored is a rule that never ran. Before it existed, a
        # rolling_window longer than an instrument's series (100 against the
        # demo's 84/80/78 buckets — nowhere near the clamp, so no clamp
        # warning) silently scored nothing and reported it as "no anomalies".
        "n_buckets_anomaly_scored": int(n_buckets_scored),
        # One record per instrument the rule could NOT be applied to, with the
        # bucket count and the reason (warm-up longer than the series, or a
        # zero residual MAD). A `n_anomalies` of 0 in instrument_summary.csv
        # for an instrument named here means "never scored", not "nothing
        # unusual". The matching data_warnings entry says the same in words.
        "anomaly_rule_skipped_instruments": list(anomaly_skipped),
        "baseline_instrument": baseline,
        # The number instrument_summary.csv's mean_vs_baseline was measured
        # from, so that column can be re-derived from the manifest alone. null
        # only when there was no usable baseline at all, and a `notes` entry
        # always says which case it was — a null here with no explanation would
        # take the key's stated purpose away without saying so.
        "baseline_mean": (None if baseline_mean is None
                          else round(float(baseline_mean), ROUND_TO)),
        "source_file": (str(source_path.relative_to(DATA_DIR))
                        if source_path is not None else None),
        "input_files": [str(p.relative_to(DATA_DIR)) for p in csvs],
        # Files that HAVE a reading column but were refused, and what they were
        # missing. Non-empty alongside `source_file: null` is the difference
        # between "there were no readings" and "there were readings this step
        # would not take" — a difference the manifest used to hide completely.
        "rejected_readings_files": list(rejected),
        # Files that had EVERYTHING this step requires and still were not the
        # one analysed, with the reason each lost. `source_file` alone cannot
        # express "there were three readings files and I took this one", and
        # while that was unrecorded this step could analyse a stale archive
        # while step 1 filtered the live file, both exiting 0. Step 1 records
        # the same list for the same mount under the same key.
        "readings_candidates_not_chosen": list(not_chosen),
        "readings_file_chosen_by": picked["chosen_reason"],
        # Input CSVs that could not be read at all: name and why. Same key,
        # same record shape as step 1. Without it, an unreadable readings.csv
        # left the manifest positively asserting that the input held no
        # readings. This step copies no input file into /results, so unlike
        # step 1's version there is no `passed_through` to report.
        "unreadable_input_files": list(unreadable),
        "required_input_columns": list(REQUIRED_INPUT_COLUMNS),
        "rows_in": int(rows_in),
        "rows_unusable": int(rows_unusable),
        # A SUBSET of rows_unusable, not an extra bucket, so the accounting
        # rows_in = rows_analyzed + rows_unusable still holds.
        "rows_non_finite_reading": int(rows_non_finite_reading),
        # Likewise a SUBSET of rows_unusable: rows whose timestamp parsed but
        # landed outside `plausible_timestamp_range`. One such value turns a
        # 12-hour series into a 326-year one, which is not a time series and
        # overflows the int64 nanosecond arithmetic behind every span, so the
        # row is dropped like an unparseable timestamp — and counted here,
        # because a drop nobody can see is worse than the crash it replaces.
        "rows_implausible_timestamp": int(ts_info["n_implausible"]),
        "plausible_timestamp_range": [
            PLAUSIBLE_MIN_TIMESTAMP.isoformat(),
            PLAUSIBLE_MAX_TIMESTAMP.isoformat(),
        ],
        # true = at least one raw timestamp carried a UTC offset (or "Z") and
        # was converted to UTC with the zone dropped, so every bucket boundary
        # in resampled.csv is UTC rather than the original local time. More
        # than one entry in `utc_offsets_seen` is the MIXED-offset case, which
        # has no single dtype and used to end the run at `resample`.
        "timestamps_normalized_to_utc": bool(ts_info["n_with_offset"]),
        "n_timestamps_with_utc_offset": int(ts_info["n_with_offset"]),
        "utc_offsets_seen": list(ts_info["offsets"]),
        # How the timestamp text was INTERPRETED. `unambiguous_iso8601: false`
        # means the run had to guess something the file does not state — bare
        # numbers read as nanoseconds since the epoch, or slash-dates whose
        # day/month order is inferred per value — and every bucket boundary
        # below rests on that guess. `null` means there was no column to look
        # at. See describe_timestamp_format; the matching data_warnings entry
        # says the same thing in words.
        "timestamp_interpretation": dict(ts_info["format"]),
        # NOT a subset of rows_unusable: these rows were kept and analysed,
        # under the label below rather than under an instrument called "nan".
        # Step 1 uses the same label for the same rows, so both capsules report
        # the same instrument list for one input file.
        "rows_unknown_instrument": int(rows_unknown_instrument),
        "unknown_instrument_label": UNKNOWN_INSTRUMENT,
        # Rows whose instrument_id in the INPUT was already the literal label
        # above. Non-zero together with a non-zero rows_unknown_instrument is a
        # COLLISION: relabelled rows and real ones merged into one instrument
        # that no output can separate, and the warning about the relabelling
        # would otherwise assert that the id is this capsule's invention when
        # part of it came from the file. Step 1 reports the same two keys.
        "n_rows_with_literal_unknown_instrument_id": int(rows_literal_unknown),
        "unknown_instrument_label_collision": bool(
            rows_literal_unknown and rows_unknown_instrument),
        "rows_analyzed": int(len(readings)),
        # Cells that were infinite after rounding and were written as BLANK.
        # `rows_non_finite_reading` only covers readings that arrived
        # non-finite; an infinity manufactured by the arithmetic (or, before
        # the fix, by pandas' round()) is a different thing and needs its own
        # line, or the manifest ends up denying what is in the CSV.
        "non_finite_cells_blanked": list(non_finite_cells),
        "n_buckets": int(n_buckets),
        "instruments": instruments,
        "n_anomalies_flagged": int(n_flagged),
        "n_anomalies_kept": int(n_anomalies),
        "upstream_manifests": [
            {"path": rel, "contents": contents} for rel, contents in manifests
        ],
        "outputs": ["resampled.csv", "instrument_summary.csv", "anomalies.csv"],
        "generated_at": generated_at,
    }
    # allow_nan=False is a backstop, not the fix: the parsers above already
    # reject every non-finite parameter value. But Python's default is to
    # write bare `NaN`/`Infinity` literals, which are NOT valid JSON — strict
    # parsers and JSON.parse both refuse them, so the app reading this file
    # would fail on a manifest that looked fine. If a future edit ever lets one
    # through, this fails loudly here instead. Serializing to a string first
    # means such a failure cannot leave a half-written manifest behind.
    text = json.dumps(manifest, indent=2, default=str, allow_nan=False)
    manifest_path = RESULTS_DIR / "manifest.json"
    with open(str(manifest_path), "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.write("\n")
    log("wrote {}".format(manifest_path))

    # Always 0: an empty input, a stuck sensor and a typo'd parameter are all
    # legitimate outcomes here, and each one still produced valid CSVs above.
    log("done: {} instrument(s), {} bucket(s), {} anomaly row(s)".format(
        len(instruments), n_buckets, n_anomalies))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - a real bug should be loud, not silent
        traceback.print_exc()
        sys.exit(1)
