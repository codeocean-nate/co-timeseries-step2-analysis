# Time-series demo — step 2: analysis (plot data, no pictures)

Part of the [Code Ocean time-series orchestration demo](https://github.com/codeocean-nate/co-timeseries-app)
— its [SETUP.md](https://github.com/codeocean-nate/co-timeseries-app/blob/main/SETUP.md)
sets the whole demo up on any Code Ocean deployment.
Resamples instrument readings into fixed time buckets, smooths them, fits a per-instrument
trend and flags the buckets that depart from it — then writes the results as **plot-ready
CSVs**. This capsule draws nothing: no images, no HTML, no plotting library. The
orchestrator app downloads these CSVs and renders the interactive charts, which keeps the
capsule tiny and the visualization genuinely interactive.

- **Environment**: Python 3.9+ with pip packages `pandas`, `numpy`.
- **Input**: a readings data asset. `/data` is scanned recursively (the mount name is not
  load-bearing, and the `.csv` extension is matched ignoring case); `clean_readings.csv`
  from step 1 is preferred, then `readings.csv`, then the largest remaining candidate —
  **the same order step 1 uses**, so the two capsules never analyse different files, and
  this capsule still runs straight on the raw readings asset with step 1 skipped. Files
  without a `reading` column (e.g. `instruments.csv`) are ignored, not fatal. A file that
  **has** a `reading` column but is missing `timestamp` or `instrument_id` is refused — and
  said so out loud, see [Readings this step refuses](#readings-this-step-refuses); every
  readings-shaped file that was not chosen is listed in the manifest, see
  [Choosing the readings file](#choosing-the-readings-file-the-same-way-step-1-does).
- **Output**: `/results/resampled.csv`, `/results/instrument_summary.csv`,
  `/results/anomalies.csv`, `/results/manifest.json`.
- **Local test**: `DATA_DIR=<dir with .csv> RESULTS_DIR=<out dir> bash code/run`

## Outputs

| File | Columns |
|---|---|
| `resampled.csv` | `timestamp, instrument_id, mean, min, max, n, rolling_mean` |
| `instrument_summary.csv` | `instrument_id, n, mean, sd, min, max, slope_per_day, r2, first_timestamp, last_timestamp, n_anomalies, mean_vs_baseline` |
| `anomalies.csv` | `timestamp, instrument_id, reading, expected, residual, robust_z` |
| `manifest.json` | `step`, `parameters` / `effective_parameters` / `parameter_warnings` / `parameters_source` / `notes` / `data_warnings`, `n_buckets` / `n_buckets_anomaly_scored` / `anomaly_rule_skipped_instruments` / `warmup_buckets_excluded_per_instrument`, `instruments`, `baseline_instrument`, `baseline_mean`, `rows_in` / `rows_unusable` / `rows_non_finite_reading` / `rows_implausible_timestamp` / `rows_unknown_instrument` / `rows_analyzed`, `unknown_instrument_label` / `n_rows_with_literal_unknown_instrument_id` / `unknown_instrument_label_collision`, `source_file` / `readings_file_chosen_by` / `readings_candidates_not_chosen`, `input_files`, `unreadable_input_files`, `rejected_readings_files` / `required_input_columns`, `non_finite_cells_blanked`, `timestamps_normalized_to_utc` / `n_timestamps_with_utc_offset` / `utc_offsets_seen` / `plausible_timestamp_range` / `timestamp_interpretation`, `dropped_duplicate_columns`, upstream manifest contents, `generated_at` |

- `resampled.csv` is one row per instrument per non-empty bucket, so `n >= 1` always.
  Buckets containing no readings (a coverage gap) are omitted rather than emitted as NaN.
- `instrument_summary.csv` is computed from the **raw** readings, so `mean`/`sd`/`min`/`max`
  and the trend fit describe the instrument itself and don't move when the bucket size
  changes. `slope_per_day` and `r2` come from a least-squares fit of reading against time
  **in days**, with `r2` as the squared Pearson correlation. Fewer than two readings, or
  readings all at one instant, give `NaN` rather than an error; a perfectly flat sensor
  gives slope `0` and `r2` `0`.

  Those `NaN`s become **blank cells** in the CSV, and every other value in this file is
  explained somewhere, so the blanks are too: an instrument with a single usable reading
  (no sample sd, no line to fit) and an instrument whose readings all share one timestamp
  (no time axis to fit against) each get a `notes` entry naming the instrument and the
  reason. An unexplained blank reads as a broken capsule rather than as a property of the
  input.
- `mean_vs_baseline` is this instrument's `mean` minus the baseline instrument's `mean`,
  and exactly `0` on the baseline row itself — "how far above or below the reference this
  instrument sits". `--baseline_instrument` picks the reference, so changing that parameter
  re-centres the column and genuinely changes the file. The baseline's own mean is in
  `manifest.json` as `baseline_mean`, so the column can be re-derived from the manifest.
  `baseline_mean` is `null` only when there was no usable baseline at all, and a `notes`
  entry always says which case it was — a bare `null` would quietly take away the one thing
  the key is for.
- **Row accounting.** `rows_in = rows_analyzed + rows_unusable`, always.
  `rows_non_finite_reading` and `rows_implausible_timestamp` are *subsets* of
  `rows_unusable` — never extra buckets — and count, respectively, the rows dropped because
  their reading was `inf`/`-inf` rather than a finite number, and the rows dropped because
  their timestamp fell outside `plausible_timestamp_range`. Each has a matching
  `data_warnings` entry, and a non-zero `rows_unusable` always has a `notes` entry
  accounting for it in words: nothing leaves the dataset without saying so.
  `rows_unknown_instrument` is *not* a subset of `rows_unusable` — those rows were kept,
  under a label (see below).
- **No non-finite value ever reaches these CSVs.** Floats are rounded to 6 decimals for
  readability, and pandas implements `.round(6)` as *multiply by 1e6, round, divide* — so a
  finite reading of `1e308` overflowed to `inf` **at write time**. The file then held `inf`
  while `manifest.json`, computed from the un-rounded numbers, reported a finite
  `baseline_mean` and `rows_non_finite_reading: 0`; the only complaint anywhere was a numpy
  `RuntimeWarning` on stderr. Rounding now skips magnitudes that would overflow (a double
  that large has no fractional part left to round anyway), and anything still infinite
  afterwards — a variance or a trend fit that overflows on its own — is written as a
  **blank** cell and recorded in `non_finite_cells_blanked` (`file`, `column`, `n_cells`)
  with a matching `data_warnings` entry. Substituting a value is allowed here; substituting
  it silently is not.
- **A row with no `instrument_id` is relabelled, not invented and not dropped.** It becomes
  `(unknown)` — `manifest.json` carries the literal as `unknown_instrument_label`, and
  **step 1 uses the same label for the same rows**, so running either capsule on one raw
  file reports the same instrument list. It used to become an instrument literally called
  `"nan"`: `astype(str)` ran *before* the blank guard, so a missing cell arrived as the
  perfectly non-empty string `nan`, the guard never fired, and `instruments` listed a
  fabricated sensor with `rows_unusable: 0`. The relabel is counted in
  `rows_unknown_instrument` and named in `data_warnings` — and if the input *itself*
  carries rows whose `instrument_id` is the literal `(unknown)`, the two groups collide
  into one instrument nothing can separate, so that is detected too
  (`n_rows_with_literal_unknown_instrument_id`, `unknown_instrument_label_collision`) and
  the warning says which rows came from the file instead of claiming they are all this
  capsule's label.
- `anomalies.csv` reports the bucket `mean` as `reading` and the `rolling_mean` as
  `expected` — what we saw versus what the smoothed trend said to expect. `n_anomalies` in
  the summary counts the rows each instrument contributed to this file.

## Readings this step refuses

This capsule needs all three of `required_input_columns` — `timestamp`, `instrument_id`,
`reading` — so a CSV with a `reading` column but no `timestamp` is skipped. That refusal
used to be a single stdout line and nothing more, which produced the worst possible
manifest: header-only outputs, `source_file: null`, `rows_in: 0`, `instruments: []` and not
one word about the file that had just been turned away. It reads as *"there were no
readings anywhere"* when the truth is *"I found readings and would not take them"* — and it
is reachable straight down the pipeline at **default settings**, because step 1 tolerates a
readings file with no `timestamp` column and writes a `clean_readings.csv` that inherits the
hole.

Every refusal is now recorded:

- a `data_warnings` entry naming each file, its row count and the columns it was missing,
  and stating explicitly that empty outputs here mean "nothing this step could read", not
  "no readings in the input";
- `rejected_readings_files` in the manifest, one record per file (`file`,
  `missing_columns`, `columns_present`, `rows`);
- `required_input_columns`, so the check can be reproduced without reading this file.

Step 1 warns from its own side too: when its `clean_readings.csv` is missing a column this
step needs, it says so in `warnings` and lists it under
`clean_readings_missing_downstream_columns`.

## Choosing the readings file (the same way step 1 does)

`/data` can hold several CSVs, and **this step and step 1 must not disagree about which one
is the readings file**. They used to. This step took the *first* CSV in sorted-path order
with the three required columns, so an `archive_2019_readings.csv` holding one stale row
beat the live 48-row `readings.csv` beside it on the alphabet alone: step 1 QC'd the real
file, step 2 analysed the archive, both exited 0, and neither manifest mentioned the file
it had passed over. For a demo whose whole subject is provenance, two capsules quietly
describing different data is the worst defect available.

The order is now step 1's, with this step's own canonical preference first:

1. a CSV must be **readable** and carry all of `required_input_columns` to be a candidate;
2. candidates carrying `qc_status`/`qc_reason` are **held back** — a previous QC run's
   output still contains the rows QC *dropped*, and analysing it would re-admit them;
3. a **canonical name** wins: `clean_readings.csv` (what step 1 writes), then `readings.csv`
   (so this capsule still runs directly on the raw asset with step 1 skipped);
4. only then, **largest file wins**.

Every other readings-shaped file is recorded in `readings_candidates_not_chosen` (`file`,
`rows`, `reason`, `is_qc_output`) and named in the run log, with `readings_file_chosen_by`
saying why the winner won. A rival readings file raises a `data_warnings` entry; a
re-mounted `qc_flags.csv` — the expected sibling of a captured step-1 result — is recorded
as a `notes` entry instead, because a warning that fires on every healthy chained run
teaches operators to ignore warnings. Step 1 records the same list for the same mount, so
the two manifests can be compared directly.

CSV discovery is recursive and matches the extension **case-insensitively**: pathlib's
`*.csv` glob is case-sensitive on POSIX whatever the filesystem does, so on the
case-sensitive filesystem Code Ocean actually runs, a `READINGS.CSV` was invisible — absent
from `input_files`, never considered, and with no warning to say so.

## Inputs that cannot be read

A zero-byte CSV, a binary file with a `.csv` name or a UTF-16 export is data, not an error:
the run carries on and exits 0. But it used to vanish — one stdout line, then nothing. On
an input holding only an unreadable `readings.csv`, the manifest said `source_file: null`,
`rows_in: 0`, `instruments: []`, `data_warnings: []` while `input_files` still listed the
file: the manifest asserting the input had no readings when in truth they could not be
read. Each one is now in `unreadable_input_files` (`file`, `error`) with a matching
`data_warnings` entry — **the same key name and the same warning shape step 1 uses** — and
when nothing usable was found at all, a `notes` entry says the outputs are header-only and
points at the reason. (Unlike step 1's version there is no `passed_through` field: this
step copies no input into `/results`.)

## Run parameters (App Panel)

`.codeocean/app-panel.json` is what makes this capsule parameterizable. Code Ocean reads
that committed file and materializes an App Panel form — no UI work required — and the
file is read-only in the capsule IDE, so it can only arrive by `git push`. Because the
panel sets `"named_parameters": true`, each value is appended to `code/run` as a single
command-line argument shaped `--param_name=value` (equals sign, not a space); `code/run`
forwards `"$@"` and `analyze.py` parses it with `argparse`. Values chosen this way are
recorded on the computation and frozen onto the captured result asset as `app_parameters`,
which is why the orchestrator app routes the user's choices through parameters instead of
keeping them to itself. Adding a parameter here makes it appear in the orchestrator's GUI
automatically — the app renders the panel it reads back from the capsule.

| Argument | Label | Default | Meaning |
|---|---|---|---|
| `--resample_interval` | Resample interval | `6H` | bucket size; one of `1H`, `6H`, `12H`, `1D` (a dropdown on the panel) |
| `--rolling_window` | Rolling window (buckets) | `4` | buckets averaged into `rolling_mean`; `1` disables smoothing |
| `--anomaly_z` | Anomaly threshold (robust z) | `3` | flag a bucket at `\|robust z\| >= this`; lower finds more |
| `--top_n_anomalies` | Top N anomalies | `20` | rows kept in `anomalies.csv`, ranked by `\|robust z\|` descending |
| `--baseline_instrument` | Baseline instrument | *(blank)* | instrument every other instrument's mean is compared against in `mean_vs_baseline`; blank = first alphabetically |

Every parameter is optional and the rules this capsule guarantees are the batch's
non-negotiables:

- **No parameters ⇒ the standard analysis.** The panel may not exist on every deployment,
  so the demo must never depend on it.
- **A bad value never fails the run.** `--rolling_window=0`, `--anomaly_z=abc`,
  `--top_n_anomalies=-1` and `--resample_interval=nonsense` each log a warning, fall back
  to the default and exit 0. A `--baseline_instrument` that isn't in the data warns and
  falls back to the first instrument alphabetically. An unrecognized parameter is noted and
  ignored (`parse_known_args`), because a panel can gain a field before the code that reads
  it is deployed. Every warning is printed to the run log **and** recorded in
  `manifest.json` under `parameter_warnings`.
- **"Bad value" includes the ones `float()` accepts.** `nan`, `inf`, `-inf`, `Infinity` and
  `1e400` (which overflows to inf) all parse without raising, and `int()` on any of them
  then raises — ending the run with an *empty* `/results`, which is the worst possible
  outcome. They are rejected before conversion, with a warning naming the value.
- **Huge finite values are clamped, which `numpy.isfinite` cannot do for you.**
  `--rolling_window=1e40` is a legal Python int and only detonates deep inside pandas as
  "Python int too large to convert to C long". `rolling_window` is capped at 100000,
  `top_n_anomalies` at 1000000 and `anomaly_z` at 1000 — each with a warning saying it was
  clamped.
- **A fractional value for a whole-number parameter warns before it is truncated.**
  `--top_n_anomalies=1.9` is used as `1` and `--rolling_window=3.7` as `3`, each with a
  message naming the value sent and the value used. Every other coercion here emits one, so
  a silent truncation is exactly the one an operator would miss. (`--top_n_anomalies=0.5`
  truncates to 0, fails the "must be positive" check and falls back to the default — that
  message already tells the whole story, so it is not doubled up.)
- **A token the run could not use never becomes a silent lie.** There are two ways a token
  ends up unused, and both are handled the same way:
  - argparse **rejects** the list — `--anomaly_z=1.5 --top_n_anomalies`, a flag with no
    value, raises;
  - argparse **accepts** the list but does not consume all of it — an unknown parameter, a
    stray word, or anything after an end-of-options `--`.

  Either way the run survives by **partial recovery**: the tokens argparse can actually use
  are re-parsed and honoured (`anomaly_z` really is 1.5), the rest are discarded, and one
  `parameter_warnings` entry quotes the raw argument list and names exactly what was thrown
  away. The second case was the dangerous one, because argparse reported *success*:
  `--  --anomaly_z=1.5` left the operator's 1.5 with no effect at all while the manifest
  read `anomaly_z "3"`, `parameter_warnings []`, `argv_parsed true` **and**
  `parameters_supplied ["anomaly_z"]` — a manifest claiming the value was supplied and
  recording the default in the same breath. The invariant now enforced everywhere:
  **`parameters_supplied` never names a parameter whose recorded value is its
  default-because-we-could-not-use-it.** `parameters_source.argv` puts the raw list on the
  record either way, so no reader can mistake a fallback for a choice; the run log likewise
  reports the parameters that were **honoured**.
- **Two input columns with the same normalized name never end the run.** Headers are
  trimmed and lower-cased so `Timestamp` is accepted as `timestamp` — which means
  `Reading,reading`, `Timestamp,timestamp`, `reading ,reading` and
  `Instrument_ID,instrument_id` all collapse onto one name. That used to make
  `df["reading"]` a *DataFrame*, blow up `pd.to_numeric` with a `TypeError`, and end the
  run with exit 1 and a **completely empty** `/results` — and it is reachable through the
  normal pipeline, because step 1 accepts such a header and carries both columns into
  `clean_readings.csv`. Now the **first** column of each name wins, the duplicates are
  dropped at read time, and the choice shows up as a run-log line, a `data_warnings` entry
  and one record per dropped column under `dropped_duplicate_columns` (`column`,
  `position`, `normalized_name`, `kept_column`).
- **Mixed UTC offsets in one timestamp column never end the run.** A real export can carry
  `2026-06-01T00:00:00+00:00` on one row and `...+05:30` on the next, and step 1 accepts
  such a file, exits 0 with `warnings []`, and copies the offsets verbatim into
  `clean_readings.csv` — so this arrives through the *normal* pipeline. Plain
  `pd.to_datetime(errors="coerce")` cannot pick one dtype for that column and hands back an
  **object**-dtype index of per-row datetimes (a `FutureWarning`, not an error);
  `errors="coerce"` is no protection, because it coerces per-*element* parse failures and
  says nothing about the dtype of the *result*. An object index has no `.dt` accessor and
  cannot be resampled, so the run died at `resample` with `TypeError: Only valid with
  DatetimeIndex ... got an instance of 'Index'` — exit 1, **completely empty** `/results`.
  The column is now parsed with `utc=True` (per-row `format="mixed"` where the installed
  pandas supports it, feature-probed rather than version-sniffed) and the zone is dropped,
  giving one tz-naive `datetime64` column; the dtype is checked before anything uses it.
  Columns with no zone at all round-trip unchanged, and two columns that mean the same
  instants bucket identically however they spell the zone. Because that **shifts bucket
  boundaries** onto UTC, it is never silent: a `data_warnings` entry and a run-log line say
  so, and `manifest.json` carries `timestamps_normalized_to_utc`,
  `n_timestamps_with_utc_offset` and `utc_offsets_seen` (more than one entry there *is* the
  mixed case). Step 1 does the identical thing, and so does batch 1's `make_report.py`.
- **An implausible timestamp is data, not a crash.** `1700-01-01T00:00:00` — a sentinel or a
  typo among ordinary 2026 readings — is inside pandas' representable range, so it parses
  cleanly. Any span measured against it is then ~326 years, and an int64 count of
  *nanoseconds* overflows at ~292 years: `fit_trend` raised `OverflowError: Overflow in
  int64 addition` **after** the resample and the anomaly scoring had already succeeded and
  been logged, leaving exit 1 and an empty `/results`. Two independent fixes, because a
  filter that only catches the cases we thought of is not the same as arithmetic that cannot
  fail:
  - every timestamp-span computation now subtracts in **microseconds** (`timestamps_as_days`),
    a unit whose int64 range (±292,471 years) cannot be overflowed by any pair of instants
    pandas can represent at all;
  - a timestamp outside `plausible_timestamp_range` (`1900-01-01` … `2200-01-01`, recorded
    in the manifest) is treated exactly like an unparseable one — the row is dropped and
    **counted** as `rows_implausible_timestamp`, a subset of `rows_unusable`, with a matching
    `data_warnings` entry and run-log line. Every other row of that instrument is analysed
    normally and the instrument still appears in `instrument_summary.csv`; it drops out only
    if it has no usable timestamp left at all.
- **How the column was READ is recorded whenever that was a choice.** ISO-8601 means one
  thing; other shapes do not. A **bare number** (`1780000000`) in a numeric column is read
  as *nanoseconds* since 1970-01-01, so an epoch value in seconds lands two seconds into
  1970 and the whole series collapses into one bucket. A **`01/02/2026`-style date** has
  its day/month order inferred **per value** (`format="mixed"`), so `01/02/2026` (read
  month-first) and `13/02/2026` (read day-first) in one column come back in two different
  calendars — a successful parse, not a failure. Neither is fixable here and neither is
  fatal, so both are *disclosed*: `timestamp_interpretation` carries `unambiguous_iso8601`,
  a count per shape, up to three example values and — taken from this run's own parse
  rather than guessed — what each example was actually read as, with a `data_warnings`
  entry saying it in words. `null` means there was no timestamp column to examine.
- **`manifest.json` is always valid JSON.** Bare `NaN`/`Infinity` literals are not — strict
  parsers and `JSON.parse` reject them — so the manifest is written with `allow_nan=False`
  as a backstop behind those parse-time checks. An *upstream* manifest that already
  contains one is not fatal: those values are carried forward as `null`, and a `notes`
  entry in **this** manifest names the file and says the copy under `upstream_manifests`
  differs from the file on disk. A run-log line was not enough: `upstream_manifests` is the
  copy people read afterwards, so an invisible substitution sits in exactly the artifact
  that is meant to carry the provenance chain. An upstream manifest that could not be
  parsed at all is likewise noted, because a missing link in that chain must not be
  silent.
- **An empty input is an answer, not an error.** No readings, one row, or a header-only CSV
  all produce the three CSVs *with their headers* and exit 0, so downstream code never has
  to special-case an empty run.
- **`baseline_instrument` changes the result, not just the manifest.** It sets the
  `mean_vs_baseline` column of `instrument_summary.csv`, so `--baseline_instrument=RX-103`
  and `--baseline_instrument=RX-101` produce genuinely different files — that is the
  "add a parameter and watch the output move" moment the demo is built around. The id that
  won is still recorded in `manifest.json` as part of the provenance.

### The manifest's parameter block

Every capsule in this demo records parameters the same three ways, so one reader handles
all of them:

| Key | Contents |
|---|---|
| `parameters` | the raw values as received *and understood* (strings), including untouched defaults |
| `effective_parameters` | the coerced values the run actually used (`null` when there was nothing to use) |
| `parameter_warnings` | one message per rejected, clamped or truncated value, the same text as the run log |
| `parameters_source` | how those values arrived: `argv` (the raw argument list, verbatim), `argv_parsed` (`true` only when the list was used **exactly as sent** — `false` if argparse rejected it *or* left any token unconsumed), `parameters_supplied` (the names actually honoured), `ignored_tokens`, `superseded_tokens` |
| `notes` | things that changed the run but rejected nothing — a blank value falling back to its default, or a blank `baseline_instrument` resolving to the first instrument alphabetically |

So `--anomaly_z=abc` shows up as `"anomaly_z": "abc"` in `parameters`, `3.0` in
`effective_parameters`, and one entry in `parameter_warnings` — and the run still exits 0.

`superseded_tokens` covers the one way a value can be discarded by a list that parsed
*perfectly*. `--anomaly_z=1 --anomaly_z=9` is a clean parse: argparse takes the last value
and never mentions the first, so the manifest read `anomaly_z "9"`, `argv_parsed true` and
— the false part — `ignored_tokens []`, positively claiming nothing had been dropped. Every
overridden token is now listed there and named in a `parameter_warnings` entry saying which
value won.

`parameters_source` exists because `parameters` **cannot** express "you sent
`--anomaly_z=1.5` and I could not use it" — in that slot it holds the *default*, which a
reader would otherwise take for the operator's choice. For a capsule whose whole purpose is
provenance, a manifest that quietly contradicts its caller is worse than a crash: a crash is
loud, and this is the artifact people trust afterwards. Between the four keys the rule is
absolute: **every** `null` or fallback in `effective_parameters` is explained by an entry in
`parameter_warnings` or `notes`, and the raw argument list is always on the record.

### Two details worth knowing when reading the numbers

- **Frequency aliases are resolved at runtime.** pandas 2.2 deprecated the upper-case `H`
  alias in favour of `h`, so `analyze.py` probes what the installed pandas accepts without
  warning and uses that; the alias actually used is in `manifest.json` as
  `pandas_freq_alias`. Runs are clean under `python -W error::FutureWarning`.
- **The interval itself is matched leniently, and says when it was.**
  `--resample_interval=1d` and `" 6h "` are obviously meant to be `1D` and `6H`, so they
  are accepted. But a lenient match still *changes* the value: `parameters` records `"1d"`
  while `effective_parameters` records `"1D"`, and every such difference owes the reader an
  explanation. So a value that only matched after case-folding or trimming leaves a `notes`
  entry — *"Resample interval "1d" was read as 1D (interval names are matched
  case-insensitively, ignoring surrounding whitespace)"* — exactly as
  `--baseline_instrument` already does for a case-insensitive instrument match.
- **A non-finite reading in the data is dropped, not analyzed.** `pd.to_numeric` is not the
  filter it looks like: it turns `abc` into `NaN` (dropped), but turns `inf`, `-inf` and
  `Infinity` into a real `inf` float, which survives `dropna`. One such row
  used to reach `resampled.csv` as an `inf` bucket mean, give that instrument an `inf`
  `mean`/`max` and a blank `sd` in `instrument_summary.csv`, and — because the baseline's
  mean was then non-finite — blank out `mean_vs_baseline` for **every** instrument while
  `manifest.json` reported `baseline_mean: null` with nothing to explain it. Such rows are
  now dropped alongside blank readings, counted in `rows_non_finite_reading`, and named in
  `data_warnings`. That guard screens the **input**; a separate one screens the
  **output**, because rounding and ordinary arithmetic can both manufacture an infinity
  from readings that arrived perfectly finite — see `non_finite_cells_blanked` above.
- **An upstream `manifest.json` that goes missing from the chain always says so.** Three
  things can happen to one: it fails to parse (recorded), it contains bare `NaN`/`Infinity`
  literals that are carried forward as `null` (recorded — the copy differs from the file on
  disk), or it parses as valid JSON that is **not an object** — a list, a string, `null`.
  That last one fell through the `isinstance(..., dict)` test with no `except` to catch it
  and no `else` to report it, so the link vanished without even the note its malformed
  sibling gets. It is now recorded like the others.
- **The rolling mean's warm-up buckets are smoothed but not anomaly-scored.** With
  `min_periods=1` the first `rolling_window - 1` buckets get a rolling mean from fewer
  points, so their residual is small *by construction*. On a steadily drifting instrument
  every later residual sits at the same non-zero trend lag, which would make those warm-up
  rows read as enormous outliers — and drift is legitimate signal that must surface as a
  slope, not as an anomaly. `resampled.csv` still carries a `rolling_mean` on every row;
  only the scoring skips the warm-up, and the count is in `manifest.json` as
  `warmup_buckets_excluded_per_instrument`.
- **…and a warm-up that swallows the whole series says so.** Raise `rolling_window` past an
  instrument's bucket count — `100` against the demo's 84/80/78 buckets is an ordinary
  value, nowhere near the clamp, so no clamp warning fires — and *every* bucket is warm-up:
  the instrument is never scored at all. That used to produce a header-only `anomalies.csv`
  with `n_anomalies_flagged: 0` and empty `parameter_warnings`, `data_warnings` and
  `notes`, so a presenter nudging the smoothing window up in the GUI watched the anomaly
  table empty itself with no explanation. **A rule that cannot run must say so**, per
  instrument: `anomaly_rule_skipped_instruments` names each one with its bucket count and
  the reason (the warm-up is longer than its series, or its scored residuals have a MAD of
  0 so no robust z exists — which is also what `rolling_window=1` does to every
  instrument), `n_buckets_anomaly_scored` says how many buckets the rule *did* judge, and a
  `data_warnings` entry spells out that a `0` in `n_anomalies` for those instruments means
  "no verdict", not "nothing unusual". `n_anomalies_flagged: 0` with
  `n_buckets_anomaly_scored: 233` is a measurement; with `0` it is the absence of one.

  The warm-up is deliberately **not** clamped to fit a short series. Clamping would score
  buckets whose rolling mean is an average of however few points came before them, which
  on the demo's own drifting instrument manufactures a run of large residuals out of
  legitimate signal — trading "the rule could not run" for "the rule ran and lied". Lower
  `rolling_window` or pick a finer `resample_interval` instead; the warning says so.

Local test with parameters:

```bash
DATA_DIR=<dir with .csv> RESULTS_DIR=<out dir> bash code/run \
  --resample_interval=1D --rolling_window=2 --anomaly_z=2 --top_n_anomalies=5
```
