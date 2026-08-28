#!/usr/bin/env bash
# Render a coloured overlay video for every batch run that tracked successfully.
#
# A run counts as successful if it has masks.npz; run_batch.sh deletes frames/
# on success, so each render decodes its pictures back out of the source video
# (--video). Playback fps is left to render_colors.py, which infers the source
# rate from tracks.csv — tracking ran unstrided, so these play in real time.
#
# Resumable — a run whose colored.mp4 exists is skipped.
#
#   ./render_batch.sh                 # render (or resume) the whole batch
#   DRY_RUN=1 ./render_batch.sh       # print the plan, render nothing

set -uo pipefail

PROJECT=/home/kyle/projects/segmentation
DATA=/home/kyle/projects/test_data
OUT_ROOT="$PROJECT/runs/batch"
LOG_DIR="$OUT_ROOT/_logs"
TRAIL_LEN=20
MIN_FREE_GB=20          # the mp4v intermediate is transient but not small

cd "$PROJECT" || exit 1
mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/render.log"

say() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER"; }

mapfile -t RUNS < <(find "$OUT_ROOT" -mindepth 3 -maxdepth 3 -name masks.npz | sort)
say "[render] ${#RUNS[@]} tracked runs, trail-len=$TRAIL_LEN"

for masks in "${RUNS[@]}"; do
    run=$(dirname "$masks")
    stem=$(basename "$run")
    day=$(basename "$(dirname "$run")")
    video="$DATA/$day/$stem.mp4"
    log="$LOG_DIR/$day.$stem.render.log"

    if [[ -f "$run/colored.mp4" ]]; then
        say "[skip] $day/$stem already has colored.mp4"
        continue
    fi
    if [[ ! -f "$video" ]]; then
        say "[FAIL] $day/$stem — source video not found at $video"
        continue
    fi

    free_gb=$(df -BG --output=avail "$PROJECT" | tail -1 | tr -dc '0-9')
    if (( free_gb < MIN_FREE_GB )); then
        say "[abort] only ${free_gb}G free, need ${MIN_FREE_GB}G — stopping before $day/$stem"
        exit 1
    fi

    if [[ -n "${DRY_RUN:-}" ]]; then
        say "[dry-run] would render $day/$stem -> $run/colored.mp4"
        continue
    fi

    say "[start] $day/$stem  (log: $log)"
    start=$SECONDS
    uv run render_colors.py "$run" --video "$video" \
        --trails --trail-len "$TRAIL_LEN" \
        --out "$run/colored.mp4" >>"$log" 2>&1
    status=$?
    mins=$(( (SECONDS - start) / 60 ))

    if (( status == 0 )) && [[ -f "$run/colored.mp4" ]]; then
        say "[done] $day/$stem in ${mins}m — $(du -h "$run/colored.mp4" | cut -f1)"
    else
        # Drop the half-written output so a rerun does not skip it.
        rm -f "$run/colored.mp4" "$run/.colored.raw.mp4"
        say "[FAIL] $day/$stem exit=$status after ${mins}m — see $log"
    fi
done

say "[render] finished"
