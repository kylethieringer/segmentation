#!/usr/bin/env bash
# Track every video under test_data/ with SAM 3, one at a time.
#
# Sequential by design: a single run already saturates the 16 GB card, so two
# at once would OOM. Resumable — a video whose tracks.csv exists is skipped, so
# the script can be killed and restarted without losing finished work.
#
#   ./run_batch.sh                 # run (or resume) the whole batch
#   DRY_RUN=1 ./run_batch.sh       # print the plan, run nothing

set -uo pipefail

PROJECT=/home/kyle/projects/segmentation
DATA=/home/kyle/projects/test_data
OUT_ROOT="$PROJECT/runs/batch"
LOG_DIR="$OUT_ROOT/_logs"
PROMPT=insect
STRIDE=1
CHUNK=450
OVERLAP=15
RETRY_CHUNK=225         # second attempt for a video that OOMs the GPU at CHUNK
MIN_FREE_GB=40          # frames for the longest video need ~26 GB

cd "$PROJECT" || exit 1
mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/batch.log"

say() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER"; }

# track VIDEO OUT LOG CHUNK
# --reuse-frames picks up a previous failed attempt's extracted frames.
track() {
    uv run track_long.py "$1" \
        --prompt "$PROMPT" --stride "$STRIDE" \
        --chunk "$4" --overlap "$OVERLAP" \
        --reuse-frames --out "$2" >>"$3" 2>&1
}

# GPU memory scales with objects x frames per session, so a video with more
# subjects than usual can OOM on chunk 1 at a chunk size that suits the rest of
# the batch (10 flies vs the usual 5-7 was enough to do it on a 16 GB card).
# Halving the chunk trades a little speed for the run finishing at all.
#
# Chunk size also selects which way identity stitching fails at a seam, so do
# not tune it on speed alone. Larger chunks mean fewer seams but a longer
# absence per missed one: a subject lost for a whole chunk returns beyond IoU
# range and is handed a new id (a split -- one animal, two ids). Smaller chunks
# double the seams, so unmatched detections show up more often as short-lived
# stubs, but nothing is ever gone long enough to split. Measured on this batch:
# 450 gave 1 split and 0 stubs, 225 gave 0 splits and 1-4 stubs per video.
# track_long.py --relink-dist/--relink-chunks recovers the split case; nothing
# recovers a stub, so prefer the smaller chunk and filter stubs by lifetime.
is_cuda_oom() { tail -40 "$1" | grep -q "torch.OutOfMemoryError"; }

mapfile -t VIDEOS < <(find "$DATA" -name '*.mp4' | sort)
say "[batch] ${#VIDEOS[@]} videos, prompt=$PROMPT stride=$STRIDE chunk=$CHUNK overlap=$OVERLAP"

for video in "${VIDEOS[@]}"; do
    day=$(basename "$(dirname "$video")")
    stem=$(basename "$video" .mp4)
    out="$OUT_ROOT/$day/$stem"
    log="$LOG_DIR/$day.$stem.log"

    if [[ -f "$out/tracks.csv" ]]; then
        say "[skip] $day/$stem already has tracks.csv"
        continue
    fi

    free_gb=$(df -BG --output=avail "$PROJECT" | tail -1 | tr -dc '0-9')
    if (( free_gb < MIN_FREE_GB )); then
        say "[abort] only ${free_gb}G free, need ${MIN_FREE_GB}G — stopping before $day/$stem"
        exit 1
    fi

    if [[ -n "${DRY_RUN:-}" ]]; then
        say "[dry-run] would track $day/$stem -> $out"
        continue
    fi

    say "[start] $day/$stem -> $out  (log: $log)"
    start=$SECONDS
    track "$video" "$out" "$log" "$CHUNK"
    status=$?

    if (( status != 0 )) && [[ ! -f "$out/tracks.csv" ]] && is_cuda_oom "$log"; then
        say "[retry] $day/$stem hit CUDA OOM at chunk $CHUNK — retrying at $RETRY_CHUNK"
        track "$video" "$out" "$log" "$RETRY_CHUNK"
        status=$?
    fi
    mins=$(( (SECONDS - start) / 60 ))

    if (( status == 0 )) && [[ -f "$out/tracks.csv" ]]; then
        rm -rf "$out/frames"          # ~26 GB for the longest video; keep only on failure
        say "[done] $day/$stem in ${mins}m — $(wc -l < "$out/tracks.csv") rows"
    else
        # Frames are deliberately left in place so a rerun resumes the extract.
        say "[FAIL] $day/$stem exit=$status after ${mins}m — see $log"
    fi
done

say "[batch] finished"
