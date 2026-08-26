import base64, csv
from pathlib import Path

FIG = Path("/home/kyle/projects/segmentation/figures")
OUT = Path("/home/kyle/projects/segmentation/figures/report.html")

def uri(name):
    return "data:image/png;base64," + base64.b64encode((FIG / name).read_bytes()).decode()

def figure(slug, caption):
    return f'''<figure class="fig">
  <img class="ph-light" src="{uri(slug + '_light.png')}" alt="{caption}">
  <img class="ph-dark" src="{uri(slug + '_dark.png')}" alt="{caption}">
  <figcaption>{caption}</figcaption>
</figure>'''

rows = list(csv.DictReader((FIG / "summary.csv").open()))
rows.sort(key=lambda r: -float(r["dist_px"]))
tbody = "\n".join(
    f'<tr><td class="fly">Fly {r["fly"]}</td>'
    f'<td>{float(r["dist_px"])/1000:.1f}k</td>'
    f'<td>{float(r["mean_speed"]):.0f}</td>'
    f'<td>{float(r["p95_speed"]):.0f}</td>'
    f'<td>{float(r["outer_third"])*100:.0f}%</td>'
    f'<td>{float(r["med_area"]):.0f}</td></tr>' for r in rows)

HTML = f'''<title>Seven Flies, Three Minutes</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;500;600&display=swap">
<style>
:root {{
  color-scheme: light;
  --ground:   #fcfcfb;
  --raised:   #f4f4f1;
  --ink:      #12141a;
  --ink-2:    #4a4f5c;
  --ink-3:    #7c8290;
  --rule:     #e2e3de;
  --accent:   #2a78d6;
  --accent-soft: #e8f0fb;
  --warn:     #8a5a00;
  --warn-soft:#f8f0dc;
  --serif: "IBM Plex Serif", Georgia, serif;
  --sans:  "IBM Plex Sans", system-ui, sans-serif;
  --mono:  "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --ground: #1a1a19; --raised: #232322; --ink: #f5f5f2; --ink-2: #c3c2b7;
    --ink-3: #8e8d84; --rule: #35352f; --accent: #3987e5; --accent-soft: #17263a;
    --warn: #e0b558; --warn-soft: #2c2618;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --ground: #1a1a19; --raised: #232322; --ink: #f5f5f2; --ink-2: #c3c2b7;
  --ink-3: #8e8d84; --rule: #35352f; --accent: #3987e5; --accent-soft: #17263a;
  --warn: #e0b558; --warn-soft: #2c2618;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--ground); color: var(--ink);
  font-family: var(--sans); font-size: 16.5px; line-height: 1.65;
  margin: 0; padding: 0 24px 96px;
  -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 1120px; margin: 0 auto; }}
.col {{ max-width: 68ch; }}
header {{ padding: 72px 0 40px; border-bottom: 1px solid var(--rule); }}
.eyebrow {{
  font-family: var(--mono); font-size: 12px; letter-spacing: .12em;
  text-transform: uppercase; color: var(--ink-3); margin: 0 0 18px;
}}
h1 {{
  font-family: var(--serif); font-weight: 600; font-size: clamp(34px, 5.2vw, 52px);
  line-height: 1.08; letter-spacing: -.02em; margin: 0 0 18px; text-wrap: balance;
}}
.lede {{ font-size: 19px; color: var(--ink-2); margin: 0; max-width: 62ch; }}
h2 {{
  font-family: var(--serif); font-weight: 600; font-size: 27px; letter-spacing: -.01em;
  margin: 0 0 6px; text-wrap: balance;
}}
h3 {{ font-family: var(--sans); font-weight: 600; font-size: 17px; margin: 0 0 6px; }}
section {{ padding-top: 60px; }}
.sec-note {{ font-family: var(--mono); font-size: 12.5px; color: var(--ink-3); margin: 0 0 28px; }}
p {{ margin: 0 0 18px; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1px;
  background: var(--rule); border: 1px solid var(--rule); margin: 40px 0 0; }}
.stat {{ background: var(--ground); padding: 22px 24px; }}
.stat .v {{ font-family: var(--mono); font-size: 30px; font-weight: 500; letter-spacing: -.02em;
  color: var(--accent); font-variant-numeric: tabular-nums; }}
.stat .k {{ font-size: 13.5px; color: var(--ink-2); margin-top: 4px; }}
.fig {{ margin: 34px 0 0; }}
.fig img {{ width: 100%; height: auto; display: block; }}
.ph-dark {{ display: none; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) .ph-light {{ display: none; }}
  :root:not([data-theme="light"]) .ph-dark  {{ display: block; }}
}}
:root[data-theme="dark"] .ph-light {{ display: none; }}
:root[data-theme="dark"] .ph-dark  {{ display: block; }}
figcaption {{ font-size: 14px; color: var(--ink-2); margin-top: 14px; max-width: 78ch;
  padding-left: 14px; border-left: 2px solid var(--rule); }}
.tablewrap {{ overflow-x: auto; margin-top: 34px; border: 1px solid var(--rule); }}
table {{ border-collapse: collapse; width: 100%; font-size: 14.5px; }}
th, td {{ text-align: right; padding: 11px 16px; border-bottom: 1px solid var(--rule); }}
th {{ font-family: var(--mono); font-size: 11.5px; letter-spacing: .07em; text-transform: uppercase;
  color: var(--ink-3); font-weight: 500; white-space: nowrap; }}
td {{ font-family: var(--mono); font-variant-numeric: tabular-nums; color: var(--ink-2); }}
td.fly, th:first-child {{ text-align: left; }}
td.fly {{ color: var(--ink); font-family: var(--sans); }}
tbody tr:last-child td {{ border-bottom: none; }}
.callout {{ background: var(--warn-soft); border-left: 3px solid var(--warn);
  padding: 20px 24px; margin: 34px 0 0; }}
.callout h3 {{ color: var(--warn); }}
.callout p:last-child {{ margin-bottom: 0; }}
.caveats {{ display: grid; gap: 1px; background: var(--rule); border: 1px solid var(--rule);
  margin-top: 30px; }}
.caveat {{ background: var(--ground); padding: 22px 24px; }}
.caveat p {{ margin: 0; color: var(--ink-2); font-size: 15px; }}
.pipe {{ font-family: var(--mono); font-size: 13px; background: var(--raised);
  border: 1px solid var(--rule); padding: 18px 20px; overflow-x: auto;
  margin-top: 26px; color: var(--ink-2); white-space: pre; }}
.k-inline {{ font-family: var(--mono); font-size: .88em; background: var(--accent-soft);
  color: var(--accent); padding: 1px 6px; }}
footer {{ margin-top: 72px; padding-top: 26px; border-top: 1px solid var(--rule);
  font-family: var(--mono); font-size: 12.5px; color: var(--ink-3); }}
a {{ color: var(--accent); }}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 3px; }}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Drosophila arena assay &middot; SAM 3 segmentation</p>
  <h1>Seven flies, three minutes</h1>
  <p class="lede">Every fly segmented and tracked in every frame of a 1120&times;1120 backlit
  arena recording &mdash; 10,722 source frames at 60&nbsp;fps &mdash; using Meta's Segment
  Anything 3 with the text prompt <span class="k-inline">insect</span>.</p>
  <div class="stats">
    <div class="stat"><div class="v">7</div><div class="k">flies, stable identities throughout</div></div>
    <div class="stat"><div class="v">3,575</div><div class="k">frames tracked at 20&nbsp;Hz</div></div>
    <div class="stat"><div class="v">100%</div><div class="k">continuity &mdash; no dropped frames</div></div>
    <div class="stat"><div class="v">0.57<span style="font-size:16px"> px</span></div><div class="k">median error vs ground truth</div></div>
  </div>
</header>

<section>
  <div class="col">
    <h2>Where they went</h2>
    <p class="sec-note">7 panels &middot; 3,575 positions each</p>
    <p>Each fly's complete path over the recording. The spread in activity is large and
    immediate: Fly&nbsp;4 covered 24.3k pixels while Fly&nbsp;5 managed 7.3k, a
    3.3&times; difference between individuals in the same arena under the same conditions.</p>
  </div>
  {figure("trajectories", "Complete walking path for each fly. Dot marks the starting position; the grey circle is the arena wall fitted to the outermost tracked positions.")}
</section>

<section>
  <div class="col">
    <h2>When they moved</h2>
    <p class="sec-note">speed, 1-second means</p>
    <p>Activity comes in bouts rather than at a steady rate. Flies&nbsp;2 and&nbsp;4 are
    near-continuously active; Fly&nbsp;5 shows long quiet stretches broken by short bursts.
    The pale vertical band around 80&ndash;100&nbsp;s in several rows is a period when most
    of the group settled at once.</p>
  </div>
  {figure("speed", "Walking speed per fly over the full recording, smoothed with a 1-second mean. Speed is in pixels per second; the recording has no spatial calibration, so this cannot be converted to mm/s without the arena's physical diameter.")}
</section>

<section>
  <div class="col">
    <h2>They hug the wall</h2>
    <p class="sec-note">radial position, normalised to arena radius</p>
    <p>Across the group, 68% of all observations fall in the outer third of the arena &mdash;
    thigmotaxis, the wall-following preference well documented in <em>Drosophila</em>. It is
    not uniform: Fly&nbsp;6 spent 96% of its time there while Fly&nbsp;1 spent 42%, so the
    group average conceals a real spread between individuals.</p>
  </div>
  {figure("radial", "Left: distribution of distance from arena centre, pooled across all flies and frames, with the outer third shaded. Right: the same measure per individual.")}

  <div class="tablewrap">
    <table>
      <thead><tr>
        <th>Fly</th><th>Distance (px)</th><th>Mean speed (px/s)</th>
        <th>p95 speed (px/s)</th><th>Time at wall</th><th>Median area (px)</th>
      </tr></thead>
      <tbody>
{tbody}
      </tbody>
    </table>
  </div>
</section>

<section>
  <div class="col">
    <h2>Why the numbers can be trusted</h2>
    <p class="sec-note">validation</p>
    <p>A single SAM&nbsp;3 session runs out of GPU memory at roughly 500 frames on a 16&nbsp;GB
    card, so the recording was tracked in nine overlapping chunks and identities were
    reconciled across the seams by mask IoU. Every seam matched all seven tracks, with
    overlap IoU between 0.77 and 0.95 against a 0.30 match threshold &mdash; a wide margin.</p>
    <p>Tracking ran at every third frame. To recover the intermediate frames, positions were
    interpolated and then checked against a separately tracked stride-1 run covering the
    first 120 frames. The error is smaller than the mask boundary noise.</p>
  </div>
  {figure("validation", "Left: cumulative distribution of interpolated position error against stride-1 ground truth, over 840 paired observations. Right: overlap IoU at each of the eight chunk boundaries.")}
</section>

<section>
  <div class="col">
    <h2>What this does not show</h2>
    <p class="sec-note">limits of the method</p>
  </div>
  <div class="caveats">
    <div class="caveat">
      <h3>Identity was never stress-tested</h3>
      <p>No two flies came within 30&nbsp;px of each other at any point in the recording, and the
      closest approach was 44.7&nbsp;px between centroids &mdash; roughly one body length. Tracking
      held perfectly, but it was never asked the hard question. Nothing here indicates how
      SAM&nbsp;3 behaves through a genuine occlusion, which is the case that breaks trackers and
      the reason tools like idtracker.ai and SLEAP exist.</p>
    </div>
    <div class="caveat">
      <h3>Distances are in pixels, not millimetres</h3>
      <p>The arena's physical diameter was not recorded here, so every distance and speed is in
      image units. One scale factor converts the whole table once that measurement exists.</p>
    </div>
    <div class="caveat">
      <h3>The true sampling rate is 20&nbsp;Hz, not 60</h3>
      <p>The upsampled track file has a row for all 10,723 source frames, but only one in three
      is a measurement &mdash; the <span class="k-inline">measured</span> column marks which.
      Velocities computed at 60&nbsp;fps are smooth between anchors by construction, so
      events faster than about 10&nbsp;Hz are attenuated. Re-track at stride&nbsp;1 for fine
      turn dynamics.</p>
    </div>
    <div class="caveat">
      <h3>Mask area is noisier than the flies are</h3>
      <p>Median mask area is 3,652&nbsp;px but the maximum reaches 6,742&nbsp;px &mdash; nearly
      double. Some of that is real (wing extension), some is the mask bleeding into adjacent
      dark regions. Centroids are unaffected; treat <span class="k-inline">area_px</span> as
      a rough signal, not a measurement.</p>
    </div>
  </div>
</section>

<section>
  <div class="col">
    <h2>How it was produced</h2>
    <p class="sec-note">pipeline</p>
    <p>SAM&nbsp;3 (<span class="k-inline">facebook/sam3</span>, the non-multiplex checkpoint &mdash;
    SAM&nbsp;3.1 does not fit in 16&nbsp;GB at this resolution) on an RTX&nbsp;4080&nbsp;SUPER.
    Tracking took 16.8&nbsp;minutes for the full recording.</p>
  </div>
  <div class="pipe">track_long.py  test_video.mp4 --prompt insect --stride 3 --chunk 450 --overlap 15
upsample_tracks.py  runs/full_s3/tracks.csv --kind pchip
render_colors.py  runs/full_s3 --fps 20 --trails
make_figures.py</div>
</section>

<footer>
  Prompt <span class="k-inline">insect</span> was chosen over <span class="k-inline">fly</span>:
  both detect all seven, but "fly" scores 0.56&ndash;0.60 against a 0.50 threshold and returns
  nothing in the video model, while "insect" scores 0.94.
</footer>
</div>
'''
OUT.write_text(HTML)
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")
