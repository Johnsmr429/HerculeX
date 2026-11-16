"""A spot's depth sweep laid out as the page showed it, stretched once per model, and written as a page to browse.

The pieces are placed where the sheet showed them (larger pieces first, as sheet_tiles paints), over the circle's box
widened by the narrowing margin. Statistics follow the PHerc0343 First Letters images (ensemble_maps.py):

    valid      pixels on a piece's nodes whose render holds CT in the five layers round the surface (d -2..+2),
               holes under 0.25 mm² closed; the PHerc0343 rule (CT in every layer) punched specks through the
               papyrus here wherever a read layer reached the masked air between wraps
    stretch    per surface and model, p1 and p99.5 of its valid inward reads over all depths, never per depth,
               so depths keep their brightness differences; the outward side is shown with the inward statistics
    ensemble   per model a z-score with its pooled inward mean and SD, then the mean over the models
    CT         layer 33 + d, one stretch, p1..p99.5 over the valid pixels of layers 18..48

Images show ink dark on light papyrus, as the site does; the page can invert them. They go to
SHEETS/spot-candidates/SPOT/ and the page there steps through depths with the arrow keys.
Each surface, model and side gets a reads row per depth: the mean and 95th percentile z inside the circle,
the share of it above z = 2, and the same mean on the other side.
"""

import json

import numpy as np
from PIL import Image
from scipy import ndimage

from spots import database
from spots.catalogue import inside
from spots.surfaces import MARGIN_MM, MIDDLE, work_of
from spots.sweep import DEPTHS, READ_BY, SIDES
from storage.data_roots import SHEETS

PAGES = SHEETS / "spot-candidates"
QUALITY = 85
LOW, HIGH = 1.0, 99.5  # the CT's stretch, as PHerc0343 showed it
INK_LOW, INK_HIGH = 50.0, 99.9  # a read's stretch, median to its top tenth of a percent, as PHerc0306B showed it
BAND = 2  # the "5 depths" stacks average the ensemble over d - BAND .. d + BAND
INKY_Z = 2.0
MARGIN_CELLS = 2
STAT_EVERY = 4
SMALL_HOLE_PX = 26  # a hole an opening of this many steps removes is wider than about 0.5 mm


def frame_of(spot, um):
    outline = np.asarray(spot["outline_mm"])
    low, high = outline.min(axis=0) - MARGIN_MM, outline.max(axis=0) + MARGIN_MM
    x0, y0 = int(np.floor(low[0] / um)), int(np.floor(low[1] / um))
    x1, y1 = int(np.ceil(high[0] / um)), int(np.ceil(high[1] / um))
    return x0, y0, x1, y1


def placement(r, shape, surface="m7"):
    """Top-left sheet pixel of a piece's render and whether it is flipped, rows then columns."""
    lay, (h, w) = r["layout"], shape
    r0, _, c0, _ = r.get("crops", {}).get(surface, r["crop"])
    rows, columns = lay["shape"]
    stride, um = lay["stride"], lay["um"]
    if lay["flip_rows"]:
        top = lay["y_mm"] / um + (rows - 1 - r0 + MARGIN_CELLS) * stride - (h - 1)
    else:
        top = lay["y_mm"] / um + (r0 - MARGIN_CELLS) * stride
    if lay["flip_cols"]:
        left = lay["x_mm"] / um + (columns - 1 - c0 + MARGIN_CELLS) * stride - (w - 1)
    else:
        left = lay["x_mm"] / um + (c0 - MARGIN_CELLS) * stride
    return round(top), round(left), lay["flip_rows"], lay["flip_cols"]


def upright(image, flips):
    if flips[0]:
        image = image[..., ::-1, :]
    if flips[1]:
        image = image[..., ::-1]
    return image


def node_mask(work, surface, r, shape):
    """Pixels of a render on the narrowed piece as the renderer reads it (bridged across small holes), not its margin."""
    from sampling import column_sampling as sampler

    folder = work / "surfaces" / surface / r["name"]
    xyz, _ = sampler.load_surface(folder)
    stride = r["layout"]["stride"]
    good = sampler.bridged_mesh(xyz, stride)[0] > 0
    h, w = shape
    rows = np.clip(np.rint(np.arange(h) / stride).astype(int) - MARGIN_CELLS, -1, good.shape[0])
    columns = np.clip(np.rint(np.arange(w) / stride).astype(int) - MARGIN_CELLS, -1, good.shape[1])
    ok_r = (rows >= 0) & (rows < good.shape[0])
    ok_c = (columns >= 0) & (columns < good.shape[1])
    mask = np.zeros((h, w), dtype=bool)
    mask[np.ix_(ok_r, ok_c)] = good[np.ix_(rows[ok_r], columns[ok_c])]
    return mask


class Mosaic:
    """Where each piece's pixels land in the spot's frame, and which piece owns each frame pixel."""

    def __init__(self, spot, records, surface, work):
        um = records[0]["layout"]["um"]
        self.x0, self.y0, x1, y1 = frame_of(spot, um)
        self.shape = (y1 - self.y0, x1 - self.x0)
        self.owner = np.full(self.shape, -1, dtype=np.int32)
        self.parts = []
        order = sorted(range(len(records)), key=lambda k: -records[k]["layout"]["area"])
        for k in order:
            r = records[k]
            stack = np.load(work / "stacks" / surface / f"{r['name']}.npy", mmap_mode="r")
            shape = stack.shape[1:]
            held = np.asarray(stack[MIDDLE - 2 : MIDDLE + 3]).min(axis=0) > 0
            held |= ndimage.binary_fill_holes(held) & ~ndimage.binary_opening(
                ndimage.binary_fill_holes(held) & ~held, iterations=SMALL_HOLE_PX
            )
            valid = node_mask(work, surface, r, shape) & held
            top, left, *flips = placement(r, shape, surface)
            valid = upright(valid, flips)
            a0, a1 = max(top, self.y0), min(top + shape[0], self.y0 + self.shape[0])
            b0, b1 = max(left, self.x0), min(left + shape[1], self.x0 + self.shape[1])
            if a1 <= a0 or b1 <= b0:
                continue
            here = (slice(a0 - self.y0, a1 - self.y0), slice(b0 - self.x0, b1 - self.x0))
            there = (slice(a0 - top, a1 - top), slice(b0 - left, b1 - left))
            free = valid[there] & (self.owner[here] < 0)
            self.owner[here][free] = k
            self.parts.append((k, flips, here, there, free))
        self.valid = self.owner >= 0
        yy, xx = np.mgrid[0 : self.shape[0] : 1, 0 : self.shape[1] : 1]
        outline = np.asarray(spot["outline_mm"]) / um - np.array([self.x0, self.y0])
        self.outline = outline
        within = np.zeros(self.shape, dtype=bool)
        sub = (slice(None, None, STAT_EVERY), slice(None, None, STAT_EVERY))
        points = np.stack([xx[sub].ravel(), yy[sub].ravel()], axis=1).astype(np.float64)
        within[sub] = inside(points, outline).reshape(xx[sub].shape)
        self.within = within & self.valid

    def laid(self, images):
        """One frame from each piece's image (indexed like records), NaN where no piece shows."""
        out = np.full(self.shape, np.nan, dtype=np.float32)
        for k, flips, here, there, free in self.parts:
            if k in images:
                out[here][free] = upright(images[k], flips)[there][free]
        return out


def pooled(values, low=LOW, high=HIGH):
    v = values[np.isfinite(values)]
    return float(v.mean()), float(v.std() or 1.0), *np.percentile(v, [low, high]).tolist()


def to_image(frame, low, high, valid, ink_dark=True):
    level = np.nan_to_num(np.clip((frame - low) / max(high - low, 1e-9), 0, 1))
    if ink_dark:
        level = 1 - level
    out = (level * 255).astype(np.uint8)
    out[~valid] = 0
    return Image.fromarray(out)


def write_spot(spot, records):
    work = work_of(spot["spot_id"])
    page = PAGES / spot["spot_id"]
    page.mkdir(parents=True, exist_ok=True)
    rows, manifest = [], {"surfaces": {}}
    for surface, models in READ_BY.items():
        if not all((work / "stacks" / surface / f"{r['name']}.npy").exists() for r in records):
            continue
        mosaic = Mosaic(spot, records, surface, work)
        names = {k: r["name"] for k, r in enumerate(records)}
        reads = {}
        for model in models:
            maps = {
                k: np.load(work / "reads" / surface / model / f"{n}.npy", mmap_mode="r")
                for k, n in names.items()
                if (work / "reads" / surface / model / f"{n}.npy").exists()
            }
            if maps:
                reads[model] = maps
        if not reads:
            continue
        sample = mosaic.valid.copy()
        sample[np.arange(sample.shape[0]) % STAT_EVERY != 0] = False
        sample[:, np.arange(sample.shape[1]) % STAT_EVERY != 0] = False

        def frame(model, s, d, maps_of=reads, mosaic=mosaic):
            return mosaic.laid({k: m[s, d].astype(np.float32) for k, m in maps_of[model].items()})

        inward = SIDES.index("inward")
        samples = {m: np.stack([frame(m, inward, d)[sample] for d in range(len(DEPTHS))]) for m in reads}
        stats = {m: pooled(v, INK_LOW, INK_HIGH) for m, v in samples.items()}
        pooled_ensemble = np.mean([(samples[m] - stats[m][0]) / stats[m][1] for m in reads], axis=0)
        stats["ensemble"] = pooled(pooled_ensemble, INK_LOW, INK_HIGH)
        band_ensemble = np.stack(
            [pooled_ensemble[max(0, d - BAND) : d + BAND + 1].mean(axis=0) for d in range(len(DEPTHS))]
        )
        stats["ensemble 5 depths"] = pooled(band_ensemble, INK_LOW, INK_HIGH)
        del samples, pooled_ensemble, band_ensemble
        shown = ["ensemble 5 depths", "ensemble", *reads]
        z_bounds = {
            k: ((s[2] - s[0]) / s[1], (s[3] - s[0]) / s[1]) if k in reads else (s[2], s[3]) for k, s in stats.items()
        }

        def write(stack, side, depth, image, folder_of=page / "images" / surface, valid=mosaic.valid, bounds=z_bounds):
            folder = folder_of / stack.replace(" ", "-") / side
            folder.mkdir(parents=True, exist_ok=True)
            low, high = bounds[stack]
            to_image(image, low, high, valid).save(folder / f"d{depth:+03d}.jpg", quality=QUALITY)

        inner = {}
        for s, side in enumerate(SIDES):
            ring = {}
            for d, depth in enumerate(DEPTHS):
                zs = {m: (frame(m, s, d) - stats[m][0]) / stats[m][1] for m in reads}
                zs["ensemble"] = np.mean([zs[m] for m in reads], axis=0)
                ring[d] = zs["ensemble"].astype(np.float16)
                for stack in shown[1:]:
                    write(stack, side, depth, zs[stack])
                    inner[(stack, side, depth)] = zs[stack][mosaic.within]
                del zs
                centres = [c for c in range(len(DEPTHS)) if c + BAND == d or (d == len(DEPTHS) - 1 and c + BAND > d)]
                for c in centres:
                    band = [ring[k].astype(np.float32) for k in range(max(0, c - BAND), min(len(DEPTHS), c + BAND + 1))]
                    image = np.mean(band, axis=0)
                    write("ensemble 5 depths", side, DEPTHS[c], image)
                    inner[("ensemble 5 depths", side, DEPTHS[c])] = image[mosaic.within]
                for k in [k for k in ring if k < d - 2 * BAND]:
                    del ring[k]
        for (stack, side, depth), values in inner.items():
            other = inner[(stack, "outward" if side == "inward" else "inward", depth)]
            rows.append(
                {
                    "spot_id": spot["spot_id"],
                    "surface": surface,
                    "model": stack,
                    "side": side,
                    "depth": depth,
                    "mean_z": round(float(np.nanmean(values)), 4) if values.size else None,
                    "p95_z": round(float(np.nanpercentile(values, 95)), 4) if values.size else None,
                    "inky_share": round(float(np.nanmean(values > INKY_Z)), 5) if values.size else None,
                    "other_side_mean_z": round(float(np.nanmean(other)), 4) if other.size else None,
                    "image": str(page / "images" / surface / stack.replace(" ", "-") / side / f"d{depth:+03d}.jpg"),
                }
            )
        stacks = {k: np.load(work / "stacks" / surface / f"{n}.npy", mmap_mode="r") for k, n in names.items()}

        def layer(depth, mosaic=mosaic, stacks=stacks):
            return mosaic.laid({k: s[MIDDLE + depth].astype(np.float32) for k, s in stacks.items()})

        ct_stats = pooled(np.stack([layer(depth)[sample] for depth in DEPTHS]))
        folder = page / "images" / surface / "ct" / "inward"
        folder.mkdir(parents=True, exist_ok=True)
        for depth in DEPTHS:
            to_image(layer(depth), ct_stats[2], ct_stats[3], mosaic.valid, ink_dark=False).save(
                folder / f"d{depth:+03d}.jpg", quality=QUALITY
            )
        manifest["surfaces"][surface] = {
            "stacks": [k.replace(" ", "-") for k in shown] + ["ct"],
            "best_depth": int(
                DEPTHS[
                    int(
                        np.argmax(
                            [np.nanpercentile(inner[("ensemble", "inward", d)], 97) for d in DEPTHS]
                            if inner[("ensemble", "inward", DEPTHS[0])].size
                            else [0]
                        )
                    )
                ]
            ),
            "stats": {k: [round(v, 4) for v in s] for k, s in stats.items()},
            "valid_share": round(float(mosaic.valid.mean()), 4),
            "within_cells": int(mosaic.within.sum()),
        }
        manifest["width"], manifest["height"] = mosaic.shape[1], mosaic.shape[0]
        manifest["outline"] = mosaic.outline.round(1).tolist()
        print(spot["spot_id"], surface, "images", len(shown), "stacks", flush=True)
    manifest |= {
        "spot": {
            k: spot[k] for k in ("spot_id", "scroll", "scan", "route", "page", "place", "area_cm2", "x_mm", "y_mm")
        },
        "depths": list(DEPTHS),
        "sides": list(SIDES),
        "pieces": [{k: r[k] for k in ("name", "share", "winding", "reading_side", "snap")} for r in records],
    }
    (page / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (page / "index.html").write_text(PAGE)
    database.write("reads", rows)
    write_index()


def write_index():
    """The catalogue of the circles as one table, each linked to its sweep page once it has one."""
    PAGES.mkdir(parents=True, exist_ok=True)
    spots = sorted(
        (r for r in database.rows("spots") if r["target"]),
        key=lambda r: (r["scroll"], r["page"], r["place"], r["spot_id"]),
    )
    found = {p.name for p in PAGES.iterdir() if (p / "manifest.json").exists()}
    verdicts = {}
    for v in database.rows("verdicts"):
        verdicts.setdefault(v["spot_id"], []).append(v)
    rows = []
    for r in spots:
        link = f'<a href="{r["spot_id"]}/index.html">{r["spot_id"]}</a>' if r["spot_id"] in found else r["spot_id"]
        said = "; ".join(f"{v['by']}: {v['readable']} {v['letters']}" for v in verdicts.get(r["spot_id"], []))
        rows.append(
            f"<tr><td>{link}</td><td>{r['scroll']}</td><td>{r['page']} {r['part']}</td><td>{r['place']}</td>"
            f"<td>{r['area_cm2']}</td><td>{r['covered_share']}</td><td>{r['n_pieces']}</td>"
            f"<td>{len(r['windings'])}</td><td>{r['main_share']}</td><td>{r.get('status', '')}</td><td>{said}</td></tr>"
        )
    (PAGES / "index.html").write_text(
        "<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Spot candidates</title><style>body{background:#111;color:#ddd;font:13px system-ui;margin:16px}"
        "a{color:#7cf}table{border-collapse:collapse}td,th{padding:2px 8px;border-bottom:1px solid #333;text-align:left}"
        "th{position:sticky;top:0;background:#181818}</style></head><body>"
        f"<h1>Circled spots: {len(spots)} circles, {len({r['place'] for r in spots})} places</h1>"
        "<p>A linked id has its surface rebuilt on m7 and read at every depth from -15 to +15.</p>"
        "<table><tr><th>spot</th><th>scroll</th><th>sheet</th><th>place</th><th>cm²</th><th>covered</th>"
        "<th>pieces</th><th>windings</th><th>main piece</th><th>status</th><th>verdicts</th></tr>"
        + "\n".join(rows)
        + "</table></body></html>"
    )


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spot depth sweep</title>
<style>
:root{--bg:#111;--fg:#ddd;--muted:#888;--accent:#39ff14}
body{background:var(--bg);color:var(--fg);font:13px/1.4 system-ui;margin:0}
header{position:sticky;top:0;background:#181818;padding:8px 12px;display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;z-index:2}
label{color:var(--muted)}select,button{background:#222;color:var(--fg);border:1px solid #444;padding:2px 6px}
#depth{width:240px}#d{font:600 15px ui-monospace,monospace;min-width:3ch;display:inline-block}
#wrap{position:relative;display:inline-block;margin:8px;transform-origin:center}#view{display:block;max-width:none}
#ol{position:absolute;inset:0;pointer-events:none}
.inv #view{filter:invert(1)}#info{color:var(--muted);padding:0 12px;white-space:pre-wrap;font:12px ui-monospace,monospace}
</style></head><body>
<header>
<b id="title"></b>
<label>surface <select id="surface"></select></label>
<label>stack <select id="stack"></select></label>
<label>side <select id="side"></select></label>
<label>depth <input id="depth" type="range" min="-15" max="15" value="0"> <span id="d">0</span></label>
<label>zoom <select id="zoom"><option>0.25</option><option selected>0.5</option><option>1</option><option>2</option></select></label>
<button id="invert">invert</button><button id="circle">circle</button>
<button id="mirror">mirror</button><button id="turn">turn 180°</button>
<span style="color:var(--muted)">← → depth · ↑ ↓ stack</span>
</header>
<div id="info"></div>
<div id="wrap"><img id="view" alt=""><svg id="ol"></svg></div>
<script>
const $=id=>document.getElementById(id);
let m,showCircle=true;
fetch('manifest.json').then(r=>r.json()).then(x=>{m=x;
 $('title').textContent=`${m.spot.spot_id} · ${m.spot.scroll} · ${m.spot.page} · ${m.spot.area_cm2} cm²`;
 for(const s of Object.keys(m.surfaces)) $('surface').add(new Option(s));
 for(const s of m.sides) $('side').add(new Option(s));
 $('info').textContent='pieces: '+m.pieces.map(p=>`${p.name} share ${p.share} winding ${p.winding} production reads ${p.reading_side||'?'} · snap median ${p.snap.median_move_vox} vox`).join('\\n');
 fillStacks();const b=m.surfaces[$('surface').value].best_depth;if(b!==undefined)$('depth').value=b;show();});
function fillStacks(){const s=$('stack'),keep=s.value;s.innerHTML='';
 for(const k of m.surfaces[$('surface').value].stacks) s.add(new Option(k));
 if([...s.options].some(o=>o.value===keep)) s.value=keep;}
function show(){const d=+$('depth').value,stack=$('stack').value,side=stack==='ct'?'inward':$('side').value;
 $('d').textContent=(d>0?'+':'')+d;
 $('view').src=`images/${$('surface').value}/${stack}/${side}/d${d<0?'-':'+'}${String(Math.abs(d)).padStart(2,'0')}.jpg`;
 const z=+$('zoom').value;$('view').style.width=(m.width*z)+'px';
 const svg=$('ol');svg.setAttribute('viewBox',`0 0 ${m.width} ${m.height}`);svg.style.width=(m.width*z)+'px';svg.style.height=(m.height*z)+'px';
 svg.innerHTML=showCircle?`<polygon points="${m.outline.map(p=>p.join(',')).join(' ')}" fill="none" stroke="#39ff14" stroke-width="${3/z}"/>`:'';}
for(const id of ['surface']) $(id).onchange=()=>{fillStacks();show();};
for(const id of ['stack','side','depth','zoom']) $(id).oninput=show;
$('invert').onclick=()=>document.body.classList.toggle('inv');
let mirror=false,turn=false;
function orient(){$('wrap').style.transform=`${turn?'rotate(180deg) ':''}${mirror?'scaleX(-1)':''}`;
 try{localStorage.setItem('spot-orient',JSON.stringify({mirror,turn}));}catch(e){}}
try{const o=JSON.parse(localStorage.getItem('spot-orient')||'{}');mirror=!!o.mirror;turn=!!o.turn;}catch(e){}
$('mirror').onclick=()=>{mirror=!mirror;orient();};$('turn').onclick=()=>{turn=!turn;orient();};orient();
$('circle').onclick=()=>{showCircle=!showCircle;show();};
addEventListener('keydown',e=>{const r=$('depth'),s=$('stack');
 if(e.key==='ArrowRight'){r.value=Math.min(15,+r.value+1);show();e.preventDefault();}
 if(e.key==='ArrowLeft'){r.value=Math.max(-15,+r.value-1);show();e.preventDefault();}
 if(e.key==='ArrowDown'){s.selectedIndex=(s.selectedIndex+1)%s.options.length;show();e.preventDefault();}
 if(e.key==='ArrowUp'){s.selectedIndex=(s.selectedIndex+s.options.length-1)%s.options.length;show();e.preventDefault();}});
</script></body></html>
"""
