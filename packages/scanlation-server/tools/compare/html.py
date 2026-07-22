"""HTML vote pages for the compare_models harness: the crop-OCR diff table and
the detector box-overlay grid, each with a click-to-tally scoring UI."""
from __future__ import annotations

from pathlib import Path

from compare.registry import all_adapters


def _diff_spans(ref: str, s: str) -> str:
    """s shown verbatim (whitespace kept), but only the NON-whitespace runs that differ
    from ref are wrapped .d (red). Whitespace is ignored when diffing, so VLM-inserted
    spaces don't count as differences."""
    import difflib
    import html

    def strip(t):  # non-whitespace chars + each one's original index in t
        idx = [i for i, c in enumerate(t) if not c.isspace()]
        return "".join(t[i] for i in idx), idx

    rs, _ = strip(ref)
    ss, spos = strip(s)
    diff = [False] * len(s)  # per original char of s; whitespace stays False (plain)
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(None, rs, ss, autojunk=False).get_opcodes():
        if tag != "equal":
            for j in range(j1, j2):
                diff[spos[j]] = True
    out, k = [], 0
    while k < len(s):  # group consecutive same-flag original chars into spans
        m = k
        while m < len(s) and diff[m] == diff[k]:
            m += 1
        seg = html.escape(s[k:m])
        out.append(f'<span class="d">{seg}</span>' if diff[k] else seg)
        k = m
    return "".join(out)


_HTML_JS = """
(function(){
  var K=VK, tally={};  // VK injected per page ('ocrsel:' for recognizer, 'boxsel:' for detector) -> separate vote namespaces
  function refresh(){
    var t=document.getElementById('tally');
    t.innerHTML='선택수 — '+engs.map(function(e){return '<b>'+e+'</b> '+(tally[e]||0);}).join(' &nbsp;·&nbsp; ');
    var cb=document.getElementById('catbreak'); if(!cb) return;
    var cat={};  // category -> engine -> votes, aggregated live from the selected cells
    document.querySelectorAll('.eng.sel').forEach(function(td){
      var c=td.getAttribute('data-cat'), e=td.getAttribute('data-eng');
      (cat[c]=cat[c]||{})[e]=(cat[c][e]||0)+1;
    });
    var h='<table class="cb"><tr><th>분류</th><th>crops</th>'+engs.map(function(e){return '<th>'+e+'</th>';}).join('')+'</tr>';
    var tot={}, totN=0;
    catList.forEach(function(c){
      var row=cat[c]||{}, n=catN[c]||0; totN+=n;
      var best=0; engs.forEach(function(e){ if((row[e]||0)>best) best=row[e]||0; });
      h+='<tr><td class="cn">'+c+'</td><td class="cc">'+n+'</td>'+engs.map(function(e){
        var v=row[e]||0; tot[e]=(tot[e]||0)+v;
        return '<td'+((v===best&&v>0)?' class="win"':'')+'>'+v+'<i>'+(n?Math.round(100*v/n):0)+'%</i></td>';
      }).join('')+'</tr>';
    });
    var tb=0; engs.forEach(function(e){ if((tot[e]||0)>tb) tb=tot[e]||0; });  // overall winner
    h+='<tr class="tot"><td class="cn">합계</td><td class="cc">'+totN+'</td>'+engs.map(function(e){
      var v=tot[e]||0;
      return '<td'+((v===tb&&v>0)?' class="win"':'')+'>'+v+'<i>'+(totN?Math.round(100*v/totN):0)+'%</i></td>';
    }).join('')+'</tr></table>';
    cb.innerHTML=h;
  }
  function set(td,on){
    var e=td.getAttribute('data-eng'), k=K+td.getAttribute('data-key');
    if(on){td.classList.add('sel');tally[e]=(tally[e]||0)+1;try{localStorage.setItem(k,'1')}catch(x){}}
    else {td.classList.remove('sel');tally[e]=(tally[e]||0)-1;try{localStorage.removeItem(k)}catch(x){}}
  }
  document.addEventListener('click',function(ev){
    var td=ev.target.closest&&ev.target.closest('.eng'); if(!td) return;
    set(td,!td.classList.contains('sel')); refresh();
  });
  document.querySelectorAll('.eng').forEach(function(td){
    var on=false; try{on=localStorage.getItem(K+td.getAttribute('data-key'))==='1'}catch(x){}
    if(on){td.classList.add('sel');var e=td.getAttribute('data-eng');tally[e]=(tally[e]||0)+1;}
  });
  var rb=document.getElementById('reset');
  if(rb) rb.addEventListener('click',function(){
    document.querySelectorAll('.eng.sel').forEach(function(td){set(td,false)}); refresh();
  });
  refresh();
})();
"""


def render_vote_page(dest, *, title, css, legend, catwrap_summary, body,
                     engs, cat_list, cat_n, vote_ns) -> None:
    """Assemble one self-contained scoring page: the shared <head>/legend/category
    scaffold + the click-to-tally script (_HTML_JS), around a caller-built body.
    _write_ocr_html and _write_box_html differ only in css, legend, summary, the
    body, the per-category denominator (cat_n), and the vote namespace."""
    import json
    P = [f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><title>{title}</title><style>{css}</style></head><body>",
         legend,
         f"<details open id='catwrap'><summary>{catwrap_summary}</summary>"
         "<div id='catbreak'></div></details>"]
    P += body
    P.append(f"<script>var VK='{vote_ns}',engs={json.dumps(engs)},"
             f"catList={json.dumps(cat_list, ensure_ascii=False)},"
             f"catN={json.dumps(cat_n, ensure_ascii=False)};{_HTML_JS}</script></body></html>")
    dest.write_text("".join(P), encoding="utf-8")


def _write_ocr_html(dest: Path, images, ref_id: str, out_root: Path, *, embed: bool = True, cap: int = 400) -> None:
    """Self-contained HTML: per image a table of crop rows — the crop image, then each
    engine's text with only the runs that DIFFER from ref_id highlighted red. Engine
    cells are clickable to tally a manual "who read this crop best" vote per model (live
    count in the sticky bar, persisted in localStorage). Crop images are base64-embedded
    (portable) unless embed=False (relative <img src>)."""
    import base64
    import html
    css = ("body{font-family:'Segoe UI',system-ui,sans-serif;margin:14px;background:#1e1e1e;color:#d4d4d4}"
           "h2{margin:22px 0 4px;font-size:14px;color:#e0e0e0;border-bottom:1px solid #3a3a3a}"
           ".legend{position:sticky;top:0;background:#1e1e1e;padding:8px 0;border-bottom:1px solid #444;font-size:13px;z-index:3}"
           "#tally{margin-left:6px;color:#cfe8ff}#reset{margin-left:10px;font-size:12px;cursor:pointer;background:#333;color:#ddd;border:1px solid #555;border-radius:3px;padding:1px 8px}"
           ".d{background:#6e1f24;color:#ffd0d0;font-weight:600}"
           "table{border-collapse:collapse;width:100%;table-layout:fixed;margin-bottom:6px}"
           "th,td{border:1px solid #3a3a3a;padding:4px 6px;vertical-align:top;font-size:13px;word-break:break-word;line-height:1.55}"
           "th{background:#2d2d2d;color:#e0e0e0;position:sticky;top:36px}"
           "td.idx,th.idx{width:30px;text-align:center;color:#888}td.ref{background:#262626}"
           "td.eng{cursor:pointer}td.eng:hover{outline:1px solid #4a5a6a}td.sel{box-shadow:inset 0 0 0 2px #4fc3f7;background:#1e3a5f !important}"
           "td.im,th.im{width:210px}img{max-width:200px;max-height:170px;object-fit:contain;display:block;background:#f5f5f5;border:1px solid #555}"
           "details#catwrap{margin:6px 0 2px}summary{cursor:pointer;color:#cfe8ff;font-size:13px}"
           "table.cb{width:auto;border-collapse:collapse;margin:6px 0 10px;font-size:12px;table-layout:auto}"
           "table.cb th,table.cb td{border:1px solid #3a3a3a;padding:3px 9px;text-align:center;position:static}"
           "table.cb td.cn{text-align:left;color:#e0e0e0}table.cb td.cc{color:#888}"
           "table.cb td.win{background:#4a3b12;color:#ffe9a8;font-weight:700}"
           "table.cb i{color:#7f93a6;font-style:normal;font-size:10px;margin-left:4px}"
           "table.cb tr.tot td{border-top:2px solid #5a5a5a;font-weight:600}")
    clip = lambda t: t if len(t) <= cap else t[:cap] + f" …(+{len(t) - cap})"  # noqa: E731
    esc_attr = lambda t: html.escape(t, quote=True)  # noqa: E731

    def crop_cell(rel: str, i: int) -> str:
        p = out_root / rel / "crops" / f"crop_{i:02d}.png"
        if not p.exists():
            return "<span style='color:#777'>—</span>"
        src = ("data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()) if embed \
            else f"{rel}/crops/crop_{i:02d}.png"
        return f"<img src='{src}' loading='lazy'>"

    engs = []  # union of engines in encounter order, for the tally
    for _rel, cols, _rows in images:
        for c in cols:
            if c not in engs:
                engs.append(c)
    cat_n, cat_list = {}, []  # crops per category (denominator for per-category acceptance %)
    for rel, _cols, rows in images:
        c = rel.split("/")[0]
        if c not in cat_list:
            cat_list.append(c)
        cat_n[c] = cat_n.get(c, 0) + len(rows)
    body = []
    for rel, cols, rows in images:
        ri = cols.index(ref_id) if ref_id in cols else 0
        cat = rel.split("/")[0]  # category = first path segment, embedded in each cell for per-category tally
        body.append(f"<h2>{html.escape(rel)}</h2><table><tr><th class='idx'>#</th><th class='im'>crop</th>"
                    + "".join(f"<th>{html.escape(c)}{' (기준)' if i == ri else ''}</th>" for i, c in enumerate(cols)) + "</tr>")
        for i, row in enumerate(rows):
            ref = clip(row[ri])
            tds = []
            for j, txt in enumerate(row):
                inner = html.escape(clip(txt)) if j == ri else _diff_spans(ref, clip(txt))
                cls = "eng ref" if j == ri else "eng"
                tds.append(f"<td class='{cls}' data-eng='{esc_attr(cols[j])}' data-cat='{esc_attr(cat)}' "
                           f"data-key='{esc_attr(f'{rel}|{i:02d}|{cols[j]}')}'>{inner}</td>")
            body.append(f"<tr><td class='idx'>{i:02d}</td><td class='im'>{crop_cell(rel, i)}</td>" + "".join(tds) + "</tr>")
        body.append("</table>")
    render_vote_page(
        dest, title="OCR compare", css=css,
        legend=(f"<div class='legend'>기준 = <b>{html.escape(ref_id)}</b> · 차이만 <span class='d'>&nbsp;빨강&nbsp;</span> "
                f"(공백 무시) · 칸 클릭 = 득표 <span id='tally'></span><button id='reset'>초기화</button></div>"),
        catwrap_summary="분류별 채택률 (분류 × 엔진 — 득표수 · %)",
        body=body, engs=engs, cat_list=cat_list, cat_n=cat_n, vote_ns="ocrsel:",
    )


def _consolidate_box_images(out_root: Path):
    """(rel, [model ids present]) per image, from the <model>.png box-overlays that
    `batch` writes under out_root/<cat>/<img>/. Models kept in canonical detector order."""
    det_ids = [a.id for a in all_adapters() if a.kind == "detect"]
    dirs = sorted({p.parent for m in det_ids for p in out_root.rglob(f"{m}.png")})
    images = []
    for d in dirs:
        present = [m for m in det_ids if (d / f"{m}.png").exists()]
        if present:
            images.append((d.relative_to(out_root).as_posix(), present))
    return images


def _write_box_html(dest: Path, images, out_root: Path, *, embed: bool = False) -> None:
    """Self-contained HTML to score DETECTORS (sibling of _write_ocr_html): per image,
    each model's box-overlay side by side, each panel clickable to vote 'this model boxed
    the page best'. Per-model tally + per-category matrix, persisted in localStorage under
    the boxsel: namespace (separate from the recognizer votes). Overlays linked by relative path by
    default (full-page PNGs are big); embed=True base64-inlines them (portable, heavy)."""
    import base64
    import html
    css = ("body{font-family:'Segoe UI',system-ui,sans-serif;margin:14px;background:#1e1e1e;color:#d4d4d4}"
           "h2{margin:22px 0 6px;font-size:15px;color:#e0e0e0;border-bottom:1px solid #3a3a3a}"
           "h3{margin:14px 0 4px;font-size:13px;color:#bbb;font-weight:600}"
           ".legend{position:sticky;top:0;background:#1e1e1e;padding:8px 0;border-bottom:1px solid #444;font-size:13px;z-index:3}"
           "#tally{margin-left:6px;color:#cfe8ff}#reset{margin-left:10px;font-size:12px;cursor:pointer;background:#333;color:#ddd;border:1px solid #555;border-radius:3px;padding:1px 8px}"
           "details#catwrap{margin:6px 0 2px}summary{cursor:pointer;color:#cfe8ff;font-size:13px}"
           "table.cb{width:auto;border-collapse:collapse;margin:6px 0 10px;font-size:12px}"
           "table.cb th,table.cb td{border:1px solid #3a3a3a;padding:3px 9px;text-align:center}"
           "table.cb td.cn{text-align:left;color:#e0e0e0}table.cb td.cc{color:#888}"
           "table.cb td.win{background:#4a3b12;color:#ffe9a8;font-weight:700}"
           "table.cb i{color:#7f93a6;font-style:normal;font-size:10px;margin-left:4px}"
           "table.cb tr.tot td{border-top:2px solid #5a5a5a;font-weight:600}"
           ".row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}"
           ".eng.box{width:360px;border:1px solid #3a3a3a;border-radius:4px;padding:3px;cursor:pointer;background:#232323}"
           ".eng.box:hover{outline:1px solid #4a5a6a}.eng.box.sel{box-shadow:inset 0 0 0 3px #4fc3f7;background:#1e3a5f}"
           ".ml{font-size:12px;color:#cfe8ff;padding:2px 4px}"
           ".eng.box img{width:100%;display:block;background:#f5f5f5;border-radius:2px}")
    esc_a = lambda t: html.escape(t, quote=True)  # noqa: E731

    engs = []
    for _rel, models in images:
        for m in models:
            if m not in engs:
                engs.append(m)
    cat_n, cat_list = {}, []  # denominator = images per category (one overlay set per image)
    for rel, _m in images:
        c = rel.split("/")[0]
        if c not in cat_list:
            cat_list.append(c)
        cat_n[c] = cat_n.get(c, 0) + 1

    def src(rel: str, m: str):
        p = out_root / rel / f"{m}.png"
        if not p.exists():
            return None
        return ("data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()) if embed else f"{rel}/{m}.png"

    body = []
    last_cat = None
    for rel, _models in images:
        cat = rel.split("/")[0]
        if cat != last_cat:
            body.append(f"<h2>{html.escape(cat)}</h2>")
            last_cat = cat
        body.append(f"<h3>{html.escape(rel.split('/', 1)[-1])}</h3><div class='row'>")
        for m in engs:
            s = src(rel, m)
            if s is None:
                continue
            body.append(f"<div class='eng box' data-eng='{esc_a(m)}' data-cat='{esc_a(cat)}' "
                        f"data-key='{esc_a(f'{rel}|{m}')}'><div class='ml'>{html.escape(m)}</div>"
                        f"<img src='{s}' loading='lazy'></div>")
        body.append("</div>")
    render_vote_page(
        dest, title="BOX compare", css=css,
        legend=("<div class='legend'>detector 박스 채점 · 오버레이 클릭 = 득표 "
                "<span id='tally'></span><button id='reset'>초기화</button></div>"),
        catwrap_summary="분류별 채택률 (분류 × 모델 — 득표수 · %)",
        body=body, engs=engs, cat_list=cat_list, cat_n=cat_n, vote_ns="boxsel:",
    )
