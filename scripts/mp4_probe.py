# -*- coding: utf-8 -*-
"""Minimal ISO-BMFF (MP4) probe: resolution, fps, frames, duration, decodability cues."""
import struct, sys, json

def boxes(buf, start, end):
    """yield (type, body_start, body_end, box_start, box_size)"""
    off = start
    while off + 8 <= end:
        size = struct.unpack(">I", buf[off:off + 4])[0]
        typ = buf[off + 4:off + 8].decode("latin-1")
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[off + 8:off + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - off
        if size < hdr or off + size > end:
            return
        yield typ, off + hdr, off + size, off, size
        off += size

def find(buf, path, start, end):
    cur = [(start, end)]
    for i, name in enumerate(path):
        nxt = []
        for (s, e) in cur:
            for typ, bs, be, _, _ in boxes(buf, s, e):
                if typ == name:
                    nxt.append((bs, be))
        if not nxt:
            return []
        cur = nxt
    return cur

def probe(path):
    with open(path, "rb") as f:
        buf = f.read()
    out = {"file": path, "bytes": len(buf), "ok": False, "notes": []}
    top = [(t, bs, be) for t, bs, be, _, _ in boxes(buf, 0, len(buf))]
    out["top_level_boxes"] = [t for t, _, _ in top]
    if not any(t == "ftyp" for t, _, _ in top):
        out["notes"].append("no ftyp box -> not a standard MP4")
        return out
    out["brand"] = buf[8:12].decode("latin-1", "replace")

    mv = find(buf, ["moov", "mvhd"], 0, len(buf))
    if mv:
        s, e = mv[0]
        ver = buf[s]
        if ver == 0:
            ts, du = struct.unpack(">II", buf[s + 12:s + 20])
        else:
            ts, du = struct.unpack(">IQ", buf[s + 20:s + 32])
        out["movie_timescale"] = ts
        out["movie_duration_s"] = round(du / ts, 3) if ts else None

    traks = find(buf, ["moov", "trak"], 0, len(buf))
    out["n_traks"] = len(traks)
    best = None
    for (ts_, te_) in traks:
        w = h = None
        tk = find(buf, ["tkhd"], ts_, te_)
        if tk:
            s, e = tk[0]
            ver = buf[s]
            base = s + (76 if ver == 0 else 88)
            if base + 8 <= e:
                wf, hf = struct.unpack(">II", buf[base:base + 8])
                w, h = round(wf / 65536.0, 1), round(hf / 65536.0, 1)
        md = find(buf, ["mdia", "mdhd"], ts_, te_)
        mts = mdu = None
        if md:
            s, e = md[0]
            ver = buf[s]
            if ver == 0:
                mts, mdu = struct.unpack(">II", buf[s + 12:s + 20])
            else:
                mts, mdu = struct.unpack(">IQ", buf[s + 20:s + 32])
        sz = find(buf, ["mdia", "minf", "stbl", "stsz"], ts_, te_)
        nframes = None
        if sz:
            s, e = sz[0]
            nframes = struct.unpack(">I", buf[s + 8:s + 12])[0]
        st = find(buf, ["mdia", "minf", "stbl", "stts"], ts_, te_)
        stts = None
        if st:
            s, e = st[0]
            n = struct.unpack(">I", buf[s + 4:s + 8])[0]
            stts = [struct.unpack(">II", buf[s + 8 + 8 * i:s + 16 + 8 * i]) for i in range(n)]
        ss = find(buf, ["mdia", "minf", "stbl", "stss"], ts_, te_)
        nkeys = None
        if ss:
            s, e = ss[0]
            nkeys = struct.unpack(">I", buf[s + 4:s + 8])[0]
        cand = {"width": w, "height": h, "media_timescale": mts, "media_duration_units": mdu,
                "n_frames": nframes, "stts": stts, "n_keyframes": nkeys}
        if nframes and (best is None or nframes > (best.get("n_frames") or 0)):
            best = cand
    if best:
        out.update(best)
        mts, mdu, nf = best.get("media_timescale"), best.get("media_duration_units"), best.get("n_frames")
        if mts and mdu:
            out["media_duration_s"] = round(mdu / mts, 3)
            if nf:
                out["fps"] = round(nf / (mdu / mts), 3)
        if best.get("stts"):
            out["stts_total_samples"] = sum(c for c, _ in best["stts"])
    mdats = [(be - bs) for t, bs, be in top if t == "mdat"]
    out["mdat_count"] = len(mdats)
    out["mdat_bytes"] = sum(mdats)
    out["ok"] = bool(out.get("n_frames") and out["mdat_bytes"] > 0)
    if out.get("n_frames") and out["mdat_bytes"] / out["n_frames"] < 100:
        out["notes"].append("bytes/frame < 100 -> possibly truncated")
    return out

if __name__ == "__main__":
    print(json.dumps([probe(p) for p in sys.argv[1:]], ensure_ascii=False, indent=2))
