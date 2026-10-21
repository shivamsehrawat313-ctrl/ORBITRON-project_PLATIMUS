"""Greedy one-to-one footprint matching with overlap/nearby and merge ambiguity review."""
from dataclasses import dataclass, asdict

@dataclass
class Change:
    id: str
    category: str
    before_index: int | None = None
    after_index: int | None = None
    area_before: float | None = None
    area_after: float | None = None
    similarity: float | None = None
    review_score: float = 1.0
    review_status: str = "needs_review"
    flags: list | None = None
    def to_dict(self): return asdict(self)

def _iou(a,b):
    if not a.is_valid or not b.is_valid: return 0.0
    union=a.union(b).area
    return a.intersection(b).area/union if union else 0.0

def compare_footprints(before, after, iou_threshold=.25, max_centroid_distance=3.0,
                       area_change_threshold=.35, reliable=True):
    """Input polygons share a projected CRS (or pixel space); units follow that CRS.

    One-to-one maximum-similarity greedy matching prevents duplicate matches.
    Nearby unmatched adjacency indicating split/merge is emitted as uncertain.
    """
    if not before and not after: return []
    scores=[]
    for i,a in enumerate(before):
        for j,b in enumerate(after):
            if a.is_empty or b.is_empty or not a.is_valid or not b.is_valid: continue
            iou=_iou(a,b); dist=a.centroid.distance(b.centroid)
            if iou>=iou_threshold or dist<=max_centroid_distance:
                proximity=(1-dist/max_centroid_distance) if max_centroid_distance>0 and dist<=max_centroid_distance else 0
                sim=.75*iou+.25*proximity
                scores.append((sim,iou,dist,i,j))
    scores.sort(reverse=True)
    used_a=set(); used_b=set(); pairs=[]
    for sim,iou,dist,i,j in scores:
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); pairs.append((i,j,sim,iou))
    out=[]; n=0
    def emit(**kwargs):
        nonlocal n
        n+=1; out.append(Change(id=f"CHG-{n:05d}",**kwargs))
    for i,j,sim,iou in pairs:
        a,b=before[i],after[j]; ratio=abs(b.area-a.area)/max(a.area,b.area,1e-12)
        flags=[]; category="unchanged"
        if ratio>=area_change_threshold: category="modified"; flags.append("area_change")
        if iou<iou_threshold:
            category="uncertain"; flags.append("low_overlap_match")
        elif ratio>=area_change_threshold:
            category="modified"
        else:
            category="unchanged"
        if sim<.45:
            category="uncertain"; flags.append("low_similarity")
        if not reliable: category="uncertain"; flags.append("alignment_or_image_reliability")
        emit(category=category,before_index=i,after_index=j,area_before=float(a.area),area_after=float(b.area),
             similarity=float(sim),review_score=float(sim),review_status="needs_review" if category=="uncertain" else "unreviewed",flags=flags)
    for side,polys,used in (("removed",before,used_a),("new",after,used_b)):
        for i,p in enumerate(polys):
            if i in used: continue
            # An unmatched object near any already matched opposite-side object can be split/merge/segmentation ambiguity.
            opposite=after if side=="removed" else before
            valid=p.is_valid and not p.is_empty
            ambiguous=any(q.is_valid and not q.is_empty and p.distance(q)<=max_centroid_distance for q in opposite)
            category="uncertain" if (ambiguous or not reliable or not valid) else side
            flags=[]
            if ambiguous: flags.append("possible_split_or_merge")
            if not reliable: flags.append("alignment_or_image_reliability")
            if not valid: flags.append("invalid_geometry")
            emit(category=category,before_index=i if side=="removed" else None,
                 after_index=i if side=="new" else None,area_before=float(p.area) if side=="removed" else None,
                 area_after=float(p.area) if side=="new" else None,review_score=.35 if category=="uncertain" else .7,
                 review_status="needs_review",flags=flags)
    return out
