"""Optional plotting helpers."""
def category_counts(rows):
    result={}
    for row in rows:
        c=row.category if hasattr(row,"category") else row.get("category","unknown")
        result[c]=result.get(c,0)+1
    return result
