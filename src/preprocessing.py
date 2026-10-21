"""Tiling primitives; model-specific normalization belongs to its adapter."""
def tile_windows(width: int, height: int, tile_size: int = 512, overlap: int = 64):
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size: raise ValueError("Require tile_size > overlap >= 0")
    step=tile_size-overlap
    for y in range(0,height,step):
        for x in range(0,width,step): yield (x,y,min(tile_size,width-x),min(tile_size,height-y))
