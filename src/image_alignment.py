"""Metadata-level alignment gates; pixel registration is deliberately not guessed."""
from .image_validation import compare_metadata

def estimate_translation(image_a,image_b,max_side=1024):
    """Estimate integer translation by phase correlation on a sampled grayscale pair.

    This is a diagnostic only: no image is shifted or resampled. Low peak quality
    means the temporal imagery is too different for this estimate to be trusted.
    """
    import numpy as np
    a=np.asarray(image_a); b=np.asarray(image_b)
    if a.shape[:2]!=b.shape[:2]: raise ValueError("Alignment images must have equal dimensions")
    if a.ndim==3: a=.299*a[...,0]+.587*a[...,1]+.114*a[...,2]
    if b.ndim==3: b=.299*b[...,0]+.587*b[...,1]+.114*b[...,2]
    h,w=a.shape; step=max(1,int(np.ceil(max(h,w)/max_side)))
    a=a[::step,::step].astype(np.float32); b=b[::step,::step].astype(np.float32)
    a-=float(a.mean()); b-=float(b.mean())
    sa=float(a.std()); sb=float(b.std())
    if sa<1e-6 or sb<1e-6:
        return {"shift_x_pixels":None,"shift_y_pixels":None,"peak_ratio":0.0,"pixel_precision":step,"reliable":False}
    a/=sa; b/=sb
    win=np.outer(np.hanning(a.shape[0]),np.hanning(a.shape[1])).astype(np.float32)
    fa=np.fft.fft2(a*win); fb=np.fft.fft2(b*win)
    cross=fa*np.conj(fb); mag=np.abs(cross)
    cross/=np.maximum(mag,1e-12)
    corr=np.abs(np.fft.ifft2(cross)); peak=np.unravel_index(int(np.argmax(corr)),corr.shape)
    sidelobe=corr.copy()
    py,px=peak
    for dy in range(-2,3):
        for dx in range(-2,3): sidelobe[(py+dy)%corr.shape[0],(px+dx)%corr.shape[1]]=0
    side_mean=float(sidelobe.mean())
    ratio=float(corr[peak]/max(side_mean,1e-12))
    sy=py if py<=corr.shape[0]//2 else py-corr.shape[0]
    sx=px if px<=corr.shape[1]//2 else px-corr.shape[1]
    sx*=step; sy*=step
    return {"shift_x_pixels":int(sx),"shift_y_pixels":int(sy),"peak_ratio":ratio,
            "pixel_precision":step,"reliable":bool(ratio>=5 and step<=2)}

def alignment_diagnostic(a,b):
    warnings=compare_metadata(a,b)
    if not a.georeferenced or not b.georeferenced:
        warnings.append("Pixel-space mode requires user-confirmed same area and alignment; geographic areas are disabled.")
        return {"reliable":False,"mode":"pixel-space","warnings":warnings}
    hard=bool(warnings)
    return {"reliable":not hard,"mode":"geospatial","warnings":warnings}
