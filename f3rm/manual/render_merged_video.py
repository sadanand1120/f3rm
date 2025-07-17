#!/usr/bin/env python3
import argparse, math
import cv2, numpy as np

def slider_path_fraction(frame_idx, num_frames):
    """
    Returns a fractional position [0–1] of the slider at this frame.
    Edit control_points to change the “wavy” path.
    """
    control_points = [0.99, 0.25, 0.75, 0.1, 0.9]
    segments = len(control_points) - 1
    seg_len = num_frames / segments
    seg = min(int(frame_idx // seg_len), segments - 1)
    t = (frame_idx - seg * seg_len) / seg_len
    # cosine easing
    t = (1 - math.cos(math.pi * t)) / 2
    return control_points[seg] * (1 - t) + control_points[seg+1] * t

def main():
    p = argparse.ArgumentParser(
        description="Create a back-and-forth sliding reveal of two videos"
    )
    p.add_argument("video1", help="Left-side video (e.g. rgb.mp4)")
    p.add_argument("video2", help="Right-side video (e.g. depth.mp4)")
    p.add_argument("output", help="Output filename (e.g. out.mp4)")
    p.add_argument("--fade-width", type=int, default=40,
                   help="Width of gradient blend zone")
    p.add_argument("--line-thick", type=int, default=2,
                   help="Slider line thickness")
    p.add_argument("--outline-thick", type=int, default=6,
                   help="Outline thickness for the slider line")
    p.add_argument("--handle-r-out", type=int, default=24,
                   help="Outer radius of the slider knob")
    p.add_argument("--handle-r-in", type=int, default=18,
                   help="Inner radius of the slider knob")
    args = p.parse_args()

    cap1 = cv2.VideoCapture(args.video1)
    cap2 = cv2.VideoCapture(args.video2)
    fps    = cap1.get(cv2.CAP_PROP_FPS)
    w      = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes= int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    for i in range(nframes):
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not (ret1 and ret2):
            break

        # get slider x by fraction
        frac = slider_path_fraction(i, nframes)
        x = int(frac * w)

        # merge and gradient fade
        merged = np.empty_like(f1)
        merged[:, :x] = f1[:, :x]
        merged[:, x:] = f2[:, x:]
        fw = args.fade_width
        x1, x2 = max(x-fw//2,0), min(x+fw//2, w)
        if x2 > x1:
            alpha = np.linspace(0,1,x2-x1)[None,:,None]
            merged[:, x1:x2] = (
                f1[:, x1:x2]*(1-alpha) + f2[:, x1:x2]*alpha
            ).astype(np.uint8)

        # draw slider line + outline
        cv2.line(merged, (x,0), (x,h),
                 (0,0,0), args.outline_thick, cv2.LINE_AA)
        cv2.line(merged, (x,0), (x,h),
                 (255,255,255), args.line_thick,    cv2.LINE_AA)

        # draw knob
        cy = h//2
        cv2.circle(merged, (x,cy), args.handle_r_out,
                   (0,0,0), -1, cv2.LINE_AA)
        cv2.circle(merged, (x,cy), args.handle_r_in,
                   (255,255,255), -1, cv2.LINE_AA)

        out.write(merged)

    cap1.release(); cap2.release(); out.release()

if __name__ == "__main__":
    main()
