from PIL import Image

gif = Image.open('demo_training.gif')
# 5 evenly spaced frames in range 190-400, but GIF has 267 frames
# So clamp to 190-266
n_frames = 0
start, end = 190, min(400, n_frames - 1) # 190-266
indices = [start + i * (end - start) // 4 for i in range(5)]
print(f'Extracting frames: {indices} (GIF has {n_frames} frames)')

for i, idx in enumerate(indices):
    gif.seek(idx)
    frame = gif.convert('RGBA')
    out = f'output/frames/frame_{i+1}_gifidx{idx}.pdf'
    frame.convert('RGB').save(out, 'PDF', resolution=150)
    print(f'Saved {out}')