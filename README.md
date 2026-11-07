# trame-quart

Slice viewer for a NIfTI image and a mask. `cases.csv` has `case,image_path,seg_path,spleen_path`.

```bash
docker build -t trame-quart .
docker run --rm -p 80:80 \
  -v /mnt/hd2/data/Totalsegmentator_dataset:/mnt/hd2/data/Totalsegmentator_dataset:ro \
  trame-quart
```

http://localhost/?image=case0

Three orthogonal views share one crosshair. `mask` is `seg` or `spleen`. `overlay=0` hides it.
