# trame-quart

Slice viewer for a NIfTI image and a mask. `cases.csv` maps the URL ids to `.nii.gz` paths.

```bash
docker build -t trame-quart .
docker run --rm -p 80:80 \
  -v /mnt/hd2/data/Totalsegmentator_dataset:/mnt/hd2/data/Totalsegmentator_dataset:ro \
  trame-quart
```

http://localhost/?image=case0&mask=case0_lung

The comma form works too: `/?image=case0,mask=case0_lung`
