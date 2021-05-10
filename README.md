# bin-picking-rmRCNN
### For Training
```bash
cd dataset
python save_json_complete.py --dataset small_for_test --phase train --oc 0.8 --hr 0.5 --is_nohead True --is_hr True 
````
args:
--oc: occlusion rate
--hr: head ratio (a threshold fo)
