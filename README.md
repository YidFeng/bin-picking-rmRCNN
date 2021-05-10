# bin-picking-rmRCNN
### For Training
```bash
cd dataset
python save_json_complete.py --dataset small_for_test --phase train --oc 0.8 --hr 0.5 --is_nohead True --is_hr True 
````
args:
- oc: occlusion rate
- hr: head ratio (a threshold for the percentage of continuous exposure to the head side)
- is_nohead: whether to label the head
- is_hr: whether to control head ratio in the labeling 

```bash
python train-rmRCNN.py --user_config_file "./configs/user_rmRCNN.yaml" --data_dir './dataset/small_for_test'
````

### For Testing
```bash
python test.py --model_path './models/05r_0057999.pth'
````
### For Downloading Trained model of object 05
google drive: https://drive.google.com/file/d/14EeWqa8i72egt51nKTGSHN3JlKzuwcK-/view?usp=sharing
