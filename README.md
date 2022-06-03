# Kaggle-IMC-28th-solution
The solution and code for Image Matching Challenge.

## Brief Summary
Ensemble of LoFTR, SuperGlue, DKM and ASLFeat with the resize calibration and appropriate number of keypoints for each model.

## Choosing a RANSAC
Set up: [baseline LoFTR](https://www.kaggle.com/code/mcwema/imc-2022-kornia-loftr-score-plateau-0-726), 100k iterations
|Approach | Threshold | Public LB |
| --- | --- | --- | 
| **MAGSAC** | **0.2**  | **0.726** |
| DEGENSAC  | 0.5 | 0.713 |
| DEGENSAC  | 0.2 | 0.684 |
| DEGENSAC  | 1.0 | 0.705 |
| GC-RANSAC| 0.2 | 0.671 |
| MAGSAC & DEGENSAC &  GC-RANSAC| 0.25, 1.0, 1.0 | 0.725 |
| MAGSAC & DEGENSAC &  GC-RANSAC| 0.25, 3.0, 3.0 | 0.727 (but too slow) |
| MAGSAC & DEGENSAC &  GC-RANSAC| 0.25, 0.5, 0.5 | 0.720 |

## Choosing an image size
Image size is crucial for LoFTR. Here is the proof.
Set up: [baseline LoFTR](https://www.kaggle.com/code/mcwema/imc-2022-kornia-loftr-score-plateau-0-726)
|Approach | Public LB |
| --- | --- |
| 640, 480 | 0.533  | 
| 840 longest size | 0.726 | 
| **Just calibrate image size in such way that both sides were divisible by 8**  | **0.755** | 

## Building a ZOO

### Models
- LoFTR
- SuperGlue
- DKM
- ASLFeat
- SGMNet (isn't used in final submission)

### Tuning
We've tried a bunch of different parameters different models. Only one worked. Switching `border_rm` parameter in LoFTR from 4 to 1 leads to generating more keypoints and gives 0.006 boost.

### Ensembling approach
1) Get the keypoints from all the models.
2) Concatenate them.
3) Run RANSAC algorithm.

We've also tried to concatenate keypoints after several RANSAC runs but it was too slow and was giving a worse score.

### Image sizes
I mentioned above that image size is crucial here, especially for LoFTR, but in the terms in ensembling it becomes even more important. So here are some points:

1) We've chosen `Just calibrate image size in such way that both sides were divisible by 8` approach for all the models and it was the only way where ensembling started giving a solid boost.
2) The first reason of it is that when LoFTR are more sensitive to image size, other models are less, so the rule `the bigger is better` works here.
3) Another reason is that in order to do a proper ensembling all keypoints should have the same scale, the same image size is the easiest way to achieve it. 
4) DKM resizes image to (512, 384) in its forward function, so it was necessarily to rewrite it in order to fix it.

### Keypoints number
The next important thing to do in order to get the most out of the ensemble is to select the number of points since the default value have wide range among the models which. **That's the reason why ensembles could not to work for some people.** 

Not all the models have a strict parameter for the number of final keypoints, so here we show how to set them up:
1) SuperGlue | `max_keypoints` in config: 1024 -> 2048
2) DKM | `num` in sample function: 2000 -> 300
3) ASLFeat | `kpt_n` in config: 8000 -> 2048

Our final keypoints' shapes for LoFTR, SuperGlue, DKM and ASLFeat accordingly look this way:
```
(871, 2) (247, 2) (300, 2) (407, 2)
(529, 2) (210, 2) (300, 2) (236, 2)
(210, 2) (42, 2) (300, 2) (193, 2)
```
We also tried to create an adaptive number of keypoints for these models. For example, based on the number of LoFTR's keypoints, but it didn't work. 

Decreasing the number of keypoints not only allows to create a stable ensemble but also gives a solid speed up.

### Overall
DR - default resize
CR - calibrated resize
| Approach | Public LB | Private LB |
| --- | --- | --- |
| LoFTR (DR) | 0.732 | 0.742 |
|  LoFTR (DR) +  SuperGlue (DR, max_keypoints=1024) | 0.701 | 0.709 |
| LoFTR  (CR) | 0.755 | 0.764 |
| LoFTR  (CR) + SuperGlue (CR, max_keypoints=1024) | 0.795 | 0.797 |
| LoFTR  (CR + SuperGlue (CR, max_keypoints=1024) + DKM (DR, num=2000)| 0.743 | 0.751 |
| LoFTR  (CR) + SuperGlue (CR, max_keypoints=1024) + DKM (CR, num=2000)| 0.786 | 0.799 |
| LoFTR  (CR) + SuperGlue (CR, max_keypoints=1024) + DKM (CR, num=200) | 0.816 | 0.820 |
| LoFTR  (CR) + SuperGlue (CR, max_keypoints=1024) + DKM (CR, num=200) + ASLFeat (CR, kpt_n=2048)| 0.819 | 0.821 |
| LoFTR  (CR) + SuperGlue (CR, max_keypoints=1024) + DKM (CR, num=200) + ASLFeat (CR, kpt_n=2048) + SGMNet | 0.818 | 0.818 |
| **LoFTR  (CR) + SuperGlue (CR, max_keypoints=2048) + DKM (CR, num=300) + ASLFeat (CR, kpt_n=2048)** | **0.820** | **0.827** |

## Handling exceptions
We haven't had any submission exceptions after adding this to our code:
```
if FM is None:
    F_dict[sample_id] = np.zeros((3, 3))
    continue
elif FM.shape != (3, 3):
    F_dict[sample_id] = np.zeros((3, 3))
    continue
```

## Another things that didn't work
- TTA (horizontal flilp)
- 90 degrees rotate
- DISK
- Inference in fp16 mode
- Tuning confidence thresholds

## Code and final keypoints
The notebook with an entire solution and keypoints examples can be found [here]([https://www.kaggle.com/code/vadimtimakin/imc-solution](https://github.com/t0efL/Kaggle-IMC-solution/blob/main/imc-solution.ipynb))!
