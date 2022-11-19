# SPIGA: Dataloaders
Currently, the repository provides a pytorch based image dataloader implementation for the task of:
* **Facial Landmarks estimation**.
* **Headpose estimation**.
* **Facial landmarks visibilities**.

The dataloader can be used for training or testing the networks and it includes general and 
specifics data augmentation for each task, as it can be image partial occlusions 
or headpose generation from facial landmarks. 

In addition, the framework provides a wide benchmark software in order to evaluate the 
different task along the following databases:
* **WFLW**.
* **MERLRAV (AFLW 68)**
* **COFW68**.
* **300W Public, Private**.
<details>
  <summary> Coming soon... </summary>
  
* AFLW, AFLW19, AFLW2000 (test only).
* Menpo and 3D Menpo.
* COFW.
* 300WLP
* 300W Masked

</details>

***Note:*** All the callable files provide a detailed parser that describes the behaviour of 
the program and their inputs. Please, check the operational modes by using the extension ```--help```.

## Training/Testing 
The dataloader structure can be found in ```./data/loaders/aligments.py``` and it can be 
manually controlled by instantiating the class ```AlignmentsDataset()``` or by using 
the ```data_config``` structure available in ```./data/loaders/dl_config.py```.

Each image sample will follow the next configuration:
```
sample = {'image': Data augmented crop image,
          'sample_idx': Image ID,
          'imgpath': Absolute path to raw image,
          'imgpath_local': Relative path to raw image,
          'ids_ldm': Landmarks ids,
          'bbox': Face bbox [x,y,w,h] (ref crop),
          'bbox_raw': Face bbox [x,y,w,h] (ref image),
          'landmarks': Augmented landmarks [[x1,y1], [x2,y2], ...] (ref crop)
          'visible': Visibilities [0,1, ...] (1 == Visible)
          'mask_ldm': Available landmarks anns [True, False, ...] <- len(ids_ldm)
          'headpose': Augmented POSIT headpose [yaw, pithc, roll]
          }

Extra features while debugging:
sample = { ...
          'landmarks_ori' = Ground truth landmarks before augmentation (ref image)
          'visible_ori' = Ground truth visibilities before augmentation
          'mask_ldm_ori' = Ground truth mask before augmentation
          'headpose_ori' = Ground truth headpose before augmentation (if available)
         }
```

## Visualizers
The dataloader framework provides complementary visualizers to further understand the databases,
datasets and their difficulties:

* ```./data/visualize/inspect_dataset.py```   
Focus on the database annotations and the data augmentations, which allows us to 
understand the training, validation and test datasets.

* ```./data/visualize/inspect_heatmaps.py```   
Extended visualizer version focus on understanding the heatmaps and boundaries features used for training.

* ```./data/model3D/visualization.py```   
Visualize the rigid facial 3D models used by SPIGA to project the initial coordinates of the GAT regressor.
