## Face Video-Analyzer Framework
This demo application provides a general framework for tracking, detecting and extracting features of human faces in images or videos. 
Currently, the following tools  are integrated in a video-demo available at ```./spiga/demo/app.py```:

* Tracking:
    * FaceTracker : RetinaFace + SORT.
    * Detectors:
        * RetinaFace (bbox + 5 landmarks).
            * Backbone Mobilenet 0.25.
            * Backbone Resnet50.
    * Associators:
        * SORT: Frame by frame
* Extractors:
    * SPIGA architecture 
        * Features: Landmarks and headpose.
        * Datasets: 300W Public, 300W Private, WFLW, MERLRAV.
* Viewers:
    * Landmarks, headpose and bbox (score + face_id).

### Demo Application

```
python ./spiga/demo/app.py \
            [--input] \      # Webcam ID or Video Path. Dft: Webcam '0'.
            [--dataset] \    # SPIGA pretrained weights per dataset. Dft: 'wflw'.
            [--tracker] \    # Tracker name. Dft: 'RetinaSort'.
            [--show] \       # Select the attributes of the face to be displayed. Dft: ['fps', 'face_id', 'landmarks', 'headpose']
            [--save] \       # Save record.
            [--noview] \     # Do not visualize window.
            [--outpath] \    # Recorded output directory. Dft: './spiga/demo/outputs'
            [--fps] \        # Frames per second.
            [--shape] \      # Visualizer shape (W,H).
```

### Code Structure
The demo framework has been organised according to the following structure:

```
./spiga/demo/
| app.py                            # Video-demo
|
└───analyze                                  
│   | analyzer.py                   # Generic video/image analyzer compositor
│   │
│   └───features                    # Object/Faces classes
│   │
│   └───track                       # Task heads 
│   |   | tracker.py                # Tracker class         
│   |   | get_tracker.py            # Get model tracker from zoo by name
│   |   └─── retinasort             # RetinaFace + SORT tracker (tracker + zoo files)
|   |
|   └───extract
|       | processor.py              # Processor classes
|       └ spiga_processor.py        # SPIGA wrapper
|   
└───visualize
|   | viewer.py                     # Viewer manager
|   | plotter.py                    # Englobe available features drawers
|   └─── layouts                    # Landmarks, bbox, headpose drawers
|
└───utils                           # Video converters
```