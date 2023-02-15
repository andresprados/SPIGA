# SPIGA: Benchmark
The benchmark evaluator can be found at ```./eval/benchmark/evaluator.py``` and it allows
to extract an extended report of metrics for each dataset. For further details,
check the parser and complete the interactive terminal procedure to specify the evaluation
characteristics.

In order to use the benchmark evaluation, the prediction file must follow the same data structure
and file extension as the ground truth annotations available in ```./data/annotations/<database_name>```. 
The data structure consist on a list of dictionaries where each one represents an image sample,
similar to the previous dataloader configuration:

```
sample = {"imgpath": Relative image path, 
          "bbox": Bounding box [x,y,w,h] (ref image), 
          "headpose": Euler angles [yaw, pitch, roll], 
          "ids": Landmarks database ids,
          "landmarks": Landmarks (ref image), 
          "visible": Visibilities [0,1, ...] (1 == Visible)
          }
```

Finally, is worth to mention that the benchmark can be easily extended for other task by 
inheriting the class structure available in ```./eval/benchmark/metrics/metrics.py``` and 
developing a new task file like the available ones: landmarks and headpose.
