# SIHE: estimation of building height from a single street view image 
This repository contains the Python implementation of a building height estimation method.

# Introduction
SIHE is a tool to estimate building height from a single street view image based on single view metrology. In the approach, geometric information and features, such as vanishing points and vertical lines, are automatically extracted through deep neural networks and are processed for height estimation. Furthermore, a simulation system is included in this repository for theoretically analysing how the factors influence single-view height measurement. The uncertainty of height can be calculated and simulated using the system.

# Setup
Simply clone this repo or download the zip file onto your local machine, then install `requirements.txt` file to install relevant python packages:

```
$ git clone https://github.com/
$ python install -r requirements.txt
```

# Code Structure
Below is a quick overview of the function of each file. The code will be available soon.

```bash
########################### height estimation code ###########################    
config/                         # configurations
    estimation_config.ini       # default parameters for height estimation
data/                           # default folder for placing the data
    imgs/                       # folder for original street view images
    lines/                      # folder for detected line segment files
    segs/                       # folder for semantic segmented image files
    vpts/                       # folder for detected vanishing point files
misc/                           # misc files
simulationSystem/               # scripts for simulation and theoretical analysis

demo.py                         # main function
filesIO.py                      # functions for loading files
heightMeasurement.py            # functions for height measurement
lineClassification.py           # functions for line segment classification
lineDrawingConfig.py            # script for line visualization configuration
lineRefinement.py               # functions for line segment refinement
```

# How to use
Use `demo.py` to run the code with sample data and default parameters. Execute the following command in the terminal, or add `img_path config_fname` in the Parameters when run `demo.py` in PyCharm:
```bash
python ./demo.py ./data/imgs/ ./config/estimation_config.ini
```

The height estimation results will be written to `./data/ht_results/` accordingly.

The main estimation function is the `heightCalc()` function and 
there are two important parameters:

* `use_pitch_only`: when the value is '1', use only the pitch angle to calculate 
  vanishing line and vertical vanishing point for height measurement
* `use_detected_vpt_only`: when the value is '1', use only the detected vanishing
  points for height measurement
  
The vanishing points can be detected from the image using neural networks, 
or calculated using pitch angle (when the rotation angles of the image are known 
from the data source, e.g. Google Street View). The setting of the two parameters
depends on how vanishing points are obtained.

In the estimation of building height, semantic segmentation map of the street view 
image, line segments, and vanishing points are used. Since they can be obtained 
through different neural networks, the codes of the networks are not integrated 
in the main stream of height estimation. Instead, the street view images are 
separately processed by three networks, and the result files are prepared for later
height measurement (as in `./data/lines`, `./data/segs`, `./data/vpts`).

When the above mentioned result files are prepared, the config file `estimation_config.ini` 
can be modified in accordance with the data and the `demo.py` can be used to estimate heights.

# Example results
![fig.1](./misc/figs/00000_1.png) ![fig.2](./misc/figs/00000_2.png) ![fig.3](./misc/figs/00002_1.png)        |


