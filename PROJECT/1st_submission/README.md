# INSTRUCTIONS

This repository contains all the files needed to run the first task of the PIV project.

The aim of this project is obtain a Birds Eye View perspective of a sheet of paper from video.\\ This is achieved in the 1st stage by determining the homography matrix by solving least squares with single value decomposition.

To run the program, go to `/PIV/1st_submission` directory and type:


```
  python pivproject2021.py 1 template_1.png output_files input_files_1
```

All dependancies can be installed with:

```
  pip install numpy scipy opencv-python opencv-contrib-python
```
