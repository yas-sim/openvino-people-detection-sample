# OpenVINO Python sample program - face detection, age/gender estimation, people detection, human pose estimation  

### Description:  
A simple OpenVINO Python sample code.    
- face detection  
- age/gender estimation  
- people detection  
- human pose estimation  
![demo](./resources/demo.gif)  

### Prerequisites:  
- '**pose_extractor**' Python module  

You need to build **pose_extractor** module with `build_demos` script in OpenVINO and copy `pose_extractor.pyd`(win) or `pose_extractor.so`(linux) to the same directory as Python script.  
 ```sh
 cd $INTEL_OPENVINO_DIR/deployment_tools/open_model_zoo/demos/build_demos.sh -DENABLE_PYTHON=ON  
 ```


