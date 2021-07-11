# OpenVINO Python sample program - face detection, age/gender estimation, people detection, human pose estimation  

### Description:  
A simple OpenVINO Python sample code.    
- face detection  
- age/gender estimation  
- people detection  
- human pose estimation  

### Prerequisites:  
- '**pose_estimation**' Python module  
You need to build **pose_estimation** module with `build_demos` script in OpenVINO and copy `build_demos.pyd`(win) or `build_demos.so`(linux) to the same directory as Python script.  
 ```sh
 cd $INTEL_OPENVINO_DIR/deployment_tools/open_model_zoo/demos/build_demos.sh -DENABLE_PYTHON=ON  
 ```
