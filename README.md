# Perfect-loop-gif-nerf
The target of this project is to create a perfect loop gif using a neural radiance fields technique.  
example of a perfect loop gif:  

<p align="center">
  <img src="extra/animatedOutput_o.gif" width="200" />
</p>

# Data 
Scenes were taken from the dataset: https://ai.meta.com/datasets/CO3D-dataset/   
Example: (notice the end of the gif) 

<p align="center">
  <img src="extra/rgb_maps.gif" width="200" />
</p>

# Approach 
  
Since a gif is a series of images, to utilize nerf, we need to:  
1) extract images from the gif.
2) generate camera position for each image (we use colmap for that).
3) train nerf on the extracted images (+ generated camera positions).
4) create a new path (new camera positions), those will be our test set.
5) run trained nerf network on the new path to generate new images
6) create a new gif from the new images.


**Camera Positions**  
This is a visualization of the camera positions around the object (the path that the person recording is doing around the object - assume object is placed at (0,0,0).   
You can see clearly that the person holding the camera is moving more than a 360deg around the object. Meaning, the person is not returning to the origin point.  
Also, the Z axis shows that the person is moving up and down while capturing the video.   

<p align="center">
  <img src="extra/cameraPositions.png" width="900" />
</p>



