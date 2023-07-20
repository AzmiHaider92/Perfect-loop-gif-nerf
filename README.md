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
A gif is a series of images.  
To utilize nerf, we need to:  
1) extract images from the gif.
2) generate camera position for each image (we use colmap for that).
3) train nerf on the extracted images (+ generated camera positions).
4) create a new path (new camera positions), those will be our test set.
5) run trained nerf network on the new path to generate new images
6) create a new gif from the new images.



