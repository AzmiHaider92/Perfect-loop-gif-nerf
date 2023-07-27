# Perfect-loop-gif using Neural Radiance Fields (NeRF)  
A perfect-loop-gif is a gif where the transition between the first and last frame is done smoothly enough the naked eye can't distinguish where the series of images begin and end. The action in the gif appears to be repeated instead of played back.  
The aim of this project is to create a *steady perfect-loop-gif* with the help of a neural radiance fields technique.  
  
Example of a perfect loop gif:  

<p align="center">
  <img src="extra/BMW-i8-360.gif" width="400" />
</p>

# Data 
The objects used in this project are scenes taken from the dataset: https://ai.meta.com/datasets/CO3D-dataset/   
Notice how at the end of the gif the last frame "jumps" to the first.  

<p align="center">
  <img src="extra/rgb_maps.gif" width="400" />
</p>

The "jump" in frames happens because the camera position (position of the person capturing photos around the object) at the first frame and the last frame are not identical. Meaning, the person didn't return to the starting point - see camera positions below.

# Approach 
  
To create the *perfect-loop-gif* using NeRF, we follow these steps:  
1) Extract images from the gif.  
   Since a gif is a series of images, we can extract and save them as individual images.
2) Calculate the camera position for each image. We use colmap for that (Colmap is a structure from motion framework that generates camera positions for a given series of images).   
3) Train NeRF on the extracted images (+ generated camera positions).
   For this step, we use the TensoRF framework to train NeRF on the specific scene (object).  
4) Create a new camera path (new camera positions).
   To create a steady smooth gif, we create a camera path (positions) that surround the object in a steady way.   
5) Run trained nerf network on the new path to generate new images to match the new camera path.  
6) Create a new gif from the new generated images.

**1. Extracting images from gif**

<p align="center">
  <img src="extra/extractedImages.png" width="600" />
</p>

**2. Camera Positions**  
We used Colmap to generate camera positions for extracted images.
Below is a visualization of the camera positions around the object (number of frame is next to the point in black).  
The blue points represent the path that the person recording is taking around the object - assume object is placed at (0,0,0).  
You can see clearly that the person holding the camera is moving more than a 360deg around the object. Meaning, the person is not returning to the origin point.  
Also, the Z axis shows that the person is moving up and down while capturing the video.   

<p align="center">
  <img src="extra/cameraPositions.png" width="900" />
</p>

**3. NeRF: Neural Radiance Fields**  
For this part, we had the freedom to use any variation of NeRF.  
We chose TensoRF: https://apchenstu.github.io/TensoRF/    
The reason we chose this one is because the pytorch implementation is very clear and the run-time is short.   
The training PSNR is 37.451 (the images are very good and no floating noise around the object).  

We also ran Instant-ngp and the result is good. It is definetly another way to go.  
<p align="center">
  <img src="extra/instantngp.png" width="500" />
</p>


**4. New camera positions**   
After training the model on the scene, we want to use it to generate images to help create a perfect-loop GIF.  
For this purpose, we need a new camera path (points of view) where the starting point and the end point are the same.

A top-view of the given camera path (assume object at (0,0))
<p align="center">
  <img src="extra/topview.png" height="250", width="250" />
</p>

We implemented two methods of creating a closed camera path:  

   
**4.1 fixing the existing camera path**  
What we mean by fixing the existing path is: 
1) Find the overlap point in the camera path:  
   Meaning, find the point where the person completed a 360 degree circle around the object.  
   Figure (a) shows the overlap point in red.  
2) Ignore all position after the overlap point.  
   Every position of camera after it completed a 360 degree is not relevant. Figure (b) shows those positions in red.  
   Also, we added some margin and Ignored positions before and after the overlapping point. Figure (b) shows the margin in yellow.  
   This way we have a circling path around the object missing some positions.  
3) In those missing positions in the circle shown missing in figure (c), we will create new camera positions (shown in green in figure (d)) and generate images there so that the transition is smooth between start and finish.   
  
<p align="center">
  <img src="extra/fixoverlap.png" width="900" />
</p>
   

**4.2 New generic camera path**  
Another way to go is to ignore the whole camera path given and simply sample 360 camera positions in a steady circle around the object.   
This creates a more smooth path than before because the camera positions are uniformly distributed and at the same height.

<p align="center">
  <img src="extra/newcameraPositions.png" width="500" />
</p>

**5. Generating new images from the new camera positions**  
We use the trained model to generate new images at given camera positions.   
In 4.1, we created only 20 camera positions and hence 20 new images.  
In 4.2, we created 360 camera position. Generating all 360 images around the object.  
<p align="center">
  <img src="extra/generatedImages.png" width="600" />
</p>

**6. New gif**  
Putting together the generated images to create a perfect-loop gif.   

Fixing the existing path:  

<p align="center">
  <img src="extra/fixing_existing_path.gif" width="400" />
</p>

New generic path (unifrom 360):    
<p align="center">
  <img src="extra/new_path.gif" width="400" />
</p>



# References  
[https://apchenstu.github.io/TensoRF/  
](https://github.com/apchenstu/TensoRF)

[https://github.com/NVlabs/instant-ngp
](https://github.com/NVlabs/instant-ngp)https://github.com/NVlabs/instant-ngp

