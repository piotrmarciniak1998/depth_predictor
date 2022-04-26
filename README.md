# 🚧 Project `depth_predictor` (in progress)
Project's goal is to predict the image without the objects that are obstructing the view on other items in the scene. 
The neural network model takes depth image of the scene as an input, and tries to remove the obstructing object creating 
a new depth image (output). Currently, the only expected target is a selected mug. The project uses the artificial 
dataset created by [`image_generator`](https://github.com/piotrmarciniak1998/image_generator). 

# ⚒️ TODO
1. Create new neural network models.
2. Tweak models' parameters.
3. Check if adding rgb images to input can improve results.

# 💻 Components
Scripts:
* `create_dataset.py` - the script takes `.zip` directories of smaller datasets to join them together. The datasets 
  should be placed inside `depth_predictor/input_dataset`. The final dataset should be located in `dataset` directory 
  with `input` and `ground_truth` directories inside.

Neural network models:
* `basic_model` - first test of neural network based on `resnet34`.
