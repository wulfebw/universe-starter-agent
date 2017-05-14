# universe-starter-agent
See https://github.com/openai/universe-starter-agent for original readme

# extensions

## action space
- how do you control the mouse in these environments?
- the options seem to be 
1. connect a dense layer to the output of the rnn with num_outputs = number of possible mouse positions
- see `DiscreteToMouseCoordVNCActions` in envs.py 
2. connect convolutional and upsampling layers to the rnn output
- gives the same number of outputs as the dense case, but with a lot fewer parameters
- also accounts for spatial correlation in action probabilities
- also uses `DiscreteToMouseCoordVNCActions` in envs.py 
3. treat the mouse action as a 2d continuous coordinate vector and map to pixel space
- see `ContinuousToMouseCoordVNCActions` in envs.py
- this is implemented in model.py itself
4. only allow relative movements (discrete or continuous), and track the position over tim
- see `DiscreteToMouseMovementVNCActions` in envs.py for discrete version with 4 actions of set distance
- extensions to this would be to allow the network to output the movement distance as well
- could also output polar coordinates for direction of movement

