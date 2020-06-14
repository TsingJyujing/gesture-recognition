# Models and Experiment

## Palm Detection

We're using BlazePalm from mediapipe for Plam Detection.
But the model in mediapipe have some problem, it's using itself defined operation: `Convolution2DTransposeBias`
For more details look [this issue](https://github.com/google/mediapipe/issues/35) and [this issue](https://github.com/google/mediapipe/issues/245).

It equals to TRANSPOSE_CONV + ADD but can make inference 2.5x faster!
But I'd like to make things simple & universe in the begining.
So I'm trying to find a solution which replaced the op with default ops:

- [wolterlw/hand_tracking](https://github.com/wolterlw/hand_tracking)
- [metalwhale/hand_tracking](https://github.com/metalwhale/hand_tracking)