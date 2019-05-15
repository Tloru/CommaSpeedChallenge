# Speed Challenge
Hello, welcome to this repo.
This is my code to attempt to solve Comma.ai's ML challenge.
Here's what the Challenge consists of:

>Welcome to the comma.ai 2017 Programming Challenge!  
Basically, your goal is to predict the speed of a car from a video.  
`data/train.mp4` is a video of driving containing `20400` frames. Video is shot at `20` fps.
`data/train.txt` contains the speed of the car at each frame, one speed on each line.  
`data/test.mp4` is a different driving video containing `10798` frames. Video is shot at `20` fps.
Your deliverable is `test.txt`
We will evaluate your test.txt using mean squared error. `<10` is good. `<5` is better. `<3` is heart.

## Plan
There were quite a few problems with my old method:

- I don't know how effective the frame difference method actually is.
  It might not be helping much.
- The model is not temporal, the only temporal information is encoded in the subtracted image.
- The model is not that big, and under fits the data.
  This is because I'm training it on an old MacBook Air, and so yeah.
- I'm normalizing the input images, but haven't gotten around to normalizing the output speeds.

How I will fix some of these issues:

- [x] Run some tests to figure out which frame preprocessing method is the best (difference, averaging, etc.).
    - Result: Frame averaging might be the best method,
      but since I'm preprocessing the images using InceptionV3,
      I'm only going to do cropping, no grayscale conversion or subtraction.
- [x] Add some RNN layers (LSTMs are probably overkill) to increase prediction accuracy.
      Since the frame rate is about `20` fps, there shouldn't be that much of a difference in speed -
      adding the previous frame's predicted speed as an input to the model might help.
    - Result: I've added 3 GRU layers, which should get the job done.
- [ ] I'll move the training to the cloud at some point to speed up the training -
      With this in place, I could also increase the complexity of the model
- [x] write a short script to normalize / denormalize the speeds.
    - Result: Integrated normalization, have not had the chance to see the training results yet.

Yeah, that's about it. I'm mainly putting this on Github to have a backup in case something goes wrong.

## Revised Method (Currently Being Implemented)
New training process:

```
get batch of frames from training video.
preprocess frames and get encodings using InceptionV3 with batch prediction.
process encodings and shape them to the right size.
→ [training input data]

read the test file for speeds
normalize the speeds by dividing by the mean (0 min, 1 average)
→ [training target data]

↓ [training input data + training target data]
assemble batches and train neural network
save the weights every few epochs
[neural network weights]
```

### Reasoning
> Why no longer use frame subtraction?

Because I'll be using inception v3 to generate encodings for each frame, and frame subtraction would mess with the encoding results.

> Why inception v3?

Inception v3, despite being a classification network, has been trained on massive amounts of data, and for this reason can easily extract details from images.

> What will happen to the temporal aspect?

Because these encodings are smaller than whole images, I plan to pass two encodings (temporally separated by one step) to the model at once.
In addition to these images, I'll also pass the previous frame's speed, because upon inspecting the data, I realized that from frame to frame, the speed didn't diverge that much.

> Where will you be working on this?

In the `develop` branch : )
