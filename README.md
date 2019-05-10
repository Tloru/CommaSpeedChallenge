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

## Method
This repo has 3 programs:

- [x] `preprocess.py`: splits the videos into a sequence of images, and applies other transformations, such as cropping.
- [x] `train.py`: defines the model and trains it.
- [ ] `test.py`: predicts and assembles a list of speeds.

Right now, the model uses frame differences to predict speed.
2 (or any number) of images are subtracted from another, creating a stack of images

The network then convolves over this to produce an image encoding.
After then, a few dense layers are used for prediction.

So far, I've been able to reach a loss of about `<15`, and the goal is `<3`.

## Plan
There are a lot of problems with the above method:

- I don't know how effective the frame difference method actually is.
  It might not be helping much.
- The model is not temporal, the only temporal information is encoded in the subtracted image.
- The model is not that big, and under fits the data.
  This is because I'm training it on an old MacBook Air, and so yeah.
- I'm normalizing the input images, but haven't gotten around to normalizing the output ones.

How I will fix some of these issues:

- [ ] Run some tests to figure out which frame preprocessing method is the best (difference, averaging, etc.).
- [ ] Add some RNN layers (LSTMs are probably overkill) to increase prediction accuracy.
      Since the frame rate is about `20` fps, there shouldn't be that much of a difference in speed -
      adding the previous frame's predicted speed as an input to the model might help.
- [ ] I'll move the training to the cloud at some point to speed up the training -
      With this in place, I could also increase the complexity of the model
- [ ] write a short script to normalize / denormalize the speeds.

Yeah, that's about it. I'm mainly putting this on Github to have a backup in case something goes wrong.

# Running It
Documentation for `preprocess.py`:

```
usage:       $ python3 preprocess.py input_path output_path
input_path:  file path of video file to preprocess       - example: ./data/train.mp4
output_path: folder path of place to save frame sequence - example: ./train/
```

Documentation for `train.py`:

```
usage:        $ python3 train.py train_inp train_target
train_inp:    folder path of training frame sequence - example: ./train/
train_target: file path of training speed sequence   - example: ./data/train.txt
```
