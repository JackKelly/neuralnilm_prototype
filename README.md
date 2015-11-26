# Neural NILM Prototype

Early prototype for the Neural NILM (non-intrusive load monitoring)
software.  This software will be completely re-written as the [Neural
NILM project](https://github.com/JackKelly/neuralnilm).

This is the software that was used to run the experiments for our 
[Neural NILM paper](http://arxiv.org/abs/1507.06594).

Note that `Neural NILM Prototype` is completely unsupported because
it is replaced by the
[Neural NILM project](https://github.com/JackKelly/neuralnilm).  Furthermore,
the code for `Neural NILM Prototype` is a bit of a mess!

Directories:

* `neuralnilm` contains re-usable library code
* `scripts` contains runnable experiments
* `notebooks` contains IPython Notebooks (mostly for testing stuff
  out)
  
The script which specified the experiments I ran in my paper is
[e567.py](https://github.com/JackKelly/neuralnilm_prototype/blob/master/scripts/e567.py).

(It's a pretty horrible bit of code!  Written in a rush!)  In that
script, you can see the `SEQ_LENGTH` for each appliance and the
`N_SEQ_PER_BATCH` (the number of training examples per batch).
Basically, the sequence length varied from 128 (for the kettle) up to
1536 (for the dish washer).  And the number of sequences per batch was
usually 64, although I had to reduce that to 16 for the RNN for the
longer sequences.

The nets took a long time to train (I don't remember exactly how long
but it was of the order of about one day per net per appliance).  You
can see exactly how long I trained each net in that `e567.py` script
(look at the `def net_dict_<architecture>` functions and look for
`epochs`.... that's the number of batches (not epochs!) given to the
net during training).  It's 300,000 for the rectangles net, 100,000
for the AE and 10,000 for the RNN (because the RNN was a *lot* slower
to train... I chose these numbers because the nets appeared to stop
learning after this number of training iterations).
