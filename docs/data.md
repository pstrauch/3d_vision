# Data

## Structure

- 'mode' is a placeholder for either train, val or test
- The data for a corresponding dataset is stored in `data/<dataset>/` (e.g. data/h2o/)
- The action labels are extracted and stored in `action_labels_mode.npy`
- We sample a fixed amount of frames from each action sequence
- The sampled sequences are stored in the subdirectory `seq_n_mode`, where n is the number of sampled frames.

## Types and Formats

- We generally store extracted data as np.NDArrays in '.npy' files
- Extracted sequences are labeled by the corresponding id (note: h2o starts with id 1)
- Arrays consisting of data for all sequences (e.g. action_labels) are in the order of the action ids. **Note:** Since h2o starts with id=1, the corresponding data is stored in the array at index=0.