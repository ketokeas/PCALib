# General description

All of the code necessary for the inference and extrapolation is stored in the `pcalib` folder.

List of files in the `pcalib` folder with brief descriptions:

- `classes.py` : Contains two classes (`Potential` and `Point`) used for the optimization.
- `functions.py` : Contains functions used for inference of model parameters from data and functions for making accuracy prediction for a given model.
- `utils.py` : Contains various helper functions.

Potential class represents the free energy landscape given in the main text of the paper [currently in writing; for the version without animal-to-animal variability see Legenkaia et al, 2025 (https://doi.org/10.1103/PhysRevE.111.044314)]. 
Point class represents a set of values of the parameters on which we try to find the optimum of the free energy landscape.

# Data formatting

Here we assume that the spiking activity has been preformatted into three-dimensional numpy array of the size `[n_trials,T,N]`, where

- `n_trials` is a number of trials;
- `T` is a number of time bins within one trial;
- `N` is a number of neurons.

To use the library, you should execute the the fuctions provided in the `functions.py`.


# Model inference

The model of a given three-dimensional dataset is represented as an instance of the `Potential` class. To create this instance, we have to fit all of the model parameters, which will correspond to class attributes. 

First, we have to fit the dimensionality K of the low-dimensional latent dynamics of the neural activity. This can be done by using the fuction determine_dimensionality from the `pcalib/functions.py`

Inputs: 
 - `non_flattened_data` : real data of the size `[n_trials,T,N]` that has been smoothed with the convolutional kernel (if used).
 - `mode` : indication whether the dataset is intended to be trial-averaged or trial-concatenated. Accepts string values `"trial-averaged"` or `"trial-concatenated"`;
 - `n_samples=1000` : number of shufflings of the original dataset for obtaining null data without any latent dynamics;

Outputs:
 - `K`: estimated dimensionality of the latent dynamics 


The inference of the model parameters from the data is done by the function `fit_statistics_from_dataset` from the `pcalib/functions.py`

Inputs:
 - `dataset` : real data of the size `[n_trials,T,N]` that has been smoothed with the convolutional kernel (if used);
 - `K` : assumed dimensionality of the latent dynamics;
 - `G` : a three-dimensional binary array of the size `[D,K,K]`, where `D` is a number of recorded animals. `G[d,i,i]=1` if neuron `i` belongs to animal `d`, zero otherwise. 
 - `gaussian_kernel_width` : width of the convolutional kernel used for data smoothing;
 - `mode` : indication whether the dataset is intended to be trial-averaged or trial-concatenated.  Accepts string values `"trial-averaged"` or `"trial-concatenated"`;
 
Outputs:
 - `potential` : an instance of the Potential class.
 
NOTE: There exists an approximate version of this function called  `fit_statistics_from_dataset_diagonal`, that treats different principal components independently, assuming there is no interaction between them. It has the same inputs as `fit_statistics_from_dataset`, and gives a list of `Potential` instances of length `K` - one instance per principal component.


# Changing parameters to adapt the dataset size

We know how our model parameters should change with the dataset size. This means that we can construct a `Potential` class for a hypothetical larger dataset based on the parameters that we have fitted from the currently available data.

For this, we use `extrapolate_potential` function from the `pcalib/functions.py`

Inputs: 
 - `original` : an instance of `Potential` with the data parameters fitted from the data;
 - `new_neurons` : new number of neurons `N`. Provide only if the number of neurons is expected to change.
 - `new_trials`: : new number of trials `n_trials`. Provide only if the number of trials is expected to change.
 - `existing_number_of_trials`: current number of trials `n_trials`. Provide only if the number of trials is expected to change.
 - `mode` : indication whether the dataset is intended to be trial-averaged or trial-concatenated. Accepts string values `"trial-averaged"` or `"trial-concatenated"`;
 - `new_bar_e` : Optional custom mode loadings `bar_e` for new neurons. Can be set if we assume something about the structure of the neural loadings, e.g. sparcity of the neural activity;
 - `new bar_sigma` : Optional custom noise strength for new neurons. Can be set if we assume something about the noise strength of the newly recorded neurons.
 - `new_G` : if we assume that more animals will be recorded, we have to provide information about which neuron belongs to which animal.

Outputs:
 - `pot` : new `Potential` instance, with the attributes set to describe the new hypothetical dataset.
	 
# Making  accuracy predictions

Accuracy measures `rho` and `epsilon` are expressed in terms of the order parameters found by searching for a saddle point of the free energy. 
The function make_predictions from `pcalib/functions.py` searches for the saddle point of a given `Potential` instance (which results in a `Point` instance), and then outputs the accuracy measures `rho` and `epsilon`.

Inputs:
 - `pot` : an instance of `Potential` class;
 - `compute_derivative` : if True, compute derivative of top eigenvalues w.r.t. sqrt(signal variances)
 - `return_R`: If True, include the `R` matrix in the output
 - `scale_factor`: Parameter included for numerical stability. Default value is None. If set as None, calculated automatically.
 
 Outputs:
 - `result` : Dictionary with keys: `"rho"`, `"epsilon"`, `"top_eigenvalues"`, and optionally `"d_lambda_d_sqrt_var"` and `"R"`
