import numpy as np
from copy import deepcopy

# tournament selection
def selection(pop, scores, k):
    # first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
	    # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if np.random.binomial(1, r_cross):
		# select crossover point that is not on the end of the string
		pt = np.random.randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
	    # check for a mutation
        if np.random.binomial(1, r_mut):
	        # flip the bit
            bitstring[i] = 1 - bitstring[i]

def generate_random_individual(example, bit_precision=8):
    n_bits = 0
    for i, key in enumerate(example):
        n_params_layer = 1
        for d in example[key].shape:
            n_params_layer *= d
        n_bits += n_params_layer * bit_precision
    return np.random.randint(0, 2, n_bits).tolist()

def convert_binary_to_float(binary_repr, weight_range=(-1, 1)):
    """ Convert a binary array into a decimal number in some range

    Parameters
    ----------
    binary_repr: list
        binary representation of the number
    weight_range: tuple
        range to which the number should be mapped
    
    Return
    ------
    float decimal number corresponding to the mapping of binary_repr 

    """
    int_repr = int("".join(str(b) for b in binary_repr), 2)
    total_range = weight_range[1] - weight_range[0]
    smallest_representable_decimal = total_range / (2**len(binary_repr) - 1)
    return weight_range[0] + int_repr * smallest_representable_decimal

def convert_binary_weight_array_to_float_weight_array(binary_weight_array, weight_array_shape):
    """ Convert a binary array into an array of weights of the given shape
    """ 
    weight_array = np.zeros(weight_array_shape)
    # assume that the array is at most 2-dimensional
    if len(weight_array_shape) == 1: 
        # if the array is 1-dimensional, then the binary array contains weight_array_shape[0] weights one after the other
        assert len(binary_weight_array) % weight_array_shape[0] == 0
        binary_repr_len = len(binary_weight_array) // weight_array_shape[0]
        # unpack the weights one by one
        for i in range(weight_array_shape[0]):
            idx_start = i*binary_repr_len
            binary_weight = binary_weight_array[idx_start:idx_start+binary_repr_len]
            weight_array[i] = convert_binary_to_float(binary_weight) # fill the weight array
    else:
        # if the array is 2-dimensional, then it contains weight_array_shape[0] * weight_array_shape[1] weights following the order: 
        # start in the top-left corner and move towards the right, when the end of the row is reached, go to the next row 
        assert len(binary_weight_array) % (weight_array_shape[0] * weight_array_shape[1]) == 0
        binary_repr_len = len(binary_weight_array) // (weight_array_shape[0] * weight_array_shape[1])
        row_len = weight_array_shape[1] * binary_repr_len
        # unpack the weights one by one
        for j in range(weight_array_shape[0]): # loop on the rows
            for i in range(weight_array_shape[1]): # loop on the columns
                idx_start = j*row_len + i*binary_repr_len
                binary_weight = binary_weight_array[idx_start:idx_start+binary_repr_len]
                weight_array[j][i] = convert_binary_to_float(binary_weight) # fill the weight array
    return weight_array

def convert_individual_to_manager_weights(individual, manager_weights_ex, bit_precision=6):
    """ Convert an individual (i.e. a long binary array) into an OrderedDict of 
        decimal weights that can fit as weight dictionary for the manager using
        the specified bit precision
    """
    manager_weights = deepcopy(manager_weights_ex) 
    used_bits = 0
    for i, key in enumerate(manager_weights_ex): # loop over all the set of parameters for each layer
        # compute the number of parameters in the current layer
        n_params_layer = 1 
        for d in manager_weights_ex[key].shape:
            n_params_layer *= d
        # take the binary subarray that corresponds to the parameters of the current layer
        binary_weight_array = individual[used_bits:used_bits+n_params_layer*bit_precision]
        # convert the binary array into an array of decimal weights
        manager_weights[key] = convert_binary_weight_array_to_float_weight_array(binary_weight_array, manager_weights_ex[key].shape)
        # update the number of used bits of the individual
        used_bits += n_params_layer*bit_precision
    return manager_weights