import numpy as np
import random
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import requests, gzip, os, hashlib




# Define a class Hopfield Net with states and weights
class Hopfield_Net:
    '''This class contains states, weights and the update functions for hopfield network'''

    def __init__(self, input_memory, init_state=np.int16(0)):
        '''
        initialize states
        initialize weights
        '''
        # make sure input_memory is a np.ndarray
        self.input_memory = np.array(input_memory)

        # pick out number of neurons and number of patterns
        self.patterns = self.input_memory.shape[0] # patterns
        self.num_neurons = self.input_memory.shape[1] # neurons
        # self.num_neurons = np.int(np.sqrt(self.neurons_squared))

        # if init_state = 0, randomly initialize the netwotk
        if type(init_state) == np.int16:
            # initialize states to random +/-1 values
            self.neuron_states = rnd.randint(-1, 2, self.num_neurons) # make an array of -1, 0, 1 values with #entries = #neurons^2
            # replace zeros by -1
            self.neuron_states[self.neuron_states == 0] = -1
        else:
            self.neuron_states = init_state
            self.neuron_states[self.neuron_states == 0] = -1
        

        # # initialize states to random +/-1 values
        # self.neuron_states = rnd.randint(-1, 2, self.num_neurons) # make an array of -1, 0, 1 values with #entries = #neurons^2
        # # replace zeros by -1
        # self.neuron_states[self.neuron_states == 0] = -1

        # store the initial random state for future reference
        self.init_random_state = np.copy(self.neuron_states)

        # initialize weights
        self.weights = rnd.rand(self.num_neurons, self.num_neurons)

        # store the initial random matrix for future reference
        self.init_weights = np.copy(self.weights)

        # compute weights
        self.compute_weights()

    def compute_weights(self):
        '''Function to compute weights i.e. finding correlation matrix'''
        # Implementing W = (1/N)Y'Y
        pattern1 = self.input_memory[0,:]/np.sqrt(np.dot(self.input_memory[0,:], self.input_memory[0,:]))
        pattern2 = self.input_memory[1,:]/np.sqrt(np.dot(self.input_memory[1,:], self.input_memory[1,:]))
        self.input_memory = np.stack((pattern1, pattern2))
        self.weights =  (self.input_memory.T @ self.input_memory)
        np.fill_diagonal(self.weights, 0)
        # plt.imshow(self.weights, cmap='Spectral')
        # plt.colorbar()
        return self.weights


    def update_network_state(self, update_neurons):
        '''
        - Function to update the state of the network 'update_neurons' neurons at a time. 
        - Also returns the trajectory of energy
        - Takes a random neuron at a time and updates it's state. Random samples are taken without replacement.
        '''
        # construct an array with random indices from 0-783 without replacement
        self.rand_idx_arr = rnd.choice(self.num_neurons, size=update_neurons, replace=False) #random.sample(range(self.num_neurons), num_iterations) 

        # # compute weights
        # self.weights = self.compute_weights()

        # initialize an empty array to store energy as a function of iterations
        self.energy_arr = []

        # iterate over the random neurons to update the states
        for i in self.rand_idx_arr:
            
            self.activation = np.dot(self.weights[i, :], self.neuron_states)

            # update states based on neuron activation
            if self.activation < 0:
                self.neuron_states[i] = -1
            else:
                self.neuron_states[i] = 1

            # compute energy
            self.energy = -(self.neuron_states.T @ self.weights @ self.neuron_states)/2
            # append to the energy array
            self.energy_arr.append(self.energy)

        return self.neuron_states, self.energy_arr
  
    # make a function to update a given neuron at a time for animation purposes
    def update_neuron_state(self, neuron_number):
        '''This function takes in a neuron number (0-784) and updates its state'''

        # make sure that the neuron_number  variable in int
        neuron_number = np.int16(neuron_number)

        activation = np.dot(self.weights[neuron_number, :], self.neuron_states)

        # update states based on neuron activation
        if activation < 0:
            self.neuron_states[neuron_number] = -1
        else:
            self.neuron_states[neuron_number] = 1

        # compute energy
        energy = -(self.neuron_states.T @ self.weights @ self.neuron_states)/2

        return self.neuron_states, energy
    
    # function to compute energy of a given state
    def compute_energy(self, input_state):
        '''define a function to compute energy of a given input state'''
        return -(input_state.T @ self.weights @ input_state)/2
    
     

 

###################################################################################
############################ INDEPENDENT FUNCTIONS ################################
###################################################################################
## Off the shelf function to import MNIST dataset and converting it to binary data
#Fetch MNIST dataset from the ~SOURCE~

def fetch_MNIST(url):
    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)

    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

def make_binary_MNIST(): #test out the Hopfield_Network object on some MNIST data

    #fetch MNIST dataset for some random memory downloads
    X = fetch_MNIST(
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
        )[0x10:].reshape((-1, 28**2))

    #convert to binary
    X_binary = np.where(X>50, 1,-1)
    return X_binary

def pick_patterns(X_binary, num_patterns):
    '''Picks two num_patterns random patterns from binary MNIST dataset'''
    random_indices = rnd.randint(0, X_binary.shape[1], num_patterns)

    # initialize a matrix to store the random patterns. dimension: num_patterns x num_neurons (784)
    patterns_mat = np.zeros((num_patterns, X_binary.shape[1]))

    # set up a for loop and append the patterns in the numpy array
    for i in range(num_patterns):
        patterns_mat[i, :] = X_binary[random_indices[i], :]

    return patterns_mat

def pick_fixed_patterns(X_binary, indices, num_patterns):
    '''Picks num_patterns  patterns from binary MNIST dataset'''
    idx = indices#rnd.randint(0, X_binary.shape[1], num_patterns)

    # initialize a matrix to store the random patterns. dimension: num_patterns x num_neurons (784)
    patterns_mat = np.zeros((num_patterns, X_binary.shape[1]))

    # set up a for loop and append the patterns in the numpy array
    for i in range(num_patterns):
        patterns_mat[i, :] = X_binary[idx[i], :]

    return patterns_mat

## Make a function to move the images towards the left or right boundaries
def shift_image(im, direction):
    '''
    Moves the image im (28, 28) to left or right
    direction: string, 'l' or 'r' for left and right respectively
    It moves the first or last 8 columns
    '''
    if im.shape != (28, 28):
        raise Exception("The dimension of input image is not (28, 28)")

    if direction == 'r':
        im_trunc = im[:, 22:]
        im = np.concatenate((im_trunc, im[:, :22]), axis = 1)

    else:
        im_trunc = im[:, 0:8]
        im = np.concatenate((im[:, 8:], im_trunc),  axis= 1)

    return im

## Make a function to output corrupted memories given a fraction of bits to be flipped
def corrupt_input(input_memory, frac_flip):
    '''
    Returns a corrupted form of the input. Make sure that the input is a 1-D array
    '''

    # pick random indices without replacement
    rand_idx = rnd.choice(input_memory.shape[0], np.int16(np.floor(frac_flip*input_memory.shape[0])), replace=False)
    
    # make a copy of the input memory to corrupt
    corr_mem = np.copy(input_memory)
    
    # flip specified pixels from corr_mem
    corr_mem[rand_idx] = -corr_mem[rand_idx]

    return corr_mem



    







