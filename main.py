import numpy as np
import matplotlib.pyplot as plt
import time
import _pickle as cPickle


# Function used for loading the CIFAR10 dataset
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

# Compute the softmax function of the output
def softmax(y):
    max_of_rows = np.max(y, 1)
    m = np.array([max_of_rows, ] * y.shape[1]).T
    y = y - m
    y = np.exp(y)
    return y / (np.array([np.sum(y, 1), ] * y.shape[1])).T


# Returns the outputs of the hidden level
def get_z(X, w1):
    a = X.dot(w1.T)

    z = activationFunction(a)

    # Z is N,M right now(since w1 is M), so we add ones at the beginning
    z = np.hstack((np.ones((z.shape[0], 1)), z))
    return z


# Returns the cost function and the gradients for w1,w2
def compute_gradients_cost(T, X, w1, w2, lamda):

    Z = get_z(X,w1)

    # The result of Z*w2
    z_w2 = Z.dot(w2.T)

    Y = softmax(z_w2)
    # Compute the cost function to check convergence
    max_error = np.max(z_w2, axis=1)
    Ew = np.sum(T * z_w2) - np.sum(max_error) - \
         np.sum(np.log(np.sum(np.exp(z_w2 - np.array([max_error, ] * z_w2.shape[1]).T), 1))) - \
         (0.5 * lamda) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))

    # Calculate gradient for w2
    grad_w2 = (T-Y).T.dot(Z) - lamda * w2

    # We remove the bias since z0 is not dependant by w1
    w2_temp = np.copy(w2[:, 1:])

    # This is the result of the derivative of the activation function
    der = activationFunctionDerivative(X.dot(w1.T))

    temp = (T-Y).dot(w2_temp) * der

    # Calculate gradient for w1
    grad_w1 = temp.T.dot(X) - lamda*w1

    return Ew, grad_w1, grad_w2


def train_neural_network(T, X, lamda, w1_init, w2_init, options):
    """inputs :
      t: N x 1 binary output data vector indicating the two classes
      X: N x (D+1) input data vector with ones already added in the first column
      lamda: the positive regularizarion parameter
      winit: D+1 dimensional vector of the initial values of the parameters
      options: options(1) is the maximum number of iterations
               options(2) is the tolerance
               options(3) is the learning rate eta
    outputs :
      w: the trained D+1 dimensional vector of the parameters"""

    w1 = np.copy(w1_init)
    w2 = np.copy(w2_init)

    # Maximum number of iteration for each season clean
    _iter = options[0]

    # Minibatch Size
    mb_size = options[1]

    n = X.shape[0]

    # Learing rate
    eta = options[2]
    # Since we apply gradients on batches the eta
    # needs to be relevant to the batch size not to the whole dataset
    eta = eta / mb_size

    # We save each cost we compute across all season in order to plot it
    costs = []

    # iter is the number of epoch the algorithm is running
    for i in range(_iter):
        # Shuffle the array's in the same order.
        #  If we don't shuffle them the same, a X_train row will not correspond to the original T row
        set = list(zip(X,T))
        np.random.shuffle(set)
        a, b = zip(*set)
        temp_X = np.asarray(a)
        temp_T = np.asarray(b)
        for e in range(0, n, mb_size):
            # Get the new elements for gradient ascent
            x_b = temp_X[e: e+mb_size, :]
            t_b = temp_T[e: e+mb_size, :]

            Ew, grad_w1, grad_w2 = compute_gradients_cost(t_b, x_b, w1, w2, lamda)

            # Save the cost
            costs.append(Ew)

            # Update parameters based on gradient ascend

            w1 += eta * grad_w1

            w2 += eta * grad_w2

    return w1, w2, costs

# Run the w1,w2 we caculcated for the test data
def run_test_final(w1, w2, x_test):
    Z = get_z(x_test, w1)

    z_w2 = Z.dot(w2.T)

    ytest = softmax(z_w2)
    # Hard classification decisions
    ttest = np.argmax(ytest, 1)
    return ttest

# Return the result of the activation function
def activationFunction(a):
    if activation_option == 0:
        return np.maximum(a, 0) + np.log(1 + np.exp(-np.abs(a)))
    elif activation_option == 1:
        return np.tanh(a)
    else:
        return np.cos(a)


# Return the result of the derivative of the activation function
def activationFunctionDerivative(a):
    if activation_option == 0:
        return np.exp(np.minimum(0,a))/(1+np.exp(-np.abs(a)))
    elif activation_option == 1:
        return 1 - np.tanh(a)**2
    else:
        return -(np.sin(a))


def load_data_mnist(data='mnist'):
    """
    Loads the MNIST dataset. Reads the training files and creates matrices.
    :return: train_data:the matrix with the training data
    test_data: the matrix with the data that will be used for testing
    train_truth: the matrix consisting of one
                        hot vectors on each row(ground truth for training)
    test_truth: the matrix consisting of one
                        hot vectors on each row(ground truth for testing)
    """
    train_files = [data+'/train%d.txt' % (i,) for i in range(10)]
    test_files = [data+'/test%d.txt' % (i,) for i in range(10)]
    tmp = []
    for i in train_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    # load train data in N*D array (60000x784 for MNIST)
    #                              divided by 255 to achieve normalization
    train_data = np.array([[j for j in i.split(" ")] for i in tmp], dtype='int') / 255
    print ("Train data array size: ", train_data.shape)
    tmp = []
    for i in test_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    # load test data in N*D array (10000x784 for MNIST)
    #                             divided by 255 to achieve normalization
    test_data = np.array([[j for j in i.split(" ")] for i in tmp], dtype='int') / 255
    print ("Test data array size: ", test_data.shape)
    tmp = []
    for i, _file in enumerate(train_files):
        with open(_file, 'r') as fp:
            for line in fp:
                tmp.append([1 if j == i else 0 for j in range(0, 10)])
    train_truth = np.array(tmp, dtype='int')
    del tmp[:]
    for i, _file in enumerate(test_files):
        with open(_file, 'r') as fp:
            for _ in fp:
                tmp.append([1 if j == i else 0 for j in range(0, 10)])
    test_truth = np.array(tmp, dtype='int')
    print ("Train truth array size: ", train_truth.shape)
    print ("Test truth array size: ", test_truth.shape)
    return train_data, test_data, train_truth, test_truth

def load_data_cifar10(data='cifar'):

    train_files = [data+'/data_batch_%d' % (i,) for i in range(1,6)]
    test_file = data+'/test_batch'

    train_data = []
    dictonaries = []
    train_truth = np.zeros((50000,10))
    k = 0

    # We store all data batch
    for i in train_files:
        dictonaries.append(unpickle(i))
    for batch in dictonaries:
        # For each input we append it
        for img in batch['data']:
            train_data.append(img)
        for label in batch['labels']:
            # for k image we put the label it belong to
            train_truth[k][label] = 1
            k += 1

    train_data = np.asarray(train_data)
    # We normalize the data. All values will be in [0,1]
    train_data = train_data/255

    #We do the same for the one test batch
    temp_dict = unpickle(test_file)
    test_data = []
    test_truth = np.zeros((10000,10))
    k = 0
    for img in temp_dict['data']:
        test_data.append(img)
    for label in temp_dict['labels']:
        #for k image we put the label it belong to
        test_truth[k][label] = 1
        k += 1
    test_data = np.asarray(test_data)
    # Normalize the test as well
    test_data = test_data/255
    return train_data, test_data, train_truth, test_truth


# Check the w1,w2 derivatives
def gradient_check(w1_init,w2_init, X, t, lamda):
    w1 = np.random.rand(*w1_init.shape)
    w2 = np.random.rand(*w2_init.shape)
    epsilon = 1e-6
    _list = np.random.randint(X.shape[0], size=5)
    x_sample = np.array(X[_list, :])
    t_sample = np.array(t[_list, :])
    Ew, gradw1, gradw2 = compute_gradients_cost(t_sample,x_sample,w1,w2, lamda)
    numericalGrad = np.zeros(gradw1.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad
    print (gradw1.shape , gradw2.shape , w1.shape, w2.shape)
    for k in range(numericalGrad.shape[0]):
        for d in range(numericalGrad.shape[1]):
            # Calculate W1 gradient
            w_tmp = np.copy(w1)
            w_tmp[k, d] += epsilon
            e_plus, _, _ = compute_gradients_cost(t_sample, x_sample, w_tmp, w2, lamda)

            w_tmp = np.copy(w1)
            w_tmp[k, d] -= epsilon
            e_minus, _, _ = compute_gradients_cost(t_sample, x_sample, w_tmp, w2, lamda)
            numericalGrad[k,d] = (e_plus - e_minus) / (2 * epsilon)

    # Absolute norm
    print ("The difference estimate for gradient of w1 is : ", np.max(np.abs(gradw1 - numericalGrad)))

    numericalGrad = np.zeros(gradw2.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad
    for k in range(numericalGrad.shape[0]):
        for d in range(numericalGrad.shape[1]):
            # Calculate W1 gradient
            w_tmp = np.copy(w2)
            w_tmp[k, d] += epsilon
            e_plus, _, _ = compute_gradients_cost(t_sample, x_sample,w1 ,w_tmp , lamda)

            w_tmp = np.copy(w2)
            w_tmp[k, d] -= epsilon
            e_minus, _, _ = compute_gradients_cost(t_sample, x_sample, w1, w_tmp, lamda)
            numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)

    # Absolute norm
    print ("The difference estimate for gradient of w2 is : ", np.max(np.abs(gradw2 - numericalGrad)))


def start(options, dataset):
    # The center of our distruption. Zero for our normalize data is perfect
    center = 0

    # The range of the distrubtion
    # Should always be relevant to the dimensions of the data
    s = 1/ np.sqrt(D+1)

    # Initialize the weights
    w_2 = np.zeros((K, M + 1))

    # We use this in order for our activation function to be more effective
    w_1 = np.random.normal(center,s,(M,D+1))

    # We add the bias
    w_1[:, 1] = 1
    w_2[:, 1] = 1

    # We use gradient check for both weights
    if i==1:
        gradient_check(w_1, w_2, X_train, y_train, lamda)

    # We use to calculate the time needed to train the model
    start_time = time.clock()

    # Start training the neural network
    w1_final, w2_final, costs = train_neural_network(y_train, X_train, lamda, w_1, w_2, options)

    # We compare the results against the real ones
    ttest = run_test_final(w1_final, w2_final, X_test)
    error_count = np.not_equal(np.argmax(y_test, 1), ttest).sum()

    print (error_count / y_test.shape[0] * 100)
    # We save the output to a file
    file = open(dataset+".txt", "a")
    file.write("\n"+str(activation_option) + "\t" + str(M) +"\t"+str(options[1])+"\t"+str(options[2])+"\t"+str(options[0]))
    file.write("\t"+str(error_count / y_test.shape[0] * 100)+"\t"+str(time.clock() - start_time))
    file.close()

    # We plot the result
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("M =" + str(M)+" Minibatch="+str(options[1]))

    # We save the plot as an image
    plt.savefig(dataset+'_Af_' + str(activation_option) + 'eta_' + str(options[2]) + 'M_' + str(M)+'mb_'+str(options[1])+'eta_'+str(options[0])+'.png', bbox_inches='tight')
    plt.clf()

##               CODE WE USED FOR RUNNING OUR EXPIREMENTS
##----------------------------------------------------------#
# Method for our expirements we were tickering values here
# to produce the results on the report
# X_train, X_test, y_train, y_test = load_data_mnist()
#
# N, D = X_train.shape
#
# # The number classes
# K = y_train.shape[1]
#
# # Adds a row of 1 in the beginning
# X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
# X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
# print "Train truth array size (with ones): ", X_train.shape
# print "Test truth array size (with ones): ", X_test.shape
# print "MNIST: "
# # Which activation function to use
# activation_options = [0, 1, 2]
# # lamda
# lamda = 0.01
# # learning rate
# # iteration
# iter_options = [400]
#
# mb_options = [100, 200]
# eta = 0.05
# # For all activation functions
# M_options = [100, 200, 300]
#
# for act in activation_options:
#     activation_option = act
#     for M in M_options:
#         for mb in mb_options:
#             for iter in iter_options:
#                 start([iter, mb, eta], "mnist")
#
# X_train, X_test, y_train, y_test = load_data_cifar10()
#
# N, D = X_train.shape
#
# # The number classes
# K = y_train.shape[1]
#
# # Adds a row of 1 in the beginning
# X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
# X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
# print "Train truth array size (with ones): ", X_train.shape
# print "Test truth array size (with ones): ", X_test.shape
#
# # Which activation function to use
# activation_options = [2]
# # lamda
# lamda = 0.01
# # learning rate
# # iteration
# iter_options = [400]
#
# mb_options = [100, 200]
# eta = 0.006
# # For all activation functions
# M_options = [100, 200, 300]
#
# for act in activation_options:
#     activation_option = act
#     for M in M_options:
#         for mb in mb_options:
#             for iter in iter_options:
#                 start([iter, mb, eta], "cifar")


# Initialize all the parameters
lamda = 0.01
eta = 0
iter = 0
M = 0
mb = 0
activation_option = -1

i = int( input('Chose a dataset: \n\t1 for MNIST \n\t2 for CIFAR-10\n>'))

if i > 2 or i < 1:
    print ("Invalid input!")
    exit()
print ("Loading data....")
# We put values to receive the optimal error score in each dataset based on our experiments
if i == 2:
    X_train, X_test, y_train, y_test = load_data_cifar10()
    eta = 0.005
    iter = 200
    M = 300
    mb = 100
    dataset = "cifar"
else:
    X_train, X_test, y_train, y_test = load_data_mnist()
    eta = 0.05
    iter = 400
    M = 300
    mb = 100
    dataset = "mnist"

i = int(input('Chose a activation option: \n\t1: log \n\t2: tanh\n\t3: cos\n>'))
activation_option = i-1
if i != 1 and i != 2 and i != 3:
    print ("Invalid Input!")
    exit()

# The optimal values here are those that obtained us the minimum score in each dataset
i = int(input('Do you want to set other variables? Press 1 for Yes (Optimal values are default): \n>'))

if i == 1:
    eta = float(input('Give the eta(float):\n>'))
    lamda = float(input('Give the lamda(float):\n>'))
    iter = int(input('Give the number of epoch(int):\n>'))
    M = int(input('Give the number of neurons(int):\n>'))
    mb = int(input('Give the size of the minibatch(int):\n>'))
N, D = X_train.shape

# The number classes
K = y_train.shape[1]
gradcheck = -1
gradcheck = int(input('Peform Gradient Check? Press 1 for Yes:\n>'))
# Adds a row of 1 in the beginning
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

start([iter, mb, eta], dataset)