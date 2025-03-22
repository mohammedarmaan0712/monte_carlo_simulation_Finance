import numpy as np

import matplotlib.pyplot as plt

def monte_carlo():
    pass
    '''
    Monte carlo simulation:
    Numpy is used to generate random numbers and perform calculations.
    find considerable random numbers to simulate a process.
    The simulation is run for a large number of iterations to get an accurate estimate.
    Then the results are used to plot a graph.
    The graph shows the distribution of the random numbers generated.
    Finding statistics on this graph can help in predicting the outcome of the process.

   '''
    #generate random numbers
    num_samples = 1000000 #(1 million samples)
    A = np.random.uniform(3,5,num_samples)
    B = np.random.uniform(1,4,num_samples)
    C = A + B
    # Plot the random numbersp
    plt.hist(C.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title("Distribution of Random Numbers")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    #find the statistics:
    mean = np.mean(C)
    std_dev = np.std(C)
    variance = np.var(C)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Variance: {variance}")

if __name__ == "__main__":
    monte_carlo()