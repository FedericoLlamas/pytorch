import matplotlib.pyplot as plt


def plot(input, output):
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(input[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
        	output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()
