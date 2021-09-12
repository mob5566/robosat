import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot(out, history):
    plt.figure()

    n = max(map(len, history.values()))

    plt.grid()

    for values in history.values():
        plt.plot(values)

    plt.xlabel("epoch")
    plt.legend(list(history))

    plt.savefig(out, format="png")
    plt.close()
