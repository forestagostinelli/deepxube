from typing import List, Union

from matplotlib import pyplot as plt

from deepxube.base.domain import EnvVizable, State, Goal


def visualize_examples(env: EnvVizable, states: Union[List[State], List[Goal]]):
    states_np = env.visualize(states)

    plt.ion()
    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    idx_stop: int
    if states_np.shape[3] == 3:
        axs = [ax1, ax2, ax3, ax4]

        for idx in range(0, len(states_np), 4):
            for ax in axs:
                ax.cla()

            idx_stop = min(idx + 4, len(states_np))
            for idx_ax, idx_show in enumerate(range(idx, idx_stop)):
                ax = axs[idx_ax]
                ax.imshow(states_np[idx_show])
                ax.set_xticks([])
                ax.set_yticks([])

            fig.canvas.draw()
            input("Enter anything: ")

    elif states_np.shape[3] == 6:
        axs1 = [ax1, ax2]
        axs2 = [ax3, ax4]
        axs_cube = [axs1, axs2]

        for idx in range(0, len(states_np), 2):
            for axs in axs_cube:
                for ax in axs:
                    ax.cla()

            idx_stop = min(idx + 2, len(states_np))
            for idx_ax, idx_show in enumerate(range(idx, idx_stop)):
                axs = axs_cube[idx_ax]
                axs[0].imshow(states_np[idx_show, :, :, :3])
                axs[1].imshow(states_np[idx_show, :, :, 3:])

                for ax in axs:
                    ax.set_xticks([])
                    ax.set_yticks([])

            fig.canvas.draw()
            input("Enter anything: ")

    plt.ioff()
    plt.close(fig)
