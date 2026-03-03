import matplotlib.pyplot as plt
import numpy

def plot_preds(example_ytrue, example_ypred, model_name: str, num_imgs: int = 4):
    layout_var = "none" if num_imgs == 1 else "constrained"
    fig, ax = plt.subplots(ncols=2, nrows=num_imgs, layout=layout_var)
    # plt.suptitle("Persistence model", fontsize=14)

    max_val = example_ytrue.max()
    min_val = example_ytrue.min()

    if num_imgs == 1:
        ax = [ax]

    row_num = 0
    col_num = 0
    for row in ax:
        for col in row:
            if col_num % 2 == 0:
                if row_num == 0:
                    col.set_title(model_name)
                im = col.imshow(
                    example_ypred[0, row_num],
                    cmap="viridis",
                    vmin=min_val,
                    vmax=max_val,
                )
                col.set_yticks([])
                col.set_xticks([])
                col.set_ylabel(f"t={row_num+1}")
            else:
                if row_num == 0:
                    col.set_title("Ground truth")
                im = col.imshow(
                    example_ytrue[0, row_num],
                    cmap="viridis",
                    vmin=min_val,
                    vmax=max_val,
                )
                col.set_yticks([])
                col.set_xticks([])
                col.set_ylabel(f"t={row_num+1}")
            col_num += 1
        row_num += 1

    # fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
    fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="[mm/h]")
    plt.show()
    return True