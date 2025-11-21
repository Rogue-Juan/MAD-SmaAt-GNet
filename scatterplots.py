import json
import matplotlib.pyplot as plt
import numpy


def per_img():
    list_names_in_order = [
        # "Persistence",
        "SmaAt-UNet",
        "MAD-SmaAt-GNet",
        "SmaAt-UNet with Evo-Net",
        "SmaAt-UNet with 2-stream",
        "Evo-Net",
    ]
    list_jsons = [
        # "./results/test_results_per_img_persistence_model.json",
        "./results/test_results_per_img_SmaAt_UNet.json",
        "./results/test_results_per_img_MAD_SmaAt_GNet.json",
        "./results/test_results_per_img_madsmaat_evo_.json",
        "./results/test_results_per_img_madsmaat_2stream.json",
        "./results/test_results_per_img_EvoNet.json",
    ]
    list_mse = []
    for i in range(len(list_jsons)):
        with open(list_jsons[i], "r") as file:
            data = json.load(file)
            list_mse.append(data["mse"])

    for i in range(len(list_names_in_order)):
        print(f"{list_names_in_order[i]}: {list_mse[i]}")

    line_styles = ["-.", "-", "--", ":", "--"]
    marker_list = ["+", "o", "^", "s", "x"]
    x = [1, 2, 3, 4]
    for i in range(len(list_names_in_order)):
        plt.plot(
            x,
            list_mse[i],
            ls=line_styles[i],
            marker=marker_list[i],
            label=f"{list_names_in_order[i]}",
        )

    plt.legend()
    plt.grid()
    plt.suptitle("MSE per time step")
    plt.xlabel("Prediction hour")
    plt.ylabel("MSE")
    plt.xticks(x)
    plt.show()

    return True


def per_thresh():
    list_names_in_order = [
        "Persistence",
        "SmaAt-UNet",
        "MAD-SmaAt-GNet",
        "SmaAt-UNet with Evo-Net",
        "SmaAt-UNet with 2-stream",
        "Evo-Net",
    ]
    list_jsons = [
        "./results/test_results_per_img_persistence_model.json",
        "./results/test_results_per_img_SmaAt_UNet.json",
        "./results/test_results_per_img_MAD_SmaAt_GNet.json",
        "./results/test_results_per_img_madsmaat_evo_.json",
        "./results/test_results_per_img_madsmaat_2stream.json",
        "./results/test_results_per_img_EvoNet.json",
    ]
    list_mcc = []
    for i in range(len(list_jsons)):
        with open(list_jsons[i], "r") as file:
            data = json.load(file)
            list_mcc.append(data["mcc"])

    for i in range(len(list_names_in_order)):
        print(f"{list_names_in_order[i]}: {list_mcc[i]}")

    line_styles = ["-", "-.", "-", "--", ":", "--"]
    marker_list = ["o", "+", "*", "^", "s", "x"]
    x = [0.5, 4, 8, 16]
    for i in range(len(list_names_in_order)):
        plt.plot(
            x,
            list_mcc[i],
            ls=line_styles[i],
            marker=marker_list[i],
            label=f"{list_names_in_order[i]}",
        )

    plt.legend()
    plt.grid()
    plt.suptitle("MCC per threshold")
    plt.xlabel("Threshold value [mm/h]")
    plt.ylabel("MCC")
    plt.xticks(x)
    plt.show()

    return True


if __name__ == "__main__":
    per_img()
    per_thresh()
