import torch


def calc_bin_classes(y_pred: torch.Tensor, yb: torch.Tensor, thresh: int = 0.5):

    y_pred_mask = y_pred > thresh
    yb_mask = yb > thresh

    true_pos = (torch.logical_and(y_pred_mask == yb_mask, yb_mask == True)).sum()
    true_negs = (torch.logical_and(y_pred_mask == yb_mask, yb_mask == False)).sum()
    false_pos = (torch.logical_and(y_pred_mask == True, yb_mask == False)).sum()
    false_negs = (torch.logical_and(y_pred_mask == False, yb_mask == True)).sum()

    return true_pos, true_negs, false_pos, false_negs


def calc_acc(true_pos: int, true_negs: int, false_pos: int, false_negs: int):
    return (true_pos + true_negs) / (true_pos + true_negs + false_pos + false_negs)


def calc_prec(true_pos: int, false_pos: int):
    return true_pos / (true_pos + false_pos)


def calc_rec(true_pos: int, false_negs: int):
    return true_pos / (true_pos + false_negs)


def calc_f1(true_pos: int, false_pos: int, false_negs: int):
    prec = calc_prec(true_pos=true_pos, false_pos=false_pos)
    rec = calc_rec(true_pos=true_pos, false_negs=false_negs)
    return 2 * (prec * rec) / (prec + rec)


def calc_csi(true_pos: int, false_pos: int, false_negs: int):
    return true_pos / (true_pos + false_negs + false_pos)


def calc_mcc(true_pos: int, true_negs: int, false_pos: int, false_negs: int):
    """This function calculates the Matthews Correlation Coefficient.
    The inputs are expected to be torch scalars; otherwise the function will fail."""
    true_pos = true_pos.item()
    true_negs = true_negs.item()
    false_pos = false_pos.item()
    false_negs = false_negs.item()

    num = true_pos * true_negs - false_pos * false_negs
    denom = (
        (true_pos + false_pos)
        * (true_pos + false_negs)
        * (true_negs + false_pos)
        * (true_negs + false_negs)
    ) ** 0.5
    if (
        denom == 0.0
    ):  # In the case that some class was never predicted or is not present
        denom += 0.00000000001
    return num / denom


def calc_all_metrics_from_bins(binObins: list):
    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    csi_list = []
    mcc_list = []

    for thresh in range(len(binObins)):
        true_pos = binObins[thresh][0]
        true_negs = binObins[thresh][1]
        false_pos = binObins[thresh][2]
        false_negs = binObins[thresh][3]

        acc = calc_acc(
            true_pos=true_pos,
            true_negs=true_negs,
            false_pos=false_pos,
            false_negs=false_negs,
        )
        acc_list.append(acc.item())

        prec = calc_prec(true_pos=true_pos, false_pos=false_pos)
        prec_list.append(prec.item())

        rec = calc_rec(true_pos=true_pos, false_negs=false_negs)
        rec_list.append(rec.item())

        f1_list.append((2 * (prec * rec) / (prec + rec)).item())

        csi = calc_csi(true_pos=true_pos, false_pos=false_pos, false_negs=false_negs)
        csi_list.append(csi.item())

        mcc = calc_mcc(
            true_pos=true_pos,
            true_negs=true_negs,
            false_pos=false_pos,
            false_negs=false_negs,
        )
        mcc_list.append(mcc)

    return acc_list, prec_list, rec_list, f1_list, csi_list, mcc_list
