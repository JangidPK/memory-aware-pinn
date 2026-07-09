
import numpy as np
import matplotlib.pyplot as plt


def plot_heatmaps(x, t, c_true, c_pred, save_path=None, save_data=None):
    """Three side-by-side heatmaps: true, predicted, |difference|."""
    diff = np.abs(c_true - c_pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["True c(x,t)", "Predicted c(x,t)", "|Difference|"]
    data = [c_true, c_pred, diff]
    cmaps = ["viridis", "viridis", "inferno"]

    for ax, d, title, cmap in zip(axes, data, titles, cmaps):
        im = ax.imshow(d, extent=[t.min(), t.max(), x.min(), x.max()],
                        origin="lower", aspect="auto", cmap=cmap)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# updated function to plot and save data for later use
# def plot_heatmaps(x, t, c_true,c_pred,save_path, save_data, 
#                   data_available=False):

#     if data_available:
#         if save_data is None:
#             raise ValueError("save_data must be provided when data_available=True.")

#         data = np.load(save_data)
#         x = data["x"]
#         t = data["t"]
#         c_true = data["c_true"]
#         c_pred = data["c_pred"]

#     # Save data for later reuse
#     elif save_data is not None:
#         np.savez(save_path[:-4], x=x,t=t,c_true=c_true,c_pred=c_pred)

#     diff = np.abs(c_true - c_pred)

#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#     titles = [
#         "True c(x,t)",
#         "Predicted c(x,t)",
#         "|Difference|",
#     ]
#     arrays = [c_true, c_pred, diff]
#     cmaps = ["viridis", "viridis", "inferno"]

#     for ax, arr, title, cmap in zip(axes, arrays, titles, cmaps):
#         im = ax.imshow(
#             arr,
#             extent=[t.min(), t.max(), x.min(), x.max()],
#             origin="lower",
#             aspect="auto",
#             cmap=cmap,
#         )
#         ax.set_xlabel("t")
#         ax.set_ylabel("x")
#         ax.set_title(title)
#         fig.colorbar(im, ax=ax)

#     fig.tight_layout()

#     if save_path is not None:
#         fig.savefig(save_path, dpi=150)

#     return fig



def plot_kernel(t, K_pred, K_true=None, save_path=None):
    """Plot the learned kernel K(t), optionally overlaid with the true kernel."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, K_pred, label="Learned K(t)", color="tab:blue")
    if K_true is not None:
        ax.plot(t, K_true, label="True K(t)", color="tab:orange", linestyle="--")
    ax.set_xlabel("t (time lag)")
    ax.set_ylabel("K(t)")
    ax.set_title("Memory Kernel")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# def plot_kernel(t, K_pred,K_true,save_path,save_data,data_available):

#     if data_available:
#         if save_data is None:
#             raise ValueError("save_data must be provided when data_available=True.")

#         data = np.load(save_data)
#         t = data["t"]
#         K_pred = data["K_pred"]

#         # K_true may or may not exist
#         K_true = data["K_true"] if "K_true" in data.files else None

#     elif save_data is not None:
#         save_dict = {
#             "t": t,
#             "K_pred": K_pred,
#         }
#         if K_true is not None:
#             save_dict["K_true"] = K_true
        
#         # same as image fiel
#         np.savez(save_path[:-4], **save_dict)

#     fig, ax = plt.subplots(figsize=(6, 4))

#     ax.plot(t, K_pred, label="Learned K(t)", color="tab:blue")

#     if K_true is not None:
#         ax.plot(t, K_true, label="True K(t)", color="tab:orange", linestyle="--",)

#     ax.set_xlabel("t (time lag)")
#     ax.set_ylabel("K(t)")
#     ax.set_title("Memory Kernel")
#     ax.legend()

#     fig.tight_layout()

#     if save_path is not None:
#         fig.savefig(save_path, dpi=150)

#     return fig




def plot_training_curves(history, save_path=None):
    """Plot physics / boundary / data (/ ic) losses over training."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in history:
        if key == "loss":
            continue
        ax.plot(history[key], label=key)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Training Curves")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# def plot_training_curves(
#     history=None,
#     save_path=None,
#     save_data=None,
#     data_available=False,
# ):
#     if data_available:
#         if save_data is None:
#             raise ValueError("save_data must be provided when data_available=True.")

#         data = np.load(save_data, allow_pickle=True)
#         history = {key: data[key] for key in data.files}

#     elif save_data is not None:
#         np.savez(save_path[:-4], **history)

#     fig, ax = plt.subplots(figsize=(7, 4))

#     for key, values in history.items():
#         if key == "loss":
#             continue
#         ax.plot(values, label=key)

#     ax.set_yscale("log")
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Loss (log scale)")
#     ax.set_title("Training Curves")
#     ax.legend()

#     fig.tight_layout()

#     if save_path is not None:
#         fig.savefig(save_path, dpi=150)

#     return fig