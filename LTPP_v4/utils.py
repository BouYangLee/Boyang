# utils.py
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(losses, outpath=None):
    plt.figure(figsize=(6, 3))
    plt.plot(losses, marker='o')
    plt.title('Training loss (avg per-seq)')
    plt.xlabel('Epoch')
    plt.grid(True)
    if outpath:
        plt.savefig(outpath, bbox_inches='tight'); plt.close()
    else:
        plt.show()

def plot_W_history(W_history, predicate_list, outdir=None):
    # W_history: dict param_name (str) -> list of arrays (epochs x P)
    for key, W_list in W_history.items():
        W = np.array(W_list)
        plt.figure(figsize=(6, 3))
        for p_idx, p in enumerate(predicate_list):
            plt.plot(W[:, p_idx], label=f'pred {p}')
        plt.title(f'W evolution for clause {key}')
        plt.xlabel('Epoch')
        plt.ylabel('W (summed over slots)')
        plt.legend(fontsize=6)
        plt.grid(True)
        if outdir:
            safe_key = str(key).replace('/', '_').replace(' ', '_')
            plt.savefig(f"{outdir}/{safe_key}_W.png", bbox_inches='tight'); plt.close()
        else:
            plt.show()
