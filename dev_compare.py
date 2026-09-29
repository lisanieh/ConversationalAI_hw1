import matplotlib.pyplot as plt

fractions = [0.05, 0.10, 0.25, 1.00]
crf_f1 = [0.8300, 0.8681, 0.9061, 0.9341]
bilstm_f1 = [0.3950, 0.6955, 0.8681, 0.9407]

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(fractions, crf_f1, marker="o", label="CRF")
ax.plot(fractions, bilstm_f1, marker="o", label="BiLSTM")

ax.set_xticks(fractions, ["5%", "10%", "25%", "100%"])
ax.set_ylim(0, 1)
ax.set_xlabel("training fraction")
ax.set_ylabel("slot F1")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig("part_c_data_efficiency.png", dpi=300)
plt.show()