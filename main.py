from training.train import run_experiment

'''
Structure 

def run_experiment(name, learn_kernel=True, obs_fraction=0.1, noise_std=0.0,
                    D=0.01, Nx=41, Nt=41, L=1.0, T=1.0, epochs=5000,
                    lr=1e-3, device="cpu", out_dir="results", print_every=500):
'''


# not learning kernel noise is zero
run_experiment(
    name="not_learning_kernel_zero_noise",
    learn_kernel=False,
    obs_fraction=0.15,
    noise_std=0.0,
    D=0.1,
    Nx=101, Nt=201,
    epochs=5000,
    lr=1e-3,
    device="cpu",
    print_every=100,
    r_seed=12
)

# learning kernel noise is zero
# run_experiment(
#     name="learning_learning_kernel_zero_noise",
#     learn_kernel=True,
#     obs_fraction=0.15,
#     noise_std=0.0,
#     D=0.1,
#     Nx=101, Nt=201,
#     epochs=5000,
#     lr=1e-3,
#     device="cpu",
#     print_every=100,
#     r_seed=12
# )

# run_experiment(
#     name="my_run",
#     learn_kernel=False,
#     obs_fraction=0.15,
#     noise_std=0.0,
#     D=0.01,
#     Nx=101, Nt=201,
#     epochs=1,
#     lr=1e-3,
#     device="cpu",
# )