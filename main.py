import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt
import multiprocessing

def run_test(cfg):
    # create random numbers between 0 and 2 pi
    x = np.random.uniform(size = cfg["samples"]) * 2 * np.pi
    inp = x.reshape(x.size, 1)

    # calculate sin e cos with noise
    sin = np.sin(x) + np.random.uniform(size = cfg["samples"]) * cfg["noise"]
    cos = np.cos(x) + np.random.uniform(size = cfg["samples"]) * cfg["noise"]

    y = np.dstack((sin, cos))[0]

    # create network with 2 layers and 10 neurons in the hiden layer
    net = nl.net.newff([[np.min(x), np.max(x)]], [cfg["neurons"], 2], cfg["activation_functions"])

    # use traingd
    net.trainf = nl.train.train_gd

    # train the network
    error = net.train(input=inp, target=y, epochs=cfg["epochs"], show=0, goal=0.001, lr=cfg["lr"])

    # simulate network
    out = net.sim(inp)

    # plot results
    ## create our test subjects
    if "extra_test_range" in cfg:
        x = np.random.uniform(-1, 1, 3 * cfg["samples"]) * 3 * np.pi
        cfg["file_name"] = "extra_test_range"
    else:
        x = np.random.uniform(size = cfg["samples"]) * 2 * np.pi
    
    ## simulate using trained neurons
    plt.plot(x, net.sim(x.reshape(x.size, 1)), '+b')
    
    ## compare with correct output (without noise)
    plt.plot(x, np.dstack((np.sin(x), np.cos(x)))[0], '.r')
    plt.legend(['sin simulation', 'cos simulation', 'sin', 'cos'])

    cfg["file_name"] += ", error: " + str(error[-1])
    plt.savefig("results/" + cfg["file_name"].replace(".", ","))
    
    print("Done: ", cfg)
    
def generate_cfg(new_values = None):
    cfg = {"epochs": 20000, "lr": 0.005, "neurons": 10, "activation_functions": [nl.trans.LogSig(), nl.trans.TanSig()], "samples": 200, "noise": 0.05, "file_name": "base"}
    
    if new_values:
        cfg.update(new_values)
        
        if "activation_functions" in new_values:
            cfg["file_name"] = str([i.__class__.__name__ for i in cfg["activation_functions"]])
        else:
            cfg["file_name"] = str(new_values)
    
    return cfg

configurations = [
    # base configuration
    generate_cfg(),
    
    # different epochs
    generate_cfg({"epochs": 2000}),
    generate_cfg({"epochs": 100000}),
    
    # more or less samples
    generate_cfg({"samples": 50}),
    generate_cfg({"samples": 400}),
    generate_cfg({"samples": 1000}),
    
    # different noise levels
    generate_cfg({"noise": 0.001}),
    generate_cfg({"noise": 0.1}),
    
    # more or less neurons
    generate_cfg({"neurons": 5}),
    generate_cfg({"neurons": 20}),
    generate_cfg({"neurons": 50}),
    
    # different learning rates
    generate_cfg({"lr": 0.001}),
    generate_cfg({"lr": 0.01}),
    generate_cfg({"lr": 0.02}),
    generate_cfg({"lr": 0.05}),
    
    # different activation functions
    generate_cfg({"activation_functions": [nl.trans.TanSig(), nl.trans.LogSig()]}),
    generate_cfg({"activation_functions": [nl.trans.TanSig(), nl.trans.TanSig()]}),
    generate_cfg({"activation_functions": [nl.trans.LogSig(), nl.trans.LogSig()]}),
    generate_cfg({"activation_functions": [nl.trans.LogSig(), nl.trans.PureLin()]}),
    
    # simulate for a larger range of values
    generate_cfg({"extra_test_range": True}),
]

with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    p.map(run_test, configurations)
