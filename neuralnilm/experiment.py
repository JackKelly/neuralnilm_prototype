from __future__ import print_function, division
import os
import logging
from sys import stdout

def init_experiment(base_path, experiment, full_exp_name):
    """
    Parameters
    ----------
    base_path : str
    full_exp_name : str

    Returns
    -------
    func_call : str
    """
    path = os.path.join(base_path, full_exp_name)
    try:
        os.mkdir(path)
    except OSError as exception:
        if exception.errno == 17:
            print(path, "already exists.  Reusing directory.")
        else:
            raise
    os.chdir(path)
    func_call = 'exp_{:s}(full_exp_name)'.format(experiment)
    logger = logging.getLogger(full_exp_name)
    fh = logging.FileHandler(full_exp_name+'.log')
    fh.setFormatter('%(asctime)s %(message)s')
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(stream=stdout))
    logger.setLevel(logging.DEBUG)
    logger.info("***********************************")
    logger.info("Preparing " + full_exp_name + "...")
    return func_call
    

def run_experiment(net, epochs):
    net.print_net()
    net.compile()
    fit(net, epochs)


def save(net):
    print("Saving plots...")
    net.plot_estimates(save=True)
    net.plot_costs(save=True)
    print("Saving params...")
    net.save_params()
    net.save_activations()
    print("Done saving.")


def fit(net, epochs):
    print("Running net.fit for", net.experiment_name)
    try:
        net.fit(epochs)
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")
        enter_debugger = raw_input("Enter debugger [N/y]? ")
        if enter_debugger.lower() == 'y':
            import ipdb; ipdb.set_trace()
        save_data = raw_input("Save latest data [Y/n]? ")
        if save_data.lower() in ["y", ""]:
            save(net)
        stop_all = raw_input("Stop all experiments [N/y]? ")
        if stop_all.lower() == "y":
            raise
        continue_fit = raw_input("Continue fitting this experiment [Y/n]? ")
        if continue_fit.lower() in ["y", ""]:
            new_epochs = raw_input("Change number of epochs [currently {}]? "
                                   .format(epochs))
            if new_epochs:
                epochs = int(new_epochs)
            fit(net, epochs)
    # except:
    #     save(net)
    #     raise
