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
    if not logger.handlers:
        fh = logging.FileHandler(full_exp_name + '.log')
        formatter = logging.Formatter('%(asctime)s %(message)s')
        fh.setFormatter(formatter)
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


def fit(net, epochs):
    print("Running net.fit for", net.experiment_name)
    try:
        net.fit(epochs)
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")
        menu(net, epochs)


def menu(net, epochs):
    # Print menu
    print("")
    print("------------------ OPTIONS ------------------")
    print("d: Enter debugger.")
    print("s: Save plots and params.")
    print("q: Quit all experiments.")
    print("e: Change number of epochs to train this net (currently {})."
          .format(epochs))
    print("c: Continue training.")
    print("")

    # Get input
    selection_str = raw_input("Please enter one or more letters: ")

    # Handle input
    for selection in selection_str:
        if selection == 'd':
            import ipdb
            ipdb.set_trace()
        elif selection == 's':
            net.save()
        elif selection == 'q':
            sure = raw_input("Are you sure you want to quit [Y/n]? ")
            if sure.lower() != 'n':
                raise
        elif selection == 'e':
            new_epochs = raw_input("New number of epochs (or 'None'): ")
            if new_epochs == 'None':
                epochs = None
            else:
                try:
                    epochs = int(new_epochs)
                except:
                    print("'{}' not an integer!".format(new_epochs))
        elif selection == 'c':
            break
        else:
            print("Selection '{}' not recognised!".format(selection))
            break
    print("Continuing training for {} epochs...".format(epochs))
    fit(net, epochs)
