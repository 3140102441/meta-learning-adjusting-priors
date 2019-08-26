
from __future__ import absolute_import, division, print_function

import timeit

from Models.stochastic_models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import get_bayes_task_objective, run_test_Bayes
from Utils.common import grad_step, count_correct, get_loss_criterion, write_to_log, load_model_state



def run_learning(task_data, prior_model, prm, init_from_prior=True, verbose=1):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------
    # Unpack parameters:
    optim_func, optim_args, lr_schedule =\
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type)

    # Create posterior model for the new task:
    post_model = get_model(prm)

    if init_from_prior:
        post_model.load_state_dict(prior_model.state_dict())

        # prior_model_dict = prior_model.state_dict()
        # post_model_dict = post_model.state_dict()
        #
        # # filter out unnecessary keys:
        # prior_model_dict = {k: v for k, v in prior_model_dict.items() if '_log_var' in k or '_mu' in k}
        # # overwrite entries in the existing state dict:
        # post_model_dict.update(prior_model_dict)
        #
        # # #  load the new state dict
        # post_model.load_state_dict(post_model_dict)

        # add_noise_to_model(post_model, prm.kappa_factor)

    # The data-sets of the new task:
    train_loader = task_data['train']
    test_loader = task_data['test']
    #import pudb
    #pudb.set_trace()
    n_train_samples = len(train_loader.dataset)
    n_batches = len(train_loader)

    #  Get optimizer:
    optimizer = optim_func(post_model.parameters(), **optim_args)


    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------

    def run_train_epoch(i_epoch):
        log_interval = 500

        post_model.train()

        train_info = {}
        train_info["task_comp"] = 0.0
        train_info["total_loss"] = 0.0

        cnt = 0
        for batch_idx, batch_data in enumerate(train_loader):
            cnt += 1

            correct_count = 0
            sample_count = 0

            # Monte-Carlo iterations:
            n_MC = prm.n_MC
            task_empirical_loss = 0
            task_complexity = 0
            for i_MC in range(n_MC):
                # get batch:
                inputs, targets = data_gen.get_batch_vars(batch_data, prm)

                # Calculate empirical loss:
                outputs = post_model(inputs)
                curr_empirical_loss = loss_criterion(outputs, targets)

                #hyper_kl = 0 when testing
                curr_empirical_loss, curr_complexity, task_info = get_bayes_task_objective(prm, prior_model, post_model,
                                                           n_train_samples, curr_empirical_loss, noised_prior=False)

                task_empirical_loss += (1 / n_MC) * curr_empirical_loss
                task_complexity += (1 / n_MC) * curr_complexity

                correct_count += count_correct(outputs, targets)
                sample_count += inputs.size(0)

            # Total objective:

            total_objective = task_empirical_loss + prm.task_complex_w * task_complexity

            train_info["task_comp"] += task_complexity.data[0]
            train_info["total_loss"] += total_objective.data[0]

            # Take gradient step with the posterior:
            grad_step(total_objective, optimizer, lr_schedule, prm.lr, i_epoch)



            # Print status:
            if batch_idx % log_interval == 0:
                batch_acc = correct_count / sample_count
                write_to_log(cmn.status_string(i_epoch, prm.n_meta_test_epochs, batch_idx, n_batches, batch_acc, total_objective.data[0]) +
                        ' Empiric Loss: {:.4}\t Intra-Comp. {:.4}, w_kld {:.4}, b_kld {:.4}'.
                      format(task_empirical_loss.data[0], task_complexity.data[0], task_info["w_kld"], task_info["b_kld"]), prm)

        train_info["task_comp"] /= cnt
        train_info["total_loss"] /= cnt
        return train_info


    # -----------------------------------------------------------------------------------------------------------#
    # Update Log file
    if verbose == 1:
        write_to_log('Total number of steps: {}'.format(n_batches * prm.n_meta_test_epochs), prm)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    # Training loop:
    best_acc = -1
    best_acc_loss = -1
    best_acc_comp = -1

    for i_epoch in range(prm.n_meta_test_epochs):
        train_info = run_train_epoch(i_epoch)
        test_acc, test_loss = run_test_Bayes(post_model, test_loader, loss_criterion, prm)
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_loss = test_loss
            best_acc_comp = train_info["task_comp"]


    # Test:
    test_acc, test_loss = run_test_Bayes(post_model, test_loader, loss_criterion, prm)

    stop_time = timeit.default_timer()
    cmn.write_final_result(best_acc, stop_time - start_time, prm, result_name=prm.test_type, verbose=verbose)

    test_err = 1 - best_acc
    return test_err, best_acc_comp, best_acc_loss, post_model
