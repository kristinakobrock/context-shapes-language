import pickle
import numpy as np


def load_accuracies(all_paths, n_runs=5, n_epochs=300, val_steps=10, zero_shot=True, context_unaware=True):
    """ loads all accuracies into a dictionary, val_steps should be set to the same as val_frequency during training
    """
    result_dict = {'train_acc': [], 'val_acc': [], 'test_acc': [],
                   'cu_train_acc': [], 'cu_val_acc': [], 'cu_test_acc': []}

    for path_idx, path in enumerate(all_paths):

        train_accs = []
        val_accs = []
        test_accs = []
        cu_train_accs = []
        cu_val_accs = []
        cu_test_accs = []

        for run in range(n_runs):

            standard_path = path + '/standard/' + str(run) + '/'
            context_unaware_path = path + '/context_unaware/' + str(run) + '/'

            # train and validation accuracy

            data = pickle.load(open(standard_path + 'loss_and_metrics.pkl', 'rb'))
            lists = sorted(data['metrics_train0'].items())
            _, train_acc = zip(*lists)
            train_accs.append(train_acc)
            lists = sorted(data['metrics_test0'].items())
            _, val_acc = zip(*lists)
            if len(val_acc) > n_epochs // val_steps:  # old: we had some runs where we set val freq to 5 instead of 10
                val_acc = val_acc[::2]
            val_accs.append(val_acc)
            test_accs.append(data['final_test_acc'])

            # context-unaware accuracy
            if context_unaware:
                cu_data = pickle.load(open(context_unaware_path + 'loss_and_metrics.pkl', 'rb'))
                lists = sorted(cu_data['metrics_train0'].items())
                _, cu_train_acc = zip(*lists)
                if len(cu_train_acc) != n_epochs:
                    print(path, run, len(cu_train_acc))
                    raise ValueError(
                        "The stored results don't match the parameters given to this function. "
                        "Check the number of epochs in the above mentioned runs.")
                cu_train_accs.append(cu_train_acc)
                lists = sorted(cu_data['metrics_test0'].items())
                _, cu_val_acc = zip(*lists)
                # for troubleshooting in case the stored results don't match the parameters given to this function
                if len(cu_val_acc) != n_epochs // val_steps:
                    print(context_unaware_path, len(cu_val_acc))
                    raise ValueError(
                        "The stored results don't match the parameters given to this function. "
                        "Check the above mentioned files for number of epochs and validation steps.")
                if len(cu_val_acc) > n_epochs // val_steps:
                    cu_val_acc = cu_val_acc[::2]
                cu_val_accs.append(cu_val_acc)
                cu_test_accs.append(cu_data['final_test_acc'])

        result_dict['train_acc'].append(train_accs)
        result_dict['val_acc'].append(val_accs)
        result_dict['test_acc'].append(test_accs)
        if context_unaware:
            result_dict['cu_train_acc'].append(cu_train_accs)
            result_dict['cu_val_acc'].append(cu_val_accs)
            result_dict['cu_test_acc'].append(cu_test_accs)

    for key in result_dict.keys():
        result_dict[key] = np.array(result_dict[key])

    return result_dict


def load_entropies(all_paths, n_runs=5, context_unaware=False, length_cost=0.001):
    """ loads all entropy scores into a dictionary"""

    if context_unaware:
        setting = 'context_unaware'
    else:
        setting = 'standard'

    result_dict = {'NMI': [], 'effectiveness': [], 'consistency': [],
                   'NMI_hierarchical': [], 'effectiveness_hierarchical': [], 'consistency_hierarchical': [],
                   'NMI_context_dep': [], 'effectiveness_context_dep': [], 'consistency_context_dep': [],
                   'NMI_concept_x_context': [], 'effectiveness_concept_x_context': [],
                   'consistency_concept_x_context': []}

    for path_idx, path in enumerate(all_paths):

        NMIs, effectiveness_scores, consistency_scores = [], [], []
        NMIs_hierarchical, effectiveness_scores_hierarchical, consistency_scores_hierarchical = [], [], []
        NMIs_context_dep, effectiveness_scores_context_dep, consistency_scores_context_dep = [], [], []
        NMIs_conc_x_cont, effectiveness_conc_x_cont, consistency_conc_x_cont = [], [], []

        for run in range(n_runs):
            standard_path = path + '/' + setting + '/' + str(run) + '/'
            data = pickle.load(open(standard_path + 'entropy_scores.pkl', 'rb'))
            NMIs.append(data['normalized_mutual_info'])
            effectiveness_scores.append(data['effectiveness'])
            consistency_scores.append(data['consistency'])
            NMIs_hierarchical.append(data['normalized_mutual_info_hierarchical'])
            effectiveness_scores_hierarchical.append(data['effectiveness_hierarchical'])
            consistency_scores_hierarchical.append(data['consistency_hierarchical'])
            NMIs_context_dep.append(data['normalized_mutual_info_context_dep'])
            effectiveness_scores_context_dep.append(data['effectiveness_context_dep'])
            consistency_scores_context_dep.append(data['consistency_context_dep'])
            NMIs_conc_x_cont.append(data['normalized_mutual_info_concept_x_context'])
            effectiveness_conc_x_cont.append(data['effectiveness_concept_x_context'])
            consistency_conc_x_cont.append(data['consistency_concept_x_context'])

        result_dict['NMI'].append(NMIs)
        result_dict['consistency'].append(consistency_scores)
        result_dict['effectiveness'].append(effectiveness_scores)
        result_dict['NMI_hierarchical'].append(NMIs_hierarchical)
        result_dict['consistency_hierarchical'].append(consistency_scores_hierarchical)
        result_dict['effectiveness_hierarchical'].append(effectiveness_scores_hierarchical)
        result_dict['NMI_context_dep'].append(NMIs_context_dep)
        result_dict['consistency_context_dep'].append(consistency_scores_context_dep)
        result_dict['effectiveness_context_dep'].append(effectiveness_scores_context_dep)
        result_dict['NMI_concept_x_context'].append(NMIs_conc_x_cont)
        result_dict['consistency_concept_x_context'].append(consistency_conc_x_cont)
        result_dict['effectiveness_concept_x_context'].append(effectiveness_conc_x_cont)

    for key in result_dict.keys():
        result_dict[key] = np.array(result_dict[key])

    return result_dict
