# copied and adapted from https://github.com/XeniaOhmer/hierarchical_reference_game/blob/master/utils/analysis_from_interaction.py

from egg.core.language_analysis import calc_entropy, _hashable_tensor, Disent
from sklearn.metrics import normalized_mutual_info_score
from language_analysis_local import MessageLengthHierarchical
import numpy as np
import itertools
import random
import torch


def k_hot_to_attributes(khots, dimsize):
    """
    Decodes many-hot-represented objects to an easy-to-interpret version.
    E.g. [0, 0, 1, 1, 0, 0, 1, 0, 0] -> [2, 0, 0]
    """
    base_count = 0
    n_attributes = khots.shape[2] // dimsize
    attributes = np.zeros((len(khots), len(khots[1]), n_attributes))
    for att in range(n_attributes):
        attributes[:, :, att] = np.argmax(khots[:, :, base_count:base_count + dimsize], axis=2)
        base_count = base_count + dimsize
    return attributes


def retrieve_fixed_vectors(target_objects):
    """
    Reconstructs fixed vectors given a list of (decoded) target objects.
    """
    n_attributes = target_objects.shape[2]
    # compare target objects to each other to find out whether and in which attribute they differ
    fixed_vectors = []
    for target_objs in target_objects:
        fixed = np.ones(n_attributes) # (1, 1, 1) -> all fixed
        for idx, target_object in enumerate(target_objs):
            # take first target_object as baseline for comparison (NOTE: could also be random)
            if idx == 0:
                concept = target_object
            else:
                for i, attribute in enumerate(target_object):
                    # if values mismatch, then the attribute is not fixed, i.e. 0 in fixed vector
                    if attribute != concept[i]:
                        fixed[i] = 0
        fixed_vectors.append(fixed)
    return fixed_vectors


def convert_fixed_to_intentions(fixed_vectors):
    """
    NOTE: not needed right now
    fixed vectors are 0: irrelevant, 1: relevant
    intentions are 1: irrelevant, 0: relevant
    """
    intentions = []
    for fixed in fixed_vectors:
        intention = np.zeros(len(fixed))
        for i, att in enumerate(fixed):
            if att == 0:
                intention[i] = 1
        intentions.append(intention)
    return np.asarray(intentions)


def retrieve_concepts_sampling(target_objects, all_targets=False):
    """
    Builds concept representations consisting of one sampled target object and a fixed vector.
    """
    fixed_vectors = retrieve_fixed_vectors(target_objects)
    if all_targets:
        target_objects_sampled = target_objects
    else:
        target_objects_sampled = [random.choice(target_object) for target_object in target_objects]
    return (np.asarray(target_objects_sampled), np.asarray(fixed_vectors))
            

def joint_entropy(xs, ys):
    xys = []

    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)

    return calc_entropy(xys)


def retrieve_context_condition(targets, fixed, distractors):
    """returns the context condition given a list of targets and a list of distractors (from interaction)"""
    context_conds = []
    # go through all observations
    for i, t_obj in enumerate(targets):
        # consider first target and first distractor
        shared = 0
        # go through attributes
        if t_obj.ndim > 1:
            for k, attr in enumerate(t_obj[0]):
                # if target attribute was fixed:
                if fixed[i][k] == 1:
                    #shared = np.zeros(len(t_obj))
                    # go through distractors
                    #for dist_obj in distractors[i]:
                    # compare target attribute with distractor attribute
                    if attr == distractors[i][0][k]:
                        # count shared attributes
                        shared = shared + 1
        else:
            for k, attr in enumerate(t_obj):
                if fixed[i][k] == 1:
                    if attr == distractors[i][0][k]:
                        shared = shared + 1
        #print("target", t_obj, "fixed", fixed[i], "distractors", distractors[i][0], "shared", shared)
        context_conds.append(shared)  
    return context_conds


def bosdis(interaction, n_dims, n_values, vocab_size):
    """
    calculate bag-of-symbol disentanglement for all concept and context conditions
    """
    # Get relevant attributes
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects/2)

    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects)
    # add one such that zero becomes an empty attribute for the calculation (_)
    objects = objects + 1
    concepts = torch.from_numpy(objects * (np.array(fixed)))

    # get distractor objects to re-construct context conditions
    distractor_objects = sender_input[:, n_targets:]
    distractor_objects = k_hot_to_attributes(distractor_objects, n_values)
    distractor_objects = distractor_objects + 1
    context_conds = retrieve_context_condition(objects, fixed, distractor_objects)

    # get messages from interaction
    messages = interaction.message.argmax(dim=-1)

    # sum of fixed vectors gives the specificity of the concept (all attributes fixed means
    # specific concept, one attribute fixed means generic concept)
    # n_relevant_idx stores the indices of the concepts on a specific level of abstraction
    n_relevant_idx = [np.where(np.sum(np.array(fixed), axis=1) == i)[0] for i in range(1, n_dims + 1)]

    context_cond_idx = [np.where(np.array(context_conds) == i)[0] for i in range(0, n_dims)]

    # Concept-context dependent Entropies:
    # go through concept conditions
    conceptxcontext_idx = []
    for i in range(len(n_relevant_idx)):
        # go through context conditions
        for j in range(len(context_cond_idx)):
            # only keep shared entries for each concept-context condition
            shared_elements = [elem for elem in n_relevant_idx[i] if elem in context_cond_idx[j]]
            conceptxcontext_idx.append(shared_elements)

    # Bosdis for each concept and context condition
    bosdis_concept_x_context = np.array([Disent.bosdis(concepts[concept_x_context], messages[concept_x_context], vocab_size) for concept_x_context in conceptxcontext_idx])  

    return bosdis_concept_x_context


def posdis(interaction, n_dims, n_values, vocab_size):
    """
    calculate positional disentanglement for all concept and context conditions
    """
    # Get relevant attributes
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects/2)

    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects)
    # add one such that zero becomes an empty attribute for the calculation (_)
    objects = objects + 1
    concepts = torch.from_numpy(objects * (np.array(fixed)))

    # get distractor objects to re-construct context conditions
    distractor_objects = sender_input[:, n_targets:]
    distractor_objects = k_hot_to_attributes(distractor_objects, n_values)
    distractor_objects = distractor_objects + 1
    context_conds = retrieve_context_condition(objects, fixed, distractor_objects)

    # get messages from interaction
    messages = interaction.message.argmax(dim=-1)

    # sum of fixed vectors gives the specificity of the concept (all attributes fixed means
    # specific concept, one attribute fixed means generic concept)
    # n_relevant_idx stores the indices of the concepts on a specific level of abstraction
    n_relevant_idx = [np.where(np.sum(np.array(fixed), axis=1) == i)[0] for i in range(1, n_dims + 1)]

    context_cond_idx = [np.where(np.array(context_conds) == i)[0] for i in range(0, n_dims)]

    # Concept-context dependent Entropies:
    # go through concept conditions
    conceptxcontext_idx = []
    for i in range(len(n_relevant_idx)):
        # go through context conditions
        for j in range(len(context_cond_idx)):
            # only keep shared entries for each concept-context condition
            shared_elements = [elem for elem in n_relevant_idx[i] if elem in context_cond_idx[j]]
            conceptxcontext_idx.append(shared_elements)

    # Posdis for each concept and context condition
    posdis_concept_x_context = np.array([Disent.posdis(concepts[concept_x_context], messages[concept_x_context]) for concept_x_context in conceptxcontext_idx])

    return posdis_concept_x_context
            


def information_scores(interaction, n_dims, n_values, normalizer="arithmetic"):
    """calculate entropy scores: mutual information (MI), effectiveness and consistency. 
    
    :param interaction: interaction (EGG class)
    :param n_dims: number of input dimensions, e.g. D(3,4) --> 3 dimensions
    :param n_values: size of each dimension, e.g. D(3,4) --> 4 values
    :param normalizer: normalizer can be either "arithmetic" -H(M) + H(C)- or "joint" -H(M,C)-
    :return: NMI, NMI per level, effectiveness, effectiveness per level, consistency, consistency per level
    """
    # Get relevant attributes
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects/2)

    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects)
    # add one such that zero becomes an empty attribute for the calculation (_)
    objects = objects + 1
    concepts = torch.from_numpy(objects * (np.array(fixed)))

    # get distractor objects to re-construct context conditions
    distractor_objects = sender_input[:, n_targets:]
    distractor_objects = k_hot_to_attributes(distractor_objects, n_values)
    distractor_objects = distractor_objects + 1
    context_conds = retrieve_context_condition(objects, fixed, distractor_objects)

    # get messages from interaction
    messages = interaction.message.argmax(dim=-1)

    # Entropies:
    # H(m), H(c), H(m,c)
    m_entropy = calc_entropy(messages)
    c_entropy = calc_entropy(concepts)
    joint_mc_entropy = joint_entropy(messages, concepts)

    # Concept-dependent Entropies:
    # sum of fixed vectors gives the specificity of the concept (all attributes fixed means
    # specific concept, one attribute fixed means generic concept)
    # n_relevant_idx stores the indices of the concepts on a specific level of abstraction
    n_relevant_idx = [np.where(np.sum(np.array(fixed), axis=1) == i)[0] for i in range(1, n_dims + 1)]
    # H(m), H(c), H(m,c) for each level of abstraction
    m_entropy_hierarchical = np.array([calc_entropy(messages[n_relevant]) for n_relevant in n_relevant_idx])    
    c_entropy_hierarchical = np.array([calc_entropy(concepts[n_relevant]) for n_relevant in n_relevant_idx])
    joint_entropy_hierarchical = np.array([joint_entropy(messages[n_relevant], concepts[n_relevant])
                                  for n_relevant in n_relevant_idx])
    
    # Context-dependent Entropies:
    context_cond_idx = [np.where(np.array(context_conds) == i)[0] for i in range(0, n_dims)]
    # H(m), H(c), H(m,c) for each context condition
    m_entropy_context_dep = np.array([calc_entropy(messages[context_cond]) for context_cond in context_cond_idx])
    c_entropy_context_dep = np.array([calc_entropy(concepts[context_cond]) for context_cond in context_cond_idx])
    joint_entropy_context_dep = np.array([joint_entropy(messages[context_cond], concepts[context_cond])
                                  for context_cond in context_cond_idx])

    # Concept-context dependent Entropies:
    # go through concept conditions
    conceptxcontext_idx = []
    for i in range(len(n_relevant_idx)):
        # go through context conditions
        for j in range(len(context_cond_idx)):
            # only keep shared entries for each concept-context condition
            shared_elements = [elem for elem in n_relevant_idx[i] if elem in context_cond_idx[j]]
            conceptxcontext_idx.append(shared_elements)

    # H(m), H(c), H(m,c) for each concept and context condition
    m_entropy_concept_x_context = np.array([calc_entropy(messages[concept_x_context]) for concept_x_context in conceptxcontext_idx])    
    c_entropy_concept_x_context = np.array([calc_entropy(concepts[concept_x_context]) for concept_x_context in conceptxcontext_idx])
    joint_entropy_concept_x_context = np.array([joint_entropy(messages[concept_x_context], concepts[concept_x_context])
                                  for concept_x_context in conceptxcontext_idx])
            

    # Normalized scores: NMI, consistency, effectiveness
    if normalizer == "arithmetic":
        normalizer = 0.5 * (m_entropy + c_entropy)
        normalizer_hierarchical = 0.5 * (m_entropy_hierarchical + c_entropy_hierarchical)
        normalizer_context_dep = 0.5 * (m_entropy_context_dep + c_entropy_context_dep)
        normalizer_conc_x_cont = 0.5 * (m_entropy_concept_x_context + c_entropy_concept_x_context)
    elif normalizer == "joint":
        normalizer = joint_mc_entropy
        normalizer_hierarchical = joint_entropy_hierarchical
        normalizer_context_dep = joint_entropy_context_dep
        normalizer_conc_x_cont = joint_entropy_concept_x_context
    else:
        raise AttributeError("Unknown normalizer")

    # normalized mutual information: H(m) - H(m|c) / normalizer, H(m|c)=H(m,c)-H(c)
    normalized_MI = (m_entropy + c_entropy - joint_mc_entropy) / normalizer
    normalized_MI_hierarchical = ((m_entropy_hierarchical + c_entropy_hierarchical - joint_entropy_hierarchical)
                                  / normalizer_hierarchical)
    normalized_MI_context_dep = ((m_entropy_context_dep + c_entropy_context_dep - joint_entropy_context_dep)
                                  / normalizer_context_dep)
    normalized_MI_conc_x_cont = ((m_entropy_concept_x_context + c_entropy_concept_x_context - joint_entropy_concept_x_context)
                                    / normalizer_conc_x_cont)

    # normalized version of h(c|m), i.e. h(c|m)/h(c)
    normalized_effectiveness = (joint_mc_entropy - m_entropy) / c_entropy
    normalized_effectiveness_hierarchical = ((joint_entropy_hierarchical - m_entropy_hierarchical) 
                                             / c_entropy_hierarchical)
    normalized_effectiveness_context_dep = ((joint_entropy_context_dep - m_entropy_context_dep) 
                                             / c_entropy_context_dep)
    normalized_effectiveness_conc_x_cont = ((joint_entropy_concept_x_context - m_entropy_concept_x_context)
                                            / c_entropy_concept_x_context)

    # normalized version of h(m|c), i.e. h(m|c)/h(m)
    normalized_consistency = (joint_mc_entropy - c_entropy) / m_entropy
    normalized_consistency_hierarchical = (joint_entropy_hierarchical - c_entropy_hierarchical) / m_entropy_hierarchical
    normalized_consistency_context_dep = (joint_entropy_context_dep - c_entropy_context_dep) / m_entropy_context_dep
    normalized_consistency_conc_x_cont = (joint_entropy_concept_x_context - c_entropy_concept_x_context) / m_entropy_concept_x_context

    score_dict = {'normalized_mutual_info': normalized_MI,
                  'normalized_mutual_info_hierarchical': normalized_MI_hierarchical,
                  'normalized_mutual_info_context_dep': normalized_MI_context_dep,
                  'normalized_mutual_info_concept_x_context': normalized_MI_conc_x_cont,
                  'effectiveness': 1 - normalized_effectiveness,
                  'effectiveness_hierarchical': 1 - normalized_effectiveness_hierarchical,
                  'effectiveness_context_dep': 1 - normalized_effectiveness_context_dep,
                  'effectiveness_concept_x_context': 1 - normalized_effectiveness_conc_x_cont,
                  'consistency': 1 - normalized_consistency,
                  'consistency_hierarchical': 1 - normalized_consistency_hierarchical,
                  'consistency_context_dep': 1 - normalized_consistency_context_dep,
                  'consistency_concept_x_context': 1 - normalized_consistency_conc_x_cont
                  }
    return score_dict


def cooccurrence_per_hierarchy_level(interaction, n_attributes, n_values, vs_factor):

    vocab_size = (n_values + 1) * vs_factor + 1

    messages = interaction.message.argmax(dim=-1)
    messages = messages[:, :-1].numpy()
    sender_input = interaction.sender_input.numpy()
    relevance_vectors = sender_input[:, -n_attributes:]

    cooccurrence = np.zeros((vocab_size, n_attributes))

    for s in range(vocab_size):
        for i, m in enumerate(messages):
            relevance = relevance_vectors[i]
            cooccurrence[s, int(sum(relevance))] += list(m).count(s)

    cooccurrence = cooccurrence[1:, :]  # remove eos symbol
    split_indices = np.array([np.sum(sender_input[:, -n_attributes:], axis=1) == i for i in range(n_attributes)])
    normalization = np.array([np.sum(split_indices[i]) for i in range(n_attributes)])
    cooccurrence = cooccurrence / normalization

    return cooccurrence


def message_length_per_hierarchy_level(interaction, n_attributes):

    # Get relevant attributes
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects/2)
    n_values = int(sender_input.shape[2]/n_attributes)

    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects)

    message = interaction.message.argmax(dim=-1)
    ml = MessageLengthHierarchical.compute_message_length(message)
    print("message length", ml)
    ml_hierarchical = MessageLengthHierarchical.compute_message_length_hierarchical(message, torch.from_numpy(fixed))
    print("hierarchical", ml_hierarchical)
    return ml_hierarchical


def symbol_frequency(interaction, n_attributes, n_values, vocab_size, is_gumbel=True):

    messages = interaction.message.argmax(dim=-1) if is_gumbel else interaction.message
    messages = messages[:, :-1]
    sender_input = interaction.sender_input
    n_objects = sender_input.shape[1]
    n_targets = int(n_objects/2)
    #k_hots = sender_input[:, :-n_attributes]
    #objects = k_hot_to_attributes(k_hots, n_values)
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    #intentions = sender_input[:, -n_attributes:]  # (0=same, 1=any)
    (objects, fixed) = retrieve_concepts_sampling(target_objects)

    objects[fixed == 1] = np.nan

    objects = objects
    messages = messages
    favorite_symbol = {}
    mutual_information = {}
    for att in range(n_attributes):
        for val in range(n_values):
            object_labels = (objects[:, att] == val).astype(int)
            max_MI = 0
            for symbol in range(vocab_size):
                symbol_indices = np.argwhere(messages == symbol)[0]
                symbol_labels = np.zeros(len(messages))
                symbol_labels[symbol_indices] = 1
                MI = normalized_mutual_info_score(symbol_labels, object_labels)
                if MI > max_MI:
                    max_MI = MI
                    max_symbol = symbol
            favorite_symbol[str(att) + str(val)] = max_symbol
            mutual_information[str(att) + str(val)] = max_MI

    sorted_objects = []
    sorted_messages = []
    for i in reversed(range(n_attributes)):
        sorted_objects.append(objects[np.sum(np.isnan(objects), axis=1) == i])
        sorted_messages.append(messages[np.sum(np.isnan(objects), axis=1) == i])

    # from most concrete to most abstract (all same to only one same)
    att_val_frequency = np.zeros(n_attributes)
    symbol_frequency = np.zeros(n_attributes)

    for level in range(n_attributes):
        for obj, message in zip(sorted_objects[level], sorted_messages[level]):
            for position in range(len(obj)):
                if not np.isnan(obj[position]):
                    att_val_frequency[level] += 1
                    fav_symbol = favorite_symbol[str(position) + str(int(obj[position]))]
                    symbol_frequency[level] += np.count_nonzero(message == fav_symbol)

    return symbol_frequency / att_val_frequency, mutual_information
