import unittest
import math
import numpy as np

from dataset import DataSet


class TestDataset(unittest.TestCase):

    def setUp(self):

        # feel free to add more data sets and game sizes
        self.possible_properties = [[2, 2],
                                    [4, 4],
                                    [3, 3, 3],
                                    [4, 4, 4, 4]]
        self.game_sizes = [1, 3, 10]

        self.datasets = []
        for props in self.possible_properties:
            for size in self.game_sizes:
                self.datasets.append(DataSet(props, size, 'cpu', testing=True))

    def test_get_all_concepts(self):
        """
        Test
        - if number of concepts is correct
        - if number of instances per concept is correct

        Functions used by get_all_concepts:
        - get_fixed_vectors
        - _get_all_possible_objects
        - get_all_objects_for_a_concepts
        - satisfies
        """

        for ds in self.datasets:
            n_atts = len(ds.properties_dim)
            n_vals = ds.properties_dim[0]
            concepts = ds.concepts

            # Test total number of concepts
            combinations_per_attribute = 0
            for i in range(1, n_atts+1):
                combinations_per_attribute += math.comb(n_atts, i)*(n_vals**i)
            total_combinations = combinations_per_attribute
            print(len(ds.concepts), total_combinations)
            self.assertEqual(total_combinations, len(concepts))

            # Test number of instances per concept
            for c in concepts:
                n_fixed = np.sum(c[1])
                n_instances = n_vals**(n_atts-n_fixed)
                self.assertEqual(n_instances, len(c[0]))

    def test_get_distractors(self):
        """
        Test
        - if the right number (all) distractors are generated for each concept
        - if at least one fixed attribute is different between target and distractor
        """

        for ds in self.datasets:
            n_atts = len(ds.properties_dim)
            n_vals = ds.properties_dim[0]
            concepts = ds.concepts
            n_objects = n_vals**n_atts

            for c_idx, concept in enumerate(concepts):
                nr_possible_contexts = sum(concepts[c_idx][1])
                # collect distractors over all possible context conditions
                distractors = []
                for context_condition in range(0, nr_possible_contexts):
                    distractors_distributed = DataSet.get_distractors(ds, c_idx, context_condition)

                    for elem in distractors_distributed:
                        distractors.append(elem)

                # make sure distractors are not counted twice
                distractors = set(distractors)

                # assert number of distractors correct
                n_fixed = np.sum(concept[1])
                n_expected_distractors = int(n_objects * (1 - (1 / n_vals) ** n_fixed))
                self.assertEqual(n_expected_distractors, len(distractors))

                # assert distractors do not correspond to the target
                # self.assertFalse(concept[0] in distractors)

                # assert that distractors differ from target in at least one fixed attribute
                for d in distractors:
                    diff = np.abs(np.array(concept[0]) - np.array(d))
                    # select the difference between target and distractor for fixed attributes
                    mask = 1 - np.tile(np.array(concept[1]), (len(diff), 1))
                    masked_difference = np.ma.masked_array(diff, mask)
                    self.assertTrue(0 not in np.sum(masked_difference, axis=1))

    def test_get_item(self):
        """
        Test
        - if the game size is correct
        - if the target inputs correspond to the concept
        - if the distractors do not correspond to the concept
        - if the distractors fulfill the context condition
        """

        for ds in self.datasets:

            dim = int(ds.properties_dim[0])

            for c_idx, concept in enumerate(ds.concepts):
                n_fixed = np.sum(concept[1])
                # only for reasonable context conditions
                for n_same in range(n_fixed):
                    item = ds.get_item(c_idx, n_same, ds._many_hot_encoding)
                    sender_input, receiver_label, receiver_input = item

                    # test whether game size correct
                    self.assertEqual(ds.game_size*2, len(sender_input))
                    self.assertEqual(ds.game_size*2, len(receiver_input))

                    # test whether targets correspond to concept

                    example_concept = concept[0][0]
                    relevant_indices = []
                    relevant_values = []
                    for i in range(len(concept[1])):
                        if concept[1][i] == 1:
                            relevant_indices.append(i)
                            relevant_values.append(example_concept[i])

                    for g in range(ds.game_size):
                        for i, idx in enumerate(relevant_indices):
                            self.assertEqual(
                                relevant_values[i],
                                np.argmax(sender_input[g][idx * dim:(idx + 1) * dim])
                            )

                    for g, l in enumerate(receiver_label):
                        if l == 1:
                            for i, idx in enumerate(relevant_indices):
                                self.assertEqual(
                                    relevant_values[i],
                                    np.argmax(receiver_input[g][idx * dim:(idx + 1) * dim])
                                )

                    # test whether distractors fulfill the context condition for sender and receiver

                    for g in range(ds.game_size, ds.game_size*2):
                        count_mismatch = 0
                        for i, idx in enumerate(relevant_indices):
                            if relevant_values[i] != np.argmax(sender_input[g][idx * dim:(idx + 1) * dim]):
                                count_mismatch += 1
                        self.assertEqual(len(relevant_indices) - n_same, count_mismatch)

                    for g, l in enumerate(receiver_label):
                        count_mismatch = 0
                        if l == 0:
                            for i, idx in enumerate(relevant_indices):
                                if relevant_values[i] != np.argmax(receiver_input[g][idx * dim:(idx + 1) * dim]):
                                    count_mismatch += 1
                            self.assertEqual(len(relevant_indices) - n_same, count_mismatch)


if __name__ == '__main__':

    unittest.main()
