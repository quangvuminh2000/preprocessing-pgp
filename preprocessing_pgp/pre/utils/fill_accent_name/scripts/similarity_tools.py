from typing import List

from rapidfuzz import fuzz
import numpy as np

from .split_name import NameProcess


def similarity(word_a: str,
               word_b: str,
               weight: int = None,
               threshold: int = None) -> float:
    if (word_a == None or word_b == None):  # Missing word
        return 100 * weight

    # |a| != |b| and a[0] != b[0] --> First word difference (Quang|g)
    if (len(word_a) != len(word_b)) and (word_a[0] != word_b[0]) and max(len(word_a.split()), len(word_b.split())) == 1:
        return 0

    sim_word = fuzz.WRatio(word_a, word_b)
    # print(word_a, word_b, sim_word)

    score = 0

    if weight != None:
        score = sim_word * weight
    else:
        score = sim_word

    if threshold != None:
        score = 0 if sim_word < threshold else score

    return score


class NameSimilarity:
    def __init__(self, base_path):
        self.name_process = NameProcess(base_path)

    def compute(self,
                a: str,
                b: str,
                weights: List[int] = None,
                threshold: int = None) -> float:
        # Preprocessing
        clean_a = self.name_process.CleanName(a)
        clean_b = self.name_process.CleanName(b)
        # print(clean_a, clean_b)
        preprocessed_a = clean_a.lower()
        preprocessed_b = clean_b.lower()

        # Split the names
        last_a, middle_a, first_a = self.name_process.SplitName(preprocessed_a)

        last_b, middle_b, first_b = self.name_process.SplitName(preprocessed_b)

#         print(last_a, middle_a, first_a)
#         print(last_b, middle_b, first_b)

        # Performing similarity calculation
        if weights != None:
            assert len(
                weights) == 4, 'Please provide 4 weights for first, middle, last names, and fullname respectively!'

            last_w, middle_w, first_w, full_w = weights
            first_sim = similarity(first_a, first_b, first_w, threshold)
            middle_sim = similarity(middle_a, middle_b, middle_w, threshold)
            last_sim = similarity(last_a, last_b, last_w, threshold)
            full_sim = fuzz.WRatio(preprocessed_a, preprocessed_b) * full_w

#             print(last_sim/last_w, middle_sim/middle_w,
#                   first_sim/first_w, full_sim/full_w)

            score = (first_sim + middle_sim +
                     last_sim + full_sim) / sum(weights)

            return score

        return np.average([first_sim, middle_sim, last_sim, full_sim])

if __name__ == '__main__':
    name_a = input('Please input the first name: ')
    name_b = input('Please input the second name: ')

    name_sim = NameSimilarity()

    print(name_sim.compute(name_a, name_b, [0.25, 0.25, 0.5], threshold=40))
