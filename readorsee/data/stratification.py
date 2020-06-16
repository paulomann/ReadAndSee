import random
import time
import numpy as np


class LocalSearch:

    """ TabuSearch to stratify the dataset."""

    def __init__(self, tr_size=0.6, val_size=0.2, te_size=0.2):
        self.tr_size = tr_size
        self.te_size = te_size
        self.val_size = val_size
        if tr_size + te_size + val_size != 1.0:
            raise ValueError
        self._original_bdi_qty = {0: 0, 1: 0}
        self._original_size = -1
        self._days = 60
        self._timer = None

    def stratify(self, participants, days):
        """ Return the participants list stratified through train/val/test
        size, class, and number of examples for each participant.

        We stratify by 3 levels: train/test/val size, class, and examples

        Params:
        participants -- A list of models.Participant subclasses
        Return:
        the stratified participants list
        """
        print("Stratifying...")
        self._timer = time.time()
        self._days = days
        prtcpts = np.array(self._get_participants_with_posts(participants))
        self._original_bdi_qty, self._original_size = self._calculate_bdi_qty(
            prtcpts)
        best = tuple(range(len(prtcpts)))
        best_candidate = best
        tabuList = [hash(best)]
        TABU_SIZE = int(0.3*len(prtcpts))

        while not self._stopping_condition(prtcpts, best):
            neighbors = self._get_neighbors(prtcpts, best_candidate)
            for n in neighbors:
                if (hash(n) not in tabuList
                   and self._fitness(prtcpts, n)
                   < self._fitness(prtcpts, best_candidate)):
                    best_candidate = n

            if (self._fitness(prtcpts, best_candidate)
                    < self._fitness(prtcpts, best)):
                best = best_candidate

            if random.random() < 0.30:
                best_candidate = random.choice(neighbors)

            tabuList.append(hash(best_candidate))
            if len(tabuList) > TABU_SIZE:
                tabuList.pop(0)

        return self._get_subsets(prtcpts, best)

    def _get_participants_with_posts(self, participants):
        """ Return participants that have, at least, one post in the previous
        days' from the questionnaire answer. """
        participants_with_posts = []
        for p in participants:
            posts = p.get_posts_from_qtnre_answer_date(self._days)
            if len(posts) > 0:
                participants_with_posts.append(p)

        return participants_with_posts

    def _stopping_condition(self, participants, best):

        fit_value = self._fitness(participants, best)

        epsilon = 0.08
        print("Fit Value : {}".format(fit_value))
        if fit_value <= epsilon or time.time() - self._timer >= 300:  # 300
            return True
        return False

    def _get_neighbors(self, participants, best_candidate):
        NEIGHBORS = 15

        tr_idx, val_idx = self._get_indexes(participants)
        neighbors = []

        for n in range(NEIGHBORS):
            if n % 2 == 0:
                indexes = [random.randrange(0, tr_idx),
                           random.randrange(tr_idx, val_idx),
                           random.randrange(val_idx, len(participants))]
                swap_idx = random.sample(indexes, 2)
                neighbor = list(best_candidate)
                temp = neighbor[swap_idx[0]]
                neighbor[swap_idx[0]] = neighbor[swap_idx[1]]
                neighbor[swap_idx[1]] = temp
            else:
                neighbor = random.sample(best_candidate, len(best_candidate))
            neighbors.append(tuple(neighbor))

        return neighbors

    def _fitness(self, participants, mask):
        """ Return the sum of differences of the maximum and minimum of: (1)
        proportions of examples for each BDI category, and (2) proportion of
        examples in each generated set (test, train, and validation)
        """
        tr_subset, val_subset, test_subset = self._get_subsets(participants,
                                                               mask)

        def get_bdi_fraction(bdi_qty, qty):
            return (bdi_qty[0] / qty), qty

        original_bdi_0_frac, total_qty = get_bdi_fraction(
            self._original_bdi_qty,
            self._original_size)

        original_tr_size = total_qty*self.tr_size
        original_val_size = total_qty*self.val_size
        original_test_size = total_qty*self.te_size

        tr_bdi_0_frac, tr_qty = get_bdi_fraction(
            *self._calculate_bdi_qty(tr_subset))
        val_bdi_0_frac, val_qty = get_bdi_fraction(
            *self._calculate_bdi_qty(val_subset))
        test_bdi_0_frac, test_qty = get_bdi_fraction(
            *self._calculate_bdi_qty(test_subset))

        bdi_proportions = [np.abs(tr_bdi_0_frac - original_bdi_0_frac),
                           np.abs(val_bdi_0_frac - original_bdi_0_frac),
                           np.abs(test_bdi_0_frac - original_bdi_0_frac)]

        sets_proportions = [np.abs(tr_qty - original_tr_size),
                            np.abs(val_qty - original_val_size),
                            np.abs(test_qty - original_test_size)]

        # Normalization process. Necessary due to the discrepancies between
        # bdi_proportions and sets_proportions values.
        bdi_proportions = bdi_proportions / np.linalg.norm(bdi_proportions,
                                                           ord=1)
        sets_proportions = sets_proportions / np.linalg.norm(sets_proportions,
                                                             ord=1)

        # Here, we use the difference between the max and the min value to
        # weight the generated solutions. More discrepancies in the generated
        # set, more weighted they become.
        return ((np.max(bdi_proportions) - np.min(bdi_proportions))
                + (np.max(sets_proportions) - np.min(sets_proportions)))

    def _get_subsets(self, participants, mask):
        tr_idx, val_idx = self._get_indexes(participants)
        chosen_sets = participants[list(mask)]
        tr_subset = chosen_sets[:tr_idx]
        val_subset = chosen_sets[tr_idx:val_idx]
        test_subset = chosen_sets[val_idx:]
        return tr_subset, val_subset, test_subset

    def _get_indexes(self, participants):
        """ Return the maximum index for training and val sets.

        It's not necessary to return training index since it's the last element
        in the array. Typically:
        training set = [00% - 60%)
        val set      = [60% - 80%)
        training set = [80% - 100%]
        """
        tr_idx = int(np.floor(self.tr_size*len(participants)))
        j = self.val_size + self.tr_size
        val_idx = int(np.floor(j*len(participants)))
        return tr_idx, val_idx

    def _calculate_bdi_qty(self, subset):
        bdi_fraction = {0: 0, 1: 0}
        for participant in subset:
            posts = participant.get_posts_from_qtnre_answer_date(self._days)
            # qty = len(posts)
            qty = self._get_total_number_of_images(posts)
            bdi = participant.questionnaire.get_binary_bdi()
            bdi_fraction[bdi] += qty

        return bdi_fraction, (bdi_fraction[0] + bdi_fraction[1])

    def _get_total_number_of_images(self, posts):
        total = 0
        for p in posts:
            total += len(p.get_img_path_list())
        return total

class LocalSearchTwitter:

    """ TabuSearch to stratify the twitter dataset."""

    def __init__(self, tr_size=0.6, val_size=0.2, te_size=0.2):
        self.tr_size = tr_size
        self.te_size = te_size
        self.val_size = val_size
        if tr_size + te_size + val_size != 1.0:
            raise ValueError
        self._original_bdi_qty = {0: 0, 1: 0}
        self._original_size = -1
        self._days = 60
        self._timer = None

    def stratify(self, participants, days):
        """ Return the participants list stratified through train/val/test
        size, class, and number of examples for each participant.

        We stratify by 3 levels: train/test/val size, class, and examples

        Params:
        participants -- A list of models.Participant subclasses
        Return:
        the stratified participants list
        """
        print("Stratifying...")
        self._timer = time.time()
        self._days = days
        prtcpts = np.array(self._get_participants_with_posts(participants))
        self._original_bdi_qty, self._original_size = self._calculate_bdi_qty(
            prtcpts)
        best = tuple(range(len(prtcpts)))
        best_candidate = best
        tabuList = [hash(best)]
        TABU_SIZE = int(0.3*len(prtcpts))

        while not self._stopping_condition(prtcpts, best):
            neighbors = self._get_neighbors(prtcpts, best_candidate)
            for n in neighbors:
                if (hash(n) not in tabuList
                   and self._fitness(prtcpts, n)
                   < self._fitness(prtcpts, best_candidate)):
                    best_candidate = n

            if (self._fitness(prtcpts, best_candidate)
                    < self._fitness(prtcpts, best)):
                best = best_candidate

            if random.random() < 0.30:
                best_candidate = random.choice(neighbors)

            tabuList.append(hash(best_candidate))
            if len(tabuList) > TABU_SIZE:
                tabuList.pop(0)

        return self._get_subsets(prtcpts, best)

    def _get_participants_with_posts(self, participants):
        """ Return participants that have, at least, one post in the previous
        days' from the questionnaire answer. """
        participants_with_posts = []
        for p in participants:
            posts = p.get_posts_from_qtnre_answer_date(self._days)
            if len(posts) > 0:
                participants_with_posts.append(p)

        return participants_with_posts

    def _stopping_condition(self, participants, best):

        fit_value = self._fitness(participants, best)

        epsilon = 0.08
        print("Fit Value : {}".format(fit_value))
        if fit_value <= epsilon or time.time() - self._timer >= 300:  # 300
            return True
        return False

    def _get_neighbors(self, participants, best_candidate):
        NEIGHBORS = 15

        tr_idx, val_idx = self._get_indexes(participants)
        neighbors = []

        for n in range(NEIGHBORS):
            if n % 2 == 0:
                indexes = [random.randrange(0, tr_idx),
                           random.randrange(tr_idx, val_idx),
                           random.randrange(val_idx, len(participants))]
                swap_idx = random.sample(indexes, 2)
                neighbor = list(best_candidate)
                temp = neighbor[swap_idx[0]]
                neighbor[swap_idx[0]] = neighbor[swap_idx[1]]
                neighbor[swap_idx[1]] = temp
            else:
                neighbor = random.sample(best_candidate, len(best_candidate))
            neighbors.append(tuple(neighbor))

        return neighbors

    def _fitness(self, participants, mask):
        """ Return the sum of differences of the maximum and minimum of: (1)
        proportions of examples for each BDI category, and (2) proportion of
        examples in each generated set (test, train, and validation)
        """
        tr_subset, val_subset, test_subset = self._get_subsets(participants,
                                                               mask)

        def get_bdi_fraction(bdi_qty, qty):
            return (bdi_qty[0] / qty), qty

        original_bdi_0_frac, total_qty = get_bdi_fraction(
            self._original_bdi_qty,
            self._original_size)

        original_tr_size = total_qty*self.tr_size
        original_val_size = total_qty*self.val_size
        original_test_size = total_qty*self.te_size

        tr_bdi_0_frac, tr_qty = get_bdi_fraction(
            *self._calculate_bdi_qty(tr_subset))
        val_bdi_0_frac, val_qty = get_bdi_fraction(
            *self._calculate_bdi_qty(val_subset))
        test_bdi_0_frac, test_qty = get_bdi_fraction(
            *self._calculate_bdi_qty(test_subset))

        bdi_proportions = [np.abs(tr_bdi_0_frac - original_bdi_0_frac),
                           np.abs(val_bdi_0_frac - original_bdi_0_frac),
                           np.abs(test_bdi_0_frac - original_bdi_0_frac)]

        sets_proportions = [np.abs(tr_qty - original_tr_size),
                            np.abs(val_qty - original_val_size),
                            np.abs(test_qty - original_test_size)]

        # Normalization process. Necessary due to the discrepancies between
        # bdi_proportions and sets_proportions values.
        bdi_proportions = bdi_proportions / np.linalg.norm(bdi_proportions,
                                                           ord=1)
        sets_proportions = sets_proportions / np.linalg.norm(sets_proportions,
                                                             ord=1)

        # Here, we use the difference between the max and the min value to
        # weight the generated solutions. More discrepancies in the generated
        # set, more weighted they become.
        return ((np.max(bdi_proportions) - np.min(bdi_proportions))
                + (np.max(sets_proportions) - np.min(sets_proportions)))

    def _get_subsets(self, participants, mask):
        tr_idx, val_idx = self._get_indexes(participants)
        chosen_sets = participants[list(mask)]
        tr_subset = chosen_sets[:tr_idx]
        val_subset = chosen_sets[tr_idx:val_idx]
        test_subset = chosen_sets[val_idx:]
        return tr_subset, val_subset, test_subset

    def _get_indexes(self, participants):
        """ Return the maximum index for training and val sets.

        It's not necessary to return training index since it's the last element
        in the array. Typically:
        training set = [00% - 60%)
        val set      = [60% - 80%)
        training set = [80% - 100%]
        """
        tr_idx = int(np.floor(self.tr_size*len(participants)))
        j = self.val_size + self.tr_size
        val_idx = int(np.floor(j*len(participants)))
        return tr_idx, val_idx

    def _calculate_bdi_qty(self, subset):
        bdi_fraction = {0: 0, 1: 0}
        for participant in subset:
            posts = participant.get_posts_from_qtnre_answer_date(self._days)
            qty = len(posts)
            #qty = self._get_total_number_of_images(posts)
            bdi = participant.questionnaire.get_binary_bdi()
            bdi_fraction[bdi] += qty

        return bdi_fraction, (bdi_fraction[0] + bdi_fraction[1])

    #def _get_total_number_of_images(self, posts):
    #    total = 0
    #    for p in posts:
    #       total += len(p.get_img_path_list())
    #    return total
    
class SimpleTwitterStratifie:

    """ 
        SimpleTwitterStratifie code to stratify the twitter data
        by number of posts and depression diagnosed frequency 
    """

    def __init__(self, n_sets, days):
        
        if n_sets != 1:
            raise ValueError

        self.n_sets = n_sets
        self.days = days

    def stratify(self, participants):

        print("Twitter Stratifying...")

        data_separeted_by_days = {}
        data_separeted_by_days['data_' + str(self.days[0])] = []
        data_separeted_by_days['data_' + str(self.days[1])] = []
        data_separeted_by_days['data_' + str(self.days[2])] = []

        final_data = {}
        final_data['data_' + str(self.days[0])] = []
        final_data['data_' + str(self.days[1])] = []
        final_data['data_' + str(self.days[2])] = []

        participants.sort(key=lambda x: len(x.posts))
        participants.sort(key=lambda x: x.depression_diagnosed)

        idx = 0

        for user in participants:
            
            if idx == 0:
                data_separeted_by_days['data_' + str(self.days[0])].append(user)
            elif idx == 1:
                data_separeted_by_days['data_' + str(self.days[1])].append(user)
            else:
                data_separeted_by_days['data_' + str(self.days[2])].append(user)
                
            idx += 1
            
            if (idx == 3):
                idx = 0
                
        participants_sub1 = []
        participants_sub2 = []
        participants_sub3 = []

        for sub_data_index in data_separeted_by_days:
            
            sud_data = data_separeted_by_days[sub_data_index]
            
            sud_data.sort(key=lambda x: len(x.posts))
            sud_data.sort(key=lambda x: x.depression_diagnosed)

            idx = 0

            for user in sud_data:

                if idx == 0:
                    participants_sub1.append(user)
                elif idx == 1:
                    participants_sub2.append(user)
                else:
                    participants_sub3.append(user)

                idx += 1

                if (idx == 3):
                    idx = 0
                
            participants_sub1_array = np.array(participants_sub1)
            participants_sub2_array = np.array(participants_sub2)
            participants_sub3_array = np.array(participants_sub3)
            
            participants_tuple = (participants_sub1_array, participants_sub2_array, participants_sub3_array)
            
            final_data[sub_data_index] = []
            final_data[sub_data_index].append(participants_tuple) 

        return final_data