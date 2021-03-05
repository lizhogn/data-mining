#   /$$$$$$  /$$$$$$$  /$$$$$$$  /$$$$$$  /$$$$$$  /$$$$$$$  /$$$$$$
#  /$$__  $$| $$__  $$| $$__  $$|_  $$_/ /$$__  $$| $$__  $$|_  $$_/
# | $$  \ $$| $$  \ $$| $$  \ $$  | $$  | $$  \ $$| $$  \ $$  | $$
# | $$$$$$$$| $$$$$$$/| $$$$$$$/  | $$  | $$  | $$| $$$$$$$/  | $$
# | $$__  $$| $$____/ | $$__  $$  | $$  | $$  | $$| $$__  $$  | $$
# | $$  | $$| $$      | $$  \ $$  | $$  | $$  | $$| $$  \ $$  | $$
# | $$  | $$| $$      | $$  | $$ /$$$$$$|  $$$$$$/| $$  | $$ /$$$$$$
# |__/  |__/|__/      |__/  |__/|______/ \______/ |__/  |__/|______/

# author: lizhogn
# time: 2021-01-04
# type: "data science 2nd practice"
# function: Apriori algrithm implement

import pandas as pd
from collections import defaultdict
from itertools import chain, combinations
import time


class Apriori(object):
    """
    run the apriori algorithm, data_iter is a record iterator
    Input:
        data: Pandas Series data, for example like these form:
                >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                0    {margarine, citrus fruit, ready soups, semi-fi...
                1                     {coffee, tropical fruit, yogurt}
                2                                         {whole milk}
                3     {meat spreads, pip fruit, yogurt, cream cheese }
                4    {whole milk, condensed milk, other vegetables,...
                <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        minSupport: the minimum support of the item
        minConf: the threshold of the Conference of the item
    Return both:
        items (tuple, support)
        rules ((pretuple, posttuple), confidence)
    """
    def __init__(self, data, minSupport=0.2, minConf=0.6, item_k=None):
        self.itemSets, self.transactionList = self.getItemSetTransactionList(data)
        self.minSupport = minSupport
        self.minConf = minConf
        self.itemk = item_k

    def getItemSetTransactionList(self, data):
        """
        parser and analysis the data
        :param data: Pandas Series data
        :return: itemSet, transactionList
        """
        transactionList = []
        itemSet = set()

        for record in data:
            transaction = frozenset(record)
            transactionList.append(transaction)
            for item in transaction:
                itemSet.add(frozenset([item]))
        return itemSet, transactionList

    def dummyApriori(self, show=True):
        # Base line algorithm.
        # Use brute force search to generate candidate k+1 item sets based on k frequent itemsets

        # start the timer
        time_start = time.time()

        self.freqSet = defaultdict(int)
        largeSet = dict()

        oneCSet = self.returnItemsWithMinSupport(k_itemSet=self.itemSets,
                                                 transactionList=self.transactionList)

        # ***********************generate all frequent candidate k-itemsets.**********************
        # brute search function to join 2 subset
        def joinSet(itemSet, length):
            """Join a set with itself and returns the n-element itemsets"""
            return set(
                [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
            )
        # ***********************generate all frequent candidate k-itemsets.**********************

        currenLSet = oneCSet
        k = 2
        while currenLSet != set([]):
            # store the k-1 item set
            largeSet[k-1] = currenLSet
            if self.itemk is not None:
                if self.itemk < k:
                    break

            # step1: generate the k item set
            currenLSet = joinSet(currenLSet, k)
            # step2: filter the k item set
            currenLSet = self.returnItemsWithMinSupport(k_itemSet=currenLSet,
                                                        transactionList=self.transactionList)
            k += 1

        # end the timer
        time_elapsed = time.time() - time_start
        print('Dummy Apriori complete in {:.5f}s'.format(time_elapsed))

        # get the result
        self.generateItemsAndRules(largeSet)

        return time_elapsed

    def advancedApriori1(self, show=True):
        """advanced Apriori 1: Reduce the size of the candidate set"""

        # start the timer
        time_start = time.time()

        self.freqSet = defaultdict(int)
        largeSet = dict()

        oneCSet = self.returnItemsWithMinSupport(k_itemSet=self.itemSets,
                                                 transactionList=self.transactionList)

        # ***********************generate all frequent candidate k-itemsets.**********************
        # the difference to dummy apriori
        def is_apriori(Ck_item, Lksub1):
            """
            Judge whether a frequent candidate k-itemset satisfy Apriori property.
            Args:
                Ck_item: a frequent candidate k-itemset in Ck which contains all frequent
                         candidate k-itemsets.
                Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.
            Returns:
                True: satisfying Apriori property.
                False: Not satisfying Apriori property.
            """
            for item in Ck_item:
                sub_Ck = Ck_item - frozenset([item])
                if sub_Ck not in Lksub1:
                    return False
            return True

        def joinSet(itemSet, k):
            """Join a set with itself and returns the n-element itemsets"""
            Ck = set()
            len_Lksub1 = len(itemSet)
            list_Lksub1 = list(itemSet)
            for i in range(len_Lksub1):
                for j in range(1, len_Lksub1):
                    l1 = list(list_Lksub1[i])
                    l2 = list(list_Lksub1[j])
                    l1.sort()
                    l2.sort()
                    if l1[0:k - 2] == l2[0:k - 2]:
                        Ck_item = list_Lksub1[i] | list_Lksub1[j]
                        # pruning
                        if is_apriori(Ck_item, itemSet):
                            Ck.add(Ck_item)
            return Ck

        # ***********************generate all frequent candidate k-itemsets.**********************

        currenLSet = oneCSet
        k = 2
        while currenLSet != set([]):
            # store the k-1 item set
            largeSet[k-1] = currenLSet
            if self.itemk is not None:
                if self.itemk < k:
                    break
            # step1: generate the k item set
            currenLSet = joinSet(currenLSet, k)
            # step2: filter the k item set
            currenLSet = self.returnItemsWithMinSupport(k_itemSet=currenLSet,
                                                        transactionList=self.transactionList)
            k += 1

        # end the timer
        time_elapsed = time.time() - time_start
        print('Advanced Apriori 1 complete in {:.5f}s'.format(time_elapsed))

        # get the result
        self.generateItemsAndRules(largeSet)

        return time_elapsed

    def advancedApriori2(self, show=True):
        """advanced Apriori 2: Reduce the size of the transaction list"""

        # start the timer
        time_start = time.time()

        self.freqSet = defaultdict(int)
        largeSet = dict()

        oneCSet = self.returnItemsWithMinSupport(k_itemSet=self.itemSets,
                                                 transactionList=self.transactionList)

        # ***********************generate all frequent candidate k-itemsets.**********************
        # brute search function to join 2 subset(the same to dummy apriori)
        def joinSet(itemSet, length):
            """Join a set with itself and returns the n-element itemsets"""
            return set(
                [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
            )

        # ***********************generate all frequent candidate k-itemsets.**********************

        currenLSet = oneCSet
        transactionList = self.transactionList.copy()
        k = 2
        while (currenLSet != set([])):
            # store the k-1 item set
            largeSet[k - 1] = currenLSet
            if self.itemk is not None:
                if self.itemk < k:
                    break
            # step2: generate the k item set
            currenLSet = joinSet(currenLSet, k)
            # step3: filter the k item set
            currenLSet, transactionList = self.returnItemsWithMinSupport2(k_itemSet=currenLSet,
                                                                         transactionList=transactionList, k=k)
            k += 1

        # end the timer
        time_elapsed = time.time() - time_start
        print('Advance Apriori 2 complete in {:.5f}s'.format(time_elapsed))

        # get the result
        self.generateItemsAndRules(largeSet)

        return time_elapsed

    def advancedApriori_12(self, show=True):
        """advanced Apriori 2: Reduce the size of the transaction list"""

        # start the timer
        time_start = time.time()

        self.freqSet = defaultdict(int)
        largeSet = dict()

        oneCSet = self.returnItemsWithMinSupport(k_itemSet=self.itemSets,
                                                 transactionList=self.transactionList)

        # ***********************generate all frequent candidate k-itemsets.**********************
        # the difference to dummy apriori
        def is_apriori(Ck_item, Lksub1):
            """
            Judge whether a frequent candidate k-itemset satisfy Apriori property.
            Args:
                Ck_item: a frequent candidate k-itemset in Ck which contains all frequent
                         candidate k-itemsets.
                Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.
            Returns:
                True: satisfying Apriori property.
                False: Not satisfying Apriori property.
            """
            for item in Ck_item:
                sub_Ck = Ck_item - frozenset([item])
                if sub_Ck not in Lksub1:
                    return False
            return True

        def joinSet(itemSet, k):
            """Join a set with itself and returns the n-element itemsets"""
            Ck = set()
            len_Lksub1 = len(itemSet)
            list_Lksub1 = list(itemSet)
            for i in range(len_Lksub1):
                for j in range(1, len_Lksub1):
                    l1 = list(list_Lksub1[i])
                    l2 = list(list_Lksub1[j])
                    l1.sort()
                    l2.sort()
                    if l1[0:k - 2] == l2[0:k - 2]:
                        Ck_item = list_Lksub1[i] | list_Lksub1[j]
                        # pruning
                        if is_apriori(Ck_item, itemSet):
                            Ck.add(Ck_item)
            return Ck

        # ***********************generate all frequent candidate k-itemsets.**********************

        currenLSet = oneCSet
        transactionList = self.transactionList.copy()
        k = 2
        while (currenLSet != set([])):
            # store the k-1 item set
            largeSet[k - 1] = currenLSet
            if self.itemk is not None:
                if self.itemk < k:
                    break
            # step2: generate the k item set
            currenLSet = joinSet(currenLSet, k)
            # step3: filter the k item set
            currenLSet, transactionList = self.returnItemsWithMinSupport2(k_itemSet=currenLSet,
                                                                          transactionList=transactionList, k=k)
            k += 1

        # end the timer
        time_elapsed = time.time() - time_start
        print('Advance Apriori 2 complete in {:.5f}s'.format(time_elapsed))

        # get the result
        self.generateItemsAndRules(largeSet)

        return time_elapsed

    def generateItemsAndRules(self, largeSet):

        RetItems = dict()
        RetRules = dict()

        if self.itemk is not None:
            # specifySet[1] = largeSet[1]
            RetItems[self.itemk] = largeSet[self.itemk]
            RetRules[self.itemk] = largeSet[self.itemk]
        else:
            RetItems = largeSet
            largeSet.pop(1)
            RetRules = largeSet


        self.toRetItems = []
        for key, value in RetItems.items():
            self.toRetItems.extend([(tuple(item), self.getSupport(item)) for item in value])

        self.toRetRules = []
        for key, value in RetRules.items():
            for item in value:
                _subsets = map(frozenset, [x for x in self.subsets(item)])
                for element in _subsets:
                    remain = item.difference(element)
                    if len(remain) > 0:
                        confidence = self.getSupport(item) / self.getSupport(element)
                        if confidence >= self.minConf:
                            self.toRetRules.append(((tuple(element), tuple(remain)), confidence))

        # show the result
        self.printResults(self.toRetItems, self.toRetRules)

    def returnItemsWithMinSupport(self, k_itemSet, transactionList):
        """calculate the support for items in the itemSet and returns a subet
        of the itemSet each of whose elements satisfies the minimum support

        :return: _itemSet: contains the item that Support > threshold
        """
        _itemSet = set()
        localSet = defaultdict(int)

        # time_start2 = time.time()
        for item in k_itemSet:
            for transaction in transactionList:
                if item.issubset(transaction):
                    localSet[item] += 1
                    self.freqSet[item] += 1
        # time_spend = time.time() - time_start2
        # print('total spend time: {:.3f}'.format(time_spend))

        for item, count in localSet.items():
            support = float(count) / len(self.transactionList)

            if support >= self.minSupport:
                _itemSet.add(item)

        return _itemSet

    def returnItemsWithMinSupport2(self, k_itemSet, transactionList, k):
        """ the same to returnItemsWithMinSupport, but add the variable to record the
        transaction matched times, for advanced apriori 2
        """
        _itemSet = set()
        localSet = defaultdict(int)

        # a list to statistic the matched times for each transaction
        matched_cnt = defaultdict(int)

        # time_start2 = time.time()
        for item in k_itemSet:
            for transaction in transactionList:
                if item.issubset(transaction):
                    localSet[item] += 1
                    self.freqSet[item] += 1
                    matched_cnt[transaction] += 1

        # time_spend = time.time() - time_start2
        # print('total spend time: {:.3f}'.format(time_spend))
        for item, count in localSet.items():
            support = float(count) / len(self.transactionList)

            if support >= self.minSupport:
                _itemSet.add(item)

        # filter out the transaction record with matching times less than k+1
        transactionList = [transaction for transaction, cnt in matched_cnt.items() if cnt > k]

        return _itemSet, transactionList


    def getSupport(self, item):
        """
        local funtion which Returns the support of an item
        :param item:
        :return:
        """
        return float(self.freqSet[item] / len(self.transactionList))

    def subsets(self, arr):
        """ Return non empty subsets of arr"""
        return chain(*[combinations(arr, i+1) for i, a in enumerate(arr)])

    def printResults(self, items, rules):
        """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
        print('\n----------------------------Items------------------------------')
        for item, support in sorted(items, key=lambda x: x[1]):
            print("item: %s, %.3f" % (str(item), support))
        print('\n----------------------------RULES------------------------------')
        for rule, confidence in sorted(rules, key=lambda x: x[1]):
            pre, post = rule
            print("Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))

    def algorithnEfficiencyAnalysis(self, max_itemk=None):

        # plot the time result
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']
        import numpy as np

        # alnalysis the time spend for each k-itemset
        time0, time1, time2 = [], [], []
        for k in range(1, max_itemk+1):
            self.itemk = k
            # dummy apriori
            time_elapsed0 = self.dummyApriori(show=False)
            # advanced apriori 1
            time_elapsed1 = self.advancedApriori1(show=False)
            # advanced apriori 2
            time_elapsed2 = self.advancedApriori_12(show=False)
            # record:
            time0.append(time_elapsed0)
            time1.append(time_elapsed1)
            time2.append(time_elapsed2)

        x = np.arange(1, max_itemk+1)
        bar_width = 0.2
        trick_label = ['{}-频繁项集'.format(k) for k in range(1, max_itemk+1)]

        # plot the bar
        plt.bar(x, time0, bar_width, color='salmon', label='DummyApriori')
        plt.bar(x+bar_width, time1, bar_width, color='orchid', label='AdvancedApriori_1')
        plt.bar(x+2*bar_width, time2, bar_width, color='lightpink', label='AdvancedApriori_12')

        plt.legend()
        plt.xticks(x+bar_width, trick_label)
        plt.ylabel('time(s)')
        plt.title('总耗时随项集增加的变化情况(支持度={})'.format(self.minSupport))
        plt.show()


def callFPGrowth(data, minSupport=0.2, minConf=0.6):
    import pyfpgrowth
    transactions = []
    for item in data:
        transactions.append(list(item))

    minSupportItems = int(len(transactions) * minSupport)

    # time start
    time_start = time.time()

    patterns = pyfpgrowth.find_frequent_patterns(transactions, minSupportItems)

    # end the timer
    time_elapsed = time.time() - time_start
    print('FP Growth complete in {:.5f}s'.format(time_elapsed))

    rules = pyfpgrowth.generate_association_rules(patterns, minConf)
    print(rules)


if __name__ == '__main__':

    # use the command line to receive parameters
    import argparse
    # create a parser
    parser = argparse.ArgumentParser(prog='Apriori')
    # add arguments to parser
    parser.add_argument('--file', '-f',
                        type=str,
                        default='GroceryStore/Groceries.csv',
                        help='dataset absolute/relative path')
    parser.add_argument('--support', '-s',
                        type=float,
                        default=0.04,
                        help='min support for the frequency itemset')
    parser.add_argument('--confidence', '-c',
                        type=float,
                        default=0.5,
                        help='min Confidence for the associate rule')
    parser.add_argument('-itemk','-k',
                        type=int,
                        default=3,
                        help='the size of frequent itemsets')
    # parse the arguments
    args = parser.parse_args()

    # 1. DATA PROCESS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ## step1.1: load the data
    data = pd.read_csv(args.file, delimiter=',')

    ## step1.2: split the string to split items
    def item_split(string: str):
        # split each item into list of items
        # step1: drop the { }
        string = string[1:-1]
        # step2: split the string by the ',' or '/'
        import re
        item_list = re.split(r'[,|/]\s*', string)
        return set(item_list)

    data = data['items'].apply(item_split)


    # # 2. Agrithm Implement>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # # initialize the Apriori class
    ap = Apriori(data,
                 minSupport=args.support,
                 minConf=args.confidence,
                 item_k=args.itemk)
    #
    # # 2.1 Dummy apriori algrithm
    ap.dummyApriori()
    # #
    # # # 2.2 Advanced Apriori 1
    ap.advancedApriori1()
    #
    # # 2.3 Advanced Apriori 2
    ap.advancedApriori2()

    # 2.4 py-FP-Growth implement
    callFPGrowth(data,
                 minSupport=args.support,
                 minConf=args.confidence)


    # 3. Time analysis>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ap.algorithnEfficiencyAnalysis(max_itemk=4)
