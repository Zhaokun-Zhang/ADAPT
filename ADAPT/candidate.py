from __future__ import annotations
from copy import deepcopy
from collections import defaultdict
from typing import Dict, Set, List, Tuple
import numpy as np


class CandidateSet():
    def __init__(self, feature_type_list:List[str] = None, **args):
        '''
        feature_type_list: valid feature type, such as ['word', 'doc', 'ext']
            default: all the feature type
        '''
        # cand:  Lid -> set of {Rids, ...}
        self.word_cand:Dict[int, Set[int]] = dict() 
        self.sent_cand:Dict[int, Set[int]] = dict() 
        self.doc_cand:Dict[int, Set[int]] = dict()
        self.ext_cand:Dict[int, Set[int]] = dict()
        self.union_cand:Dict[int, Set[int]] = dict()

        self.quota:Dict[str, int] = None

        self.word_list:List[Tuple[int, int, int]] = list() 
        self.sent_list:List[Tuple[int, int, int]] = list() 
        self.doc_list:List[Tuple[int, int, int]] = list()
        self.ext_list:List[Tuple[int, int, int]] = list()

        # evaluate_result_dict
        self.eval_dict:Dict[str, float] = None

        if feature_type_list is None:
            self.feature_type_list:List[str] = ['word', 'sent', 'doc', 'ext']
        else: self.feature_type_list = list(feature_type_list)
    
    def trans_list2cand(self):
        for feature_type in self.feature_type_list:
            cand_dict:Dict[int, Set[int]] = defaultdict(set)
            feat_list:List[Tuple[int, int, int]] = getattr(self, f'{feature_type}_list')
            for l in feat_list:
                cand_dict[int(l[0])].add(int(l[1]))
            setattr(self, f'{feature_type}_cand', dict(cand_dict))  

    @staticmethod
    def candidate2set(cand:Dict[int, Set[int]]) -> Set[int]:
        s = set()
        for Lid, Ridset in cand.items(): 
            s |= set(((int(Lid), int(Rid)) for Rid in Ridset))
        return s
    @staticmethod
    def set2candidate(s:Set[int]) -> Dict[int, Set[int]]:
        cand = defaultdict(set)
        for Lid, Rid in s: cand[Lid].add(Rid)
        return dict(cand)
    @staticmethod
    def dif(candA:Dict[int, Set[int]], candB:Dict[int, Set[int]]):
        s = CandidateSet.candidate2set(candA) - CandidateSet.candidate2set(candB)
        return CandidateSet.set2candidate(s)

    @staticmethod
    def eval_one_candidate_with_GT(cand:Dict[int, Set[int]], joined_pairs:np.ndarray):
        TP, FP, FN = 0, 0, 0
        for docid, Ridxset in cand.items():
            actual_idxset = set(joined_pairs[joined_pairs[:, 0] == docid][:, 1].tolist())
            TP += len(Ridxset & actual_idxset)
            FP += len(Ridxset - actual_idxset)
            FN += len(actual_idxset - Ridxset)
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        return {
            'cmp_num': TP + FP, 
            'pos_num': TP,
            'all_pos_num': TP + FN,
            'recall': recall, 
            'precision': precision, 
        }
    
    @staticmethod
    def eval_one_candidate(cand_dict:Dict[int, Set[int]], llm_joined:np.ndarray):
        P, N = 0, 0
        for Lid, Rid_set in cand_dict.items():
            for Rid in Rid_set:
                if llm_joined[Lid, Rid] == 1:   P += 1
                elif llm_joined[Lid, Rid] == 0: N += 1
        num = P + N
        return {
            'cmp_num': num, 
            'pos_num': P, 
            'pos_ratio': P / num if num != 0 else 0 
        }

    def union_all_feature_cand(self):
        self.union_cand = dict()
        for feature_type in self.feature_type_list:
            cand:Dict[int, Set[int]] = getattr(self, f'{feature_type}_cand')
            for Lid, Ridset in cand.items():
                self.union_cand[Lid] = self.union_cand.get(Lid, set()) | Ridset
    
    def union_other(self, other:CandidateSet):
        for feature_type in self.feature_type_list:
            cand:Dict[int, Set[int]] = getattr(self, f'{feature_type}_cand')
            other_cand:Dict[int, Set[int]] = getattr(other, f'{feature_type}_cand')
        
            for Lid, Ridset in other_cand.items():
                cand[Lid] = cand.get(Lid, set()) | Ridset
            setattr(self, f'{feature_type}_cand', cand)
        
        self.union_all_feature_cand()
            

    def evaluate_with_GT(self, joined_pairs:np.ndarray):
        '''
            joined_pairs: [[Lid, Rid], ...], np.array([[int, int], ...]) # GroudTruth
        return 
            eval_dict = {
                'word': {
                    'cmp_num': TP + FP, 
                    'pos_num': TP,
                    'all_pos_num': TP + FN,
                    'recall': recall, 
                    'precision': precision, 
                }, 
            ...
        }
        '''
        # evaluate the candidate set with the joined ground truth
        # joined = joined_pairs
        eval_dict = dict()
        for feature_type in self.feature_type_list + ['union']:
            cand = getattr(self, f'{feature_type}_cand')
            # remove '_cand' 
            feature_type = feature_type.replace('_cand', '')
            d = CandidateSet.eval_one_candidate_with_GT(cand, joined_pairs)
            eval_dict[feature_type] = d
            print(f'{feature_type} : {d}')
        return eval_dict

    def evaluate(self, llm_joined:np.ndarray):
        '''
            {
                feature_type:{
                    'total': results_dict, 
                    'unique': results_dict
                }
            }
        '''
        if self.eval_dict is not None:
            return deepcopy(self.eval_dict)
        
        eval_results = dict()
        # union_num = sum(len(s) for s in self.union_cand.values()) 
        for feature_type in self.feature_type_list:
            cand = getattr(self, f'{feature_type}_cand')
            results = CandidateSet.eval_one_candidate(cand, llm_joined)
            # results['cost'] = results['cmp_num'] / union_num * self.total_matching_cost 
            unique_cand = cand 
            for other_feature_type in self.feature_type_list:
                if other_feature_type == feature_type: continue
                other_cand = getattr(self, f'{other_feature_type}_cand')
                unique_cand = CandidateSet.dif(unique_cand, other_cand)
            unique_results = CandidateSet.eval_one_candidate(unique_cand, llm_joined)

            # clean the suffix '_cand' in feature_type
            feature_type = feature_type.replace('_cand', '')
            eval_results[feature_type] = {
                'total': results, 
                'unique': unique_results
            }
        
        self.eval_dict = eval_results
        
        return eval_results



        