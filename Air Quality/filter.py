import numpy as np 
from copy import deepcopy 

class Filter():
    def __init__(self, prem_terms, complete_rules, rules):
        self.prem_terms = prem_terms
        self.complete_rules = complete_rules
        self.rules = rules 
        
    def run(self,s = 0.5, detailed_results = False):
        return self.filter_rules(s = s, dr = detailed_results)



    def filter_rules(self, s = 0.5, dr = False):
        list_keep = []

        dict_val = {}

        for i in range(self.prem_terms.shape[0]):
            except_one = deepcopy(self.prem_terms)
            v = except_one[i,:]
            idx = np.argwhere(v > 0.5).ravel()
            v = v[idx]
            rest = np.delete(except_one, i, axis=0)
            rest = rest[:,idx]
            cpare = np.tile(v,(rest.shape[0],1))
            m = np.minimum(rest,cpare) 
            M = np.maximum(rest,cpare) + 10e-15
            res = m/M
            mean = np.mean(res,axis=1)
            
            vv = np.argwhere(mean > s).ravel()

            if vv.shape[0] > 0:    
                vv[vv > i] += 1
                vv2 = np.append(vv,np.array([i]))

                eval_v = self.prem_terms[vv2][:,idx]

                t = np.mean(eval_v,axis=1)

                keep_val = vv2[np.argmax(t)]
                
                if keep_val not in list_keep:
                    dict_val[keep_val] = 1
                    list_keep.append(keep_val)
                else:
                    dict_val[keep_val] = dict_val[keep_val] + 1

        if len(list_keep) == 0:
            return self.complete_rules, self.prem_terms

        filtered_rules = deepcopy(self.complete_rules[list_keep,:])
        filtered_prems = deepcopy(self.prem_terms[list_keep,:])
        filtered_antecedents = deepcopy(self.rules[list_keep,:])

        if dr:
            return filtered_rules, filtered_prems, filtered_antecedents, dict_val
        else:
            return filtered_rules, filtered_prems, filtered_antecedents
