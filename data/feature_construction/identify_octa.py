import pandas as pd
import os
from pymatgen.analysis import local_env as le
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition 

def get_A_B_sites_from_oct(pymatgen_structure,formula, approach = 'min_dist', delta = 0.1, cutoff = 10.0):
    B_element = []
    X_element = []
    sites = []
    is_oct = []
    oxid = get_oxid_element_from_formula(formula) 
    oxid_judge = []
    if len(oxid) > 0 :
        
        oxid_judge=1

        for i in range(len(pymatgen_structure.sites)):
            cns, pt = le.site_is_of_motif_type(struct = pymatgen_structure, 
                                            n = i, 
                                            approach = approach,
                                            delta = delta,
                                            cutoff = cutoff
                                            )
            
            if pt == 'octahedral' :
                sites.append(i)
                a_ele = pymatgen_structure.sites[i].species_string

                if oxid[a_ele] >0 : 
                
                    B_element.append(a_ele)
                
                is_oct.append('yes')
                
                for j in range(6) : 
                    b_ele = cns[j].species.chemical_system
                    if oxid[b_ele] <0 : 
                        X_element.append(b_ele)
                    if len(X_element) == 0: 
                        X_element.append(b_ele)
                
                perovskite_judge = 'perovskite'


        if len(B_element)==0 :  
            for i in sites:
                cns, pt = le.site_is_of_motif_type(struct = pymatgen_structure, 
                                                n = i, 
                                                approach = approach,
                                                delta = delta,
                                                cutoff = cutoff
                                                )
                a_ele = pymatgen_structure.sites[i].species_string
                B_element.append(a_ele)      
                for j in range(6) : 
                    b_ele = cns[j].species.chemical_system
                    X_element.append(b_ele)

            perovskite_judge = 'anti-perovskite'


    else: 
        oxid_judge=0
        perovskite_judge = 'anti-perovskite'

        for i in range(len(pymatgen_structure.sites)):
            cns, pt = le.site_is_of_motif_type(struct = pymatgen_structure, 
                                            n = i, 
                                            approach = approach,
                                            delta = delta,
                                            cutoff = cutoff
                                            )
            if pt == 'octahedral' :
                sites.append(i)
                a_ele = pymatgen_structure.sites[i].species_string
                B_element.append(a_ele)
                
                is_oct.append('yes')
                
                for j in range(6) : 
                    b_ele = cns[j].species.chemical_system
                    X_element.append(b_ele)
  

    if 'yes' not in is_oct :     
        B_element = ['No']
        X_element = ['No']
        sites = 0
        B_element_num=0
        X_element_num=0
        perovskite_judge = 'non'  

    B_element = list(set(B_element)) 
    X_element = list(set(X_element))
    B_element_num = len(list(set(B_element))) 
    X_element_num = len(list(set(X_element)))

    return B_element,X_element,sites,B_element_num,X_element_num,oxid_judge,perovskite_judge

def get_oxid_element_from_formula(formula) :

    oxid = Composition(formula).oxi_state_guesses() 
    if len(oxid)!=0:
        return oxid[0] 
    else:
        return oxid


if __name__ == "__main__":
    
    data = pd.read_csv('../dataset.csv')
    print(data)

    path = '../structure_files/'
    files = os.listdir(path)

    filename = data['file_name']


    # get octahedral B and X site from structure
    B_element_list = []
    X_element_list = []
    sites_list = []
    B_element_num_list = []
    X_element_num_list = []
    f = []
    oxid_judge_list,perovskite_judge_list = [],[]

    for i in range(len(data)):
        print(i,data['file_name'][i])
        file = os.path.join(path,filename[i])
        B_element,X_element,sites,B_element_num,X_element_num,oxid_judge,perovskite_judge = get_A_B_sites_from_oct(
                                            pymatgen_structure=Structure.from_file(file),
                                            formula = data['formula'][i],
                                            delta = 0.1,
                                            )
        if B_element == ['No'] : 
            B_element,X_element,sites,B_element_num,X_element_num,oxid_judge,perovskite_judge = get_A_B_sites_from_oct(
                                            pymatgen_structure=Structure.from_file(file),
                                            formula = data['formula'][i],
                                            delta = 0.2,
                                            )

        


        print(B_element)
        print(X_element)

        B_element_list.append(B_element)
        X_element_list.append(X_element)
        sites_list.append(sites)
        B_element_num_list.append(B_element_num)
        X_element_num_list.append(X_element_num)
        oxid_judge_list.append(oxid_judge)
        perovskite_judge_list.append(perovskite_judge)


    data['B_element'] = B_element_list
    data['X_element'] = X_element_list
    data['B_element_num'] = B_element_num_list
    data['X_element_num'] = X_element_num_list
    data['sites'] = sites_list  
    data['oxid_judge'] = oxid_judge_list  
    data['perovskite_judge'] = perovskite_judge_list  
    print(data) 
    data.to_csv('dataset_octa.csv',index=False)