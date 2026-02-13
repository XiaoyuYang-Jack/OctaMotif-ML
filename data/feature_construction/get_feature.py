import pandas as pd
import os
from pymatgen.analysis import local_env as le
from pymatgen.analysis import structure_analyzer as sa
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN
from matminer.featurizers.site.chemical import SiteElementalProperty
from matminer.utils.data import MagpieData
import re
from pymatgen.core.composition import Composition 
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.core.periodic_table import Element,Species
from matminer.featurizers.site import chemical

def get_oxid_element_from_formula(formula) :

    oxid = Composition(formula).oxi_state_guesses()
    if len(oxid)!=0:
        return oxid[0] # 随便选一个价态
    else:
        return oxid

def get_pymatgen_element_from_str(ele_str):
    ele = ''
    for j in re.findall('[A-Z][a-z]?',ele_str) : 
        ele = ele + j
    return Composition(ele).elements

def get_pymatgen_composation_from_str(ele_str):
    ele = ''
    for j in re.findall('[A-Z][a-z]?',ele_str) : 
        ele = ele + j
    return Composition(ele)

def get_site_from_str(site_str):
    site = []
    for i in re.findall('[0-9][0-9]?',site_str) : 
        site.append(i)
    return site

magpie_elementfeatures = [
    "Number",
    "MendeleevNumber",
    "AtomicWeight",
    "MeltingT",
    "Column",
    "Row",
    "CovalentRadius",
    "Electronegativity",
    "NsValence",
    "NpValence",
    "NdValence",
    # "NfValence",
    "NValence",
    "NsUnfilled",
    "NpUnfilled",
    "NdUnfilled",
    # "NfUnfilled",
    "NUnfilled",
    "GSvolume_pa",
    "GSbandgap",
    "GSmagmom",
    # "SpaceGroupNumber",
    ]

pymatgen_elementfeatures = [
    # "X",
    # "row",
    # "group",
    # "block",
    "atomic_mass",
    "atomic_radius",
    # "mendeleev_no",
    "electrical_resistivity",
    "velocity_of_sound",
    "thermal_conductivity",
    # "melting_point",
    "bulk_modulus",
    "coefficient_of_linear_thermal_expansion",
    ]
        
deml_elementfeatures = [
    # "atom_num",
    # "atom_mass",
    # "row_num",
    # "col_num",
    # "atom_radius",
    # "molar_vol",
    "heat_fusion",
    "melting_point",
    "boiling_point",
    "heat_cap",
    "first_ioniz",
    "electronegativity",
    "electric_pol",
    # "GGAU_Etot",
    # "mus_fere",
    # "FERE correction",
    ]


class oct_element_feature():
    
    def __init__(self):
        pass

    def matminer_ElementProperty(self,data,column_composition,feature=magpie_elementfeatures,source='magpie',stats='mean', feature_name='name',):

        from matminer.featurizers.composition import ElementProperty 
        
        ele_property  = ElementProperty(
            data_source=source,
            features = feature,
            stats = stats,
            name=feature_name, 
            )
        
        data = ele_property.featurize_dataframe(df=data,col_id=column_composition)
        return data



    def mendeleev_element_featurize(self,element_name):
        from mendeleev import element
        ele = element(element_name)
        # atomic_number = ele.atomic_number
        # atomic_radius = ele.atomic_radius
        # block = ele.block
        density = ele.density
        electron_affinity = ele.electron_affinity
        # econf = ele.econf
        # covalent_radius = ele.covalent_radius

        electrons = ele.electrons
        mass = ele.mass
        # nvalence = ele.nvalence()
        electrophilicity = ele.electrophilicity()
        vdw_radius = ele.vdw_radius
        # oxistates = ele.oxistates 
        # ionenergies = ele.ionenergies 
        proton_affinity = ele.proton_affinity 
        
        # from mendeleev.fetch import fetch_ionization_energies
        # fetch_ionization_energies(degree=1)
        
        return  density ,electron_affinity,\
                electrons,mass,electrophilicity,\
                vdw_radius,proton_affinity,#ionenergies,econf,block,
    

    def pyamtgen_element_feature(self,element) : 
        from pymatgen.core.periodic_table import Element
        B_element = Element(element)
        # atomic_mass = str(B_element.atomic_mass).split(' ')[0]
        # ionic_radii = float(str(B_element.ionic_radii))
        # Z = str(B_element.Z)
        # print(B_element.electrical_resistivity)
        electrical_resistivity = B_element.electrical_resistivity

        # van_der_waals_radius = str(B_element.van_der_waals_radius)
        # reflectivity = B_element.reflectivity
        poissons_ratio = (B_element.poissons_ratio)
        # molar_volume = str(B_element.molar_volume).split(' ')[0]
        # electronic_structure = B_element.electronic_structure
        thermal_conductivity = (B_element.thermal_conductivity)
        boiling_point       = (B_element.boiling_point)
        melting_point       = (B_element.melting_point)
        density_of_solid    = (B_element.density_of_solid)
        # ionization_energies = B_element.ionization_energies

        element_features = []
        # element_features.append(ionic_radii)
        element_features.append(electrical_resistivity)
        element_features.append(poissons_ratio)
        element_features.append(thermal_conductivity)
        element_features.append(boiling_point)
        element_features.append(melting_point)
        element_features.append(density_of_solid)

        return element_features



class oct_site_feature():
    def __init__(self) :
        self.filename = filename
        self.B_element = B_element_list
        self.X_element = X_element_list
        self.sites = sites
        
    def oxid_structure(self,data):
        self.pymatgen_structure = []
        for i in range(len(data)) : 
            vaspfile = os.path.join(path,self.filename[i])
            self.pymatgen_structure.append(Structure.from_file(vaspfile))
        data['pymatgen_structure'] = self.pymatgen_structure

        from matminer.featurizers.conversions import StructureToOxidStructure
        data = StructureToOxidStructure().featurize_dataframe(data,'pymatgen_structure',ignore_errors=True)
        
        return data


    def __matminer_bond_length_angle(self,dataset_column,site,pymatgen_method):
        from matminer.featurizers.site.bonding import AverageBondAngle,AverageBondLength
        Average_bond_angle = AverageBondAngle(method=pymatgen_method).featurize(strc=dataset_column,idx=site)
        Average_bond_length = AverageBondLength(method=pymatgen_method).featurize(strc=dataset_column,idx=site)
        return Average_bond_angle,Average_bond_length

    def bond_length_angle(self,data) : 
        site_angle_mean = []
        site_bond_mean = []
        for i in range(len(data)) : 
            site_angle_list = []
            site_bond_list = []
            for j in get_site_from_str(data['sites'][i]) :
                j = int(j)
                site_angle,site_bond = self.__matminer_bond_length_angle(dataset_column=data['structure_oxid'][i],
                                                                         site=j,
                                                                         pymatgen_method=MinimumDistanceNN(),
                                                                         )
                site_angle_list.append(site_angle)
                site_bond_list.append(site_bond)
            site_angle_mean.append(np.mean(site_angle_list))
            site_bond_mean.append(np.mean(site_bond_list))
        data['oct_mean_angle'] = site_angle_mean
        data['oct_mean_bond'] = site_bond_mean
        
        return data
            
    def site_EwaldSiteEnergy(self,data):
        from matminer.featurizers.site.chemical import EwaldSiteEnergy
        '''Compute site energy from Coulombic interactions'''
        print(EwaldSiteEnergy(accuracy=3).feature_labels())
        site_Energy_list = []

        for i in range(len(data)) : 
            site_Energy = []
            for j in get_site_from_str(data['sites'][i]) :
                j = int(j)
                site_Energy.append(EwaldSiteEnergy(accuracy=3).featurize(strc=data['structure_oxid'][i], idx=j))

            site_Energy_list.append(np.mean(site_Energy))
        data['site_EwaldSiteEnergy'] = site_Energy_list
        return data
    

    def site_EwaldSiteEnergy_allsites(self,data):
        from matminer.featurizers.site.chemical import EwaldSiteEnergy
        '''Compute site energy from Coulombic interactions'''
        print(EwaldSiteEnergy(accuracy=3).feature_labels())
        site_Energy_list = []

        for i in range(len(data)) : 
            site_Energy = []
            for j in  range(int(str(data['atom_num'][i]))):
                j = int(j)
                site_Energy.append(EwaldSiteEnergy(accuracy=3).featurize(strc=data['structure_oxid'][i], idx=j))

            site_Energy_list.append(np.mean(site_Energy))
        data['site_EwaldSiteEnergy_allsites'] = site_Energy_list
        return data

    def covalent_radius_ratio(self,data):
        coval_r_B_coval_r_X = []
        for i in range(len(data)) : 
            coval_r_B_coval_r_X.append(data['MagpieData mean CovalentRadius _B'][i]/data['MagpieData mean CovalentRadius _X'][i])
        data['coval_r_B_coval_r_X'] = coval_r_B_coval_r_X

        return data


    def matmanier_LocalPropertyDifference(self,dataset):
        '''local difference in Electronegativity'''
        print(chemical.LocalPropertyDifference().feature_labels())
        Electronegativity,CovalentRadius,NValence,GSvolume_pa,NdValence = [],[],[],[],[]

        for i in range(len(dataset)):
            print(dataset['file_name'][i])
            site_int = []
            for s in get_site_from_str(dataset['sites'][i]):
                site_int.append(int(s))
            feature,feature1,feature2,feature3,feature4 = [],[],[],[],[]

            for j in site_int:

                feature.append(chemical.LocalPropertyDifference(properties=['Electronegativity']).featurize(strc=dataset['pymatgen_structure'][i], idx=j))
                feature1.append(chemical.LocalPropertyDifference(properties=['CovalentRadius']).featurize(strc=dataset['pymatgen_structure'][i], idx=j))
                feature2.append(chemical.LocalPropertyDifference(properties=['NValence']).featurize(strc=dataset['pymatgen_structure'][i], idx=j))
                feature3.append(chemical.LocalPropertyDifference(properties=['GSvolume_pa']).featurize(strc=dataset['pymatgen_structure'][i], idx=j))
                feature4.append(chemical.LocalPropertyDifference(properties=['NdValence']).featurize(strc=dataset['pymatgen_structure'][i], idx=j))

            Electronegativity.append(np.mean(feature))
            CovalentRadius.append(np.mean(feature1))
            NValence.append(np.mean(feature2))
            GSvolume_pa.append(np.mean(feature3))
            NdValence.append(np.mean(feature4))

        dataset['local_difference_in_Electronegativity'] = Electronegativity
        dataset['local_difference_in_CovalentRadius'] = CovalentRadius
        dataset['local_difference_in_NValence'] = NValence
        dataset['local_difference_in_GSvolume_pa'] = GSvolume_pa
        dataset['local_difference_in_NdValence'] = NdValence

        return dataset

    def matmanier_LocalPropertyDifference_allsites(self,dataset):
        print(chemical.LocalPropertyDifference().feature_labels())
        Electronegativity,CovalentRadius,NdValence,GSbandgap = [],[],[],[]

        for i in range(len(dataset)):
            print(dataset['file_name'][i])
            
            feature,feature1,feature2,feature3 = [],[],[],[]

            for j in range(int(str(dataset['atom_num'][i]))):

                feature.append(chemical.LocalPropertyDifference(properties=['Electronegativity']).featurize(strc=dataset['pymatgen_structure'][i], idx=j))
                feature1.append(chemical.LocalPropertyDifference(properties=['CovalentRadius']).featurize(strc=dataset['pymatgen_structure'][i], idx=j))
                feature2.append(chemical.LocalPropertyDifference(properties=['NdValence']).featurize(strc=dataset['pymatgen_structure'][i], idx=j))
                feature3.append(chemical.LocalPropertyDifference(properties=['GSbandgap']).featurize(strc=dataset['pymatgen_structure'][i], idx=j))

            Electronegativity.append(np.mean(feature))
            CovalentRadius.append(np.mean(feature1))
            NdValence.append(np.mean(feature2))
            GSbandgap.append(np.mean(feature3))

        dataset['local_difference_in_Electronegativity_allsites'] = Electronegativity
        dataset['local_difference_in_CovalentRadius_allsites'] = CovalentRadius
        dataset['local_difference_in_NdValence_allsites'] = NdValence
        dataset['local_difference_in_GSbandgap_allsites'] = GSbandgap

        return dataset

    def ionic_radius_ratio(self,data):
        B_ionic_radius_list = []
        X_ionic_radius_list = []
        r_B_r_X = []
        for i in range(len(data)) : 
            B_ionic_radius = []
            X_ionic_radius = []
            
            for bele in get_pymatgen_element_from_str(self.B_element[i]) : 
                if (float((str(bele.average_cationic_radius).split(' ')[0]))) < 0.001:
                    B_ionic_radius.append(float((str(bele.average_anionic_radius).split(' ')[0])))
                else :
                    B_ionic_radius.append(float((str(bele.average_cationic_radius).split(' ')[0])))
            for xele in get_pymatgen_element_from_str(self.X_element[i]) : 
                if float((str(xele.average_anionic_radius).split(' ')[0])) < 0.001 : 
                    X_ionic_radius.append(float((str(xele.average_cationic_radius).split(' ')[0]))) 
                else :
                    X_ionic_radius.append(float((str(xele.average_anionic_radius).split(' ')[0])))
            
            B_ionic_radius_list.append(np.mean(B_ionic_radius))
            X_ionic_radius_list.append(np.mean(X_ionic_radius))
            r_B_r_X.append(np.mean(B_ionic_radius)/np.mean(X_ionic_radius))


        data['B_ionic_radius'] = B_ionic_radius_list
        data['X_ionic_radius'] = X_ionic_radius_list
        data['r_B_r_X'] = r_B_r_X
        
        return data
        
    def pymatgen_shannon_rad(self,data):
        B_shannon_rad_list = []       
        for i in range(len(data)) : 

            B_shannon_list = []
            for j in get_site_from_str(sites[i]) : 
                j = int(j)
                element = data['pymatgen_structure'][i][j].species_string

                oxid_element = data['structure_oxid'][i][j].species_string

                oxid = re.findall(r'\d',oxid_element)                
                if oxid == [] :
                    if '+' in oxid_element : 
                        oxid = ['1']
                    elif '-' in oxid_element : 
                        oxid = ['-1']
                
                try : 
                    B_shannon_list.append(Species(symbol=element,oxidation_state=float(oxid[0])).get_shannon_radius(cn='VI',spin=''))
                except : 
                    B_shannon_list.append(-1)
                        
            B_shannon_rad = np.mean(np.array(B_shannon_list))
            if -1 in B_shannon_list : 
                B_shannon_rad = 'No'
            B_shannon_rad_list.append(B_shannon_rad)  
        
        data['B_shannon_rad'] = B_shannon_rad_list

        return data
    

    def get_shannon_rad_from_csv(self,data):

        B_shannon_rad_list = []
        X_shannon_rad_list = []
        
        shannon_rad_csv = pd.read_csv('./shannon_rad_csv.csv')

        for i in range(len(data)) : 
            oxid = get_oxid_element_from_formula(data['formula'][i])
            
            B_shannon_list,X_shannon_list=[],[]

            for e in re.findall('[A-Z][a-z]?',B_element_list[i]) : 
                if oxid[e]>0:
                    ion = e+'+'+str(oxid[e]).split('.')[0]
                else:
                    ion = e+str(oxid[e]).split('.')[0]
                try:
                    B_shannon_csv = shannon_rad_csv.loc[shannon_rad_csv[shannon_rad_csv['ION'] == ion].index]
                    B_shannon = B_shannon_csv.loc[B_shannon_csv[B_shannon_csv['Coord'] == 6].index,'Ionic_Radius'].values[0]
                except: 
                    B_shannon = shannon_rad_csv.loc[shannon_rad_csv[shannon_rad_csv['ele'] == e].index,'Ionic_Radius']
                    B_shannon = np.mean(B_shannon)

                B_shannon_list.append(B_shannon)

            B_shannon_rad_list.append(np.mean(B_shannon_list))

            for e in re.findall('[A-Z][a-z]?',X_element_list[i]) : 
                if oxid[e]>0:
                    ion = e+'+'+str(oxid[e]).split('.')[0]
                else:
                    ion = e+str(oxid[e]).split('.')[0]

                X_shannon = shannon_rad_csv.loc[shannon_rad_csv[shannon_rad_csv['ION'] == ion].index,'Ionic_Radius']
                
                if np.isnan(np.mean(X_shannon)): 
                    X_shannon = shannon_rad_csv.loc[shannon_rad_csv[shannon_rad_csv['ele'] == e].index,'Ionic_Radius']
                
                X_shannon_list.append(np.mean(X_shannon))

            X_shannon_rad_list.append(np.mean(X_shannon_list))

        data['B_shannon_rad'] = B_shannon_rad_list
        data['X_shannon_rad'] = X_shannon_rad_list

        return data
    
    def pymatgen_bond_length_angle(self,data) :
        
        mean_delta_d_list = []
        mean_oct_g_list = []
        max_bondlength_list = []
        min_bondlength_list = []
        mean_max_oct_angle_list = []
        mean_min_oct_angle_list = []
        
        for i in range(len(data)) : 
            delta_d = []
            oct_g = []
            max_bondlength = []
            min_bondlength = []
            oct_angle_min_mean = []
            oct_angle_max_mean = []
            
            for j in get_site_from_str(sites[i]) :
                j = int(j)
                B_coords = data['structure_oxid'][i][j].coords 
                
                nn_info = le.MinimumDistanceNN(tol=0.1, cutoff=10.0).get_nn_info(data['structure_oxid'][i], j)
                
                # bond length
                oct_bondlenth = np.linalg.norm(
                            np.subtract([site["site"].coords for site in nn_info], B_coords), axis=1) 
                
                max_bondlength.append(max(oct_bondlenth))
                min_bondlength.append(min(oct_bondlenth))
                
                mean_oct_bondlenth = np.mean(oct_bondlenth)
                d = []
                for y in oct_bondlenth : 
                    d.append(((mean_oct_bondlenth - y)/y)**2)
                delta_d.append(np.mean(np.array(d)))
                
                oct_g.append((max(oct_bondlenth)-min(oct_bondlenth))/(max(oct_bondlenth)+min(oct_bondlenth)))
                
                
                # bond angle
                B_sites = [i["site"].coords for i in nn_info]
                # Calculate bond angles for each neighbor
                bond_angles = np.empty((len(B_sites), len(B_sites)))
                bond_angles.fill(np.nan)
                
                for a, a_site in enumerate(B_sites):
                    for b, b_site in enumerate(B_sites):
                        if b == a:
                            continue
                        dot = np.dot(a_site - B_coords, b_site - B_coords) / (
                            np.linalg.norm(a_site - B_coords) * np.linalg.norm(b_site - B_coords)
                        )
                        if np.isnan(np.arccos(dot)):
                            bond_angles[a, b] = bond_angles[b, a] = np.arccos(round(dot, 5))
                        else:
                            bond_angles[a, b] = bond_angles[b, a] = np.arccos(dot)
                # Take the minimum bond angle of each neighbor

                oct_angle_min = np.nanmin(bond_angles, axis=1)
                oct_angle_max = np.nanmax(bond_angles, axis=1)
                
                oct_angle_min_mean.append(np.mean(oct_angle_min))
                oct_angle_max_mean.append(np.mean(oct_angle_max))
        
            mean_delta_d = np.mean(delta_d)*1e4
            mean_oct_g = np.mean(oct_g)*1e2

            mean_max_oct_angle = np.mean(oct_angle_max_mean)
            mean_min_oct_angle = np.mean(oct_angle_min_mean)
            
            mean_delta_d_list       .append(mean_delta_d)
            mean_oct_g_list         .append(mean_oct_g)
            max_bondlength_list     .append(np.mean(max_bondlength))
            min_bondlength_list     .append(np.mean(min_bondlength))
            mean_max_oct_angle_list .append(mean_max_oct_angle)
            mean_min_oct_angle_list .append(mean_min_oct_angle) 
        
        
        data['mean_delta_d']        = mean_delta_d_list       
        data['mean_oct_g']          = mean_oct_g_list         
        data['max_bondlength']      = max_bondlength_list     
        data['min_bondlength']      = min_bondlength_list     
        data['mean_max_oct_angle']  = mean_max_oct_angle_list 
        data['mean_min_oct_angle']  = mean_min_oct_angle_list    
        
        return data



class oct_structure_feature():
    def __init__(self) :
        pass
    

    def matminer_DensityFeatures(self,dataset,column_structure):
        '''density、vpa、packing fraction'''
        from matminer.featurizers.structure import DensityFeatures
        print(DensityFeatures().feature_labels())
        data = DensityFeatures().featurize_dataframe(df=dataset,col_id=column_structure)
        return data


    def matminer_StructuralHeterogeneity(self,dataset,column_structure):
        '''Variance in the bond lengths and atomic volumes in a structure'''
        from matminer.featurizers.structure.bonding import StructuralHeterogeneity
        print(StructuralHeterogeneity().feature_labels())
        data = StructuralHeterogeneity().featurize_dataframe(df=dataset,col_id=column_structure,ignore_errors=True,return_errors=True)
        data = data.drop(columns='StructuralHeterogeneity Exceptions')
        return data 


    def matminer_StructuralComplexity(self,dataset,column_structure):
        '''Shannon information entropy of a structure.'''
        from matminer.featurizers.structure import StructuralComplexity
        print(StructuralComplexity().feature_labels())
        data = StructuralComplexity().featurize_dataframe(df=dataset,col_id=column_structure)
        return data


    def matminer_ChemicalOrdering(self,dataset,column_structure):
        '''How much the ordering of species in the structure differs from random'''
        from matminer.featurizers.structure import ChemicalOrdering
        print(ChemicalOrdering().feature_labels())
        data = ChemicalOrdering().featurize_dataframe(df=dataset,col_id=column_structure,ignore_errors=True,return_errors=True)
        data = data.drop(columns='ChemicalOrdering Exceptions')
        return data


    def matminer_MaximumPackingEfficiency(self,dataset,column_structure):
        '''Maximum possible packing efficiency'''
        from matminer.featurizers.structure import MaximumPackingEfficiency
        print(MaximumPackingEfficiency().feature_labels())
        data = MaximumPackingEfficiency().featurize_dataframe(df=dataset,col_id=column_structure,ignore_errors=True,return_errors=True)
        data = data.drop(columns='MaximumPackingEfficiency Exceptions')
        return data


    def matminer_GlobalSymmetryFeatures_Dimensionality(self,dataset,column_structure):
        '''结构对称性与维度'''
        from matminer.featurizers.structure import GlobalSymmetryFeatures,Dimensionality
        print(GlobalSymmetryFeatures().feature_labels())
        print(Dimensionality().feature_labels())
        data = GlobalSymmetryFeatures().featurize_dataframe(df=dataset,col_id=column_structure)
        data = Dimensionality().featurize_dataframe(df=dataset,col_id=column_structure)
        return data


    def matminer_EwaldEnergy(self,dataset,column_structure):
        '''Compute the energy from Coulombic interactions.'''
        from matminer.featurizers.structure import EwaldEnergy
        print(EwaldEnergy().feature_labels())
        data = EwaldEnergy().featurize_dataframe(df=dataset,col_id=column_structure)
        return data


    def matminer_SiteStatsFingerprint(self,dataset,column_structure):
        '''Computes statistics of properties across all sites in a structure.'''
        from matminer.featurizers.site.bonding import AverageBondLength,AverageBondAngle
        from matminer.featurizers.structure import SiteStatsFingerprint
        
        print(SiteStatsFingerprint(site_featurizer=AverageBondLength(method=MinimumDistanceNN())).feature_labels())
        data = SiteStatsFingerprint(site_featurizer=AverageBondLength(method=MinimumDistanceNN())).featurize_dataframe(df=dataset,col_id=column_structure)
        
        print(SiteStatsFingerprint(site_featurizer=AverageBondAngle(method=MinimumDistanceNN())).feature_labels())
        data = SiteStatsFingerprint(site_featurizer=AverageBondAngle(method=MinimumDistanceNN())).featurize_dataframe(df=data,col_id=column_structure)
        
        from matminer.featurizers.site.chemical import EwaldSiteEnergy
        print(SiteStatsFingerprint(site_featurizer=EwaldSiteEnergy(accuracy=3)).feature_labels())
        data = SiteStatsFingerprint(site_featurizer=EwaldSiteEnergy(accuracy=3),
                                    stats = ["mean", "std_dev"],
                                    ).featurize_dataframe(df=data,col_id=column_structure)
        return data


    # Unused:
    def matminer_GlobalSymmetryFeatures(self,dataset,column_structure):
        '''Determines symmetry features, e.g. spacegroup number and crystal system'''
        from matminer.featurizers.structure import GlobalSymmetryFeatures
        print(GlobalSymmetryFeatures().feature_labels())
        data = GlobalSymmetryFeatures().fit(dataset[column_structure]).featurize_dataframe(df=dataset,col_id=column_structure)
        return data

    def matminer_Dimensionality(self,dataset,column_structure):
        '''Returns dimensionality of structure: 1 means linear chains of atoms OR'''
        from matminer.featurizers.structure import Dimensionality
        data = Dimensionality().featurize_dataframe(df=dataset,col_id=column_structure)
        return data

    def matminer_bonding_Features(self,dataset,column_oxid_structure):
        '''The global instability index of a structure. 空值太多,未使用'''
        from matminer.featurizers.structure import bonding
        print(bonding.GlobalInstabilityIndex().fit(dataset[column_oxid_structure]).feature_labels())
        dataset = bonding.GlobalInstabilityIndex().fit(dataset[column_oxid_structure]).featurize_dataframe(df=dataset,col_id=column_oxid_structure,ignore_errors=True) #pymatgen structure
        
        '''Variance in the bond lengths and atomic volumes in a structure'''
        print(bonding.StructuralHeterogeneity().fit(dataset[column_oxid_structure]).feature_labels())
        dataset = bonding.StructuralHeterogeneity().fit(dataset[column_oxid_structure]).featurize_dataframe(df=dataset,col_id=column_oxid_structure,ignore_errors=True)
        return dataset


def construct_element_feature_matminer(data) : 

    B_element_list_,X_element_list_,formula_composition = [0] * len(data), [0] * len(data),[0] * len(data)
    for i in range(len(data)) :  
        B_element_list_[i] = get_pymatgen_composation_from_str(B_element_list[i])
        X_element_list_[i] = get_pymatgen_composation_from_str(X_element_list[i])
        formula_composition[i] = Composition(data['formula'][i])
    data['B_element_'] = B_element_list_
    data['X_element_'] = X_element_list_
    data['formula_composition'] = formula_composition

    # 1. the features based on formula
    # matminer-pymatgen
    data = oct_element_feature().matminer_ElementProperty(data,column_composition='formula_composition',
                                                            feature=pymatgen_elementfeatures,source='pymatgen',stats=["minimum", "maximum", "range", "mean", "std_dev"],feature_name='formula')
    # matminer-deml
    data = oct_element_feature().matminer_ElementProperty(data,column_composition='formula_composition',
                                                            feature=deml_elementfeatures,source='deml',stats=["minimum", "maximum", "range", "mean", "std_dev"],feature_name='formula')
    # matminer-magpie 
    data = oct_element_feature().matminer_ElementProperty(data,column_composition='formula_composition',stats = ["minimum", "maximum", "range", "mean", "avg_dev", "mode"],feature_name='formula')
    
    # 2. the features based on octahedral B and X element
    data = oct_element_feature().matminer_ElementProperty(data,column_composition='B_element_',stats = ["minimum", "maximum", "range", "mean", "avg_dev", "mode"],feature_name='_B')
    data = oct_element_feature().matminer_ElementProperty(data,column_composition='X_element_',stats = ["minimum", "maximum", "range", "mean", "avg_dev", "mode"],feature_name='_X')

    data = oct_element_feature().matminer_ElementProperty(
                                                        data,
                                                        column_composition='B_element_',
                                                        feature=pymatgen_elementfeatures,
                                                        source='pymatgen',
                                                        stats=["minimum", "maximum", "range", "mean", "std_dev"],  
                                                        feature_name='_B',
                                                        )
    data = oct_element_feature().matminer_ElementProperty(
                                                        data,
                                                        column_composition='X_element_',
                                                        feature=pymatgen_elementfeatures,
                                                        source='pymatgen',
                                                        stats=["minimum", "maximum", "range", "mean", "std_dev"],  
                                                        feature_name='_X',
                                                        )
    data = oct_element_feature().matminer_ElementProperty(
                                                        data,
                                                        column_composition='B_element_',
                                                        feature=deml_elementfeatures,
                                                        source='deml',
                                                        stats=["minimum", "maximum", "range", "mean", "std_dev"],  
                                                        feature_name='_B',
                                                        )
    data = oct_element_feature().matminer_ElementProperty(
                                                        data,
                                                        column_composition='X_element_',
                                                        feature=deml_elementfeatures,
                                                        source='deml',
                                                        stats=["minimum", "maximum", "range", "mean", "std_dev"], 
                                                        feature_name='_X',
                                                        )
    
    data= data.drop(columns=['B_element_','X_element_',])

    return data


def construct_element_feature_mendeleev(data):
    feature_B_sum = []
    feature_X_sum = []
    for i in range(len(data)) : 
        feature_B = []
        feature_X = []
        aelement = get_pymatgen_element_from_str(B_element_list[i])
        belement = get_pymatgen_element_from_str(X_element_list[i])

        for i in range(len(aelement)) : 
            list_feature = list(oct_element_feature().mendeleev_element_featurize(str(aelement[i])))
            for x in range(len(list_feature)) : 
                if list_feature[x] is None :
                    list_feature[x]=0.0
            feature_B.append(list_feature)

        feature_B_sum.append(np.array(feature_B).sum(axis=0)) 

        for j in range(len(belement)) : 
            list_feature = list(oct_element_feature().mendeleev_element_featurize(str(belement[j])))
            for x in range(len(list_feature)) : 
                if list_feature[x] is None :
                    list_feature[x]=0.0
            feature_X.append(list_feature)
            
        feature_X_sum.append(np.array(feature_X).sum(axis=0))

    data[oct_element_feature().mendeleev_B_elementfeatures] = feature_B_sum
    data[oct_element_feature().mendeleev_X_elementfeatures] = feature_X_sum
    print(data)

    return data


def construct_element_feature_pymatgen(data):
    feature_B_sum = []
    feature_X_sum = []
    for i in range(len(data)) : 
        # print(filename[i])
        feature_B = []
        feature_X = []
        aelement = get_pymatgen_element_from_str(B_element_list[i])
        belement = get_pymatgen_element_from_str(X_element_list[i])

        for i in range(len(aelement)) : 
            list_feature = list(oct_element_feature().pyamtgen_element_feature(str(aelement[i])))
            for x in range(len(list_feature)) : 
                if list_feature[x] is None :
                    list_feature[x]='0.0 '
                list_feature[x] = float(str(list_feature[x]).split(' ')[0])
            feature_B.append(list_feature)
            
        feature_B_sum.append(np.array(feature_B).sum(axis=0)) 

        for j in range(len(belement)) : 
            list_feature = list(oct_element_feature().pyamtgen_element_feature(str(belement[j])))
            for x in range(len(list_feature)) : 
                if list_feature[x] is None :
                    list_feature[x]='0.0 '
                list_feature[x] = float(str(list_feature[x]).split(' ')[0])
            feature_X.append(list_feature)
        feature_X_sum.append(np.array(feature_X).sum(axis=0))

    data[oct_element_feature().pymatgen_B_elementfeatures] = feature_B_sum
    data[oct_element_feature().pymatgen_X_elementfeatures] = feature_X_sum
    print(data)


    return data
    

def local_feature_construct(data) : 
    data = oct_site_feature().oxid_structure(data)
    
    data = oct_site_feature().bond_length_angle(data)
    data = oct_site_feature().covalent_radius_ratio(data)
    data = oct_site_feature().ionic_radius_ratio(data)
    # data = oct_site_feature().pymatgen_shannon_rad(data)
    data = oct_site_feature().get_shannon_rad_from_csv(data)
    data = oct_site_feature().pymatgen_bond_length_angle(data)
    data = oct_site_feature().site_EwaldSiteEnergy(data)
    data = oct_site_feature().matmanier_LocalPropertyDifference(data)
    data = data.drop(columns=['pymatgen_structure','structure_oxid'])
    return data 
    

def structure_feature_construct(data):
    pymatgen_structure = []
    for i in range(len(data)) : 
        vaspfile = os.path.join(path,filename[i])
        pymatgen_structure.append(Structure.from_file(vaspfile))
    data['pymatgen_structure'] = pymatgen_structure

    from matminer.featurizers.conversions import StructureToOxidStructure
    data = StructureToOxidStructure().featurize_dataframe(data,'pymatgen_structure',ignore_errors=True)

    data = oct_structure_feature().matminer_DensityFeatures(dataset=data,column_structure='pymatgen_structure')
    data = oct_structure_feature().matminer_StructuralHeterogeneity(dataset=data,column_structure='pymatgen_structure')
    data = oct_structure_feature().matminer_StructuralComplexity(dataset=data,column_structure='pymatgen_structure')
    data = oct_structure_feature().matminer_ChemicalOrdering(dataset=data,column_structure='pymatgen_structure')
    data = oct_structure_feature().matminer_MaximumPackingEfficiency(dataset=data,column_structure='pymatgen_structure')
    # data = oct_structure_feature().matminer_GlobalSymmetryFeatures_Dimensionality(dataset=data,column_structure='structure_oxid')
    data = oct_structure_feature().matminer_EwaldEnergy(dataset=data,column_structure='structure_oxid')
    data = oct_structure_feature().matminer_SiteStatsFingerprint(dataset=data,column_structure='structure_oxid')
    # data = oct_structure_feature().matminer_GlobalSymmetryFeatures(dataset=data,column_structure='structure_oxid')

    data = data.drop(columns=['pymatgen_structure','structure_oxid'])

    return data

if __name__ == '__main__' : 

    # 0. load dataset
    data = pd.read_csv('dataset_octa.csv')
    print(data)
    B_element_list = data['B_element']
    X_element_list = data['X_element']
    sites = data['sites']
    filename = data['file_name']
    path = '../structure_files/'


    # 1. contructure element features
    # contructure element features by matminer
    data = construct_element_feature_matminer(data)
    # contructure element features by mendeleev
    # data = construct_element_feature_mendeleev(data)
    # contructure element features by pymatgen
    # data = construct_element_feature_pymatgen(data)

    # 2. contructure local motif features
    data = local_feature_construct(data)

    # 3. contructure global features
    data = structure_feature_construct(data)

    print(data)
    data.to_csv('feature_set.csv',index=False)
