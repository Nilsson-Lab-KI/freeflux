#!/usr/bin/env pyhton
# -*- coding: UTF-8 -*-


__author__ = 'Chao Wu'
__date__ = '02/16/2020'




from collections import ChainMap, Counter
from itertools import product
from functools import reduce, lru_cache
import numpy as np
import pandas as pd
from sympy import Symbol
from .emu import EMU




class Reaction():
    '''
    Duplicate substrates or products could appear in one reaction, but with different atoms
    
    Attributes
    id: str
        reaction ID
    
    reversible: bool
        reversibility
    
    substrates_info: df
        index are substrate IDs (there could be duplicate substrates), 
        columns are Metabolite and stoichiometry
    
    products_info: df
        index are product IDs (there could be duplicate products), 
        columns are Metabolite and stoichiometry
    
    substrates: list
        unique substrate IDs, in order of alphabet
    
    products: list
        unique product IDs, in order of alphabet
    
    substrates_with_atoms: list
        unique IDs of substrates with atoms, in order of alphabet
    
    products_with_atoms: list    
        unique IDs of products with atoms, in order of alphabet
    
    _substrates_atom_mapping: list of dict or None
        reactants like: A({'ab': 0.5, 'ba': 0.5}) + B({'c': 1}) will be transformed to
        [{'a': [A, 1, 0.5], 'b': [A, 2, 0.5], 'c': [C, 1, 1]},
         {'a': [A, 2, 0.5], 'b': [A, 1, 0.5], 'a': [C, 1, 1]}]
    
    _products_atom_mapping: list of dict or None
        reactants like: A({'ab': 0.5, 'ba': 0.5}) + B({'c': 1}) will be transformed to
        [{'a': [A, 1, 0.5], 'b': [A, 2, 0.5], 'c': [C, 1, 1]},
         {'a': [A, 2, 0.5], 'b': [A, 1, 0.5], 'a': [C, 1, 1]}]    
    
    flux (fflux and bflux for reversible reaction): sym
        reaction flux (forward flux and backward flux for reversible reaction)
        
    host_models: set of Model or None
        model reaction is involved in
    '''
    
    def __init__(self, id, reversible = True):
        '''
        Parameters
        id: str
            reaction ID
        reversible: bool
            reversibility
        '''
        
        self.id = id
        self.reversible = reversible
        self.substrates_info = pd.DataFrame(columns = ['metab', 'stoy'])
        self.products_info = pd.DataFrame(columns = ['metab', 'stoy'])
        if self.reversible:
            self.fflux = Symbol(self.id+'_f')
            self.bflux = Symbol(self.id+'_b')
        else:
            self.flux = Symbol(self.id)
        self.host_models = None
    
    
    def add_substrates(self, substrates, stoichiometry):
        '''
        Parameters
        substrates: Metabolite or list of Metabolites
        stoichiometry: float or list of float
            stoichiometry of corresponding substrates
        '''
        
        if not isinstance(substrates, list):
            substrates = [substrates]
            stoichiometry = [stoichiometry]
            
        newSubs = pd.DataFrame({'metab': substrates, 'stoy': np.array(stoichiometry).astype(np.float)}, 
                               index = [sub.id for sub in substrates])    
            
        self.substrates_info = pd.concat((self.substrates_info, newSubs))
        
        
        for sub in substrates:
            if sub.host_reactions is None:
                sub.host_reactions = set([self])
            else:
                sub.host_reactions.add(self)
        

    def add_products(self, products, stoichiometry):
        '''
        Parameters
        products: Metabolite or list of Metabolites
        stoichiometry: float or list of float
            stoichiometry of corresponding products
        '''
        
        if not isinstance(products, list):
            products = [products]
            stoichiometry = [stoichiometry]
            
        newPros = pd.DataFrame({'metab': products, 'stoy': np.array(stoichiometry).astype(np.float)}, index = [pro.id for pro in products])    
            
        self.products_info = pd.concat((self.products_info, newPros))    
        
        
        for pro in products:
            if pro.host_reactions is None:
                pro.host_reactions = set([self])
            else:
                pro.host_reactions.add(self)
                

    def remove_substrates(self, substrates):
        '''
        Parameters
        substrates: Metabolite or list of Metabolites
        '''
        
        if not isinstance(substrates, list):
            substrates = [substrates]
        
        self.substrates_info.drop([sub.id for sub in substrates], inplace = True)
        
        
        for sub in substrates:
            sub.host_reactions.remove(self)
            
            if not sub.host_reactions:
                sub.host_reactions = None
                break
        
    
    def remove_products(self, products):
        '''
        Parameters
        products: Metabolite or list of Metabolites
        '''
        
        if not isinstance(products, list):
            products = [products]
        
        self.products_info.drop([pro.id for pro in products], inplace = True)
        
        
        for pro in products:
            pro.host_reactions.remove(self)
            
            if not pro.host_reactions:
                pro.host_reactions = None
                break
    
    
    @property
    @lru_cache()
    def substrates(self):
        
        return sorted(self.substrates_info.index.unique().tolist())
        
        
    @property
    @lru_cache()
    def products(self):
        
        return sorted(self.products_info.index.unique().tolist())
        
        
    @property
    @lru_cache()
    def substrates_with_atoms(self):
        
        return sorted({sub for sub, (metab, row) in self.substrates_info.iterrows() if metab.atoms})
        
        
    @property
    @lru_cache()
    def products_with_atoms(self):
        
        return sorted({pro for pro, (metab, row) in self.products_info.iterrows() if metab.atoms})    
        
    
    def _atom_mapping(self, reactant):
        '''
        reactants like: A({'ab': 0.5, 'ba': 0.5}) + B({'c': 1}) will be transformed to
        [{'a': [A, 1, 0.5], 'b': [A, 2, 0.5], 'c': [C, 1, 1]},
         {'a': [A, 2, 0.5], 'b': [A, 1, 0.5], 'a': [C, 1, 1]}]
        
        Parameters
        reactant: str,
            'substrate' or 'product'
        '''
        
        if reactant == 'substrate':
            reacsInfo = self.substrates_info.loc[self.substrates_with_atoms, 'metab']
        elif reactant == 'product':
            reacsInfo = self.products_info.loc[self.products_with_atoms, 'metab']
        else:
            raise ValueError('must be "substrate" or "product"')
        
        reacsAtomInfo = []
        for _, Metab in reacsInfo.iteritems():
            
            reacAtomInfo = []
            for atoms, coe in Metab.atoms_info.items():
                
                atomInfo = {atom: [Metab, no+1, coe] for no, atom in enumerate(atoms)}
                reacAtomInfo.append(atomInfo)
        
            reacsAtomInfo.append(reacAtomInfo)
        
        reacsAtomMappingRaw = list(product(*reacsAtomInfo))
        
        reacsAtomMapping = [ChainMap(*scenario) for scenario in reacsAtomMappingRaw]
        
        return reacsAtomMapping
        
    
    @property
    @lru_cache()
    def _substrates_atom_mapping(self):
        
        if self.substrates_with_atoms:
            return self._atom_mapping('substrate')
        else:
            return None
        
    
    @property
    @lru_cache()
    def _products_atom_mapping(self):
        
        if self.products_with_atoms:
            return self._atom_mapping('product')
        else:
            return None
            
    
    def _find_precursor_EMUs(self, emu, direction = 'forward'):
        '''
        for reaction like: A({'ab': 0.5, 'ba': 0.5}) + B({'c': 1}) -> C({'abc': 0.5, 'cba': 0.5})
        _find_precursor_EMUs(C12) returns
        [[[A_12], 0.5],
         [[B_1, A_2], 0.25],
         [[B_1, A_1], 0.25]]
        
        Parameters
        emu: EMU
        direction: str
            for reversible reaction,
            'forward' if emu is product and precursor emu(s) are substrates,
            'backward' if emu is substrate and precursor emu(s) are products;
            for irreversible reaction, only 'forward' is acceptable
            
        Returns
        preEMUsInfo: list
        '''        
        
        if self.reversible:
            if direction == 'forward':
                atomMapping = self._substrates_atom_mapping
                reacInfo = self.substrates_info['metab']
            
            elif direction == 'backward':
                atomMapping = self._products_atom_mapping
                reacInfo = self.products_info['metab']
        
        else:
            if direction != 'forward':
                raise ValueError('only "forward" is acceptable for irreversible reaction')
            
            atomMapping = self._substrates_atom_mapping
            reacInfo = self.substrates_info['metab']
        
        
        preEMUsInfoRaw = []
        for scenario in atomMapping:
            for atoms, coe in emu.metabolite.atoms_info.items():   # emu could have equivalents
                
                preAtoms = {}
                uniCoe = coe
                for atom in [atoms[no-1] for no in emu.atom_nos]:
                    
                    pre, preAtomNO, preCoe = scenario[atom]   # pre is Metabolite
                    
                    if pre not in preAtoms:
                        uniCoe *= preCoe
                        preAtoms[pre] = [preAtomNO]
                    else:
                        preAtoms[pre].append(preAtomNO)
                
                preEMUs = [EMU(pre.id+'_'+''.join(map(str, sorted(preAtomNOs))), pre, preAtomNOs) 
                           for pre, preAtomNOs in preAtoms.items()]
                preEMUsInfoRaw.append([preEMUs, uniCoe])
        
        preEMUsInfoRaw = [Counter({tuple(sorted(preEMUs)): coe}) 
                          for preEMUs, coe in preEMUsInfoRaw]   # use tuple not frozenset as key because 
                                                                # there could be identical EMUs
                                                                # remrember to sort
        preEMUsInfo = reduce(lambda x, y: x+y, preEMUsInfoRaw)   # combine identical precursor EMUs
        preEMUsInfo = [[list(preEMUs), coe] for preEMUs, coe in preEMUsInfo.items()]
        
        return preEMUsInfo
        
    
    def __repr__(self):
        
        arrow = '<->' if self.reversible else '->'
        subsStr = '+'.join(self.substrates)
        prosStr = '+'.join(self.products)
        
        return '%s %s: %s%s%s' % (self.__class__.__name__, self.id, subsStr, arrow, prosStr)        








