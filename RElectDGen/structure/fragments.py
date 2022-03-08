import os
import pandas as pd
import numpy as np

class FragmentDB():
    def __init__(
        self,
        filename: str
    ) -> None:

        self.filename = filename
        self.fragment_dir = os.path.dirname(filename)

        if os.path.isfile(filename):
            self.fragment_db = pd.read_csv(filename)
        else:
            self.fragment_db = None
    
    def is_valid_cluster(self, cluster) -> bool:

        total_charge = cluster.get_initial_charges().sum()
        integer_charge = np.round(total_charge,0) == np.round(total_charge,4)

        return integer_charge

    def is_valid_fragment(self, cluster,raw=False):

        if self.fragment_db is None:
            return False
        else:
            charge_i = cluster.get_initial_charges().sum()
                    
            db_ind = np.isclose(self.fragment_db['origin_charge'],charge_i)

            atom_symbols = np.array(cluster.get_chemical_symbols())
            for i, sym in enumerate(np.unique(atom_symbols)):
                col = 'n_' + sym
                
                nsym = np.sum(atom_symbols==sym)
                try:
                    db_ind = np.logical_and(db_ind,self.fragment_db[col] == nsym)
                except KeyError as e:
                    print(e)
                    print(cluster)
            if raw:
                return db_ind
            else:
                if db_ind.sum()==1:
                    return True
                else:
                    return False