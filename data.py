import itertools
import random
import sys
import time
from pathlib import Path
from typing import Optional

import os
import torch
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms

from jarvis.core.graphs import nearest_neighbor_edges, build_undirected_edgedata
from jarvis.db.figshare import data as jdata
from jarvis.core.specie import chem_data, get_node_attributes

# from torch.utils.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.loader import DataLoader

from tqdm import tqdm
import math
from jarvis.db.jsonutils import dumpjson

from pandarallel import pandarallel
import periodictable
import algorithm

pandarallel.initialize(progress_bar=True)

tqdm.pandas()

torch.set_printoptions(precision=10)


def find_index_array(A, B):
    _, n = B.shape
    index_array = torch.zeros(n, dtype=torch.long)

    for i in range(n):
        idx = torch.where((A == B[:, i].unsqueeze(1)).all(dim=0))[0]
        index_array[i] = idx

    return index_array

class StructureDataset(InMemoryDataset):

    def __init__(self, df, data_path, processdir, target, name, atom_features="atomic_number",
                 id_tag="jid", root='./', transform=None, pre_transform=None, pre_filter=None,
                 mean=None, std=None, normalize=False):

        self.df = df
        self.data_path = data_path
        self.processdir = processdir
        self.target = target
        self.name = name
        self.atom_features = atom_features
        self.id_tag = id_tag
        self.ids = self.df[self.id_tag]
        self.labels = torch.tensor(self.df[self.target]).type(
            torch.get_default_dtype()
        )
        if mean is not None:
            self.mean = mean
        elif normalize:
            self.mean = torch.mean(self.labels)
        else:
            self.mean = 0.0
        if std is not None:
            self.std = std
        elif normalize:
            self.std = torch.std(self.labels)
        else:
            self.std = 1.0

        self.group_id = {
            "H": 0,
            "He": 1,
            "Li": 2,
            "Be": 3,
            "B": 4,
            "C": 0,
            "N": 0,
            "O": 0,
            "F": 5,
            "Ne": 1,
            "Na": 2,
            "Mg": 3,
            "Al": 6,
            "Si": 4,
            "P": 0,
            "S": 0,
            "Cl": 5,
            "Ar": 1,
            "K": 2,
            "Ca": 3,
            "Sc": 7,
            "Ti": 7,
            "V": 7,
            "Cr": 7,
            "Mn": 7,
            "Fe": 7,
            "Co": 7,
            "Ni": 7,
            "Cu": 7,
            "Zn": 7,
            "Ga": 6,
            "Ge": 4,
            "As": 4,
            "Se": 0,
            "Br": 5,
            "Kr": 1,
            "Rb": 2,
            "Sr": 3,
            "Y": 7,
            "Zr": 7,
            "Nb": 7,
            "Mo": 7,
            "Tc": 7,
            "Ru": 7,
            "Rh": 7,
            "Pd": 7,
            "Ag": 7,
            "Cd": 7,
            "In": 6,
            "Sn": 6,
            "Sb": 4,
            "Te": 4,
            "I": 5,
            "Xe": 1,
            "Cs": 2,
            "Ba": 3,
            "La": 8,
            "Ce": 8,
            "Pr": 8,
            "Nd": 8,
            "Pm": 8,
            "Sm": 8,
            "Eu": 8,
            "Gd": 8,
            "Tb": 8,
            "Dy": 8,
            "Ho": 8,
            "Er": 8,
            "Tm": 8,
            "Yb": 8,
            "Lu": 8,
            "Hf": 7,
            "Ta": 7,
            "W": 7,
            "Re": 7,
            "Os": 7,
            "Ir": 7,
            "Pt": 7,
            "Au": 7,
            "Hg": 7,
            "Tl": 6,
            "Pb": 6,
            "Bi": 6,
            "Po": 4,
            "At": 5,
            "Rn": 1,
            "Fr": 2,
            "Ra": 3,
            "Ac": 9,
            "Th": 9,
            "Pa": 9,
            "U": 9,
            "Np": 9,
            "Pu": 9,
            "Am": 9,
            "Cm": 9,
            "Bk": 9,
            "Cf": 9,
            "Es": 9,
            "Fm": 9,
            "Md": 9,
            "No": 9,
            "Lr": 9,
            "Rf": 7,
            "Db": 7,
            "Sg": 7,
            "Bh": 7,
            "Hs": 7
        }
        # Ensure the processed directory exists
        os.makedirs(os.path.join(root, processdir), exist_ok=True)

        super(StructureDataset, self).__init__(root, transform, pre_transform, pre_filter)
        print(f"DIAGNOSTIC: Checking if processed file exists: {self.processed_paths[0]}")
        #if os.path.exists(self.processed_paths[0]):
            #print(f"DIAGNOSTIC: Found existing processed file: {self.processed_paths[0]}")
        #self.data, self.slices = torch.load(self.processed_paths[0])

        # Load processed data if it exists #### new add
        processed_path = self.processed_paths[0]
        if os.path.exists(processed_path):
            try:
                self.data, self.slices = torch.load(processed_path)
                print(f"Loaded processed data from {processed_path}")
            except Exception as e:
                print(f"Error loading processed data: {str(e)}")
                print("Regenerating processed data...")
                self.process()
        else:
            self.process()
        ###  new add
    @property
    def raw_file_names(self):
        return os.path.join(self.root, self.data_path)

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.processdir)

    @property
    def processed_file_names(self):
        return self.name + '.pt'

    def process(self):
        try:
            process_start = time.time()
            print(f"DIAGNOSTIC: Starting StructureDataset.process for {self.name}")
            mat_data = torch.load(self.raw_file_names)
            print(f"DIAGNOSTIC: Loaded raw data with {len(mat_data)} structures")

            data_list = []
            features = self._get_attribute_lookup(self.atom_features)

            count = 0
            for i in tqdm(range(len(mat_data))):
                try:
                    z = mat_data[i].x
                    mat_data[i].atom_numbers = z

                    group_feats = []
                    for atom in z:
                        group_feats.append(self.group_id[periodictable.elements[int(atom)].symbol])
                    group_feats = torch.tensor(np.array(group_feats)).type(torch.LongTensor)
                    identity_matrix = torch.eye(10)
                    g_feats = identity_matrix[group_feats]
                    if len(list(g_feats.size())) == 1:
                        g_feats = g_feats.unsqueeze(0)

                    f = torch.tensor(features[mat_data[i].atom_numbers.long().squeeze(1)]).type(torch.FloatTensor)
                    if len(mat_data[i].atom_numbers) == 1:
                        f = f.unsqueeze(0)

                    mat_data[i].x = f
                    mat_data[i].g_feats = g_feats

                    mat_data[i].y = (self.labels[i] - self.mean) / self.std
                    mat_data[i].label = self.labels[i]

                    data_list.append(mat_data[i])

                    count += 1  # Increment counter
                    if count % 50 == 0:  # Log every 50 structures
                        print(f"DIAGNOSTIC: Processed {count}/{len(mat_data)} structures")
                except Exception as e:
                    print(f"ERROR processing structure {i}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue  # Skip this structure and try the next one

            print(f"DIAGNOSTIC: Completed processing loop - processed {count}/{len(mat_data)} structures")

            if self.pre_filter is not None:

                try:
                    data_list = [data for data in data_list if self.pre_filter(data)]
                    print(f"DIAGNOSTIC: Applied pre_filter, now have {len(data_list)} structures")
                except Exception as e:
                    print(f"ERROR during pre_filter: {str(e)}")
                    import traceback
                    traceback.print_exc()

            if self.pre_transform is not None:

                try:
                    data_list = [self.pre_transform(data) for data in data_list]
                    print(f"Processing: Applied pre_transform, still have {len(data_list)} structures")
                except Exception as e:
                    print(f"ERROR during pre_transform: {str(e)}")
                    import traceback
                    traceback.print_exc()


            print(f"Processing: About to collate and save {len(data_list)} structures")
            try:
                mat_data, slices = self.collate(data_list)
                print(f"Processing: Collation successful")
            except Exception as e:
                print(f"ERROR during collation: {str(e)}")
                import traceback
                traceback.print_exc()
                raise

            save_start = time.time()
            print(f"Processing: About to save to {self.processed_paths[0]}")
            try:
                # Make sure the directory exists
                import os
                os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)

                # Check file path
                print(f"Processing: Saving to path: {self.processed_paths[0]}")

                # Save the file
                print('Saving...')
                torch.save((mat_data, slices), self.processed_paths[0])
                print(f"Processing: Successfully saved processed data, took {time.time() - save_start:.2f}s")
            except Exception as e:
                print(f"ERROR during save: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            print(f"PROCESSING: Total processing time: {time.time() - process_start:.2f}s")
        except Exception as e:
            print(f"UNHANDLED ERROR in processing {self.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        max_z = max(v["Z"] for v in chem_data.values())

        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features


def load_radius_graphs(
        df: pd.DataFrame,
        name: str = "dft_3d",
        target: str = "",
        radius: float = 4.0,
        max_neighbors: int = 16,
        cachedir: Optional[Path] = None,
):
    print(f"GRAPH BUILDING: Starting radius graph construction with r={radius}, max_neighbors={max_neighbors}")

    def atoms_to_graph(atoms):
        try:
            structure = Atoms.from_dict(atoms)
            sps_features = []
            for ii, s in enumerate(structure.elements):
                feat = list(get_node_attributes(s, atom_features="atomic_number"))
                sps_features.append(feat)
            sps_features = np.array(sps_features)
            node_features = torch.tensor(sps_features).type(
                torch.get_default_dtype()
            )
            edges = nearest_neighbor_edges(atoms=structure, cutoff=radius, max_neighbors=max_neighbors)
            u, v, r = build_undirected_edgedata(atoms=structure, edges=edges)

            data = Data(x=node_features, edge_index=torch.stack([u, v]), edge_attr=r.norm(dim=-1))
            return data
        except Exception as e:
            print(print(f"Error in atoms_to_graph: {str(e)}"))
            return None

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{target}-radius.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        print(f"Graph Building: Loading existing cache file: {cachefile}")
        try:
            return torch.load(cachefile)
        except Exception as e:
            print(f"Error loading cached file {cachefile}: {str(e)}")
            # If loading fails, regenerate the data
            pass
    #else:
        #graphs = df["atoms"].parallel_apply(atoms_to_graph).values
        #torch.save(graphs, cachefile)
    print(f"Graph Building:Generating radius graphs for {name}-{target}")
    # Reduce chunk size for parallel processing to avoid memory issues
    # Process in smaller chunks
    chunk_size = min(300, len(df))
    all_graphs = []
    graph_build_start = time.time()
    ########## below is new add
    # Process in smaller chunks to avoid memory issues
    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:i + chunk_size]
        print(f"Processing chunk {i // chunk_size + 1}/{(len(df) + chunk_size - 1) // chunk_size}")
        # Use apply instead of parallel_apply for more stability
        graphs = chunk_df["atoms"].apply(atoms_to_graph).values
        # Filter out None values that might be returned from errors
        graphs = [g for g in graphs if g is not None]
        all_graphs.extend(graphs)
    print(f"GRAPH BUILDING: Graph generation completed in {time.time() - graph_build_start:.2f}s")
    # Save the data
    if cachefile is not None:
        try:
            print(f"Graph Building: Saving {len(all_graphs)} graphs to {cachefile}")
            save_start = time.time()
            torch.save(all_graphs, cachefile)
            print(f"GRAPH BUILDING: Saved in {time.time() - save_start:.2f}s")
        except Exception as e:
            print(f"Error saving to {cachefile}: {str(e)}")

    return all_graphs
    ##### above is new add

def load_infinite_graphs(
        df: pd.DataFrame,
        name: str = "dft_3d",
        target: str = "",
        cachedir: Optional[Path] = Path('cache'),
        infinite_funcs=[],
        infinite_params=[],
        R=5,
):
    def atoms_to_graph(atoms):
        try:
            """Convert structure dict to DGLGraph."""
            structure = Atoms.from_dict(atoms)

            # build up atom attribute tensor
            sps_features = []
            for ii, s in enumerate(structure.elements):
                feat = list(get_node_attributes(s, atom_features="atomic_number"))
                sps_features.append(feat)

            sps_features = np.array(sps_features)
            node_features = torch.tensor(sps_features).type(
                torch.get_default_dtype()
            )

            u = torch.arange(0, node_features.size(0), 1).unsqueeze(1).repeat((1, node_features.size(0))).flatten().long()
            v = torch.arange(0, node_features.size(0), 1).unsqueeze(0).repeat((node_features.size(0), 1)).flatten().long()

            edge_index = torch.stack([u, v])

            lattice_mat = structure.lattice_mat.astype(dtype=np.double)

            vecs = structure.cart_coords[u.flatten().numpy().astype(np.int)] - structure.cart_coords[
                v.flatten().numpy().astype(np.int)]

            inf_edge_attr = torch.FloatTensor(np.stack([getattr(algorithm, func)(vecs, lattice_mat, param=param, R=R)
                                         for func, param in zip(infinite_funcs, infinite_params)], 1))
            edges = nearest_neighbor_edges(atoms=structure, cutoff=4, max_neighbors=16)
            u, v, r = build_undirected_edgedata(atoms=structure, edges=edges)

            data = Data(x=node_features, edge_attr=r.norm(dim=-1), edge_index=torch.stack([u, v]), inf_edge_index=edge_index,
                        inf_edge_attr=inf_edge_attr)
            return data
        except Exception as e:
            print(f"Error in infinite_atoms_to_graph: {str(e)}")
            return None

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{target}-infinite.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        print(f"DIAGNOSTIC: Loading existing cache file: {cachefile}")
    #else:
        #graphs = df["atoms"].parallel_apply(atoms_to_graph).values
        #torch.save(graphs, cachefile)
        try:
            return torch.load(cachefile)
        except Exception as e:
            print(f"Error loading cached file {cachefile}: {str(e)}")
            # If loading fails, regenerate the data
            pass
    ####  new add
    print(f"Generating infinite graphs for {name}-{target}")
    # Process in smaller chunks to avoid memory issues
    chunk_size = min(100, len(df))
    all_graphs = []

    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:i + chunk_size]
        print(f"Processing chunk {i // chunk_size + 1}/{(len(df) + chunk_size - 1) // chunk_size}")
        # Use apply instead of parallel_apply for more stability
        graphs = chunk_df["atoms"].apply(atoms_to_graph).values
        # Filter out None values
        graphs = [g for g in graphs if g is not None]
        all_graphs.extend(graphs)

    # Save the data
    if cachefile is not None:
        try:
            print(f"Saving {len(all_graphs)} graphs to {cachefile}")
            torch.save(all_graphs, cachefile)
        except Exception as e:
            print(f"Error saving to {cachefile}: {str(e)}")

    return all_graphs
    #### new add

def get_id_train_val_test(
        total_size=1000,
        split_seed=123,
        train_ratio=None,
        val_ratio=0.1,
        test_ratio=0.1,
        n_train=None,
        n_test=None,
        n_val=None,
        keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
            train_ratio is None
            and val_ratio is not None
            and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1

    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)

    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test): -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


def get_torch_dataset(
        dataset=None,
        root="",
        cachedir="",
        processdir="",
        name="",
        id_tag="jid",
        target="",
        atom_features="",
        normalize=False,
        euclidean=False,
        cutoff=4.0,
        max_neighbors=16,
        infinite_funcs=[],
        infinite_params=[],
        R=5,
        mean=0.0,
        std=1.0,
):
    """Get Torch Dataset."""
    df = pd.DataFrame(dataset)
    # Verify target values
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    vals = df[target].values
    print(f"Target '{target}' data range: min={np.min(vals)}, max={np.max(vals)}")

    cache = os.path.join(root, cachedir)
    #if not os.path.exists(cache):
        #os.makedirs(cache)
    os.makedirs(cache, exist_ok=True) ####   new add

    # Ensure process directory exists   #### new add
    process_dir = os.path.join(root, processdir)
    os.makedirs(process_dir, exist_ok=True)

    if euclidean:
        load_radius_graphs(
            df,
            radius=cutoff,
            max_neighbors=max_neighbors,
            name=name + "-" + str(cutoff),
            target=target,
            cachedir=Path(cache),
        )

        data = StructureDataset(
            df,
            os.path.join(cachedir, f"{name}-{cutoff}-{target}-radius.bin"),
            processdir,
            target=target,
            name=f"{name}-{cutoff}-{target}-radius",
            atom_features=atom_features,
            id_tag=id_tag,
            root=root,
            mean=mean,
            std=std,
            normalize=normalize,
        )
    else:
        load_infinite_graphs(
            df,
            name=name,
            target=target,
            cachedir=Path(cache),
            infinite_funcs=infinite_funcs,
            infinite_params=infinite_params,
            R=R,
        )

        data = StructureDataset(
            df,
            os.path.join(cachedir, f"{name}-{target}-infinite.bin"),
            processdir,
            target=target,
            name=f"{name}-{target}-infinite",
            atom_features=atom_features,
            id_tag=id_tag,
            root=root,
            mean=mean,
            std=std,
            normalize=normalize,
        )
    print(f"Successfully created {name} dataset with {len(data)} samples")
    return data


def get_train_val_loaders(
        dataset: str = "dft_3d",
        root: str = "",
        cachedir: str = "",
        processdir: str = "",
        dataset_array=None,
        target: str = "target",
        atom_features: str = "cgcnn",
        n_train=None,
        n_val=None,
        n_test=None,
        train_ratio=None,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size: int = 64,
        split_seed: int = 123,
        keep_data_order=False,
        workers: int = 4,
        pin_memory: bool = True,
        id_tag: str = "jid",
        normalize=False,
        euclidean=False,
        cutoff: float = 4.0,
        max_neighbors: int = 16,
        infinite_funcs=[],
        infinite_params=[],
        R=5,
):
    print("Data Loading: Starting get_train_val_loaders function")
    print(f"DATA LOADING: Configuration - euclidean={euclidean}, normalize={normalize}, batch_size={batch_size}")
    """Get train, val, test data loaders for a dataset."""
    try:
        # load dataset
        data_load_start = time.time()
        if not dataset_array:
            d = jdata(dataset)
        else:
            print(f"DATA LOADING: Using provided dataset array with {len(dataset_array)} entries")
            d = dataset_array
        print(f"DATA LOADING: Dataset loaded in {time.time() - data_load_start:.2f}s")

        filtering_start = time.time()
        print(f"DATA LOADING: Filtering dataset for valid '{target}' values")
        dat = []
        all_targets = []
        nan_count = 0
        none_count = 0
        na_count = 0
        list_target_count = 0

        for i in d:
            if isinstance(i[target], list):
                all_targets.append(torch.tensor(i[target]))
                dat.append(i)
                list_target_count += 1
            elif (
                    i[target] is not None
                    and i[target] != "na"
                    and not math.isnan(i[target])
            ):
                dat.append(i)
                all_targets.append(i[target])
            else:
                if i[target] is None:
                    none_count += 1
                elif i[target] == "na":
                    na_count += 1
                else:
                    nan_count += 1
        print(f"DATA LOADING: Found {len(dat)} valid entries out of {len(d)}")
        print(f"DATA LOADING: Filtered out - NaN: {nan_count}, None: {none_count}, 'na': {na_count}")
        print(f"DATA LOADING: {list_target_count} targets are lists")
        print(f"DATA LOADING: Data filtering completed in {time.time() - filtering_start:.2f}s")

        # Split data into train/val/test
        split_start = time.time()
        print(f"DATA LOADING: Splitting data with seed {split_seed}, train/val/test ratios: {train_ratio}/{val_ratio}/{test_ratio}")
        print(f"DATA LOADING: Target counts - train: {n_train}, val: {n_val}, test: {n_test}")
        id_train, id_val, id_test = get_id_train_val_test(
            total_size=len(dat),
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            n_train=n_train,
            n_test=n_test,
            n_val=n_val,
            keep_data_order=keep_data_order,
        )
        # Save split information
        ids_train_val_test = {}
        ids_train_val_test["id_train"] = [dat[i][id_tag] for i in id_train]
        ids_train_val_test["id_val"] = [dat[i][id_tag] for i in id_val]
        ids_train_val_test["id_test"] = [dat[i][id_tag] for i in id_test]

        dumpjson(
            data=ids_train_val_test,
            filename=os.path.join(root, "ids_train_val_test.json"),
        )

        dataset_train = [dat[x] for x in id_train]
        dataset_val = [dat[x] for x in id_val]
        dataset_test = [dat[x] for x in id_test]

        print(f"DATA LOADING: Split datasets - train: {len(dataset_train)}, val: {len(dataset_val)}, test: {len(dataset_test)}")
        print(f"DATA LOADING: Data splitting completed in {time.time() - split_start:.2f}s")

        train_process_start = time.time()
        print("DATA LOADING: Starting train data processing")
        try:
            train_data = get_torch_dataset(
                dataset=dataset_train,
                root=root,
                cachedir=cachedir,
                processdir=processdir,
                name=dataset + "_train",
                id_tag=id_tag,
                target=target,
                atom_features=atom_features,
                normalize=normalize,
                euclidean=euclidean,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                infinite_funcs=infinite_funcs,
                infinite_params=infinite_params,
                R=R,
            )

            mean = train_data.mean
            std = train_data.std
            print("Data Loading: Finished train data processing")
            print(f"DATA LOADING: Training data processing completed in {time.time() - train_process_start:.2f}s")
        except Exception as e:
            print(f"Error processing training data: {str(e)}")
            raise

        # Process validation dataset
        val_process_start = time.time()
        print("Data Loading: Starting validation data processing")
        try:
            val_data = get_torch_dataset(
                dataset=dataset_val,
                root=root,
                cachedir=cachedir,
                processdir=processdir,
                name=dataset + "_val",
                id_tag=id_tag,
                target=target,
                atom_features=atom_features,
                normalize=normalize,
                euclidean=euclidean,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                infinite_funcs=infinite_funcs,
                infinite_params=infinite_params,
                R=R,
                mean=mean,
                std=std
            )
            print(f"DATA LOADING: Validation data processing completed in {time.time() - val_process_start:.2f}s")
        except Exception as e:
            print(f"Error processing validation data: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        #val_data = None

        # Process test dataset
        test_process_start = time.time()
        print("Data Loading: Starting testing data processing")
        try:
            test_data = get_torch_dataset(
                dataset=dataset_test,
                root=root,
                cachedir=cachedir,
                processdir=processdir,
                name=dataset + "_test",
                id_tag=id_tag,
                target=target,
                atom_features=atom_features,
                normalize=normalize,
                euclidean=euclidean,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                infinite_funcs=infinite_funcs,
                infinite_params=infinite_params,
                R=R,
                mean=mean,
                std=std,
            )
            print(f"DATA LOADING: Test data processing completed in {time.time() - test_process_start:.2f}s")
        except Exception as e:
            print(f"Error processing test data: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        #test_data = None


        # Create a regular pytorch dataloader
        loader_creation_start = time.time()
        print("DATA LOADING: Creating data loaders...")
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=workers,
            pin_memory=pin_memory,
        )

        print("n_train:", len(train_loader.dataset))
        print("n_val:", 0 if val_loader is None else len(val_loader.dataset))
        print("n_test:", 0 if test_loader is None else len(test_loader.dataset))
        print(f"DATA LOADING: Complete data loading process took {time.time() - loader_creation_start:.2f}s")
        return (
            train_loader,
            val_loader,
            test_loader,
            mean,
            std,
        )
    except Exception as e:
        print(f"Error in get_train_val_loaders: {str(e)}")
        import traceback
        traceback.print_exc()
        raise