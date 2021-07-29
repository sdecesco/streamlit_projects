#!/usr/bin/env python

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS, AllChem
import mols2grid
import streamlit as st
import sys
from rdkit import Chem
import pandas as pd
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Draw import MolsToGridImage
from tqdm.auto import tqdm

# Set the layout of the page yo wide
# st.set_page_config(layout='wide')
"""
# App to navigate through Cluster members 
# 
"""


@st.cache
class ButinaCluster:
    def __init__(self, fp_type="rdkit"):
        self.fp_type = fp_type

    @st.cache
    def cluster_smiles(self, smi_list, sim_cutoff=0.8):
        mol_list = [Chem.MolFromSmiles(x) for x in tqdm(smi_list, desc="Calculating Fingerprints")]
        return self.cluster_mols(mol_list, sim_cutoff)

    @st.cache
    def get_fps(self, mol_list):
        fp_dict = {
            "morgan2": [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in mol_list],
            "rdkit": [Chem.RDKFingerprint(x) for x in mol_list],
            "maccs": [MACCSkeys.GenMACCSKeys(x) for x in mol_list],
            "ap": [Pairs.GetAtomPairFingerprint(x) for x in mol_list]
        }
        if fp_dict.get(self.fp_type) is None:
            print(f"No fingerprint method defined for {fp_type}")
            sys.exit(0)

        return fp_dict[self.fp_type]

    def cluster_mols(self, mol_list, sim_cutoff=0.8):
        dist_cutoff = 1.0 - sim_cutoff
        # fp_list = [rdmd.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in mol_list]
        fp_list = self.get_fps(mol_list)
        dists = []
        nfps = len(fp_list)
        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
            dists.extend([1 - x for x in sims])
        mol_clusters = Butina.ClusterData(dists, nfps, dist_cutoff, isDistData=True)
        cluster_id_list = [0] * nfps
        for idx, cluster in enumerate(mol_clusters, 1):
            for member in cluster:
                cluster_id_list[member] = idx
        return [x - 1 for x in cluster_id_list]


def get_largest_fragment(mol):
    frags = list(Chem.GetMolFrags(mol, asMols=True))
    frags.sort(key=lambda x: x.GetNumAtoms(), reverse=True)
    return frags[0]


# class Session:
#     pass
#
#
# @st.cache(allow_output_mutation=True)
# def fetch_session(clusters=None):
#     session = Session()
#     session.page = 1
#     session.global_answers = {y:False for y in clusters.Cluster}
#     return session


def display_cluster_members(df, sel, align_mcs=True, strip_counterions=True, mol_per_row=4):
    beta_col = [1] * mol_per_row
    if len(sel):
        sel_df = df.query("Cluster in @sel")
        mol_list = [Chem.MolFromSmiles(x) for x in sel_df.SMILES]
        # strip counterions
        if strip_counterions:
            mol_list = [get_largest_fragment(x) for x in mol_list]
        # Align structures on the MCS
        if align_mcs and len(mol_list) > 1:
            mcs = rdFMCS.FindMCS(mol_list)
            mcs_query = Chem.MolFromSmarts(mcs.smartsString)
            AllChem.Compute2DCoords(mcs_query)
            for m in mol_list:
                AllChem.GenerateDepictionMatching2DStructure(m, mcs_query)
        legends = list(sel_df.Name.astype(str))
        for i in range(len(mol_list)):
            for j in range(mol_per_row):
                if i % mol_per_row == j:
                    if j == 0:
                        mol_grid = st.beta_columns(beta_col)
                    mol_grid[j].image(get_img(mol_list[i]))
                    mol_grid[j].write(legends[i])


@st.cache
def get_img(mol):
    return Draw.MolToImage(mol)


def update_box(clust_id):
    st.session_state[clust_id] = st.session_state[clust_id]


def display_cluster_card(df, mol_per_row=4):
    mol_list = [Chem.MolFromSmiles(x) for x in df.SMILES]
    mol_count = [x for x in df.Num]
    cluster_name = [str(y) for y in df.Cluster]
    beta_col = [1] * mol_per_row
    for i in range(len(mol_list)):
        for j in range(mol_per_row):
            if i % mol_per_row == j:
                if j == 0:
                    checkboxes = st.beta_columns(beta_col)
                checkboxes[j].checkbox(label='Cluster ' + str(cluster_name[i]), key=cluster_name[i],on_change=update_box(str(cluster_name[i])))
                checkboxes[j].image(get_img(mol_list[i]))
                checkboxes[j].write("mol in cluster: "+str(mol_count[i]))


butina_cluster = ButinaCluster("rdkit")
df = pd.read_csv("test.smi", sep=" ", names=["SMILES", "Name"])
df['Cluster'] = butina_cluster.cluster_smiles(df.SMILES, sim_cutoff=0.7)
cluster_rows = []
for k, v in df.groupby("Cluster"):
    cluster_rows.append([v.SMILES.values[0], k, len(v)])

cluster_df = pd.DataFrame(cluster_rows, columns=["SMILES", "Cluster", "Num"])

mol_per_page = 20
idx = [x for x in range(0, len(cluster_df), mol_per_page)]
idx_list = []
for i in range(len(idx)):
    if i != len(idx) - 1:
        idx_list.append((idx[i], idx[i + 1]))
    else:
        if idx[i] == len(cluster_df):
            continue
        elif idx[i] < len(cluster_df):
            remainder = len(cluster_df) % mol_per_page
            idx_list.append((idx[i], idx[i] + remainder))

page_name = ['Page ' + str(x) for x in range(1, len(idx_list) + 1)]


first, previous, next_page, last, page_number = st.beta_columns([1, 1, 1, 1, 1])
deselect_all, select_all, p1, p2 = st.beta_columns([1, 1, 1, 1])

if 'page' not in st.session_state:
    st.session_state.page = 1
cluster_name = [str(y) for y in cluster_df.Cluster]
for i in cluster_name:
    if i not in st.session_state:
        st.session_state[i] = False


with previous:
    if st.button("PREVIOUS"):
        st.session_state.page -= 1
        if st.session_state.page == 0:
            st.session_state.page = 1
with next_page:
    if st.button("NEXT"):
        st.session_state.page += 1
        if st.session_state.page > len(idx_list):
            st.session_state.page = len(idx_list)
with first:
    if st.button("FIRST"):
        st.session_state.page = 1
with last:
    if st.button("LAST"):
        st.session_state.page = len(idx_list)
with page_number:
    selected_page = st.selectbox('', page_name, index=st.session_state.page - 1)
    st.session_state.page = int(selected_page.split(' ')[1])

with deselect_all:
    if st.button("DESELECT ALL"):
        for i in cluster_name:
            st.session_state[i] = False

with select_all:
    if st.button("SELECT ALL"):
        for i in cluster_name:
            st.session_state[i] = True


display_cluster_card(cluster_df[idx_list[st.session_state.page - 1][0]:idx_list[st.session_state.page - 1][1]], mol_per_row=4)


cluster_selected = [i for i in cluster_name if st.session_state[i]]


st.markdown("---")
for i in cluster_selected:
    with st.beta_expander("Cluster "+str(i)):
        display_cluster_members(df, i, strip_counterions=True, align_mcs=True, mol_per_row=4)

st.markdown("---")
#display_cluster_members(df,cluster_selected,strip_counterions=True,align_mcs=True,mol_per_row=4)
