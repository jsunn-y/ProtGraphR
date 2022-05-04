"""Helper functions"""
from __future__ import annotations
from lib2to3.pgen2.pgen import DFAState

import os
import pickle
from itertools import product

import pandas as pd
from pandas.api.types import is_numeric_dtype

import Bio.Seq

def std_folder_path(folder_path: str) -> str:
    """Check if the folder path ends with /"""
    if folder_path[-1] != "/":
        folder_path = folder_path + "/"
    return folder_path


def checkNgen_folder(folder_path: str, sub_folder_name: str = "") -> str:
    """Check if the folder or the subfolder exists
    and create a new directory"""

    folder_path = os.path.join(folder_path, sub_folder_name)

    if not os.path.exists(folder_path):
        print(f"Making {folder_path}...")
        os.mkdir(folder_path)

    return folder_path


def pickle_save(what2save, where2save):
    with open(where2save, "wb") as f:
        pickle.dump(what2save, f)


class PDBPath:
    """Class take care of file names and paths"""

    # TODO generate new folders if does not exist

    def __init__(self, file_path: str):

        # init variables
        self._file_path = file_path

        self._file_name = self._get_file_name()
        self._mut_name = self._get_mut_name()

    def _get_file_name(self):
        """Returns file name without extension."""
        return os.path.splitext(os.path.basename(self._file_path))[0]

    def _get_jy_struct_mutant_name(pdb_path: str, seq: Bio.Seq.Seq) -> str:
        """Extract the mutant name.
        TODO standardize graph names from the input pdb instead

        '2GI9_prepared_designed_100109_A_39W+A_40K+A_41M+A_54I.pdb'
        => 'V39W:D40K:G41M:V54I'

        Args
        - pdb_path: str, path to traid generated pdb file
        - seq: Bio.Seq.Seq, mutant amino acid sequence

        Returns: str, mutant name
        """
        if "WT" in pdb_path:
            return ""

        else:
            mut_list = pdb_path.split("_A_")[-1].split(".pdb")[0].split("+A_")
            for m, mut in enumerate(mut_list):
                # from 1 index to 0 index
                mut_list[m] = seq[int(mut[:-1]) - 1] + mut

            return ":".join(mut_list)
    
    def _get_mut_name(self) -> str:
        """Gets mutant name from the file name.

        Examples:
        (parent) 'GB1_MinRun__Minimized_StandardAlign' => ''
        (mutant) 'GB1_MinRun_D40A_Minimized_StandardAlign' => 'D40A'

        or 

        2GI9_prepared_designed_100109_A_39W+A_40K+A_41M+A_54I'
        => 'V39W:D40K:G41M:V54I
        """
        # the way that bruce renamed files
        if "MinRun" in self._file_name:
            return self._file_name.split("_")[2]
        elif "designed" in self._file_name:
            return self._get_jy_struct_mutant_name(self._file_name)

    def gen_path(self, output_folder: str, new_file_name: str, file_ext: str = "") -> str:
        """Generate path for given file"""

        # check if extension start with dot
        if file_ext != "":
            if file_ext[0] != ".":
                file_ext = "." + file_ext

        return os.path.join(output_folder, new_file_name + file_ext)

    @property
    def file_name(self) -> str:
        return self._file_name

    @property
    def mut_name(self) -> str:
        return self._mut_name


def split_df_unique_col_entry(
    df: pd.DataFrame, val_col_name: str, select_col_name: str
) -> dict:

    """select the dataframe based on selected column and
    return the column value in a list"""

    col_of_interest = df[select_col_name]
    select_groups = col_of_interest.unique()

    df_dict = {}
    for unique_entry in select_groups:
        df_dict[unique_entry] = df[col_of_interest == unique_entry][
            val_col_name
        ].to_list()

    return df_dict


def get_explode_df(
    df: pd.DataFrame, explode_col_name: str, save_col_names: list
) -> pd.DataFrame:

    """get tidy if backbone dataframe for plotting"""

    return df[save_col_names].join(df[explode_col_name].apply(pd.Series), how="left")


class DataFrame2PlotLabels:
    """A class dealing with plot labels from a dataframe"""

    def __init__(self, df: pd.DataFrame, col_name_list: list):
        self._df = df
        self._col_name_list = col_name_list

    def sort_df_labels(self, col_name: str):
        """sort column values"""

        col_entries = self._df[col_name].unique()

        if is_numeric_dtype(col_entries):
            return [str(entry) for entry in sorted(col_entries)]
        else:
            return sorted(col_entries, key=lambda s: str(s).lower())

    def gen_nested_labels(self):
        """generate sorted labels for bokeh plots for given unknown length of col_name_list"""
        if len(self._col_name_list) == 1:
            return self.sort_df_labels(self._col_name_list[0])
        elif "mut_loc" in self._col_name_list:
            if len(self._col_name_list) == 2:
                rest_sorted = self.sort_df_labels(
                    list(set(self._col_name_list) - {"mut_loc"})[0]
                )
                sorted_mut = self.sort_df_labels("mut_loc")
                return [
                    (r, m)
                    for r in rest_sorted
                    for m in sorted_mut
                    if int(r) == len(m.split(","))
                ]
            else:
                rest_sorted = list(
                    product(
                        *[
                            self.sort_df_labels(col_name)
                            for col_name in self._col_name_list
                            if col_name != "mut_loc"
                        ]
                    )
                )
                return [
                    (r, m) for r in rest_sorted for m in self.sort_df_labels("mut_loc")
                ]
        else:
            return list(
                product(
                    *[self.sort_df_labels(col_name) for col_name in self._col_name_list]
                )
            )