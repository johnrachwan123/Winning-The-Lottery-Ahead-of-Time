import _pickle as pickle
import json
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt


class DataManager:
    """
    Handles saving and loading of pickle files
    also implements a faster deepcopy
    and defines the savename in the results folder
    """

    def __init__(self, directory):

        # determines relative disk directory for saving/loading
        self.directory = directory
        self.stamp = ""
        self.actual_date = None

    def save_python_obj(self, obj, name, print_success=True):
        """ Saves python object to disk in pickle """

        try:
            filepath = os.path.join(self.directory, f'{name}.pickle')
            with open(filepath, 'wb') as handle:
                pickle.dump(obj, handle, protocol=-1)

                if (print_success):
                    print("Saved {}".format(name))
        except Exception as e:
            print(e)
            print("Failed saving {}, continue anyway".format(name))

    def load_python_obj(self, name):
        """ Loads python object from disk if pickle """

        obj = None
        try:
            filepath = os.path.join(self.directory, f'{name}.pickle')
            with (open(filepath, "rb")) as openfile:
                obj = pickle.load(openfile)
        except FileNotFoundError:
            print("{} not loaded because file is missing".format(name))
            return
        print("Loaded {}".format(name))
        return obj

    def personal_deepcopy(self, obj):
        """ Deep copies any object faster than builtin """

        return pickle.loads(pickle.dumps(obj, protocol=-1))

    def duplicate_list(self, lst: list) -> list:
        """ shallow copies list """

        return [x for x in lst]

    def duplicate_set(self, st: set) -> set:
        """ shallow copies set """

        return {x for x in st}

    def duplicate_dict(self, dc) -> dict:
        """ shallow copies dictionary """
        return {key: dc[key] for key in dc}

    def duplicate_default_dict(self, dfdc, type_func, typ) -> defaultdict:
        """ shallow copies a defualtdictionary but gives tha chance to also shallow copy its members """

        output = defaultdict(typ)
        for key in dfdc:
            output[key] = type_func(dfdc[key])
        return output

    def dump_only(self, obj):
        return pickle.dumps(obj, protocol=-1)

    def load_only(self, obj):
        return pickle.loads(obj)

    def save_figure(self, name, no_axis=True):
        if (no_axis):
            plt.axis('off')
        plt.savefig(self.directory + name + ".png", bbox_inches='tight')

    def set_date_stamp(self, addition=""):
        """ generates printable date stamp"""

        if (len(self.stamp) > 2):
            raise Exception("Attempting to reset datestamp, but it was already set")

        self.actual_date = datetime.now()
        self.stamp = str(self.actual_date).split(".")[0].replace(" ", "_").replace(':', '.') + addition
        print(f"Made datestamp: {self.stamp}")
        return self.stamp

    def create_dir(self, name: str):
        os.makedirs(os.path.join(self.directory, name), exist_ok=True)

    def load_json(self, name: str) -> dict:
        if not (name.endswith(".json")):
            name = name + ".json"
        with open(os.path.join('./Image Classification/utils/', name)) as json_file:
            data = json.load(json_file)
            return data

    def save_json(self, name: str, data: dict):
        if not (name.endswith(".json")):
            name = name + ".json"
        with open(os.path.join(self.directory, name), 'w') as outfile:
            json.dump(data, outfile)

    def write_to_file(self, name, content):
        with open(os.path.join(self.directory, name), 'w') as outfile:
            outfile.write(content)
