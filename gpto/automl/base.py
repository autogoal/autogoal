# -*- coding: utf-8 -*-

import json
import os
import time

class Metaheuristic:
    def save(self, encoder=json.JSONEncoder):
        """Salva los datos actuales de la metaheurística"""
        p = {}

        for k,v in self.__dict__.items():
            if k.startswith("_"):
                continue
            p[k]=v

        with open(self.__class__.__name__ + "-" + str(int(time.time())) + ".json","a") as f:
            json.dump(p,f,cls=encoder)
            f.write("\n")

    def load(self):
        """Carga la metaheurística en el estado en el que se quedó"""
        if not os.path.exists(self.__class__.__name__ + ".json"):
            return False

        p= {}

        with open(self.__class__.__name__ + ".json","r") as f:
            p = json.load(f)

        for k,v in p.items():
            self.__dict__[k]= v

        return True
