import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import pickle

from .deductive_reasoner import DeductiveReasoner
from TruthValue import TruthValue


class Reasoner:

    def __init__(self, top_k=5):
        path = Path(__file__).parent
        self.reasoner = DeductiveReasoner((path/"diseases_reasons.pickle").absolute())
        self.top_k = top_k
    
    def reason(self, proteins: list[str], top_k=None) -> list[tuple[str, TruthValue]]:
        proteins = [(p, TruthValue(1.0, 0.9)) for p in proteins if self.reasoner.valid_protein(p)]
        results, intermediate_results = self.reasoner.deductive_reasoning(proteins, return_intermediate_results=True)
        results = sorted(results, key=lambda x: x[1].e, reverse=True)
        results = results[:self.top_k]
        intermediate_results = sorted(intermediate_results, key=lambda x: x[1].e, reverse=True)
        intermediate_results = [r for r in intermediate_results if r[1].c > 0.01]
        return results, intermediate_results