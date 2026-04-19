import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from TruthValue import TruthValue


class DeductiveReasoner:

    def __init__(self, kg_dir = "oregano/diseases_reasons.pickle"):
        kg = pickle.load(open(kg_dir, "rb"))
        self.G = self.build_graph(kg)

        proteins_all = []
        for v in kg.values():
            for _, proteins in v:
                proteins_all.extend(proteins)
        self.proteins_all = set(proteins_all)


    @staticmethod
    def build_graph(data):
        G = nx.DiGraph()
        
        for key, gene_protein_pairs in data.items():
            G.add_node(key, label=f'Key:{key}', color='red', layer=3)  
            
            for gene, proteins in gene_protein_pairs:
                G.add_node(gene, label=gene, color='blue', layer=2) 
                G.add_edge(gene, key)  
                
                for protein in proteins:
                    G.add_node(protein, label=protein, color='green', layer=1) 
                    G.add_edge(protein, gene) 
        
        return G
    

    def deductive_reasoning(self, proteins: list[tuple[str, TruthValue]], return_intermediate_results=False) -> list[tuple[str, TruthValue]]:
        """
        Perform deductive reasoning on the knowledge graph.
        
        Args:
            G (networkx.DiGraph): The knowledge graph.
            proteins (list): List of proteins to reason about."

        Returns:
            list: List of diseases associated with truth values.
        """
        G = self.G
        for node in G.nodes:
            G.nodes[node]['truth'] = TruthValue(0.0, 0.0, 1)

        
        for protein, truth_value in proteins:
            if not G.has_node(protein):
                continue
            G.nodes[protein]['truth'] = truth_value

        # propagate from layer 1 to layer 2
        for node in G.nodes:
            if G.nodes[node]['layer'] == 1:
                truth1: TruthValue = G.nodes[node]['truth']
                for neighbor in G.neighbors(node):
                    if G.nodes[neighbor]['layer'] == 2:
                        truth2: TruthValue = G.nodes[neighbor]['truth']
                        # Update the truth value of the neighbor
                        TruthValue.deduction(TruthValue(1.0, 0.9), truth1)
                        truth2.revision(truth1)
        # propagate from layer 2 to layer 3
        for node in G.nodes:
            if G.nodes[node]['layer'] == 2:
                truth1: TruthValue = G.nodes[node]['truth']
                for neighbor in G.neighbors(node):
                    if G.nodes[neighbor]['layer'] == 3:
                        truth2: TruthValue = G.nodes[neighbor]['truth']
                        # Update the truth value of the neighbor
                        TruthValue.deduction(TruthValue(1.0, 0.9), truth1)
                        truth2.revision(truth1)
        # collect the results in layer3
        results = []
        for node in G.nodes:
            if G.nodes[node]['layer'] == 3:
                truth: TruthValue = G.nodes[node]['truth']
                results.append((node, truth))
        results = [(node, truth) for node, truth in results if truth.w > 1e-3]
        if return_intermediate_results:
            layer2_results = []
            for node in G.nodes:
                if G.nodes[node]['layer'] == 2:
                    truth1: TruthValue = G.nodes[node]['truth']
                    layer2_results.append((node, truth1))
            return results, layer2_results
        else:
            return results
        
    
    def valid_protein(self, protein: str) -> bool:
        return protein in self.proteins_all