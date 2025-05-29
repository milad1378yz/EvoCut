from typing import Dict, Any, List
from uuid import uuid4


class Individual:
    def __init__(
        self, chromosome: Dict[str, Any], fitness: float = None, solver_reports: List[Dict] = []
    ):
        self.id = str(uuid4())  # Assign a unique complex ID
        self.chromosome = chromosome
        self.fitness = fitness
        self.solver_reports = solver_reports  # Initialize solver_reports as an empty list

    def __repr__(self):
        return f"Individual(id={self.id}, chromosome={self.chromosome}, fitness={self.fitness}, solver_reports={self.solver_reports})"

    def add_parents(self, parents: List["Individual"], generator: str):
        self.generator = generator
        if parents:
            self.parents_id = [parent.id for parent in parents]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "chromosome": self.chromosome,
            "fitness": self.fitness,
            "solver_reports": self.solver_reports,
            "generator": getattr(self, "generator", None),
            "parents_id": getattr(self, "parents_id", None),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Individual":
        individual = cls(chromosome=data.get("chromosome"), fitness=data.get("fitness"))
        individual.id = data.get("id", individual.id)  # Restore ID if provided
        individual.solver_reports = data.get("solver_reports", [])
        if "generator" in data:
            individual.generator = data["generator"]
        if "parents_id" in data:
            individual.parents_id = data["parents_id"]
        return individual
