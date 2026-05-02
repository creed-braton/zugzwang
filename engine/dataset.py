import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch


@dataclass
class GameMetadata:
    game_id: str
    num_moves: int
    outcome: float
    termination: str


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        model_id: str,
        num_simulations: int,
        iteration: int,
        run_id: str,
        worker_id: int,
    ):
        self.model_id = model_id
        self.num_simulations = num_simulations
        self.iteration = iteration
        self.run_id = run_id
        self.worker_id = worker_id
        self.created_at = datetime.now(timezone.utc).isoformat()

        self.states: list[torch.Tensor] = []
        self.policies: list[torch.Tensor] = []
        self.values: list[float] = []
        self.search_values: list[float] = []
        self.tree_sizes: list[int] = []
        self.top_line_depths: list[int] = []
        self.game_indices: list[int] = []
        self.move_indices: list[int] = []
        self.games: list[GameMetadata] = []

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return self.states[idx], self.policies[idx], self.values[idx]

    def add_game(
        self,
        states: list[torch.Tensor],
        policies: list[torch.Tensor],
        values: torch.Tensor,
        search_values: list[float],
        tree_sizes: list[int],
        top_line_depths: list[int],
        outcome: float,
        termination: str,
    ) -> str:
        n = len(states)
        if (
            len(policies) != n
            or len(values) != n
            or len(search_values) != n
            or len(tree_sizes) != n
            or len(top_line_depths) != n
        ):
            raise ValueError(
                f"add_game: length mismatch — states={n}, "
                f"policies={len(policies)}, values={len(values)}, "
                f"search_values={len(search_values)}, "
                f"tree_sizes={len(tree_sizes)}, top_line_depths={len(top_line_depths)}"
            )

        game_id = uuid.uuid4().hex
        game_idx = len(self.games)
        self.games.append(
            GameMetadata(
                game_id=game_id,
                num_moves=n,
                outcome=outcome,
                termination=termination,
            )
        )

        self.states.extend(states)
        self.policies.extend(policies)
        self.values.extend(values.tolist())
        self.search_values.extend(search_values)
        self.tree_sizes.extend(tree_sizes)
        self.top_line_depths.extend(top_line_depths)
        self.game_indices.extend([game_idx] * n)
        self.move_indices.extend(range(n))

        return game_id

    def __add__(self, other: "Dataset") -> "Dataset":
        if self.model_id != other.model_id:
            raise ValueError(
                f"cannot merge datasets with different model_id: "
                f"{self.model_id!r} vs {other.model_id!r}"
            )
        if self.num_simulations != other.num_simulations:
            raise ValueError(
                f"cannot merge datasets with different num_simulations: "
                f"{self.num_simulations} vs {other.num_simulations}"
            )
        if self.iteration != other.iteration:
            raise ValueError(
                f"cannot merge datasets from different iterations: "
                f"{self.iteration} vs {other.iteration}"
            )
        if self.run_id != other.run_id:
            raise ValueError(
                f"cannot merge datasets from different runs: "
                f"{self.run_id!r} vs {other.run_id!r}"
            )

        merged = Dataset(
            self.model_id,
            self.num_simulations,
            self.iteration,
            self.run_id,
            self.worker_id,
        )
        merged.created_at = self.created_at

        merged.states = self.states + other.states
        merged.policies = self.policies + other.policies
        merged.values = self.values + other.values
        merged.search_values = self.search_values + other.search_values
        merged.tree_sizes = self.tree_sizes + other.tree_sizes
        merged.top_line_depths = self.top_line_depths + other.top_line_depths
        merged.games = self.games + other.games

        offset = len(self.games)
        merged.game_indices = self.game_indices + [
            i + offset for i in other.game_indices
        ]
        merged.move_indices = self.move_indices + other.move_indices

        return merged

    def save(self, path: str | Path) -> None:
        if self.states:
            states = torch.stack(self.states)
            policies = torch.stack(self.policies)
        else:
            states = torch.empty(0)
            policies = torch.empty(0)

        torch.save(
            {
                "states": states,
                "policies": policies,
                "values": torch.tensor(self.values, dtype=torch.float32),
                "search_values": torch.tensor(
                    self.search_values, dtype=torch.float32
                ),
                "tree_sizes": torch.tensor(self.tree_sizes, dtype=torch.long),
                "top_line_depths": torch.tensor(
                    self.top_line_depths, dtype=torch.long
                ),
                "game_indices": torch.tensor(
                    self.game_indices, dtype=torch.long
                ),
                "move_indices": torch.tensor(
                    self.move_indices, dtype=torch.long
                ),
                "games": [asdict(g) for g in self.games],
                "metadata": {
                    "model_id": self.model_id,
                    "num_simulations": self.num_simulations,
                    "iteration": self.iteration,
                    "run_id": self.run_id,
                    "worker_id": self.worker_id,
                    "created_at": self.created_at,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        blob = torch.load(path, weights_only=False)
        meta = blob["metadata"]
        ds = cls(
            meta["model_id"],
            meta["num_simulations"],
            meta["iteration"],
            meta["run_id"],
            meta["worker_id"],
        )
        ds.created_at = meta["created_at"]

        states = blob["states"]
        policies = blob["policies"]
        ds.states = list(states) if len(states) > 0 else []
        ds.policies = list(policies) if len(policies) > 0 else []
        ds.values = blob["values"].tolist()
        ds.search_values = blob["search_values"].tolist()
        ds.tree_sizes = blob["tree_sizes"].tolist()
        ds.top_line_depths = blob["top_line_depths"].tolist()
        ds.game_indices = blob["game_indices"].tolist()
        ds.move_indices = blob["move_indices"].tolist()
        ds.games = [GameMetadata(**g) for g in blob["games"]]

        return ds

    @property
    def num_games(self) -> int:
        return len(self.games)

    @property
    def mean_value_error(self) -> float:
        if not self.values:
            return 0.0
        total = sum(
            abs(sv - v) for sv, v in zip(self.search_values, self.values)
        )
        return total / len(self.values)

    @property
    def mean_tree_size(self) -> float:
        if not self.tree_sizes:
            return 0.0
        return sum(self.tree_sizes) / len(self.tree_sizes)

    @property
    def mean_top_line_depth(self) -> float:
        if not self.top_line_depths:
            return 0.0
        return sum(self.top_line_depths) / len(self.top_line_depths)

    @property
    def termination_histogram(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for g in self.games:
            hist[g.termination] = hist.get(g.termination, 0) + 1
        return hist

    @property
    def outcome_histogram(self) -> dict[float, int]:
        hist: dict[float, int] = {}
        for g in self.games:
            hist[g.outcome] = hist.get(g.outcome, 0) + 1
        return hist
