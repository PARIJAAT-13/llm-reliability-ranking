"""
Helpers for statistics unit tests.
"""

from llm_reliability.records.ranking import RankingRecord


def create_mock_ranking(
    agent_scores: dict[str, float],
    ranking_type: str = "success",
    benchmark: str = "mock-bench",
) -> RankingRecord:
    """Create a RankingRecord with given agents and scores."""
    # Convert dict to rankings tuple
    rankings_list = [(agent, score) for agent, score in agent_scores.items()]
    # Sort rankings by score desc, then name asc to match standard structure
    sorted_rankings = sorted(rankings_list, key=lambda x: (-x[1], x[0]))
    rankings_tuple = tuple(sorted_rankings)
    
    # Create rank_map
    rank_map = {agent: idx + 1 for idx, (agent, _) in enumerate(rankings_tuple)}

    return RankingRecord(
        ranking_type=ranking_type,  # type: ignore
        benchmark=benchmark,
        rankings=rankings_tuple,
        rank_map=rank_map,
        computed_at="2026-07-21T02:00:00Z",
    )
