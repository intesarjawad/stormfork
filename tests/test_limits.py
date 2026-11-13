import pytest

from stormfork.export.manifest import HINDCAST_LIMIT
from stormfork.scenarios import Scenario, load_scenarios

TRAINING_LIMIT = (
    "This storm is inside the model's training period (data through 2023), so the baseline "
    "is not an independent forecast; the fork shows how the trained model responds to the edit."
)
TRAINING_END_YEAR = 2023
INSIDE_TRAINING = {"sandy_east_300km", "sandy_sweep", "katrina_scale_0p7"}


@pytest.fixture(scope="module")
def scenarios() -> dict[str, Scenario]:
    return {scenario.id: scenario for scenario in load_scenarios()}


def test_the_storms_before_2024_are_sandy_and_katrina(scenarios: dict[str, Scenario]):
    before = {s.id for s in scenarios.values() if s.init[0].year <= TRAINING_END_YEAR}
    assert before == INSIDE_TRAINING


def test_every_storm_before_2024_states_the_training_period_and_no_other_does(
    scenarios: dict[str, Scenario],
):
    for scenario in scenarios.values():
        stated = TRAINING_LIMIT in scenario.limits
        assert stated == (scenario.init[0].year <= TRAINING_END_YEAR), scenario.id


def test_every_hindcast_states_how_its_initial_conditions_were_made(
    scenarios: dict[str, Scenario],
):
    hindcasts = [s for s in scenarios.values() if s.kind == "hindcast"]
    assert len(hindcasts) == 3
    for scenario in hindcasts:
        assert HINDCAST_LIMIT in scenario.limits, scenario.id


def test_every_published_experiment_has_a_title_and_distinct_limits(
    scenarios: dict[str, Scenario],
):
    for scenario in scenarios.values():
        assert scenario.title, scenario.id
        assert scenario.limits, scenario.id
        assert len(set(scenario.limits)) == len(scenario.limits), scenario.id
