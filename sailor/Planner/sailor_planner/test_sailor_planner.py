from os.path import expanduser

from sailor.Planner.sailor_planner.cpp_src.planner import SailorPlanner


def test_planner():
    home_dir = expanduser("~")

    planner = SailorPlanner(
        profile_file=f"{home_dir}/elastic-spot-ml/sailor/Planner/sailor_planner/profiles/OPT-350/",
        training_config_path= f"{home_dir}/elastic-spot-ml/sailor/Planner/simulations/tests/training_config_opt_350.json",
        quotas_path_dict=f"{home_dir}/elastic-spot-ml/sailor/Planner/sailor_planner/dummy_quotas_dict.json",
        objective="throughput",
        max_cost=1000,
        min_throughput_percentage=0.1,
        fp16=False
    )

    cluster_config = {
        "A100-40": 64,
        "V100-16": 64
    }

    plans = planner.get_sorted_plans(training_config={}, cluster_config=cluster_config)
    if len(plans) > 0:
        print(plans[0])

test_planner()
