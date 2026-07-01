# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Fruit-Harvest environment.
"""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Fruit-Harvest-Direct-v0",
    entry_point=f"{__name__}.fruit_harvest_env:FruitHarvestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fruit_harvest_env:FruitHarvestEnvCfg",
    },
)

gym.register(
    id="Isaac-Fruit-Harvest-Greenhouse-Direct-v0",
    entry_point=f"{__name__}.fruit_harvest_env:FruitHarvestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fruit_harvest_env:FrankaGreenhouseFruitHarvestEnvCfg",
    },
)

gym.register(
    id="Isaac-Fruit-Harvest-UR5e-Direct-v0",
    entry_point=f"{__name__}.fruit_harvest_env:FruitHarvestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fruit_harvest_env:UR5eFruitHarvestEnvCfg",
    },
)

gym.register(
    id="Isaac-Tomato-Harvest-Direct-v0",
    entry_point=f"{__name__}.tomato_harvest_env:TomatoHarvestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tomato_harvest_env:TomatoHarvestEnvCfg",
    },
)
