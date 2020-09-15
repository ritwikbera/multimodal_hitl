from gym.envs.registration import register

register(
    id='MultimodalAirSim-v0',
    entry_point='airsim_env.envs:MultimodalAirSimEnv',
)

register(
    id='MultimodalAirSimMountains-v0',
    entry_point='airsim_env.envs:MultimodalAirSimMountainsEnv',
)
