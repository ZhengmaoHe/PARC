import yaml

import learning.ppo_agent as ppo_agent
# import learning.dqn_agent as dqn_agent
# import learning.sac_agent as sac_agent
# import learning.awr_agent as awr_agent
# import learning.amp_agent as amp_agent
# import learning.add_agent as add_agent
# import learning.iql_agent as iql_agent
import learning.dm_ppo_agent as dm_ppo_agent
# import learning.dm_bc_agent as dm_bc_agent
from util.logger import Logger

def build_agent(agent_file, env, device):
    agent_config = load_agent_file(agent_file)
    
    agent_name = agent_config["agent_name"]
    Logger.print("Building {} agent".format(agent_name))

    if (agent_name == ppo_agent.PPOAgent.NAME):
        agent = ppo_agent.PPOAgent(config=agent_config, env=env, device=device)
    elif (agent_name == dm_ppo_agent.DMPPOAgent.NAME):
        agent = dm_ppo_agent.DMPPOAgent(config=agent_config, env=env, device=device)
    else:
        assert(False), "Unsupported agent: {}".format(agent_name)

    num_params = agent.calc_num_params()
    Logger.print("Total parameter count: {}".format(num_params))

    return agent

def load_agent_file(file):
    with open(file, "r") as stream:
        agent_config = yaml.safe_load(stream)
    return agent_config
