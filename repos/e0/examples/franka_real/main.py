import dataclasses
import logging
import tyro

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent

import env as _env

@dataclasses.dataclass
class Args:

    host : str = "0.0.0.0"
    port : int = 10000

    max_hz: int = 30
    num_episodes: int = 20
    max_episode_steps: int = 2000
    camera_num : int = 2
    action_horizon : int = 10 # 推理的action步长

    final_retry_times : int = 0
    task_name: str = "pick_block"
    task_prompt : str = "Pick up the object on the table."
    motion_mode : str = "abs"

    smooth_actions : bool = False
    start_wo_reset : bool = False
    convert_bgr_to_rgb: bool = True

    use_policy_metadata : bool = False 
    action_space : str = "joint"
    state_space : str = "joint"
    use_quat : bool = False
    cal_euler_order : str = "zyx"
    use_degrees : bool = False



def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port,)
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()

    if args.use_policy_metadata:
        print("Use metadata infomation !")
        args.action_space = metadata["action_space"]
        args.state_space = metadata["state_space"]
        args.use_quat = metadata["use_quat"]
        args.cal_euler_order = metadata["cal_euler_order"]
        args.use_degrees = metadata["use_degrees"]


    env = _env.FrankaRealEnvironment(**dataclasses.asdict(args))

    agent = _policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        )

    runtime = _runtime.Runtime(
        environment = env,
        agent = agent,
        subscribers = [],
        max_hz = args.max_hz,
        num_episodes = args.num_episodes,
        max_episode_steps = args.max_episode_steps,
    )
    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)