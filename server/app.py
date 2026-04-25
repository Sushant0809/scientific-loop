from openenv.core.env_server.http_server import create_app

try:
    from ..models import ScientificLoopAction, ScientificLoopObservation
    from .ScientificLoop_environment import ScientificLoopEnvironment
except ImportError:
    from models import ScientificLoopAction, ScientificLoopObservation
    from server.ScientificLoop_environment import ScientificLoopEnvironment


app = create_app(
    ScientificLoopEnvironment,
    ScientificLoopAction,
    ScientificLoopObservation,
    env_name="ScientificLoop",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)  # also callable as main()
