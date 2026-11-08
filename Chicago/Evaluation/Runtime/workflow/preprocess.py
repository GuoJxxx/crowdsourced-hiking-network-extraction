"""Regenerate the cleaned regional inputs from the published observations."""
import argparse
from .common import finish, path, intermediate
from .pipeline import mainling_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args(argv)
    steps = mainling_pipeline(intermediate('mainling')/'proposed')[:1]
    return finish({'group': 'mainling', 'steps': steps}, args)


if __name__ == '__main__':
    raise SystemExit(main())
