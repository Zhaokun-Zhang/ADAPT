"""Entry point. Run from Adapt-code/: python -m ADAPT.main --dataset MovieActor -T 2500"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ADAPT.args import get_exc_args
from ADAPT.dataset import JoinDataset
from ADAPT.iteration import join_iteration
from ADAPT.preprocess import data_preprocess
from ADAPT.utils import show_time


def main():
    show_time("Start.")
    args = get_exc_args()
    dataset = JoinDataset(**args)
    glbinfo = data_preprocess(dataset, **args)
    join_iteration(glbinfo, **args)
    show_time("Finish.")


if __name__ == "__main__":
    main()
