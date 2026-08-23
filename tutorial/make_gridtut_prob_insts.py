from typing import Dict
from domains.grid_tutorial import GridState, GridGoal
import pickle


def main():
    data: Dict = dict()
    data['states'] = [GridState(0,0), GridState(1,1)]
    data['goals'] = [GridGoal(6,6), GridGoal(5,5)]

    pickle.dump(data, open("tutorial/grid_tut/custom_insts.pkl", "wb"), protocol=-1)


if __name__ == "__main__":
    main()
