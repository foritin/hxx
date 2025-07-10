from vnpy.alpha import AlphaLab
from tools import PathManager


class Lab:

    def __init__(self, lab_path: str = None):
        if not lab_path:
            lab_path = PathManager.get_alphalab_path()
        self.lab = AlphaLab(lab_path=lab_path)


if __name__ == "__main__":
    lab = Lab()
    print(lab.lab)
